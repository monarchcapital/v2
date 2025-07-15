# pages/Backtesting.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px # Added for potential new charts
from datetime import timedelta, date

import config
from utils import (
    download_data, create_features,
    train_models_pipeline, generate_predictions_pipeline,
    parse_int_list, add_market_data_features,
    add_fundamental_features
)

st.set_page_config(page_title="Monarch: Model Backtest", layout="wide")

st.header("🏆 Walk-Forward Model Backtest")
st.markdown("""
This page performs a rigorous **walk-forward backtest** for multiple models simultaneously.
This method simulates real-world performance by training on a historical period, predicting the next day, then incrementally adding the new day's data to the training set before the next prediction.
The results reveal which model is most historically accurate for the selected stock and feature configuration.
""")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("🛠️ Backtest Configuration")
    ticker = st.text_input("Enter Stock Ticker:", value="AAPL").upper()

    st.subheader("🗓️ Initial Training & Backtest Period")
    st.markdown("Models first train on `t1` to `t2`, then walk forward day-by-day.")
    today = date.today()
    default_t2 = today - timedelta(days=45)
    default_t1 = default_t2 - timedelta(days=365 * 2)
    t1_start_date = st.date_input("Training Start Date (t1)", value=default_t1, key="t1_start")
    t2_end_date = st.date_input("Training End Date (t2)", value=default_t2, key="t2_end")
    backtest_days = st.slider("Days to Backtest (Walk-Forward):", 1, 90, 30)

    if t1_start_date >= t2_end_date:
        st.error("Training Start Date (t1) must be before Training End Date (t2).")
        st.stop()

    st.subheader("🤖 Model Selection")
    selected_models = st.multiselect(
        "Choose Models to Compare:",
        options=[m for m in config.MODEL_CHOICES if m != 'Prophet'],
        default=[m for m in config.MODEL_CHOICES if m not in ['Prophet', 'KNN']]
    )

    st.subheader("➕ Feature Selection")
    with st.expander("🌍 Global & 🔬 Fundamental Features"):
        available_indices_bt = list(config.GLOBAL_MARKET_TICKERS.keys())
        selected_indices_bt = st.multiselect("Select Global Indices:", options=available_indices_bt, default=available_indices_bt[:3], key="bt_global_select")
        selected_tickers_bt = [config.GLOBAL_MARKET_TICKERS[name] for name in selected_indices_bt]

        # Static Fundamentals
        available_fundamentals_static_bt = list(config.FUNDAMENTAL_METRICS.keys())
        select_all_fundamentals_static_bt = st.checkbox("Select All Static Fundamentals", value=False, key="bt_select_all_fundamentals_static")
        default_fundamentals_static_bt = available_fundamentals_static_bt if select_all_fundamentals_static_bt else []
        selected_fundamental_names_static_bt = st.multiselect("Select Static Fundamental Metrics:", options=available_fundamentals_static_bt, default=default_fundamentals_static_bt, key="bt_static_fundamental_select")
        selected_fundamentals_static_bt = {name: config.FUNDAMENTAL_METRICS[name] for name in selected_fundamental_names_static_bt}

        # Historical/Derived Fundamentals
        available_fundamentals_derived_bt = ['Historical P/E Ratio', 'Historical P/S Ratio', 'Historical Debt to Equity']
        select_all_fundamentals_derived_bt = st.checkbox("Select All Historical/Derived Fundamentals", value=False, key="bt_select_all_fundamentals_derived")
        default_fundamentals_derived_bt = available_fundamentals_derived_bt if select_all_fundamentals_derived_bt else []
        selected_fundamental_names_derived_bt = st.multiselect("Select Historical/Derived Metrics:", options=available_fundamentals_derived_bt, default=default_fundamentals_derived_bt, key="bt_derived_fundamental_select")
        selected_fundamentals_derived_bt = {name: name for name in selected_fundamental_names_derived_bt}

    # Combine all selected fundamental metrics
    combined_selected_fundamentals_bt = {**selected_fundamentals_static_bt, **selected_fundamentals_derived_bt}


    st.subheader("⚙️ Technical Indicator Settings")
    selected_indicator_params = {}
    with st.expander("Show Indicator Settings"):
        for indicator_name, (default_value, default_enabled) in config.TECHNICAL_INDICATORS_DEFAULTS.items():
            if st.checkbox(f"Enable {indicator_name.replace('_', ' ')}", value=default_enabled, key=f"enable_{indicator_name.lower()}_backtest"):
                if isinstance(default_value, list):
                    selected_indicator_params[indicator_name] = parse_int_list(st.text_input(f"  {indicator_name.replace('_', ' ')} (days):", value=", ".join(map(str, default_value)), key=f"input_{indicator_name.lower()}_backtest"), default_value, st.sidebar.error)
                elif isinstance(default_value, (int, float)):
                    min_val, step_val = (0.01, 0.01) if 'ACCEL' in indicator_name or isinstance(default_value, float) else (1.0, 1.0)
                    selected_indicator_params[indicator_name] = st.number_input(f"  {indicator_name.replace('_', ' ')}:", min_value=min_val, value=float(default_value), step=step_val, key=f"input_{indicator_name.lower()}_backtest")

    # --- New: Transaction Cost Input ---
    st.subheader("💸 Transaction Costs")
    transaction_cost_pct = st.slider("Transaction Cost per Trade (%)", 0.0, 1.0, 0.1, 0.01, help="Cost as a percentage of trade value (e.g., 0.1 for 0.1%)") / 100

def dummy_log(message):
    """A dummy logging function that does nothing, to keep utils clean."""
    pass

# --- Main Content Area ---
if st.button(f"Run Backtest Comparison", type="primary"):
    if not ticker:
        st.warning("Please enter a stock ticker."); st.stop()
    if not selected_models:
        st.warning("Please select at least one model to compare."); st.stop()

    st.warning(f"Starting walk-forward backtest for {ticker}. This is computationally intensive and may take several minutes.")
    with st.spinner(f"Running multi-model backtest..."):
        # Download all necessary data in one go
        end_of_backtest_period = t2_end_date + timedelta(days=backtest_days + 60) # Add buffer for non-trading days
        num_years = max(1, int(np.ceil((end_of_backtest_period - t1_start_date).days / 365.25)))
        full_data_raw = download_data(ticker, period=f"{num_years}y")
        if full_data_raw.empty:
            st.error(f"Could not download data for {ticker}."); st.stop()

        # Create a comprehensive feature set
        data_for_features = full_data_raw.copy()
        # Pass the combined selected fundamentals
        if combined_selected_fundamentals_bt: data_for_features = add_fundamental_features(data_for_features, ticker, combined_selected_fundamentals_bt, dummy_log)
        if selected_tickers_bt: data_for_features = add_market_data_features(data_for_features, f"{num_years}y", dummy_log, selected_tickers=selected_tickers_bt)
        features_df = create_features(data_for_features, selected_indicator_params)

        # Isolate the actual period for backtesting
        backtest_period_df = features_df[features_df['Date'] > pd.to_datetime(t2_end_date)].copy()
        if len(backtest_period_df) < backtest_days:
            st.warning(f"Only found {len(backtest_period_df)} trading days for backtesting. Adjusting from {backtest_days} days.")
            backtest_days = len(backtest_period_df)
        if backtest_days == 0:
            st.error(f"No trading data available after {t2_end_date} to perform the backtest."); st.stop()

        # --- Walk-Forward Loop ---
        all_results = {model: [] for model in selected_models}
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_predictions = backtest_days * len(selected_models)
        predictions_done = 0

        for i in range(backtest_days):
            # The day we are trying to predict
            current_day_for_prediction = backtest_period_df.iloc[i]
            prediction_date = pd.to_datetime(current_day_for_prediction['Date'])
            
            # The training data is all data *before* the day we are trying to predict
            training_data = features_df[features_df['Date'] < prediction_date]
            if training_data.empty: continue
            
            # The input for the prediction is the last known day of data
            input_for_prediction = features_df[features_df['Date'] < prediction_date].tail(1)
            if input_for_prediction.empty: continue

            for model_name in selected_models:
                status_text.text(f"Day {i+1}/{backtest_days} | Model: {model_name} | Predicting for {prediction_date.date()}...")
                
                trained_models, _ = train_models_pipeline(training_data.copy(), model_name, False, dummy_log, selected_indicator_params)
                if not trained_models: continue

                prediction_dict = generate_predictions_pipeline(input_for_prediction, trained_models, dummy_log)

                predicted_close = prediction_dict.get('Close', {}).get('Predicted Close', pd.Series([np.nan])).iloc[0]

                # --- New: Directional Prediction and PnL Simulation ---
                actual_close = current_day_for_prediction['Close']
                previous_actual_close = input_for_prediction['Close'].iloc[0]

                predicted_direction = np.sign(predicted_close - previous_actual_close)
                actual_direction = np.sign(actual_close - previous_actual_close)

                trade_pnl = 0.0
                if not np.isnan(predicted_close) and not np.isnan(previous_actual_close):
                    # Simple strategy: If predicted up, "buy" and hold for one day. If predicted down, "sell" (short) and hold.
                    # This is a simplified PnL calculation assuming entry at previous_actual_close and exit at actual_close
                    # with a single unit of investment.
                    if predicted_direction > 0: # Predicted up, simulate a "buy"
                        gross_pnl = (actual_close - previous_actual_close)
                        trade_pnl = gross_pnl - (previous_actual_close * transaction_cost_pct) # Cost on entry
                    elif predicted_direction < 0: # Predicted down, simulate a "short"
                        gross_pnl = (previous_actual_close - actual_close) # Profit if price drops
                        trade_pnl = gross_pnl - (previous_actual_close * transaction_cost_pct) # Cost on entry (or exit, depends on broker)
                    # If predicted_direction is 0 (no change), no trade, pnl = 0.

                all_results[model_name].append({
                    'Date': prediction_date.date(),
                    'Predicted Close': predicted_close,
                    'Actual Close': actual_close,
                    'Previous Actual Close': previous_actual_close,
                    'Predicted Direction': predicted_direction,
                    'Actual Direction': actual_direction,
                    'Trade PnL': trade_pnl # New PnL metric
                })
                
                predictions_done += 1
                progress_bar.progress(predictions_done / total_predictions)

        status_text.success(f"Backtest comparison complete!")
        progress_bar.empty()

        # --- Performance Calculation ---
        model_performance_data = []
        for model_name, results_list in all_results.items():
            if not results_list: continue
            
            results_df = pd.DataFrame(results_list).dropna()
            if results_df.empty: continue

            results_df['Model Error'] = (results_df['Predicted Close'] - results_df['Actual Close']).abs()
            results_df['Persistence Error'] = (results_df['Previous Actual Close'] - results_df['Actual Close']).abs()
            
            mae = results_df['Model Error'].mean()
            rmse = np.sqrt((results_df['Model Error']**2).mean())
            
            # Directional accuracy
            dir_accuracy = (results_df['Predicted Direction'] == results_df['Actual Direction']).mean() * 100
            correct_direction_trades = (results_df['Predicted Direction'] == results_df['Actual Direction']).sum()
            incorrect_direction_trades = (results_df['Predicted Direction'] != results_df['Actual Direction']).sum()

            # Win Rate vs Persistence
            win_rate = (results_df['Model Error'] < results_df['Persistence Error']).mean() * 100

            # Net PnL and PnL per trade
            net_pnl = results_df['Trade PnL'].sum()
            num_trades = (results_df['Predicted Direction'] != 0).sum() # Count trades where a direction was predicted
            pnl_per_trade = net_pnl / num_trades if num_trades > 0 else 0

            model_performance_data.append({
                'Model': model_name,
                'MAE': mae,
                'RMSE': rmse, 
                'Directional Accuracy': dir_accuracy,
                'Correct Direction Trades': int(correct_direction_trades), # New
                'Incorrect Direction Trades': int(incorrect_direction_trades), # New
                'Win Rate vs Persistence': win_rate,
                'Net PnL': net_pnl, # New
                'PnL per Trade': pnl_per_trade # New
            })

        if not model_performance_data:
            st.error("Could not generate any backtest results."); st.stop()

        # --- Display Results ---
        st.subheader("🏆 Model Performance Ranking")
        perf_df = pd.DataFrame(model_performance_data).sort_values(by='MAE').reset_index(drop=True)
        st.dataframe(perf_df.style.format({
            'MAE': '{:.4f}', 'RMSE': '{:.4f}', 
            'Directional Accuracy': '{:.2f}%',
            'Win Rate vs Persistence': '{:.2f}%',
            'Net PnL': '{:,.2f}', # Format for currency
            'PnL per Trade': '{:,.2f}' # Format for currency
        }).highlight_min(subset=['MAE', 'RMSE'], color='lightgreen')
          .highlight_max(subset=['Directional Accuracy', 'Win Rate vs Persistence', 'Net PnL', 'PnL per Trade'], color='lightgreen'), # Highlight max for PnL
        use_container_width=True)
        best_model = perf_df.iloc[0]['Model']
        st.success(f"**Conclusion:** Based on the lowest Mean Absolute Error (MAE), **{best_model}** was the best performing model during the backtest period.")

        # --- Visualizations ---
        st.subheader("📊 Visual Analysis of Backtest Results")
        col1, col2 = st.columns(2)

        with col1:
            # Price Comparison Chart
            fig_prices = go.Figure()
            actuals_df = pd.DataFrame(all_results[best_model])[['Date', 'Actual Close']].drop_duplicates()
            fig_prices.add_trace(go.Scatter(x=actuals_df['Date'], y=actuals_df['Actual Close'], mode='lines', name='Actual Close Price', line=dict(color='black', width=3)))
            for model_name in perf_df['Model']:
                model_preds_df = pd.DataFrame(all_results[model_name])
                fig_prices.add_trace(go.Scatter(x=model_preds_df['Date'], y=model_preds_df['Predicted Close'], mode='lines', name=f'Pred. ({model_name})', line=dict(dash='dot')))
            fig_prices.update_layout(title="Actual vs. Predicted Close Prices", yaxis_title="Price", legend_title="Legend", height=500)
            st.plotly_chart(fig_prices, use_container_width=True)

        with col2:
            # Cumulative Error Chart
            fig_error = go.Figure()
            for model_name in perf_df['Model']:
                model_results_df = pd.DataFrame(all_results[model_name]).dropna()
                # Recalculate 'Model Error' here, as model_results_df is a fresh DataFrame
                if not model_results_df.empty: # Ensure there's data to calculate
                    model_results_df['Model Error'] = (model_results_df['Predicted Close'] - model_results_df['Actual Close']).abs()
                    model_results_df['Cumulative Error'] = model_results_df['Model Error'].cumsum()
                    fig_error.add_trace(go.Scatter(x=model_results_df['Date'], y=model_results_df['Cumulative Error'], mode='lines', name=f'Error ({model_name})'))
            fig_error.update_layout(title="Cumulative Absolute Error Over Time", yaxis_title="Cumulative Error", legend_title="Legend", height=500)
            st.plotly_chart(fig_error, use_container_width=True)
        
        # --- New: PnL Chart ---
        st.subheader("💰 Cumulative PnL Over Backtest Period")
        fig_pnl = go.Figure()
        for model_name in perf_df['Model']:
            model_results_df = pd.DataFrame(all_results[model_name]).dropna()
            if not model_results_df.empty: # Ensure there's data to calculate
                model_results_df['Cumulative PnL'] = model_results_df['Trade PnL'].cumsum()
                fig_pnl.add_trace(go.Scatter(x=model_results_df['Date'], y=model_results_df['Cumulative PnL'], mode='lines', name=f'PnL ({model_name})'))
        fig_pnl.update_layout(title="Cumulative PnL Over Time (Simulated)", yaxis_title="Cumulative PnL", legend_title="Legend", height=500)
        st.plotly_chart(fig_pnl, use_container_width=True)

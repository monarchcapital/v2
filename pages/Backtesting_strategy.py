# pages/backtesting.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta, date, datetime

import config
from utils import (
    download_data, create_features, train_models_pipeline,
    generate_predictions_pipeline, add_market_data_features,
    make_ensemble_prediction,
    parse_int_list,
    add_fundamental_features
)

st.set_page_config(page_title="Monarch: Backtesting", layout="wide")

st.header("🔍 Model Backtesting")
st.markdown("Evaluate model performance by training on a historical period and predicting on a subsequent backtesting period. This helps to understand model accuracy and deviation without look-ahead bias.")

# --- Sidebar ---
st.sidebar.header("🛠️ Backtesting Configuration")
ticker = st.sidebar.text_input("Enter Stock Ticker:", value="AAPL").upper()

# --- Date Inputs ---
st.sidebar.subheader("🗓️ Training Period")
today = date.today()
default_training_end = today - timedelta(days=91)
default_training_start = default_training_end - timedelta(days=5*365)
training_start_date = st.sidebar.date_input("Training Start Date:", value=default_training_start)
training_end_date = st.sidebar.date_input("Training End Date:", value=default_training_end)

st.sidebar.subheader("🗓️ Backtesting Period")
default_backtesting_start = training_end_date + timedelta(days=1)
default_backtesting_end = today - timedelta(days=1)
backtesting_start_date = st.sidebar.date_input("Backtesting Start Date:", value=default_backtesting_start)
backtesting_end_date = st.sidebar.date_input("Backtesting End Date:", value=default_backtesting_end)

# --- Date Validation ---
if training_end_date >= backtesting_start_date:
    st.sidebar.error("The training period must end before the backtesting period begins.")
    st.stop()
if training_start_date >= training_end_date:
    st.sidebar.error("Training start date must be before the training end date.")
    st.stop()
if backtesting_start_date >= backtesting_end_date:
    st.sidebar.error("Backtesting start date must be before the backtesting end date.")
    st.stop()

# --- Feature Selection ---
st.sidebar.subheader("➕ Add Features")
with st.sidebar.expander("Feature & Indicator Settings", expanded=True):
    # Global Context
    available_indices = list(config.GLOBAL_MARKET_TICKERS.keys())
    selected_indices = st.multiselect("Select Global Indices:", options=available_indices, default=available_indices[:2], key="backtest_global_select")
    selected_tickers = [config.GLOBAL_MARKET_TICKERS[name] for name in selected_indices]

    # Fundamentals
    available_fundamentals_static = list(config.FUNDAMENTAL_METRICS.keys())
    select_all_static = st.checkbox("Select All Static Fundamentals", value=False, key="backtest_select_all_static")
    default_static = available_fundamentals_static if select_all_static else []
    selected_fundamental_names_static = st.multiselect("Select Static Fundamental Metrics:", options=available_fundamentals_static, default=default_static, key="backtest_static_fundamental_select")
    selected_fundamentals_static = {name: config.FUNDAMENTAL_METRICS[name] for name in selected_fundamental_names_static}
    
    available_fundamentals_derived = ['Historical P/E Ratio', 'Historical P/S Ratio', 'Historical Debt to Equity']
    select_all_derived = st.checkbox("Select All Historical/Derived Fundamentals", value=False, key="backtest_select_all_derived")
    default_derived = available_fundamentals_derived if select_all_derived else []
    selected_fundamental_names_derived = st.multiselect("Select Historical/Derived Metrics:", options=available_fundamentals_derived, default=default_derived, key="backtest_derived_fundamental_select")
    selected_fundamentals_derived = {name: name for name in selected_fundamental_names_derived}
    combined_selected_fundamentals = {**selected_fundamentals_static, **selected_fundamentals_derived}

    # Technical Indicators
    st.markdown("---")
    st.markdown("**Technical Indicators**")
    selected_indicator_params = {}
    for indicator_name, (default_value, default_enabled) in config.TECHNICAL_INDICATORS_DEFAULTS.items():
        if st.checkbox(f"Enable {indicator_name.replace('_', ' ')}", value=default_enabled, key=f"enable_{indicator_name.lower()}_bt"):
            if isinstance(default_value, list):
                selected_indicator_params[indicator_name] = parse_int_list(st.text_input(f"  {indicator_name} (days):", ", ".join(map(str, default_value)), key=f"input_{indicator_name.lower()}_bt"), default_value, st.sidebar.error)
            elif isinstance(default_value, (int, float)):
                min_val, step_val = (0.01, 0.01) if 'ACCEL' in indicator_name or isinstance(default_value, float) else (1.0, 1.0)
                selected_indicator_params[indicator_name] = st.number_input(f"  {indicator_name}:", min_value=min_val, value=float(default_value), step=step_val, key=f"input_{indicator_name.lower()}_bt")

# --- Model Selection ---
st.sidebar.subheader("🤖 Model Selection")
available_models = [m for m in config.MODEL_CHOICES if m != 'Prophet']
selected_models = st.sidebar.multiselect("Select Models to Backtest:", options=available_models, default=available_models[:2])
perform_tuning = st.sidebar.checkbox("Perform Hyperparameter Tuning", value=False, help="Optimizes model parameters for each model selected. May significantly increase processing time.")

# --- Ensemble Configuration ---
st.sidebar.subheader("🤖 Ensemble Prediction")
enable_ensemble = st.sidebar.checkbox("Enable Ensemble Prediction", value=True)
ensemble_models = []
if enable_ensemble:
    ensemble_models = st.sidebar.multiselect(
        "Select Models for Ensemble:",
        options=available_models,
        default=available_models[:3],
        key="backtest_ensemble_models"
    )
enable_confidence_interval = st.sidebar.checkbox("Enable Confidence Intervals for Ensemble", value=True, key="backtest_ci")
confidence_level_pct = 90
if enable_confidence_interval:
    confidence_level_pct = st.sidebar.slider("Confidence Level (%):", 70, 99, 90, 1, key="backtest_ci_slider")

# --- Main Application Flow ---
if st.button("Run Backtest", type="primary"):
    if not selected_models and not (enable_ensemble and ensemble_models):
        st.error("Please select at least one individual model or configure an ensemble to backtest.")
        st.stop()

    log_expander = st.expander("📜 Processing Log", expanded=True)
    log_area = log_expander.empty()
    log_messages = []
    def update_log(message):
        log_messages.insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        log_area.code("\n".join(log_messages))

    update_log(f"Starting backtest for {ticker}...")
    data = download_data(ticker)
    if data.empty:
        st.error(f"Could not load data for {ticker}. Please check the ticker symbol."); st.stop()
    update_log(f"Data loaded: {len(data)} rows.")

    with st.spinner("Processing data and creating features..."):
        data_for_features = data.copy()
        if combined_selected_fundamentals: data_for_features = add_fundamental_features(data_for_features, ticker, combined_selected_fundamentals, _update_log_func=update_log)
        if selected_tickers: data_for_features = add_market_data_features(data_for_features, "10y", update_log, selected_tickers=selected_tickers)
        df_features_full = create_features(data_for_features, selected_indicator_params)
        update_log(f"Features created. Rows after NaN drop: {len(df_features_full)}")

    # --- Data Slicing ---
    df_train_period = df_features_full[(df_features_full['Date'] >= pd.to_datetime(training_start_date)) & (df_features_full['Date'] <= pd.to_datetime(training_end_date))].copy()
    df_backtest_period = df_features_full[(df_features_full['Date'] >= pd.to_datetime(backtesting_start_date)) & (df_features_full['Date'] <= pd.to_datetime(backtesting_end_date))].copy()
    
    if df_train_period.empty or df_backtest_period.empty:
        st.error("No data in the selected training or backtesting date ranges. Please adjust the dates."); st.stop()
    update_log(f"Training data: {len(df_train_period)} rows. Backtesting data: {len(df_backtest_period)} rows.")

    # --- Model Training and Prediction ---
    all_trained_models = {}
    all_backtest_predictions = {}
    model_performance_data = []

    models_to_train = sorted(list(set(selected_models + (ensemble_models if enable_ensemble else []))))

    for model_name in models_to_train:
        with st.spinner(f"Processing {model_name}..."):
            update_log(f"Training {model_name}...")
            trained_models, _ = train_models_pipeline(df_train_period.copy(), model_name, perform_tuning, update_log, selected_indicator_params)
            if not trained_models:
                update_log(f"✗ Failed to train {model_name}."); continue
            
            all_trained_models[model_name] = trained_models
            update_log(f"✓ {model_name} trained successfully.")

            if model_name in selected_models:
                update_log(f"Generating backtest predictions with {model_name}...")
                preds_dict = generate_predictions_pipeline(df_backtest_period.copy(), trained_models, update_log)
                if 'Close' in preds_dict and not preds_dict['Close'].empty:
                    merged_df = pd.merge(df_backtest_period[['Date', 'Close']], preds_dict['Close'][['Date', 'Predicted Close']], on='Date', how='inner')
                    all_backtest_predictions[model_name] = merged_df
                    
                    mae = np.mean(np.abs(merged_df['Close'] - merged_df['Predicted Close']))
                    rmse = np.sqrt(np.mean((merged_df['Close'] - merged_df['Predicted Close'])**2))
                    model_performance_data.append({'Model': model_name, 'MAE': mae, 'RMSE': rmse})
                    update_log(f"✓ Backtest predictions generated for {model_name}. MAE: {mae:.4f}")
                else:
                    update_log(f"✗ Failed to generate predictions for {model_name}.")

    # --- Ensemble Prediction Logic ---
    ensemble_backtest_prediction = None
    if enable_ensemble and ensemble_models:
        update_log("Creating ensemble prediction...")
        ensemble_trained_list = [all_trained_models[m] for m in ensemble_models if m in all_trained_models]
        if ensemble_trained_list:
            ens_close, ens_lower, ens_upper = make_ensemble_prediction(
                ensemble_trained_list, 
                df_backtest_period.copy(), 
                update_log, 
                confidence_level_pct if enable_confidence_interval else None
            )
            
            ensemble_backtest_prediction = pd.DataFrame({
                'Date': df_backtest_period['Date'].values,
                'Ensemble Predicted Close': ens_close,
                'Ensemble Lower CI': ens_lower,
                'Ensemble Upper CI': ens_upper
            })
            
            merged_ens_df = pd.merge(df_backtest_period[['Date', 'Close']], ensemble_backtest_prediction, on='Date', how='inner')
            mae_ens = np.mean(np.abs(merged_ens_df['Close'] - merged_ens_df['Ensemble Predicted Close']))
            rmse_ens = np.sqrt(np.mean((merged_ens_df['Close'] - merged_ens_df['Ensemble Predicted Close'])**2))
            model_performance_data.append({'Model': 'Ensemble', 'MAE': mae_ens, 'RMSE': rmse_ens})
            update_log(f"✓ Ensemble predictions generated. MAE: {mae_ens:.4f}")
        else:
            update_log("✗ Could not create ensemble, not enough models were trained successfully.")

    # --- Section 1: Backtest Performance Chart ---
    st.subheader("Backtest Performance: Actual vs. Predicted Prices")
    st.markdown("This chart compares the models' predictions against the actual stock price over the defined backtesting period.")
    
    if all_backtest_predictions or ensemble_backtest_prediction is not None:
        fig = go.Figure()
        actuals_df = df_backtest_period[['Date', 'Close']].copy()
        fig.add_trace(go.Scatter(x=actuals_df['Date'], y=actuals_df['Close'], mode='lines', name='Actual Price', line=dict(color='black', width=3)))
        
        for model_name, preds_df in all_backtest_predictions.items():
            fig.add_trace(go.Scatter(x=preds_df['Date'], y=preds_df['Predicted Close'], mode='lines', name=f'{model_name} Prediction', line=dict(dash='dot')))
        
        if ensemble_backtest_prediction is not None:
            fig.add_trace(go.Scatter(x=ensemble_backtest_prediction['Date'], y=ensemble_backtest_prediction['Ensemble Predicted Close'], mode='lines', name='Ensemble Prediction', line=dict(color='blue', width=2)))
            if enable_confidence_interval and 'Ensemble Lower CI' in ensemble_backtest_prediction.columns:
                fig.add_trace(go.Scatter(
                    x=ensemble_backtest_prediction['Date'], 
                    y=ensemble_backtest_prediction['Ensemble Upper CI'],
                    mode='lines', name='Ensemble Upper CI', line=dict(width=0), showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=ensemble_backtest_prediction['Date'], 
                    y=ensemble_backtest_prediction['Ensemble Lower CI'],
                    mode='lines', name='Ensemble Confidence Interval', line=dict(width=0),
                    fill='tonexty', fillcolor='rgba(0,100,80,0.2)'
                ))

        fig.update_layout(title=f'{ticker} Backtest Results', yaxis_title='Price', height=600, template='plotly_white', legend_title="Legend")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not generate any backtest predictions. Please check the logs.")

    # --- Section 2: Detailed Comparison Table ---
    st.subheader("Detailed Prediction Comparison")
    st.markdown("The table below shows the actual price, predicted price, and the deviation for each model on each day of the backtest.")

    if all_backtest_predictions or ensemble_backtest_prediction is not None:
        final_table = df_backtest_period[['Date', 'Close']].rename(columns={'Close': 'Actual Close'}).copy()
        for model_name, preds_df in all_backtest_predictions.items():
            preds_subset = preds_df[['Date', 'Predicted Close']].rename(columns={'Predicted Close': f'Predicted ({model_name})'})
            final_table = pd.merge(final_table, preds_subset, on='Date', how='left')
            final_table[f'Deviation ({model_name})'] = final_table[f'Predicted ({model_name})'] - final_table['Actual Close']
        
        if ensemble_backtest_prediction is not None:
            final_table = pd.merge(final_table, ensemble_backtest_prediction, on='Date', how='left')
            final_table['Ensemble Deviation'] = final_table['Ensemble Predicted Close'] - final_table['Actual Close']
            final_table['Ensemble Error %'] = (final_table['Ensemble Deviation'] / final_table['Actual Close']) * 100

        format_dict = {'Actual Close': '{:.2f}'}
        for col in final_table.columns:
            if 'Predicted' in col or 'Deviation' in col or 'CI' in col:
                format_dict[col] = '{:,.2f}'
            if 'Error' in col:
                format_dict[col] = '{:,.2f}%'
        
        st.dataframe(final_table.style.format(format_dict), use_container_width=True)
    else:
        st.info("No data to display. Run the backtest to see detailed results.")
        
    # --- Section 3: Performance Metrics Summary ---
    st.subheader("Model Performance Metrics")
    st.markdown("Lower Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) values indicate better model performance over the backtesting period.")
    
    if model_performance_data:
        perf_df = pd.DataFrame(model_performance_data).sort_values(by='MAE')
        st.dataframe(perf_df.style.format({'MAE': '{:.4f}', 'RMSE': '{:.4f}'}), use_container_width=True)
    else:
        st.info("No performance metrics to display.")

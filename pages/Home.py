# pages/Home.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import timedelta, date, datetime
import yfinance as yf

import config
from utils import (
    download_data, create_features, train_models_pipeline,
    generate_predictions_pipeline, add_market_data_features,
    generate_iterative_forecast,
    parse_int_list,
    add_fundamental_features
)

st.set_page_config(page_title="Monarch: Stock Price Predictor", layout="wide")

st.header("📈 Single Stock Analysis & Predictor")
st.markdown("Analyze a single stock in-depth, visualize its historical data with technical indicators, and forecast future prices using machine learning.")

# --- Helper Functions for New Sections ---
def get_feature_importance_explanation(df_importance, ticker):
    """Generates a plain-English explanation of the top features."""
    if df_importance.empty:
        return "Feature importance data is not available for this model."

    top_feature = df_importance.iloc[0]['Feature']
    top_importance = df_importance.iloc[0]['Importance']
    
    explanation = f"The model's prediction for **{ticker}** is most heavily influenced by the **'{top_feature.replace('_', ' ')}'** feature, which has an importance score of {top_importance:.2f}. "

    if 'Lag' in top_feature:
        explanation += "This means the model is primarily relying on the stock's own price from a few days ago, indicating strong short-term momentum is a key factor."
    elif 'MA_' in top_feature:
        explanation += "This suggests that the stock's recent trend, as captured by this moving average, is the most critical factor in its future price direction."
    elif 'RSI' in top_feature:
        explanation += "This indicates that the model considers the stock's overbought or oversold condition to be the primary driver of its next price movement."
    elif 'Volume' in top_feature:
        explanation += "This suggests that recent trading volume is a crucial indicator for the model, likely signaling the strength or conviction behind price movements."
    elif any(sub in top_feature for sub in ['P/E', 'P/S', 'ROE', 'Debt_to_Equity']):
         explanation += "This highlights that a core fundamental metric is driving the prediction, meaning the model sees the company's underlying financial health as key."
    else:
        explanation += "This technical indicator is playing a crucial role in the model's decision-making process."

    return explanation

def display_fundamental_analysis(ticker_symbol, info):
    """Displays a comprehensive summary of fundamental data and valuation outlook."""
    st.subheader("🔬 Fundamental Analysis & Long-Term Outlook")
    
    if not info or info.get('quoteType') != 'EQUITY':
        st.warning(f"Could not retrieve sufficient fundamental data for {ticker_symbol} to perform analysis.")
        return

    tab1, tab2, tab3 = st.tabs(["Key Ratios & Metrics", "Valuation Outlook", "Analyst Consensus"])

    with tab1:
        # Key Ratios
        col1, col2, col3 = st.columns(3)
        pe = info.get('trailingPE')
        ps = info.get('priceToSalesTrailing12Months')
        pb = info.get('priceToBook')
        peg = info.get('pegRatio')
        roe = info.get('returnOnEquity')
        debt_equity = info.get('debtToEquity')
        profit_margin = info.get('profitMargins')
        dividend_yield = info.get('dividendYield')

        col1.metric("P/E Ratio", f"{pe:.2f}" if isinstance(pe, (int, float)) else "N/A")
        col1.metric("P/S Ratio", f"{ps:.2f}" if isinstance(ps, (int, float)) else "N/A")
        col1.metric("P/B Ratio", f"{pb:.2f}" if isinstance(pb, (int, float)) else "N/A")
        
        col2.metric("PEG Ratio", f"{peg:.2f}" if isinstance(peg, (int, float)) else "N/A")
        col2.metric("Return on Equity", f"{roe*100:.2f}%" if isinstance(roe, (int, float)) else "N/A")
        col2.metric("Profit Margin", f"{profit_margin*100:.2f}%" if isinstance(profit_margin, (int, float)) else "N/A")

        col3.metric("Debt-to-Equity", f"{debt_equity/100:.2f}" if isinstance(debt_equity, (int, float)) else "N/A")
        col3.metric("Dividend Yield", f"{dividend_yield*100:.2f}%" if isinstance(dividend_yield, (int, float)) else "N/A")

    with tab2:
        # Valuation Outlook
        st.markdown("##### Illustrative Value Estimates")
        col1, col2 = st.columns(2)
        
        # P/E Based Valuation
        eps = info.get('trailingEps')
        sector = info.get('sector', 'N/A')
        sector_pe_map = {
            'Technology': 28, 'Healthcare': 25, 'Financial Services': 15,
            'Consumer Cyclical': 20, 'Industrials': 22, 'Energy': 12,
            'Consumer Defensive': 20, 'Real Estate': 18, 'Utilities': 19,
            'Communication Services': 18, 'Basic Materials': 16
        }
        peer_pe = sector_pe_map.get(sector, 20)
        
        with col1:
            st.markdown("**Based on P/E Ratio**")
            if eps:
                estimated_value_pe = eps * peer_pe
                st.metric(f"Value (Sector P/E of {peer_pe})", f"${estimated_value_pe:.2f}")
            else:
                st.warning("EPS data unavailable.")
        
        # P/S Based Valuation
        revenue_per_share = info.get('revenuePerShare')
        industry_ps_map = { # Using different values for P/S
            'Technology': 8, 'Healthcare': 5, 'Financial Services': 3,
            'Consumer Cyclical': 2, 'Industrials': 1.5, 'Energy': 1,
            'Consumer Defensive': 1.2, 'Real Estate': 4, 'Utilities': 2,
            'Communication Services': 4, 'Basic Materials': 1.5
        }
        peer_ps = industry_ps_map.get(info.get('industry', ''), 3)

        with col2:
            st.markdown("**Based on P/S Ratio**")
            if revenue_per_share:
                estimated_value_ps = revenue_per_share * peer_ps
                st.metric(f"Value (Industry P/S of {peer_ps})", f"${estimated_value_ps:.2f}")
            else:
                st.warning("Revenue/Share data unavailable.")

        st.info("These are simplified valuations based on industry averages and are for illustrative purposes only. They are not financial advice.")

    with tab3:
        # Analyst Consensus
        st.markdown("##### Wall Street Analyst Ratings")
        recommendation = info.get('recommendationKey', 'N/A').replace('_', ' ').title()
        target_mean = info.get('targetMeanPrice')
        target_high = info.get('targetHighPrice')
        target_low = info.get('targetLowPrice')
        num_analysts = info.get('numberOfAnalystOpinions')

        if recommendation != 'N/A':
            st.metric(f"Consensus Recommendation ({num_analysts} Analysts)", recommendation)
        else:
            st.info("No analyst recommendations available.")

        if target_mean and target_low and target_high:
            fig = go.Figure()
            current_price = info.get('currentPrice', target_mean)
            fig.add_trace(go.Indicator(
                mode = "number+gauge+delta",
                value = target_mean,
                delta = {'reference': current_price},
                domain = {'x': [0.25, 1], 'y': [0.7, 1]},
                title = {'text': "Mean Target Price vs Current"},
                gauge = {
                    'shape': "bullet",
                    'axis': {'range': [target_low, target_high]},
                    'threshold': {
                        'line': {'color': "red", 'width': 2},
                        'thickness': 0.75,
                        'value': current_price},
                    'steps': [
                        {'range': [target_low, target_mean], 'color': "lightgray"},
                        {'range': [target_mean, target_high], 'color': "darkgray"}],
                    'bar': {'color': 'blue'}}))
            fig.update_layout(height=150, margin={'t':0, 'b':0, 'l':0, 'r':0})
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"Analysts have set price targets ranging from a low of **${target_low:.2f}** to a high of **${target_high:.2f}**.")
        else:
            st.info("No analyst price targets available.")

# --- Sidebar ---
st.sidebar.header("🛠️ Configuration Panel")
ticker = st.sidebar.text_input("Enter Stock Ticker:", value="AAPL").upper()
st.sidebar.subheader("🗓️ Training Period")
today = date.today()
default_end_bt = today - timedelta(days=1)
default_start_bt = default_end_bt - timedelta(days=5*365)
start_bt = st.sidebar.date_input("Training Start Date (t1):", value=default_start_bt)
end_bt = st.sidebar.date_input("Training End Date (t2):", value=default_end_bt)

if start_bt >= end_bt: st.sidebar.error("Start Date must be before End Date."); st.stop()

st.sidebar.subheader("➕ Add Features")
with st.sidebar.expander("🌍 Global Context Features"):
    available_indices = list(config.GLOBAL_MARKET_TICKERS.keys())
    select_all_globals = st.checkbox("Select All Global Indices", value=False, key="home_select_all_globals")
    default_globals = available_indices if select_all_globals else available_indices[:3]
    selected_indices = st.multiselect("Select Global Indices:", options=available_indices, default=default_globals)
    selected_tickers = [config.GLOBAL_MARKET_TICKERS[name] for name in selected_indices]

with st.sidebar.expander("🔬 Fundamental Features"):
    available_fundamentals_static = list(config.FUNDAMENTAL_METRICS.keys())
    select_all_fundamentals_static = st.checkbox("Select All Static Fundamentals", value=False, key="home_select_all_fundamentals_static")
    default_fundamentals_static = available_fundamentals_static if select_all_fundamentals_static else []
    selected_fundamental_names_static = st.multiselect("Select Static Fundamental Metrics:", options=available_fundamentals_static, default=default_fundamentals_static, key="home_static_fundamental_select")
    selected_fundamentals_static = {name: config.FUNDAMENTAL_METRICS[name] for name in selected_fundamental_names_static}
    available_fundamentals_derived = ['Historical P/E Ratio', 'Historical P/S Ratio', 'Historical Debt to Equity']
    select_all_fundamentals_derived = st.checkbox("Select All Historical/Derived Fundamentals", value=False, key="home_select_all_fundamentals_derived")
    default_fundamentals_derived = available_fundamentals_derived if select_all_fundamentals_derived else []
    selected_fundamental_names_derived = st.multiselect("Select Historical/Derived Metrics:", options=available_fundamentals_derived, default=default_fundamentals_derived, key="home_derived_fundamental_select")
    selected_fundamentals_derived = {name: name for name in selected_fundamental_names_derived}
    combined_selected_fundamentals = {**selected_fundamentals_static, **selected_fundamentals_derived}

st.sidebar.subheader("🤖 Model Selection")
model_choice = st.sidebar.selectbox("Select Main Model:", config.MODEL_CHOICES)
perform_tuning = st.sidebar.checkbox("Perform Hyperparameter Tuning", value=False, help="Optimizes model parameters. May significantly increase training time.")
n_future = st.sidebar.slider("Predict Future Days:", 1, 90, config.DEFAULT_N_FUTURE_DAYS)

st.sidebar.subheader("📊 Model Comparison")
available_models_for_comparison = [m for m in config.MODEL_CHOICES if m != 'Prophet']
select_all_models = st.sidebar.checkbox("Select All Models for Comparison", value=False, key="home_select_all_models")
default_compare_models = available_models_for_comparison if select_all_models else [m for m in config.MODEL_CHOICES if m not in ['Prophet', 'KNN']][:3]
compare_models = st.sidebar.multiselect("Select Models to Compare:", options=available_models_for_comparison, default=default_compare_models)
train_days_comparison = st.sidebar.slider("Recent Data for Comparison (days):", 30, 1000, config.DEFAULT_RECENT_DATA_FOR_COMPARISON, 10)

st.sidebar.subheader("⚙️ Technical Indicator Settings")
selected_indicator_params = {}
with st.sidebar.expander("Show Indicator Settings"):
    for indicator_name, (default_value, default_enabled) in config.TECHNICAL_INDICATORS_DEFAULTS.items():
        if st.checkbox(f"Enable {indicator_name.replace('_', ' ')}", value=default_enabled, key=f"enable_{indicator_name.lower()}_home"):
            if isinstance(default_value, list):
                selected_indicator_params[indicator_name] = parse_int_list(st.text_input(f"  {indicator_name} (days):", ", ".join(map(str, default_value)), key=f"input_{indicator_name.lower()}_home"), default_value, st.sidebar.error)
            elif isinstance(default_value, (int, float)):
                min_val, step_val = (0.01, 0.01) if 'ACCEL' in indicator_name or isinstance(default_value, float) else (1.0, 1.0)
                selected_indicator_params[indicator_name] = st.number_input(f"  {indicator_name}:", min_value=min_val, value=float(default_value), step=step_val, key=f"input_{indicator_name.lower()}_home")

if 'training_log' not in st.session_state: st.session_state.training_log = []
def clear_log(): st.session_state.training_log = []
def update_log(message): st.session_state.training_log.insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

# --- Main Application Flow ---
if st.button("Run Analysis", type="primary", on_click=clear_log):
    update_log(f"Starting analysis for {ticker}...")
    
    # Download data and info concurrently
    with st.spinner("Downloading data..."):
        data = download_data(ticker)
        try:
            stock_info = yf.Ticker(ticker).info
        except Exception:
            stock_info = {}
    
    if data.empty: st.error(f"Could not load data for {ticker}. Please check the ticker symbol."); st.stop()
    update_log(f"Data loaded: {len(data)} rows.")
    
    with st.spinner("Processing data and creating features..."):
        data_for_features = data.copy()
        if combined_selected_fundamentals: data_for_features = add_fundamental_features(data_for_features, ticker, combined_selected_fundamentals, _update_log_func=update_log)
        if selected_tickers: data_for_features = add_market_data_features(data_for_features, "10y", update_log, selected_tickers=selected_tickers)
        df_features_full = create_features(data_for_features, selected_indicator_params)
        update_log(f"Features created. Rows after NaN drop: {len(df_features_full)}")
        df_train_period = df_features_full[(df_features_full['Date'] >= pd.to_datetime(start_bt)) & (df_features_full['Date'] <= pd.to_datetime(end_bt))].copy()
        if df_train_period.empty: st.error("No data in selected training range. Please adjust the dates."); st.stop()
        update_log(f"Training data prepared: {len(df_train_period)} rows.")

    all_trained_models, all_future_forecasts, all_historical_predictions, model_performance_data = {}, {}, {}, []
    models_to_run = sorted(list(set([model_choice] + compare_models)))

    for model_name in models_to_run:
        with st.spinner(f"Processing {model_name}..."):
            update_log(f"Training {model_name}...")
            trained_models, _ = train_models_pipeline(df_train_period.copy(), model_name, (model_name == model_choice and perform_tuning), update_log, selected_indicator_params)
            if not trained_models: update_log(f"✗ Failed to train {model_name}."); continue
            all_trained_models[model_name] = trained_models
            update_log(f"✓ {model_name} trained successfully.")
            if model_name != 'Prophet':
                update_log(f"Generating {n_future}-day forecast with {model_name}...")
                future_df = generate_iterative_forecast(data, trained_models, ticker, n_future, end_bt, selected_indicator_params, update_log, selected_tickers, combined_selected_fundamentals)
                if not future_df.empty: all_future_forecasts[model_name] = future_df; update_log(f"✓ Forecast generated for {model_name}.")
                df_comparison_period = df_features_full[(df_features_full['Date'] >= pd.to_datetime(end_bt) - timedelta(days=train_days_comparison)) & (df_features_full['Date'] <= pd.to_datetime(end_bt))]
                if len(df_comparison_period) > 1:
                    preds_dict = generate_predictions_pipeline(df_comparison_period.copy(), trained_models, (lambda x: None))
                    if 'Close' in preds_dict and not preds_dict['Close'].empty:
                        merged_df = pd.merge(df_comparison_period[['Date', 'Close']], preds_dict['Close'], on='Date', how='inner')
                        all_historical_predictions[model_name] = merged_df
                        mae = np.mean(np.abs(merged_df['Close'] - merged_df['Predicted Close']))
                        rmse = np.sqrt(np.mean((merged_df['Close'] - merged_df['Predicted Close'])**2))
                        model_performance_data.append({'Model': model_name, 'MAE': mae, 'RMSE': rmse})

    # --- Section 1: Main Chart with Volume and Forecasts ---
    st.subheader(f"Price & Volume Forecasts for {ticker}")
    st.markdown("This chart displays recent historical price, volume, and future price forecasts generated by the selected models.")
    
    df_plot_main = df_features_full.tail(180)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df_plot_main['Date'], open=df_plot_main['Open'], high=df_plot_main['High'], low=df_plot_main['Low'], close=df_plot_main['Close'], name='Price'), row=1, col=1)
    for model_name, forecast_df in all_future_forecasts.items():
        fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Close'], mode='lines+markers', name=f'Forecast ({model_name})'), row=1, col=1)
    
    fig.add_trace(go.Bar(x=df_plot_main['Date'], y=df_plot_main['Volume'], name='Volume'), row=2, col=1)
    
    fig.update_layout(xaxis_rangeslider_visible=False, title=f'{ticker} Price Prediction', yaxis_title='Price', height=700, template='plotly_white', legend_title="Legend")
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Section 2: Prediction Analysis ---
    st.subheader(f"Why is the model predicting this price?")
    st.markdown(f"This section reveals which data points the main model **({model_choice})** considered most important when making its predictions.")
    col1, col2 = st.columns([2, 1])
    with col1:
        main_model_info = all_trained_models.get(model_choice, {}).get('Close')
        if main_model_info and hasattr(main_model_info['model'], 'feature_importances_'):
            imp_df = pd.DataFrame({'Feature': main_model_info['features'], 'Importance': main_model_info['model'].feature_importances_}).sort_values(by='Importance', ascending=False).head(20)
            fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title=f"Top 20 Features for {model_choice}")
            fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info(f"Feature importance is not available for the '{model_choice}' model type.")
    with col2:
        st.markdown("##### Prediction Rationale")
        if 'imp_df' in locals():
            st.markdown(get_feature_importance_explanation(imp_df, ticker))
        else:
            st.info("Run analysis to see the prediction rationale.")

    # --- Section 3: Fundamental Analysis ---
    display_fundamental_analysis(ticker, stock_info)

    # --- Section 4: Future Prices Data Table ---
    if all_future_forecasts:
        st.subheader("Consolidated Future Price Forecasts")
        st.markdown("The table below shows the predicted closing prices for the upcoming days from each model.")
        
        forecast_dfs_list = []
        for model_name, forecast_df in all_future_forecasts.items():
            forecast_subset = forecast_df[['Date', 'Close']].rename(columns={'Close': model_name}).set_index('Date')
            forecast_dfs_list.append(forecast_subset)

        if forecast_dfs_list:
            final_forecast_table = pd.concat(forecast_dfs_list, axis=1)
            st.dataframe(final_forecast_table.style.format("{:.2f}"), use_container_width=True)
    
    # --- Section 5: Historical Performance Comparison ---
    if all_historical_predictions:
        st.subheader(f"Historical Model Performance ({train_days_comparison} days)")
        st.markdown("This chart compares the models' past predictions against the actual stock price to evaluate their historical accuracy.")
        comparison_fig = go.Figure()
        df_comparison_period = df_features_full[(df_features_full['Date'] >= pd.to_datetime(end_bt) - timedelta(days=train_days_comparison)) & (df_features_full['Date'] <= pd.to_datetime(end_bt))]
        comparison_fig.add_trace(go.Scatter(x=df_comparison_period['Date'], y=df_comparison_period['Close'], mode='lines', name='Actual Price', line=dict(color='black', width=3)))
        for model_name, merged_df in all_historical_predictions.items():
            comparison_fig.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Predicted Close'], mode='lines', name=f'{model_name} Prediction', line=dict(dash='dot')))
        comparison_fig.update_layout(title="Model Comparison on Historical Data", yaxis_title="Price", height=500, template='plotly_white')
        st.plotly_chart(comparison_fig, use_container_width=True)
        
        st.subheader("Model Performance Metrics (on Recent Data)")
        st.markdown("Lower MAE and RMSE values indicate better performance.")
        st.dataframe(pd.DataFrame(model_performance_data).sort_values(by='MAE').style.format({'MAE': '{:.4f}', 'RMSE': '{:.4f}'}), use_container_width=True)

    # --- Section 6: Training Log ---
    with st.sidebar.expander("📜 Training Log", expanded=False):
        st.code("\n".join(st.session_state.training_log), language=None)

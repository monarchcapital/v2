# pages/Live_Predictor.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta, date, datetime
import time
import yfinance as yf # Explicitly import yfinance here for live data fetching

import config
from utils import (
    download_data, create_features, train_models_pipeline,
    generate_predictions_pipeline, add_market_data_features,
    parse_int_list,
    add_fundamental_features
)

st.set_page_config(page_title="Monarch: Live Predictor", layout="wide")

st.header("⚡ Live Stock Price & Next-Day Predictor")
st.markdown("Get the current market price and a next-day forecast for any stock using your configured models and features.")

# --- Sidebar ---
st.sidebar.header("🛠️ Configuration Panel")
ticker = st.sidebar.text_input("Enter Stock Ticker:", value="AAPL").upper()

# --- Date Inputs ---
st.sidebar.subheader("🗓️ Training Period")
today = date.today()
default_end_bt = today - timedelta(days=1)
default_start_bt = default_end_bt - timedelta(days=5*365) # Default to 5 years historical data
start_bt = st.sidebar.date_input("Training Start Date (t1):", value=default_start_bt)
end_bt = st.sidebar.date_input("Training End Date (t2):", value=default_end_bt)

if start_bt >= end_bt:
    st.sidebar.error("Training Start Date must be before End Date.")
    st.stop()

# --- Feature Selection ---
st.sidebar.subheader("➕ Add Features")
# Global Context Selector
with st.sidebar.expander("🌍 Global Context Features"):
    available_indices = list(config.GLOBAL_MARKET_TICKERS.keys())
    select_all_globals = st.checkbox("Select All Global Indices", value=False, key="live_select_all_globals")
    default_globals = available_indices if select_all_globals else available_indices[:3]
    selected_indices = st.multiselect("Select Global Indices:", options=available_indices, default=default_globals, key="live_global_select")
    selected_tickers = [config.GLOBAL_MARKET_TICKERS[name] for name in selected_indices]

# Fundamental Features Selector
with st.sidebar.expander("🔬 Fundamental Features"):
    # Static Fundamentals
    available_fundamentals_static = list(config.FUNDAMENTAL_METRICS.keys())
    select_all_fundamentals_static = st.checkbox("Select All Static Fundamentals", value=False, key="live_select_all_fundamentals_static")
    default_fundamentals_static = available_fundamentals_static if select_all_fundamentals_static else []
    selected_fundamental_names_static = st.multiselect("Select Static Fundamental Metrics:", options=available_fundamentals_static, default=default_fundamentals_static, key="live_static_fundamental_select")
    selected_fundamentals_static = {name: config.FUNDAMENTAL_METRICS[name] for name in selected_fundamental_names_static}

    # Historical/Derived Fundamentals
    available_fundamentals_derived = ['Historical P/E Ratio', 'Historical P/S Ratio', 'Historical Debt to Equity']
    select_all_fundamentals_derived = st.checkbox("Select All Historical/Derived Fundamentals", value=False, key="live_select_all_fundamentals_derived")
    default_fundamentals_derived = available_fundamentals_derived if select_all_fundamentals_derived else []
    selected_fundamental_names_derived = st.multiselect("Select Historical/Derived Metrics:", options=available_fundamentals_derived, default=default_fundamentals_derived, key="live_derived_fundamental_select")
    selected_fundamentals_derived = {name: name for name in selected_fundamental_names_derived}

# Combine selected fundamentals for passing to the utility function
combined_selected_fundamentals = {**selected_fundamentals_static, **selected_fundamentals_derived}

# --- Model Selection ---
st.sidebar.subheader("🤖 Model Selection")
model_choice = st.sidebar.selectbox("Select Prediction Model:", [m for m in config.MODEL_CHOICES if m != 'Prophet'], key="live_model_select")
perform_tuning = st.sidebar.checkbox("Perform Hyperparameter Tuning", value=False, help="Optimizes model parameters. May significantly increase training time.", key="live_tuning")

# --- Technical Indicator Settings ---
st.sidebar.subheader("⚙️ Technical Indicator Settings")
selected_indicator_params = {}
with st.sidebar.expander("Show Indicator Settings"):
    for indicator_name, (default_value, default_enabled) in config.TECHNICAL_INDICATORS_DEFAULTS.items():
        if st.checkbox(f"Enable {indicator_name.replace('_', ' ')}", value=default_enabled, key=f"enable_{indicator_name.lower()}_live"):
            if isinstance(default_value, list):
                selected_indicator_params[indicator_name] = parse_int_list(st.text_input(f"  {indicator_name.replace('_', ' ')} (days):", value=", ".join(map(str, default_value)), key=f"input_{indicator_name.lower()}_live"), default_value, st.sidebar.error)
            elif isinstance(default_value, (int, float)):
                min_val, step_val = (0.01, 0.01) if 'ACCEL' in indicator_name or isinstance(default_value, float) else (1.0, 1.0)
                selected_indicator_params[indicator_name] = st.number_input(f"  {indicator_name.replace('_', ' ')}:", min_value=min_val, value=float(default_value), step=step_val, key=f"input_{indicator_name.lower()}_live")

# --- Logging Setup ---
if 'live_log' not in st.session_state: st.session_state.live_log = []
def clear_log(): st.session_state.live_log = []
def update_log(message): st.session_state.live_log.insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
with st.sidebar.expander("📜 Process Log", expanded=True):
    st.code("\n".join(st.session_state.live_log), language=None)

# --- Main Application Flow ---
if st.button("Get Live Price & Predict Next Day", type="primary", on_click=clear_log):
    if not ticker:
        st.error("Please enter a stock ticker."); st.stop()

    update_log(f"Starting live analysis for {ticker}...")

    # 1. Fetch Live Price
    with st.spinner(f"Fetching live price for {ticker}..."):
        try:
            live_ticker_info = yf.Ticker(ticker).info
            current_price = live_ticker_info.get('currentPrice')
            if current_price is None:
                st.warning(f"Could not fetch live price for {ticker}. It might be a delisted stock or real-time data is unavailable.")
                current_price = "N/A"
            update_log(f"Live price fetched: {current_price}")
        except Exception as e:
            st.error(f"Error fetching live price for {ticker}: {e}")
            current_price = "N/A"
            update_log(f"Error fetching live price: {e}")

    # 2. Download Historical Data for Training
    with st.spinner(f"Downloading historical data for {ticker}..."):
        # Download historical data up to yesterday to ensure complete features for prediction
        # Use a period that covers the training range + buffer
        data = download_data(ticker, period=f"{int((today - start_bt).days / 365) + 2}y") 
        if data.empty:
            st.error(f"Could not load historical data for {ticker}. Cannot train model."); st.stop()
        update_log(f"Historical data loaded: {len(data)} rows.")
    
    # 3. Prepare Features
    with st.spinner("Processing data and creating features..."):
        data_for_features = data.copy()
        if combined_selected_fundamentals: data_for_features = add_fundamental_features(data_for_features, ticker, combined_selected_fundamentals, _update_log_func=update_log)
        if selected_tickers: data_for_features = add_market_data_features(data_for_features, "10y", update_log, selected_tickers=selected_tickers)
        df_features_full = create_features(data_for_features, selected_indicator_params)
        update_log(f"Features created. Rows after NaN drop: {len(df_features_full)}")
        
        # Ensure training data is within selected range
        df_train_period = df_features_full[(df_features_full['Date'] >= pd.to_datetime(start_bt)) & (df_features_full['Date'] <= pd.to_datetime(end_bt))].copy()
        if df_train_period.empty:
            st.error("No data in selected training range for feature creation. Please adjust dates."); st.stop()
        update_log(f"Training data prepared: {len(df_train_period)} rows.")

    # 4. Train Model
    with st.spinner(f"Training {model_choice} model..."):
        trained_models, _ = train_models_pipeline(df_train_period.copy(), model_choice, perform_tuning, update_log, selected_indicator_params)
        if not trained_models or 'Close' not in trained_models:
            st.error(f"Failed to train {model_choice} model for 'Close' price prediction."); st.stop()
        update_log(f"✓ {model_choice} trained successfully.")

    # 5. Generate Next-Day Prediction
    with st.spinner("Generating next-day prediction..."):
        # The input for prediction is the last available day of historical features
        # which represents the data known *before* the next trading day.
        last_day_features = df_features_full.tail(1)
        if last_day_features.empty:
            st.error("Not enough historical data to generate a prediction."); st.stop()

        next_day_preds_dict = generate_predictions_pipeline(last_day_features, trained_models, update_log)
        predicted_next_day_close = next_day_preds_dict.get('Close', {}).get('Predicted Close', pd.Series([np.nan])).iloc[0]
        
        if pd.isna(predicted_next_day_close):
            st.warning("Could not generate a valid next-day close price prediction.")
            predicted_next_day_close = "N/A"
        update_log(f"Next-day prediction generated: {predicted_next_day_close}")
    
    st.success("Analysis complete.")

    # --- Display Results ---
    st.subheader(f"Current Market & Next-Day Forecast for {ticker}")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Live Price", f"${current_price:,.2f}" if isinstance(current_price, (int, float)) else current_price)
    with col2:
        st.metric("Predicted Next Trading Day Close", f"${predicted_next_day_close:,.2f}" if isinstance(predicted_next_day_close, (int, float)) else predicted_next_day_close)
    
    if isinstance(current_price, (int, float)) and isinstance(predicted_next_day_close, (int, float)):
        predicted_change_pct = ((predicted_next_day_close - current_price) / current_price) * 100
        st.metric("Predicted Change from Live Price", f"{predicted_change_pct:,.2f}%", delta=f"{predicted_change_pct:,.2f}%")

    # --- Visualization ---
    st.subheader("Price Trend & Forecast")
    fig = go.Figure()

    # Historical prices (last 90 days for clarity)
    historical_plot_data = data.tail(90)
    fig.add_trace(go.Scatter(x=historical_plot_data['Date'], y=historical_plot_data['Close'], mode='lines', name='Historical Close', line=dict(color='blue', width=2)))

    # Add Live Price Point
    if isinstance(current_price, (int, float)):
        fig.add_trace(go.Scatter(x=[datetime.now()], y=[current_price], mode='markers', name='Live Price',
                                 marker=dict(size=10, color='green', symbol='circle'),
                                 hoverinfo='text', text=f'Live: ${current_price:,.2f}'))
        
        # Draw a line from last historical close to live price if available
        if not historical_plot_data.empty:
            last_hist_date = historical_plot_data['Date'].iloc[-1]
            last_hist_close = historical_plot_data['Close'].iloc[-1]
            fig.add_trace(go.Scatter(x=[last_hist_date, datetime.now()], y=[last_hist_close, current_price],
                                     mode='lines', line=dict(color='green', dash='dot'), showlegend=False))


    # Add Predicted Next-Day Price Point
    if isinstance(predicted_next_day_close, (int, float)):
        # The prediction date is the day after the last historical date
        prediction_date_obj = df_features_full['Date'].iloc[-1] + timedelta(days=1)
        fig.add_trace(go.Scatter(x=[prediction_date_obj], y=[predicted_next_day_close], mode='markers', name='Predicted Next Day Close',
                                 marker=dict(size=10, color='red', symbol='star'),
                                 hoverinfo='text', text=f'Predicted: ${predicted_next_day_close:,.2f}'))
        
        # Draw a line from live price to predicted price if both are available
        if isinstance(current_price, (int, float)):
             fig.add_trace(go.Scatter(x=[datetime.now(), prediction_date_obj], y=[current_price, predicted_next_day_close],
                                     mode='lines', line=dict(color='red', dash='dash'), showlegend=False))
        elif not historical_plot_data.empty: # If no live price, draw from last historical to prediction
            last_hist_date = historical_plot_data['Date'].iloc[-1]
            last_hist_close = historical_plot_data['Close'].iloc[-1]
            fig.add_trace(go.Scatter(x=[last_hist_date, prediction_date_obj], y=[last_hist_close, predicted_next_day_close],
                                     mode='lines', line=dict(color='red', dash='dash'), showlegend=False))


    fig.update_layout(
        title=f'{ticker} Price Trend & Next-Day Forecast',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=600,
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Enter a stock ticker in the sidebar and click 'Get Live Price & Predict Next Day' to see the current market price and a next-day forecast.")


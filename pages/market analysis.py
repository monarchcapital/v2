# pages/market_analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import timedelta, date
from functools import reduce
import config
from utils import (
    download_data, create_features, train_models_pipeline,
    generate_iterative_forecast
)

st.set_page_config(page_title="Monarch: Market Analysis", layout="wide")

st.header("📈 Comprehensive Market Analysis")
st.markdown("Analyze global indices, forecast the NIFTY 50 using multiple methods, and screen for opportunities within NIFTY sectors.")

# --- Helper Functions ---
@st.cache_data(ttl=86400) # Cache for a day
def get_nifty50_constituents_and_weights():
    """
    Fetches NIFTY 50 constituents and their weights reliably.
    Uses a fallback list of all 50 stocks if the primary method fails.
    """
    try:
        # Primary Method: Use a known NIFTY 50 ETF to get holdings and weights
        etf_ticker = yf.Ticker("NIFTYBEES.NS")
        holdings = etf_ticker.info.get('holdings')
        if not holdings or len(holdings) < 45: # Check if we got a reasonable number of holdings
             raise ValueError("Could not retrieve sufficient holdings from ETF.")
        
        data = []
        for stock in holdings:
            symbol = stock.get('symbol')
            # Ensure .NS suffix for NSE stocks
            if not symbol.endswith('.NS'):
                 symbol += '.NS'
            data.append({
                'Symbol': symbol,
                'Company Name': stock.get('longName', symbol),
                'Weight': stock.get('holdingPercent', 0) * 100,
                'Industry': 'N/A' # Industry info is not in holdings
            })
        
        df = pd.DataFrame(data)
        # Fetch industry for each stock (can be slow, but is cached)
        df['Industry'] = df['Symbol'].apply(lambda s: yf.Ticker(s).info.get('industry', 'N/A'))
        return df.set_index('Symbol')

    except Exception as e:
        st.warning(f"Primary method failed ({e}). Using a comprehensive fallback list for NIFTY 50.")
        # Fallback: A complete hardcoded list of NIFTY 50 stocks
        fallback_stocks = {
            'RELIANCE.NS': 11.59, 'HDFCBANK.NS': 8.79, 'ICICIBANK.NS': 7.87, 'INFY.NS': 5.02, 
            'LT.NS': 4.50, 'TCS.NS': 4.00, 'BHARTIARTL.NS': 3.70, 'ITC.NS': 3.50, 'KOTAKBANK.NS': 3.00,
            'HINDUNILVR.NS': 2.53, 'AXISBANK.NS': 2.40, 'BAJFINANCE.NS': 2.20, 'MARUTI.NS': 2.16,
            'M&M.NS': 2.10, 'SBIN.NS': 1.98, 'SUNPHARMA.NS': 1.88, 'TATAMOTORS.NS': 1.85, 
            'NTPC.NS': 1.62, 'POWERGRID.NS': 1.50, 'TATASTEEL.NS': 1.35, 'ADANIENT.NS': 1.29,
            'ULTRACEMCO.NS': 1.25, 'ASIANPAINT.NS': 1.19, 'COALINDIA.NS': 1.17, 'BAJAJFINSV.NS': 1.13,
            'HCLTECH.NS': 1.11, 'NESTLEIND.NS': 0.95, 'JSWSTEEL.NS': 0.94, 'INDUSINDBK.NS': 0.92,
            'ADANIPORTS.NS': 0.89, 'GRASIM.NS': 0.81, 'HINDALCO.NS': 0.79, 'EICHERMOT.NS': 0.75,
            'DRREDDY.NS': 0.73, 'SBILIFE.NS': 0.71, 'TITAN.NS': 0.69, 'CIPLA.NS': 0.68,
            'TECHM.NS': 0.65, 'BAJAJ-AUTO.NS': 0.61, 'WIPRO.NS': 0.59, 'SHREECEM.NS': 0.57,
            'HEROMOTOCO.NS': 0.55, 'DIVISLAB.NS': 0.53, 'APOLLOHOSP.NS': 0.51, 'LTIM.NS': 0.49,
            'BRITANNIA.NS': 0.47, 'ONGC.NS': 0.45, 'BPCL.NS': 0.43, 'HDFCLIFE.NS': 0.41, 'SHRIRAMFIN.NS': 0.38
        }
        df = pd.DataFrame(list(fallback_stocks.items()), columns=['Symbol', 'Weight'])
        df['Industry'] = df['Symbol'].apply(lambda s: yf.Ticker(s).info.get('industry', 'N/A'))
        return df.set_index('Symbol')

# --- Main App ---
tab1, tab2, tab3 = st.tabs(["🌍 Global Index Analysis", "🇮🇳 NIFTY 50 Forecast", "🏢 NIFTY Sector Screener"])

# --- Tab 1: Global Index Analysis ---
with tab1:
    st.subheader("Global Index Analysis & Multi-Model Forecast")
    
    with st.expander("Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        selected_index_name = col1.selectbox("Select Index:", options=list(config.GLOBAL_MARKET_TICKERS.keys()), key="global_index_select")
        ticker_global = config.GLOBAL_MARKET_TICKERS[selected_index_name]
        models_to_compare_global = col2.multiselect("Select Models to Compare:", options=[m for m in config.MODEL_CHOICES if m != 'Prophet'], default=['Random Forest', 'XGBoost'], key="global_models")
        n_future_global = col3.slider("Forecast Days:", 1, 90, 15, key="global_future")

    if st.button(f"▶️ Run Analysis for {selected_index_name}", key="run_global"):
        if not models_to_compare_global:
            st.warning("Please select at least one model to run the analysis.")
        else:
            indicator_params_global = {k: v[0] for k, v in config.TECHNICAL_INDICATORS_DEFAULTS.items()}
            with st.spinner(f"Fetching data for {selected_index_name}..."):
                data_global = download_data(ticker_global)
            
            if data_global.empty:
                st.error(f"Could not download data for {ticker_global}.")
            else:
                all_forecasts_global = {}
                for model_name in models_to_compare_global:
                    with st.spinner(f"Training {model_name} model..."):
                        df_features_global = create_features(data_global.copy(), indicator_params_global)
                        trained_models, _ = train_models_pipeline(df_features_global, model_name, False, (lambda x: None), indicator_params_global)
                        future_df = generate_iterative_forecast(data_global, trained_models, ticker_global, n_future_global, data_global['Date'].iloc[-1], indicator_params_global, (lambda x: None))
                        if not future_df.empty:
                            all_forecasts_global[model_name] = future_df

                st.success("Analysis Complete!")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data_global['Date'], y=data_global['Close'], mode='lines', name='Historical Close', line=dict(width=2.5)))
                for model_name, forecast_df in all_forecasts_global.items():
                    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Close'], mode='lines', name=f'{model_name} Forecast', line=dict(dash='dot')))
                fig.update_layout(title=f"{selected_index_name} Price History and Forecasts", xaxis_title="Date", yaxis_title="Price", legend_title="Legend")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Detailed Forecast Prices")
                if all_forecasts_global:
                    forecast_dfs_list = [df[['Date', 'Close']].rename(columns={'Close': name}).set_index('Date') for name, df in all_forecasts_global.items()]
                    consolidated_forecasts = pd.concat(forecast_dfs_list, axis=1)
                    st.dataframe(consolidated_forecasts.style.format("{:.2f}"), use_container_width=True)
                else:
                    st.info("No successful forecasts were generated.")

# --- Tab 2: NIFTY 50 Forecast ---
with tab2:
    st.subheader("NIFTY 50 Forecast: Direct vs. Component-based")
    st.info("The component-based forecast predicts each of the 50 stocks and aggregates them by weight. This is a thorough but very slow process.")

    with st.expander("Configuration", expanded=True):
        t2_col1, t2_col2 = st.columns(2)
        nifty_models = t2_col1.multiselect("Select Models:", options=[m for m in config.MODEL_CHOICES if m != 'Prophet'], default=['Random Forest'], key="nifty_models")
        n_future_nifty = t2_col2.slider("Forecast Days:", 1, 30, 7, key="nifty_future_days")

    if st.button("▶️ Run NIFTY 50 Forecast", key="run_nifty_forecast"):
        if not nifty_models:
            st.warning("Please select at least one model.")
        else:
            nifty_ticker = "^NSEI"
            indicator_params_nifty = {k: v[0] for k, v in config.TECHNICAL_INDICATORS_DEFAULTS.items()}
            data_nifty = download_data(nifty_ticker)
            
            all_direct_forecasts = {}
            all_component_forecasts = {}

            for model_name in nifty_models:
                # 1. Direct Forecast
                with st.spinner(f"Running direct forecast for {model_name}..."):
                    df_features_nifty = create_features(data_nifty.copy(), indicator_params_nifty)
                    trained_models_nifty, _ = train_models_pipeline(df_features_nifty, model_name, False, (lambda x: None), indicator_params_nifty)
                    direct_forecast_df = generate_iterative_forecast(data_nifty, trained_models_nifty, nifty_ticker, n_future_nifty, data_nifty['Date'].iloc[-1], indicator_params_nifty, (lambda x: None))
                    if not direct_forecast_df.empty:
                        all_direct_forecasts[model_name] = direct_forecast_df
                st.success(f"Direct forecast for {model_name} complete.")

                # 2. Component-based Forecast
                with st.spinner(f"Running component-based forecast for {model_name} on all 50 stocks... (This will take several minutes)"):
                    constituents_df = get_nifty50_constituents_and_weights()
                    constituents_df['Weight'] /= constituents_df['Weight'].sum() # Normalize weights to sum to 1
                    
                    component_forecasts_agg = pd.DataFrame(index=pd.date_range(start=data_nifty['Date'].iloc[-1] + timedelta(days=1), periods=n_future_nifty))
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    for i, (ticker, row) in enumerate(constituents_df.iterrows()):
                        status_text.text(f"Processing {i+1}/{len(constituents_df)}: {ticker}")
                        try:
                            stock_data = download_data(ticker)
                            if stock_data.empty or len(stock_data) < 200: continue
                            
                            stock_features = create_features(stock_data.copy(), indicator_params_nifty)
                            stock_models, _ = train_models_pipeline(stock_features, model_name, False, (lambda x: None), indicator_params_nifty)
                            stock_forecast = generate_iterative_forecast(stock_data, stock_models, ticker, n_future_nifty, stock_data['Date'].iloc[-1], indicator_params_nifty, (lambda x: None))
                            
                            if not stock_forecast.empty:
                                weighted_forecast = stock_forecast.set_index('Date')['Close'] * row['Weight']
                                component_forecasts_agg[ticker] = weighted_forecast
                        except Exception as e:
                            st.warning(f"Could not process {ticker}: {e}")
                        progress_bar.progress((i + 1) / len(constituents_df))
                    
                    status_text.text("Aggregating component forecasts...")
                    if not component_forecasts_agg.empty:
                        component_forecast_series = component_forecasts_agg.sum(axis=1) * (data_nifty['Close'].iloc[-1] / component_forecasts_agg.sum(axis=1).iloc[0])
                        all_component_forecasts[model_name] = component_forecast_series.reset_index().rename(columns={'index': 'Date', 0: 'Close'})
                        st.success(f"Component-based forecast for {model_name} complete.")
                    else:
                        st.error(f"Component-based forecast failed for {model_name}.")

            # Plotting results
            st.subheader("NIFTY 50 Forecast Comparison")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data_nifty['Date'], y=data_nifty['Close'], mode='lines', name='Historical NIFTY 50', line=dict(width=3)))
            for model_name, forecast_df in all_direct_forecasts.items():
                fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Close'], mode='lines', name=f'Direct ({model_name})', line=dict(dash='dot')))
            for model_name, forecast_df in all_component_forecasts.items():
                fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Close'], mode='lines', name=f'Component ({model_name})', line=dict(dash='dash')))
            fig.update_layout(title="NIFTY 50 Forecast Comparison", xaxis_title="Date", yaxis_title="Index Level", legend_title="Forecast Type")
            st.plotly_chart(fig, use_container_width=True)
            
            # Data tables
            st.subheader("Detailed Forecast Prices")
            direct_dfs = [df[['Date', 'Close']].rename(columns={'Close': f'Direct ({name})'}).set_index('Date') for name, df in all_direct_forecasts.items()]
            comp_dfs = [df[['Date', 'Close']].rename(columns={'Close': f'Component ({name})'}).set_index('Date') for name, df in all_component_forecasts.items()]
            final_df = pd.concat(direct_dfs + comp_dfs, axis=1)
            st.dataframe(final_df.style.format("{:.2f}"), use_container_width=True)

# --- Tab 3: NIFTY Sector Screener ---
with tab3:
    st.subheader("NIFTY 50 Sector-wise Screener & Mini-Forecast")
    
    with st.expander("Screener Criteria", expanded=True):
        s_col1, s_col2, s_col3 = st.columns(3)
        min_roe = s_col1.slider("Min. Return on Equity (ROE %)", -20, 50, 15, key="nifty_roe") / 100
        max_pe = s_col2.slider("Max. P/E Ratio", 5.0, 100.0, 30.0, step=1.0, key="nifty_pe")
        ma_signal = s_col3.selectbox("Price vs MA Signal", ["Price > 50-day MA", "Price > 200-day MA"], key="nifty_ma")
        ma_window = 50 if "50" in ma_signal else 200
        
        st.markdown("---")
        f_col1, f_col2 = st.columns(2)
        forecast_model = f_col1.selectbox("Model for Mini-Forecast:", options=[m for m in config.MODEL_CHOICES if m != 'Prophet'], key="screener_model")
        forecast_days = f_col2.slider("Days to Forecast for Screened Stocks:", 1, 10, 5, key="screener_days")

    if st.button("▶️ Run NIFTY 50 Screener", key="run_sector_analysis"):
        constituents = get_nifty50_constituents_and_weights().reset_index()
        results = []
        
        with st.spinner("Analyzing all 50 constituents... This may take a few minutes..."):
            progress_bar_sector = st.progress(0)
            status_text_sector = st.empty()
            for i, row in constituents.iterrows():
                ticker = row['Symbol']
                status_text_sector.text(f"Screening {i+1}/{len(constituents)}: {ticker}")
                try:
                    stock_info = yf.Ticker(ticker).info
                    
                    roe = stock_info.get('returnOnEquity', -999)
                    pe = stock_info.get('trailingPE', 9999)
                    if not (isinstance(roe, (int, float)) and isinstance(pe, (int, float)) and roe > min_roe and pe < max_pe):
                        continue
                        
                    hist_data = download_data(ticker, period=f"{ma_window+10}d")
                    if hist_data.empty or len(hist_data) < ma_window: continue
                    
                    current_price = hist_data['Close'].iloc[-1]
                    ma_value = hist_data['Close'].rolling(window=ma_window).mean().iloc[-1]
                    
                    if current_price > ma_value:
                        # Passed all filters, now run forecast
                        status_text_sector.text(f"Passed! Forecasting {ticker}...")
                        indicator_params = {k: v[0] for k, v in config.TECHNICAL_INDICATORS_DEFAULTS.items()}
                        full_hist_data = download_data(ticker)
                        features = create_features(full_hist_data.copy(), indicator_params)
                        models, _ = train_models_pipeline(features, forecast_model, False, (lambda x: None), indicator_params)
                        forecast_df = generate_iterative_forecast(full_hist_data, models, ticker, forecast_days, full_hist_data['Date'].iloc[-1], indicator_params, (lambda x: None))
                        
                        forecast_price = forecast_df['Close'].iloc[-1] if not forecast_df.empty else 'N/A'
                        
                        results.append({
                            "Ticker": ticker,
                            "Industry": stock_info.get('industry', 'N/A'),
                            "Price": current_price,
                            "ROE": roe,
                            "P/E Ratio": pe,
                            f"{forecast_days}-Day Forecast": forecast_price
                        })
                except Exception:
                    continue
                progress_bar_sector.progress((i + 1) / len(constituents))
            status_text_sector.text("Screening complete.")

        if results:
            st.success(f"Found {len(results)} stocks matching your criteria.")
            df_results = pd.DataFrame(results)
            st.dataframe(df_results.style.format({
                'Price': '{:.2f}',
                'ROE': '{:.2%}',
                'P/E Ratio': '{:.2f}',
                f'{forecast_days}-Day Forecast': '{:.2f}'
            }), use_container_width=True, hide_index=True)
        else:
            st.info("No stocks in the NIFTY 50 matched all your criteria.")

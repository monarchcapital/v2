# pages/Nifty_analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import timedelta, date

import config
from utils import (
    download_data, create_features, train_models_pipeline,
    generate_iterative_forecast, parse_int_list,
    add_market_data_features, add_fundamental_features
)

st.set_page_config(page_title="Monarch: NIFTY 50 Analysis", layout="wide")

st.header("🇮🇳 NIFTY 50 Forecast Comparison")
st.markdown("""
This page forecasts the NIFTY 50 index using two different methodologies and compares the results:
1.  **Direct Analysis (Top-Down):** A forecast on the NIFTY 50 index (`^NSEI`) as a single entity.
2.  **Component-based Analysis (Bottom-Up):** A forecast calculated from the weighted predictions of each individual constituent stock.
""")

# A hardcoded fallback list of all 50 NIFTY constituents in case the live fetch fails.
# This list is updated to be complete and accurate.
FALLBACK_NIFTY50_STOCKS = {
    'HDFCBANK.NS': 11.59,
    'RELIANCE.NS': 9.69,
    'ICICIBANK.NS': 7.87,
    'INFY.NS': 5.02,
    'LT.NS': 4.50,
    'TCS.NS': 4.00,
    'BHARTIARTL.NS': 3.70,
    'ITC.NS': 3.50,
    'KOTAKBANK.NS': 3.00,
    'HINDUNILVR.NS': 2.53,
    'AXISBANK.NS': 2.40,
    'BAJFINANCE.NS': 2.20,
    'MARUTI.NS': 2.16,
    'M&M.NS': 2.10,
    'SBIN.NS': 1.98,
    'SUNPHARMA.NS': 1.88,
    'TATAMOTORS.NS': 1.85,
    'NTPC.NS': 1.62,
    'POWERGRID.NS': 1.50,
    'TATASTEEL.NS': 1.35,
    'ADANIENT.NS': 1.29,
    'ULTRACEMCO.NS': 1.25,
    'ASIANPAINT.NS': 1.19,
    'COALINDIA.NS': 1.17,
    'BAJAJFINSV.NS': 1.13,
    'HCLTECH.NS': 1.11,
    'NESTLEIND.NS': 0.95,
    'JSWSTEEL.NS': 0.94,
    'INDUSINDBK.NS': 0.92,
    'ADANIPORTS.NS': 0.89,
    'GRASIM.NS': 0.81,
    'HINDALCO.NS': 0.79,
    'EICHERMOT.NS': 0.75,
    'DRREDDY.NS': 0.73,
    'SBILIFE.NS': 0.71,
    'TITAN.NS': 0.69,
    'CIPLA.NS': 0.68,
    'TECHM.NS': 0.65,
    'BAJAJ-AUTO.NS': 0.61,
    'WIPRO.NS': 0.59,
    'SHREECEM.NS': 0.57,
    'HEROMOTOCO.NS': 0.55,
    'DIVISLAB.NS': 0.53,
    'APOLLOHOSP.NS': 0.51,
    'LTIM.NS': 0.49,
    'BRITANNIA.NS': 0.47,
    'ONGC.NS': 0.45,
    'BPCL.NS': 0.43,
    'HDFCLIFE.NS': 0.41,
    'SHRIRAMFIN.NS': 0.38
}


@st.cache_data(ttl=86400) # Cache for one day
def get_nifty50_constituents():
    """
    Fetches NIFTY 50 constituents and weights from a reliable source (ETF holdings).
    Falls back to a hardcoded list on failure.
    """
    try:
        # Using an ETF that tracks the NIFTY 50 to get its holdings
        etf_ticker = yf.Ticker("NIFTYBEES.NS")
        holdings = etf_ticker.info.get('holdings')
        if not holdings:
            raise ValueError("Could not retrieve holdings from NIFTYBEES.NS ETF info.")
        
        weights_dict = {}
        for stock in holdings:
            symbol = stock.get('symbol')
            weight = stock.get('holdingPercent')
            if symbol and weight:
                # Ensure the ticker has the .NS suffix for NSE stocks
                if not symbol.endswith(('.NS', '.BO')):
                     symbol += '.NS'
                weights_dict[symbol] = weight * 100 # Convert decimal to percentage
        
        if not weights_dict or len(weights_dict) < 45: # Sanity check for at least 45 stocks
            st.warning("Could not parse a complete list from ETF holdings, using fallback list.")
            return FALLBACK_NIFTY50_STOCKS
        
        st.success(f"Successfully fetched {len(weights_dict)} live NIFTY 50 constituents and weights.")
        return weights_dict
    except Exception as e:
        st.error(f"Could not fetch live NIFTY 50 constituents: {e}. Using a fallback list of 50 stocks.")
        return FALLBACK_NIFTY50_STOCKS

nifty50_stocks = get_nifty50_constituents()
with st.expander(f"View NIFTY 50 Constituent Stocks and Weights ({len(nifty50_stocks)} stocks found)"):
    st.dataframe(pd.DataFrame(nifty50_stocks.items(), columns=['Ticker', 'Weight (%)']))

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Analysis Configuration")
    st.subheader("🗓️ Training & Prediction Period")
    today = date.today()
    default_end_date = today - timedelta(days=1)
    default_start_date = default_end_date - timedelta(days=5*365)
    start_date = st.date_input("Training Start Date:", value=default_start_date, key="nifty_start")
    end_date = st.date_input("Training End Date:", value=default_end_date, key="nifty_end")
    n_future = st.slider("Predict Future Days:", 1, 90, 15, key="nifty_future")

    st.subheader("🤖 Model Selection")
    selected_models = st.multiselect("Select Models (for Direct Analysis):", options=[m for m in config.MODEL_CHOICES if m != 'Prophet'], default=['Random Forest', 'XGBoost'])
    component_model = st.selectbox("Select Model (for Component Analysis):", options=[m for m in config.MODEL_CHOICES if m != 'Prophet'])

    st.subheader("➕ Feature Selection")
    with st.expander("🌍 Global Context Features"):
        available_indices = list(config.GLOBAL_MARKET_TICKERS.keys())
        selected_indices = st.multiselect("Select Global Indices:", options=available_indices, default=available_indices[:3], key="nifty_global_select")
        selected_tickers = [config.GLOBAL_MARKET_TICKERS[name] for name in selected_indices]
    
    with st.expander("🔬 Fundamental Features (for Component Analysis)"):
        # Static Fundamentals
        available_fundamentals_static_nifty = list(config.FUNDAMENTAL_METRICS.keys())
        select_all_fundamentals_static_nifty = st.checkbox("Select All Static Fundamentals", value=False, key="nifty_select_all_fundamentals_static")
        default_fundamentals_static_nifty = available_fundamentals_static_nifty if select_all_fundamentals_static_nifty else []
        selected_fundamental_names_static_nifty = st.multiselect("Select Static Fundamental Metrics:", options=available_fundamentals_static_nifty, default=default_fundamentals_static_nifty, key="nifty_static_fundamental_select")
        selected_fundamentals_static_nifty = {name: config.FUNDAMENTAL_METRICS[name] for name in selected_fundamental_names_static_nifty}

        # Historical/Derived Fundamentals
        available_fundamentals_derived_nifty = ['Historical P/E Ratio', 'Historical P/S Ratio', 'Historical Debt to Equity']
        select_all_fundamentals_derived_nifty = st.checkbox("Select All Historical/Derived Fundamentals", value=False, key="nifty_select_all_fundamentals_derived")
        default_fundamentals_derived_nifty = available_fundamentals_derived_nifty if select_all_fundamentals_derived_nifty else []
        selected_fundamental_names_derived_nifty = st.multiselect("Select Historical/Derived Metrics:", options=available_fundamentals_derived_nifty, default=default_fundamentals_derived_nifty, key="nifty_derived_fundamental_select")
        selected_fundamentals_derived_nifty = {name: name for name in selected_fundamental_names_derived_nifty}

    # Combine all selected fundamental metrics
    combined_selected_fundamentals_nifty = {**selected_fundamentals_static_nifty, **selected_fundamentals_derived_nifty}

    st.subheader("⚙️ Technical Indicator Settings")
    selected_indicator_params = {}
    with st.expander("Show Indicator Settings"):
        for indicator_name, (default_value, default_enabled) in config.TECHNICAL_INDICATORS_DEFAULTS.items():
            if st.checkbox(f"Enable {indicator_name.replace('_', ' ')}", value=default_enabled, key=f"enable_{indicator_name.lower()}_nifty"):
                if isinstance(default_value, list):
                    selected_indicator_params[indicator_name] = parse_int_list(st.text_input(f"  {indicator_name.replace('_', ' ')} (days):", value=", ".join(map(str, default_value)), key=f"input_{indicator_name.lower()}_nifty"), default_value, st.sidebar.error)
                elif isinstance(default_value, (int, float)):
                    min_val, step_val = (0.01, 0.01) if 'ACCEL' in indicator_name or isinstance(default_value, float) else (1.0, 1.0)
                    selected_indicator_params[indicator_name] = st.number_input(f"  {indicator_name.replace('_', ' ')}:", min_value=min_val, value=float(default_value), step=step_val, key=f"input_{indicator_name.lower()}_nifty")

# --- Main Application Flow ---
if st.button("Run NIFTY 50 Forecast Comparison", type="primary"):
    # --- 1. Direct Analysis (Top-Down) ---
    all_direct_forecasts = {}
    with st.spinner(f"Running direct forecast for {len(selected_models)} models..."):
        nifty_data = download_data("^NSEI")
        if not nifty_data.empty:
            data_for_features = add_market_data_features(nifty_data.copy(), "10y", (lambda x: None), selected_tickers=selected_tickers)
            df_features = create_features(data_for_features, selected_indicator_params)
            df_train = df_features[(df_features['Date'] >= pd.to_datetime(start_date)) & (df_features['Date'] <= pd.to_datetime(end_date))]
            
            if not df_train.empty:
                for model_name in selected_models:
                    st.write(f"Training {model_name} for direct analysis...")
                    trained_models, _ = train_models_pipeline(df_train.copy(), model_name, False, (lambda x: None), selected_indicator_params)
                    future_df = generate_iterative_forecast(nifty_data, trained_models, "^NSEI", n_future, end_date, selected_indicator_params, (lambda x: None), selected_tickers, combined_selected_fundamentals_nifty)
                    all_direct_forecasts[model_name] = future_df
                st.success("Direct index forecasts complete.")
            else: st.warning("Not enough data for direct index forecast in the selected date range.")
        else: st.error("Could not download NIFTY 50 Index (^NSEI) data.")

    # --- 2. Component-based Analysis (Bottom-Up) ---
    st.warning(f"Component analysis is starting. This will process {len(nifty50_stocks)} stocks and may take a significant amount of time.")
    combined_forecast_component = pd.DataFrame()
    total_processed_weight = 0.0
    with st.spinner(f"Analyzing {len(nifty50_stocks)} constituent stocks with {component_model}..."):
        all_future_predictions, progress_bar, status_text = {}, st.progress(0), st.empty()
        for i, (ticker_item, weight) in enumerate(nifty50_stocks.items()):
            status_text.text(f"Processing ({i+1}/{len(nifty50_stocks)}): {ticker_item}")
            data_stock = download_data(ticker_item)
            if data_stock.empty: continue
            
            data_stock_for_features = data_stock.copy()
            # Pass the combined selected fundamentals
            if combined_selected_fundamentals_nifty: data_stock_for_features = add_fundamental_features(data_stock_for_features, ticker_item, combined_selected_fundamentals_nifty, (lambda x: None))
            if selected_tickers: data_stock_for_features = add_market_data_features(data_stock_for_features, "10y", (lambda x: None), selected_tickers=selected_tickers)
            
            df_features_stock = create_features(data_stock_for_features, selected_indicator_params)
            df_train_stock = df_features_stock[(df_features_stock['Date'] >= pd.to_datetime(start_date)) & (df_features_stock['Date'] <= pd.to_datetime(end_date))]
            if df_train_stock.empty: continue
            
            trained_models, _ = train_models_pipeline(df_train_stock.copy(), component_model, False, (lambda x: None), selected_indicator_params)
            future_df = generate_iterative_forecast(data_stock, trained_models, ticker_item, n_future, end_date, selected_indicator_params, (lambda x: None), selected_tickers, combined_selected_fundamentals_nifty)
            
            if not future_df.empty:
                all_future_predictions[ticker_item] = {'forecast': future_df, 'weight': weight, 'last_price': data_stock.loc[data_stock['Date'] <= pd.to_datetime(end_date), 'Close'].iloc[-1]}
                total_processed_weight += weight # Add to the successfully processed weight
            
            progress_bar.progress((i + 1) / len(nifty50_stocks))
        
        status_text.success(f"Processed {len(all_future_predictions)} of {len(nifty50_stocks)} stocks."); progress_bar.empty()

        # --- Calculate the weighted component forecast ---
        if all_future_predictions:
            last_nifty_close = nifty_data.loc[nifty_data['Date'] <= pd.to_datetime(end_date), 'Close'].iloc[-1]
            # Use the first successfully processed ticker to establish the date index for the forecast
            first_ticker = list(all_future_predictions.keys())[0]
            combined_forecast_component = pd.DataFrame({'Date': all_future_predictions[first_ticker]['forecast']['Date']})
            total_contribution = pd.Series(0.0, index=range(len(combined_forecast_component)))

            for ticker_item, info in all_future_predictions.items():
                # Renormalize weight based on the total weight of *successfully processed* stocks
                if total_processed_weight > 0:
                    renormalized_weight = info['weight'] / total_processed_weight
                else:
                    renormalized_weight = 0
                # Calculate the percentage change of the stock and multiply by its renormalized weight
                price_contribution = (info['forecast']['Close'] / info['last_price']) * renormalized_weight
                # Align and add the contribution
                total_contribution += price_contribution.reset_index(drop=True)

            combined_forecast_component['Forecasted Index Level'] = last_nifty_close * total_contribution
            st.success("Component-based forecast complete.")

    # --- 3. Display Results ---
    st.subheader("📈 Forecast Comparison Results")
    col1, col2 = st.columns([3, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=nifty_data['Date'], y=nifty_data['Close'], mode='lines', name='Actual NIFTY 50 History', line=dict(color='black', width=2)))
        for model_name, forecast_df in all_direct_forecasts.items():
            if not forecast_df.empty: fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Close'], mode='lines', name=f'Direct ({model_name})', line=dict(dash='dot')))
        if not combined_forecast_component.empty: fig.add_trace(go.Scatter(x=combined_forecast_component['Date'], y=combined_forecast_component['Forecasted Index Level'], mode='lines', name=f'Component ({component_model})', line=dict(dash='dash', color='red', width=3)))
        fig.update_layout(title='NIFTY 50 Forecast: Direct vs. Component-based', yaxis_title='Index Level', height=600, legend=dict(x=0.01, y=0.99))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("##### Analysis Summary")
        st.metric(
            label="Processed Index Weight",
            value=f"{total_processed_weight:.2f}%",
            help="The percentage of the NIFTY 50's total weight that was successfully processed for the component-based forecast."
        )
        st.markdown("###### Direct Forecasts")
        if all_direct_forecasts:
            # --- START MODIFICATION ---
            final_direct_table_list = []
            for model_name, forecast_df in all_direct_forecasts.items():
                if not forecast_df.empty:
                    # Ensure Date is datetime and unique, then set as index
                    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
                    forecast_df = forecast_df.drop_duplicates(subset=['Date']) 
                    forecast_subset = forecast_df[['Date', 'Close']].rename(columns={'Close': model_name}).set_index('Date')
                    final_direct_table_list.append(forecast_subset)
            
            final_direct_table = pd.DataFrame()
            if final_direct_table_list:
                # Concatenate all forecast DataFrames along columns, aligning by Date index
                final_direct_table = pd.concat(final_direct_table_list, axis=1, join='outer')
                final_direct_table.reset_index(inplace=True) # Convert Date index back to column
            # --- END MODIFICATION ---

            st.dataframe(final_direct_table.set_index('Date').style.format('{:.2f}'), height=200)
        else: st.info("No direct forecasts generated.")
        
        st.markdown(f"###### Component Forecast ({component_model})")
        if not combined_forecast_component.empty:
            st.dataframe(combined_forecast_component.set_index('Date').style.format('{:.2f}'), height=200)
        else: st.info("Component forecast not generated.")

else:
    st.info("Configure settings in the sidebar and click the button to run the NIFTY 50 forecast comparison.")
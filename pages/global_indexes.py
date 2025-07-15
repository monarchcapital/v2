# pages/Global_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta, date, datetime

import config
from utils import (
    download_data, create_features, train_models_pipeline,
    generate_iterative_forecast,
    parse_int_list, add_market_data_features
)

st.set_page_config(page_title="Monarch: Global Index Analysis", layout="wide")

st.header("🌍 Global Index Analysis & Forecast")
st.markdown("Analyze historical data, compare models, and forecast future prices for major global indices. Use other indices as contextual features to improve model accuracy.")

# --- Sidebar ---
with st.sidebar:
    st.header("🛠️ Configuration Panel")
    
    # Dropdown for selecting a primary global index
    selected_index_name = st.selectbox(
        "Select Primary Index to Analyze:",
        options=list(config.GLOBAL_MARKET_TICKERS.keys())
    )
    ticker = config.GLOBAL_MARKET_TICKERS[selected_index_name]
    st.info(f"Selected Ticker: `{ticker}`")

    st.subheader("🗓️ Training & Prediction Period")
    today = date.today()
    default_end_bt = today - timedelta(days=1)
    default_start_bt = default_end_bt - timedelta(days=5*365)
    start_bt = st.date_input("Training Start Date (t1):", value=default_start_bt, key="ga_start_date")
    end_bt = st.date_input("Training End Date (t2):", value=default_end_bt, key="ga_end_date")
    n_future = st.slider("Predict Future Days:", 1, 90, config.DEFAULT_N_FUTURE_DAYS, key="ga_n_future")

    if start_bt >= end_bt:
        st.error("Start Date must be before End Date.")
        st.stop()

    st.subheader("🤖 Model Selection")
    selected_models_ga = st.multiselect(
        "Select Models to Compare:",
        options=[m for m in config.MODEL_CHOICES if m != 'Prophet'],
        default=[m for m in config.MODEL_CHOICES if m not in ['Prophet', 'KNN']][:3],
        key="ga_model_select"
    )

    st.subheader("➕ Feature Selection")
    with st.expander("🌐 Add Other Indices as Features"):
        # Filter out the primary index from the context selection list
        available_context_indices = [name for name in config.GLOBAL_MARKET_TICKERS.keys() if name != selected_index_name]
        selected_context_indices = st.multiselect(
            "Select Contextual Indices:",
            options=available_context_indices,
            default=[],
            key="ga_global_select"
        )
        selected_context_tickers = [config.GLOBAL_MARKET_TICKERS[name] for name in selected_context_indices]

    st.subheader("⚙️ Technical Indicator Settings")
    selected_indicator_params = {}
    with st.expander("Show Indicator Settings"):
        for indicator_name, (default_value, default_enabled) in config.TECHNICAL_INDICATORS_DEFAULTS.items():
            if st.checkbox(f"Enable {indicator_name.replace('_', ' ')}", value=default_enabled, key=f"enable_{indicator_name.lower()}_ga"):
                if isinstance(default_value, list):
                    selected_indicator_params[indicator_name] = parse_int_list(st.text_input(f"  {indicator_name.replace('_', ' ')} (days):", value=", ".join(map(str, default_value)), key=f"input_{indicator_name.lower()}_ga"), default_value, st.sidebar.error)
                elif isinstance(default_value, (int, float)):
                    min_val, step_val = (0.01, 0.01) if 'ACCEL' in indicator_name or isinstance(default_value, float) else (1.0, 1.0)
                    selected_indicator_params[indicator_name] = st.number_input(f"  {indicator_name.replace('_', ' ')}:", min_value=min_val, value=float(default_value), step=step_val, key=f"input_{indicator_name.lower()}_ga")

# --- Main Application Flow ---
if st.button(f"Run Analysis for {selected_index_name}", type="primary"):
    if not selected_models_ga:
        st.warning("Please select at least one model to compare."); st.stop()

    st.subheader(f"📈 Analysis & Forecast for {selected_index_name} ({ticker})")
    
    with st.spinner(f"Running analysis for {ticker}..."):
        data = download_data(ticker)
        if data.empty:
            st.error(f"Could not load data for {ticker}."); st.stop()
        
        # Add contextual features before creating technical indicators
        data_for_features = data.copy()
        if selected_context_tickers:
            data_for_features = add_market_data_features(data_for_features, "10y", (lambda x: None), selected_tickers=selected_context_tickers)
        
        df_features_full = create_features(data_for_features, selected_indicator_params)
        df_train_period = df_features_full[(df_features_full['Date'] >= pd.to_datetime(start_bt)) & (df_features_full['Date'] <= pd.to_datetime(end_bt))].copy()
        
        if df_train_period.empty:
            st.error("No data available in the selected training range."); st.stop()

        # --- Train models and generate forecasts ---
        all_forecasts = {}
        all_trained_models_ga = {}
        for model_name in selected_models_ga:
            st.write(f"Training {model_name}...")
            trained_models, _ = train_models_pipeline(df_train_period.copy(), model_name, False, (lambda x: None), selected_indicator_params)
            if trained_models:
                all_trained_models_ga[model_name] = trained_models
                future_df = generate_iterative_forecast(data, trained_models, ticker, n_future, end_bt, selected_indicator_params, (lambda x: None), selected_context_tickers)
                all_forecasts[model_name] = future_df
    
    st.success("Analysis complete.")
    
    # --- Display Forecast Chart and Data ---
    st.subheader("Model Forecast Comparison")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Historical Price', line=dict(color='black', width=2)))
    for model_name, forecast_df in all_forecasts.items():
        if not forecast_df.empty:
            fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Close'], mode='lines+markers', name=f'Forecast ({model_name})'))
    fig.update_layout(title=f'{selected_index_name} ({ticker}) Price Prediction', yaxis_title='Price / Level', height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    if all_forecasts:
        st.subheader("Consolidated Future Price Forecasts")
        final_table = None
        for model_name, forecast_df in all_forecasts.items():
            if not forecast_df.empty:
                forecast_subset = forecast_df[['Date', 'Close']].rename(columns={'Close': model_name})
                final_table = forecast_subset if final_table is None else pd.merge(final_table, forecast_subset, on='Date', how='outer')
        if final_table is not None:
            st.dataframe(final_table.set_index('Date').style.format("{:.2f}"), use_container_width=True)
    else:
        st.warning("Could not generate any future forecasts.")

    # --- Feature Importance Analysis ---
    st.subheader("Feature Importance Analysis")
    st.markdown("This section reveals which data points the model found most influential for its predictions.")
    if selected_models_ga and all_trained_models_ga:
        model_to_display = selected_models_ga[0]
        model_info = all_trained_models_ga.get(model_to_display, {}).get('Close')

        if model_info and hasattr(model_info.get('model'), 'feature_importances_'):
            imp_df = pd.DataFrame({
                'Feature': model_info['features'],
                'Importance': model_info['model'].feature_importances_
            }).sort_values(by='Importance', ascending=False).head(20)

            fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title=f"Top 20 Features for {model_to_display} on {selected_index_name}")
            fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)

            contextual_features_found = [feat for feat in imp_df['Feature'] if any(name.replace(' ', '_').replace('^', '') in feat for name in selected_context_indices)]
            if contextual_features_found:
                st.success("🌐 Contextual index features were influential in the forecast!")
                for feature in contextual_features_found: st.markdown(f"- **{feature}**")
            else:
                st.info("The most influential features were primarily technical indicators derived from the main index's price history.")
        else:
            st.info(f"Feature importance is not available for the '{model_to_display}' model type.")
    else:
        st.warning("Could not generate feature importance as no models were successfully trained.")

else:
    st.info("Select a global index from the sidebar and click 'Run Analysis' to see the forecast.")

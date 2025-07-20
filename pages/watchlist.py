# pages/watchlist.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta, date, datetime
import os
import time
import plotly.express as px

import config
from utils import (
    download_data, create_features, train_models_pipeline,
    generate_predictions_pipeline, add_market_data_features,
    make_ensemble_prediction,
    parse_int_list,
    add_fundamental_features
)

st.set_page_config(page_title="Monarch: My Watchlist", layout="wide")

st.header("📋 My Watchlist")
st.markdown("Add stocks to your watchlist for quick, next-day forecasts. This page allows for sophisticated analysis using individual or ensemble models with confidence intervals.")

# --- Watchlist Management in Sidebar---
WATCHLIST_FILE = "monarch_watchlist.txt"

def load_watchlist():
    """Loads the watchlist from a local text file."""
    if not os.path.exists(WATCHLIST_FILE): return []
    with open(WATCHLIST_FILE, "r") as f:
        return sorted(list(set([line.strip().upper() for line in f if line.strip()])))

def save_watchlist(tickers):
    """Saves the watchlist to a local text file."""
    with open(WATCHLIST_FILE, "w") as f:
        for ticker in sorted(list(set(tickers))):
            f.write(f"{ticker}\n")

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = load_watchlist()

with st.sidebar:
    st.header("🛠️ Watchlist Configuration")
    
    with st.expander("➕ Manage Watchlist", expanded=True):
        new_ticker = st.text_input("Add Ticker to Watchlist:", key="new_ticker_input").upper()
        if st.button("Add Ticker") and new_ticker:
            if new_ticker not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_ticker)
                save_watchlist(st.session_state.watchlist)
                st.success(f"'{new_ticker}' added!")
                st.rerun()
            else:
                st.warning(f"'{new_ticker}' is already in the watchlist.")

        if st.session_state.watchlist:
            ticker_to_remove = st.selectbox("Remove Ticker from Watchlist:", [""] + st.session_state.watchlist)
            if st.button("Remove Ticker") and ticker_to_remove:
                st.session_state.watchlist.remove(ticker_to_remove)
                save_watchlist(st.session_state.watchlist)
                st.success(f"'{ticker_to_remove}' removed.")
                st.rerun()

    # --- Watchlist Model Settings ---
    st.subheader("⚙️ Model & Feature Settings")
    today = date.today()
    default_end_bt_wl = today - timedelta(days=1)
    default_start_bt_wl = default_end_bt_wl - timedelta(days=3*365)

    start_date_watchlist = st.date_input("Training Start Date:", value=default_start_bt_wl, key="wl_start_date")
    end_date_watchlist = st.date_input("Training End Date:", value=default_end_bt_wl, key="wl_end_date")

    if start_date_watchlist >= end_date_watchlist:
        st.error("Start Date must be before End Date.")
        st.stop()

    # --- Global & Fundamental Context Selectors ---
    with st.expander("🌍 Global & 🔬 Fundamental Features"):
        available_indices_wl = list(config.GLOBAL_MARKET_TICKERS.keys())
        select_all_globals_wl = st.checkbox("Select All Global Indices", value=False, key="wl_select_all_globals")
        default_globals_wl = available_indices_wl if select_all_globals_wl else available_indices_wl[:3]
        selected_indices_wl = st.multiselect("Select Global Indices:", options=available_indices_wl, default=default_globals_wl, key="wl_global_select")
        selected_tickers_wl = [config.GLOBAL_MARKET_TICKERS[name] for name in selected_indices_wl]

        # Static Fundamentals
        available_fundamentals_static_wl = list(config.FUNDAMENTAL_METRICS.keys())
        select_all_fundamentals_static_wl = st.checkbox("Select All Static Fundamentals", value=False, key="wl_select_all_fundamentals_static")
        default_fundamentals_static_wl = available_fundamentals_static_wl if select_all_fundamentals_static_wl else []
        selected_fundamental_names_static_wl = st.multiselect("Select Static Fundamental Metrics:", options=available_fundamentals_static_wl, default=default_fundamentals_static_wl, key="wl_static_fundamental_select")
        selected_fundamentals_static_wl = {name: config.FUNDAMENTAL_METRICS[name] for name in selected_fundamental_names_static_wl}

        # Historical/Derived Fundamentals
        available_fundamentals_derived_wl = ['Historical P/E Ratio', 'Historical P/S Ratio', 'Historical Debt to Equity']
        select_all_fundamentals_derived_wl = st.checkbox("Select All Historical/Derived Fundamentals", value=False, key="wl_select_all_fundamentals_derived")
        default_fundamentals_derived_wl = available_fundamentals_derived_wl if select_all_fundamentals_derived_wl else []
        selected_fundamental_names_derived_wl = st.multiselect("Select Historical/Derived Metrics:", options=available_fundamentals_derived_wl, default=default_fundamentals_derived_wl, key="wl_derived_fundamental_select")
        selected_fundamentals_derived_wl = {name: name for name in selected_fundamental_names_derived_wl}

    # Combine all selected fundamental metrics
    combined_selected_fundamentals_wl = {**selected_fundamentals_static_wl, **selected_fundamentals_derived_wl}


    # --- Model Selection ---
    st.subheader("🤖 Model Selection")
    models_for_watchlist = st.multiselect(
        "Select Individual Models to Display:",
        options=[m for m in config.MODEL_CHOICES if m != 'Prophet'],
        default=['Random Forest', 'LightGBM']
    )
    
    # Ensemble Prediction
    enable_ensemble_wl = st.checkbox("Enable Ensemble Prediction (Recommended)", value=True, key="wl_ensemble")
    ensemble_models_wl = []
    if enable_ensemble_wl:
        ensemble_models_wl = st.multiselect(
            "Select Models for Ensemble:",
            [m for m in config.MODEL_CHOICES if m != 'Prophet'],
            default=['Random Forest', 'LightGBM', 'XGBoost'],
            key="wl_ensemble_models"
        )

    # Confidence Intervals
    enable_confidence_interval_wl = st.checkbox("Enable Confidence Intervals", value=True, key="wl_ci")
    confidence_level_pct_wl = 90
    if enable_confidence_interval_wl:
        confidence_level_pct_wl = st.slider("Confidence Level (%):", 70, 99, 90, 1, key="wl_ci_slider")

    watchlist_perform_tuning = st.checkbox("Perform Hyperparameter Tuning", value=False, key="wl_tuning", help="May significantly slow down processing.")

    # --- Technical Indicator Settings ---
    st.subheader("📊 Technical Indicator Settings")
    selected_indicator_params_watchlist = {}
    with st.expander("Show Indicator Settings"):
        for indicator_name, (default_value, default_enabled) in config.TECHNICAL_INDICATORS_DEFAULTS.items():
            if st.checkbox(f"Enable {indicator_name.replace('_', ' ')}", value=default_enabled, key=f"enable_{indicator_name.lower()}_wl"):
                if isinstance(default_value, list):
                    selected_indicator_params_watchlist[indicator_name] = parse_int_list(
                        st.text_input(f"  {indicator_name.replace('_', ' ')} (days):", value=", ".join(map(str, default_value)), key=f"input_{indicator_name.lower()}_wl"),
                        default_value, st.sidebar.error
                    )
                elif isinstance(default_value, (int, float)):
                    min_val = 0.01 if 'ACCEL' in indicator_name or isinstance(default_value, float) else 1.0
                    step_val = 0.01 if 'ACCEL' in indicator_name or isinstance(default_value, float) else 1.0
                    selected_indicator_params_watchlist[indicator_name] = st.number_input(
                        f"  {indicator_name.replace('_', ' ')}:", min_value=min_val, value=float(default_value), step=step_val, key=f"input_{indicator_name.lower()}_wl"
                    )

# --- Main Page Content ---

# Helper function to style the dashboard
def style_dashboard(df):
    """Applies color coding and formatting to the forecast dashboard."""
    def color_sentiment(val):
        color = 'green' if val == 'Bullish' else 'red' if val == 'Bearish' else 'grey'
        return f'color: {color}; font-weight: bold;'

    return df.style.applymap(color_sentiment, subset=['Sentiment']).format({
        'Predicted Change %': '{:+.2f}%',
        'Lower CI %': '{:+.2f}%',
        'Upper CI %': '{:+.2f}%',
    })

if not st.session_state.watchlist:
    st.info("Your watchlist is empty. Please add tickers using the sidebar to get started.")
elif not models_for_watchlist and not (enable_ensemble_wl and ensemble_models_wl):
    st.warning("Please select at least one individual model or enable and select models for the ensemble prediction.")
else:
    # Placeholders for real-time updates
    st.subheader("🚀 Forecast Dashboard")
    dashboard_placeholder = st.empty()
    
    st.subheader("Detailed Forecast Results")
    results_placeholder = st.empty()

    if st.button("Update Watchlist Forecasts", type="primary"):
        st.warning("Processing a large watchlist can be time-consuming. Please be patient.")
        
        all_results_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, ticker_item in enumerate(st.session_state.watchlist):
            status_text.text(f"Processing {ticker_item} ({idx+1}/{len(st.session_state.watchlist)})...")
            
            item_data = {'Ticker': ticker_item, 'Status': 'Processing'}
            df_raw = download_data(ticker_item)
            if df_raw.empty:
                item_data['Status'] = "No data"
                all_results_data.append(item_data)
                continue

            data_for_features = df_raw.copy()
            if combined_selected_fundamentals_wl:
                data_for_features = add_fundamental_features(data_for_features, ticker_item, combined_selected_fundamentals_wl, _update_log_func=(lambda x: None))
            if selected_tickers_wl:
                data_for_features = add_market_data_features(data_for_features, "10y", (lambda x: None), selected_tickers=selected_tickers_wl)

            item_data['Last Close'] = df_raw['Close'].iloc[-1]
            df_features = create_features(data_for_features, selected_indicator_params_watchlist)
            
            if df_features.empty:
                item_data['Status'] = "Not enough data for features"
                all_results_data.append(item_data)
                continue

            last_day_features = df_features.tail(1)
            df_train_period_watchlist = df_features[(df_features['Date'] >= pd.to_datetime(start_date_watchlist)) & (df_features['Date'] <= pd.to_datetime(end_date_watchlist))].copy()

            if df_train_period_watchlist.empty:
                item_data['Status'] = "No data in training range"
                all_results_data.append(item_data)
                continue
            
            # --- Ensemble Prediction Logic ---
            if enable_ensemble_wl and ensemble_models_wl:
                ensemble_trained_models = []
                for ens_model_name in ensemble_models_wl:
                    trained_ens_model, _ = train_models_pipeline(df_train_period_watchlist.copy(), ens_model_name, watchlist_perform_tuning, (lambda x: None), selected_indicator_params_watchlist)
                    if trained_ens_model:
                        ensemble_trained_models.append(trained_ens_model)
                
                if ensemble_trained_models:
                    ens_close, ens_lower, ens_upper = make_ensemble_prediction(ensemble_trained_models, last_day_features.copy(), (lambda x: None), confidence_level_pct_wl if enable_confidence_interval_wl else None)
                    item_data['Ensemble Close'] = ens_close[0] if isinstance(ens_close, np.ndarray) else ens_close
                    item_data['Ensemble Lower'] = ens_lower[0] if isinstance(ens_lower, np.ndarray) else ens_lower
                    item_data['Ensemble Upper'] = ens_upper[0] if isinstance(ens_upper, np.ndarray) else ens_upper

            # --- Individual Model Prediction Logic ---
            for model_name in models_for_watchlist:
                trained_models, _ = train_models_pipeline(df_train_period_watchlist.copy(), model_name, watchlist_perform_tuning, (lambda x: None), selected_indicator_params_watchlist)
                
                if trained_models:
                    preds_dict = generate_predictions_pipeline(last_day_features.copy(), trained_models, (lambda x: None), confidence_level_pct_wl if enable_confidence_interval_wl else None)
                    for target in ['Close', 'Open', 'High', 'Low']:
                        if target in preds_dict:
                            item_data[f'{target} ({model_name})'] = preds_dict[target][f'Predicted {target}'].iloc[0]
                            if pd.notna(preds_dict[target][f'Predicted {target} Lower'].iloc[0]):
                                item_data[f'{target} Lower ({model_name})'] = preds_dict[target][f'Predicted {target} Lower'].iloc[0]
                            if pd.notna(preds_dict[target][f'Predicted {target} Upper'].iloc[0]):
                                item_data[f'{target} Upper ({model_name})'] = preds_dict[target][f'Predicted {target} Upper'].iloc[0]

            item_data['Status'] = 'OK'
            all_results_data.append(item_data)
            
            # --- UPDATE DASHBOARD AND TABLES IN REAL-TIME ---
            # 1. Create the detailed results dataframe
            detailed_df = pd.DataFrame(all_results_data)
            
            # 2. Create and populate the dashboard dataframe
            dashboard_data = []
            primary_model_col = 'Ensemble Close' if (enable_ensemble_wl and 'Ensemble Close' in detailed_df.columns) else f'Close ({models_for_watchlist[0]})' if models_for_watchlist else None
            
            if primary_model_col and primary_model_col in detailed_df.columns:
                for _, row in detailed_df.iterrows():
                    if row['Status'] == 'OK':
                        pred_change = ((row[primary_model_col] - row['Last Close']) / row['Last Close']) * 100
                        
                        lower_ci_pct, upper_ci_pct = np.nan, np.nan
                        lower_col = 'Ensemble Lower' if 'Ensemble Lower' in row else f'{primary_model_col.split(" ")[0]} Lower ({primary_model_col.split(" ")[1][1:-1]})'
                        upper_col = 'Ensemble Upper' if 'Ensemble Upper' in row else f'{primary_model_col.split(" ")[0]} Upper ({primary_model_col.split(" ")[1][1:-1]})'

                        if enable_confidence_interval_wl and lower_col in row and pd.notna(row[lower_col]):
                            lower_ci_pct = ((row[lower_col] - row['Last Close']) / row['Last Close']) * 100
                        if enable_confidence_interval_wl and upper_col in row and pd.notna(row[upper_col]):
                             upper_ci_pct = ((row[upper_col] - row['Last Close']) / row['Last Close']) * 100

                        dashboard_data.append({
                            'Ticker': row['Ticker'],
                            'Sentiment': 'Bullish' if pred_change > 0 else 'Bearish',
                            'Predicted Change %': pred_change,
                            'Lower CI %': lower_ci_pct,
                            'Upper CI %': upper_ci_pct
                        })
            
            dashboard_df = pd.DataFrame(dashboard_data)
            
            # 3. Update placeholders
            if not dashboard_df.empty:
                dashboard_placeholder.dataframe(style_dashboard(dashboard_df[['Ticker', 'Sentiment', 'Predicted Change %', 'Lower CI %', 'Upper CI %']]), use_container_width=True, hide_index=True)

            results_placeholder.dataframe(detailed_df, use_container_width=True, hide_index=True)
            progress_bar.progress((idx + 1) / len(st.session_state.watchlist))

        progress_bar.empty()
        status_text.empty()

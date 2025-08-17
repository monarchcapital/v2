# home.py - Main Streamlit application page for stock price prediction

import streamlit as st
from datetime import date, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
import tensorflow as tf
import os
import shutil
import traceback
import json
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# --- IMPROVEMENT: Centralized pipeline import ---
from pipeline import run_prediction_pipeline
from utils import plot_predictions, display_log

warnings.filterwarnings('ignore', category=FutureWarning)
tf.get_logger().setLevel('ERROR')

st.set_page_config(
    page_title="Stock Price Predictor (Deep Learning Hybrid)",
    page_icon="✅",
    layout="wide"
)

# --- Hide Streamlit Menu & Footer ---
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- API Key Configuration ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except (KeyError, FileNotFoundError):
    st.sidebar.warning("Google AI API Key not found. News analysis will be disabled.", icon="⚠️")
    GOOGLE_API_KEY = None

st.title("✅ Stock Price Predictor (Deep Learning Hybrid)")

# --- Paths for saving models and history ---
BEST_MODELS_DIR = "best_models"
MANUAL_RUNS_DIR = "manual_runs"

# --- Refactored State Management ---
def initialize_session_state():
    """Initializes all required session state keys with default values."""
    defaults = {
        'run_analysis': False, 'log_history': [],
        'analysis_results': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_analysis_state():
    """Resets the state for a new prediction run, clearing the log."""
    keys_to_reset = list(st.session_state.keys())
    for key in keys_to_reset:
        del st.session_state[key]
    initialize_session_state()
    st.session_state['log_history'] = []

# Call initialization once at the start
initialize_session_state()

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Configuration")
ticker_symbol = st.sidebar.text_input("Select Stock Ticker", value='AAPL')
today = date.today()
start_date = st.sidebar.date_input("Start Date", value=today - timedelta(days=5*365))
end_date = st.sidebar.date_input("End Date", value=today)

st.sidebar.subheader("Model Parameters")
training_window_size = st.sidebar.slider("Time steps (look-back days)", 5, 60, 30)
test_set_split_ratio = st.sidebar.slider("Test set split ratio", 0.10, 0.50, 0.20, 0.05)
prediction_horizon = st.sidebar.slider("Prediction Horizon (days)", 1, 30, 5)
model_type = st.sidebar.radio("Select Model Type", ('LSTM', 'Transformer', 'Hybrid (LSTM + Transformer)'), index=2)

st.sidebar.subheader("Optional Features")
all_indicators = ["SMA_20", "SMA_50", "RSI", "MACD", "OBV", "Bollinger Bands", "ATR", "MFI"]
selected_indicators = st.sidebar.multiselect("Technical Indicators", all_indicators, default=["SMA_20", "RSI", "MACD"])

all_macro_indicators = ['S&P 500', 'Crude Oil', 'DXY', '10-Year Yield', 'VIX', 'Nifty 50']
selected_macro_indicators = st.sidebar.multiselect("Macroeconomic Indicators", all_macro_indicators, default=[])

all_fundamental_indicators = ['Total Revenue', 'Net Income', 'EBITDA', 'Total Assets', 'Total Liabilities', 'Operating Cash Flow', 'Capital Expenditure']
selected_fundamental_indicators = st.sidebar.multiselect("Fundamental Indicators", all_fundamental_indicators, default=[])

apply_differencing = st.sidebar.checkbox("Apply Differencing for Stationarity", value=False)
enable_pca = st.sidebar.checkbox("Enable PCA", value=False)
if enable_pca:
    pca_mode = st.sidebar.radio("PCA Mode", ["Automatic (95% variance)", "Manual"], index=0)
    if pca_mode == "Manual":
        n_components_manual = st.sidebar.slider("Number of Components", 2, 20, 5, 1)
    else:
        n_components_manual = 0
else:
    n_components_manual = 0

st.sidebar.subheader("Hyperparameter Mode")
hp_mode = st.sidebar.radio("Select Mode", ["Automatic (Tuner)", "Manual"])
force_retune = st.sidebar.checkbox("Force Hyperparameter Retune (if Automatic)", value=False)

manual_params = {}
if hp_mode == "Automatic (Tuner)":
    st.sidebar.warning("⚠️ Automatic tuning is very time-consuming.")
    epochs = st.sidebar.slider("Max Epochs for Tuning", 10, 100, 30, 10)
    num_trials = st.sidebar.slider("Number of Tuning Trials", 5, 100, 30, 5)
    executions_per_trial = st.sidebar.number_input("Executions per Trial", 1, 3, 1)
    batch_size, learning_rate = 32, 0.001 # Defaults, will be overridden by tuner
else:
    st.sidebar.subheader("Manual Hyperparameters")
    experiment_note = st.sidebar.text_input("Experiment Note", value="Manual Run")
    epochs = st.sidebar.number_input("Epochs (Manual)", 1, 100, 30)
    batch_size = st.sidebar.number_input("Batch Size (Manual)", 1, 256, 32)
    learning_rate = st.sidebar.number_input("Learning Rate (Manual)", 0.0001, 0.1, 0.001, format="%.4f")
    if model_type == 'LSTM':
        manual_params['num_lstm_layers'] = st.sidebar.slider("LSTM Layers", 1, 3, 2)
        manual_params['lstm_units_1'] = st.sidebar.slider("LSTM Layer 1 Units", 32, 256, 100, 32)
        manual_params['dropout_1'] = st.sidebar.slider("LSTM Layer 1 Dropout", 0.0, 0.5, 0.2, 0.05)
    elif model_type == 'Transformer':
        manual_params['embed_dim'] = st.sidebar.slider("Embedding Dimension", 32, 128, 64, 16)
        manual_params['num_transformer_blocks'] = st.sidebar.slider("Transformer Blocks", 1, 4, 2)
        manual_params['num_heads'] = st.sidebar.slider("Attention Heads", 1, 8, 4)
        manual_params['ff_dim'] = st.sidebar.slider("Feed Forward Dim", 16, 64, 32, 16)
    elif model_type == 'Hybrid (LSTM + Transformer)':
        manual_params['lstm_units'] = st.sidebar.slider("LSTM Units (Hybrid)", 32, 256, 64, 32)
        manual_params['lstm_dropout'] = st.sidebar.slider("LSTM Dropout (Hybrid)", 0.0, 0.5, 0.2, 0.05)
        manual_params['num_transformer_blocks'] = st.sidebar.slider("Transformer Blocks (Hybrid)", 1, 4, 1)
        manual_params['num_heads'] = st.sidebar.slider("Attention Heads (Hybrid)", 1, 8, 2)
    num_trials = 0
    executions_per_trial = 0

use_ensemble = st.sidebar.checkbox("Use Ensemble of Models", value=True)
if use_ensemble:
    num_ensemble_models = st.sidebar.slider("Ensemble Size", 2, 10, 5)
else:
    num_ensemble_models = 1

if st.sidebar.button("🚀 Run Prediction"):
    reset_analysis_state()
    st.session_state['run_analysis'] = True
    st.rerun()

if st.sidebar.button("🧹 Clear Cache"):
    reset_analysis_state()
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Cache cleared!")
    st.rerun()

# --- Main Content Area ---
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Analysis & Results")
    progress_bar = st.empty()

with col2:
    st.header("📰 AI News Analysis")
    news_container = st.container(border=True)

# --- Main Prediction Logic ---
if st.session_state['run_analysis']:
    with st.spinner(f"Running full analysis pipeline for {ticker_symbol}..."):
        # --- IMPROVEMENT: Call the centralized pipeline ---
        results = run_prediction_pipeline(
            ticker_symbol, start_date, end_date, selected_indicators,
            selected_macro_indicators, selected_fundamental_indicators,
            apply_differencing, enable_pca, n_components_manual,
            training_window_size, test_set_split_ratio, prediction_horizon,
            model_type, hp_mode, force_retune, epochs, num_trials,
            executions_per_trial, manual_params, batch_size, learning_rate,
            num_ensemble_models
        )
        st.session_state['analysis_results'] = results
        st.session_state['run_analysis'] = False # Prevent re-running on next interaction

# --- Display Results ---
results = st.session_state.get('analysis_results')
if results:
    if 'error' in results:
        st.error(f"An error occurred during analysis: {results['error']}")
    else:
        with col1:
            if results['model_metrics']:
                st.subheader("📈 Performance Metrics (Test Set - Day 1 Forecast)")
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("MAE", f"{results['model_metrics']['MAE']:.2f}")
                m_col2.metric("RMSE", f"{results['model_metrics']['RMSE']:.2f}")
                m_col3.metric("R²", f"{results['model_metrics']['R2']:.2f}")

            if results['future_predicted_prices'] is not None:
                st.subheader("🔮 Future Predicted Prices")
                fig = plot_predictions(
                    actual_prices=results['data']['Close'][-results['y_test_len'] - training_window_size:],
                    predicted_prices=results['test_predicted_prices'],
                    ticker=ticker_symbol,
                    future_predictions=results['future_predicted_prices']
                )
                st.plotly_chart(fig, use_container_width=True)

            if results.get('pca_object') is not None:
                with st.expander("🔬 Principal Component Analysis (PCA) Insights", expanded=False):
                    pca = results['pca_object']
                    feature_names = results['original_feature_names']
                    exp_var_ratio = pca.explained_variance_ratio_
                    cum_exp_var = np.cumsum(exp_var_ratio)
                    
                    pca_df = pd.DataFrame({'Principal Component': [f'PC_{i+1}' for i in range(len(exp_var_ratio))],
                                           'Explained Variance': exp_var_ratio, 'Cumulative Explained Variance': cum_exp_var})
                    
                    fig_var = px.bar(pca_df, x='Principal Component', y='Explained Variance', text=[f'{x:.1%}' for x in exp_var_ratio],
                                     title="Explained Variance by Principal Component")
                    fig_var.add_trace(go.Scatter(x=pca_df['Principal Component'], y=cum_exp_var, name='Cumulative Variance', mode='lines+markers'))
                    st.plotly_chart(fig_var, use_container_width=True)

                    st.markdown("##### Component Loadings Heatmap")
                    st.markdown("Shows how original features contribute to each principal component. Bright red indicates strong positive correlation, bright blue strong negative.")
                    loadings_df = pd.DataFrame(pca.components_.T, columns=[f'PC_{i+1}' for i in range(pca.n_components_)], index=feature_names)
                    fig_loadings = px.imshow(loadings_df, text_auto='.2f', aspect="auto",
                                             title="Heatmap of PCA Component Loadings",
                                             color_continuous_scale=px.colors.diverging.RdBu_r,
                                             color_continuous_midpoint=0,
                                             height=max(400, 25 * len(feature_names)))
                    st.plotly_chart(fig_loadings, use_container_width=True)
        with col2:
            if results.get('news_summary'):
                news_container.markdown(results['news_summary'])
            else:
                news_container.info("News summary could not be generated.")

elif not st.session_state.get('run_analysis') and not st.session_state.get('analysis_results'):
     col1.info("Configure parameters in the sidebar and click 'Run Prediction' to start.")
     col2.info("Run an analysis to see the latest AI-powered news summary.")

with st.expander("See Logs", expanded=False):
    for log in st.session_state.get('log_history', []):
        if log['level'] == 'info': st.info(log['message'])
        elif log['level'] == 'warning': st.warning(log['message'])
        else: st.error(log['message'])

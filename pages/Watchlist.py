# pages/watchlist.py - A Streamlit page for a stock watchlist with multiple tabs.

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import warnings
import os
import json
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# --- IMPROVEMENT: Import the centralized pipeline ---
from pipeline import run_prediction_pipeline

# Suppress warnings to keep the UI clean
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Page Configuration ---
st.set_page_config(page_title="Stock Watchlists", page_icon="📈", layout="wide")

# --- Hide Streamlit Menu & Footer ---
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- File-Based Persistence Functions ---
WATCHLIST_FILE = "watchlists.txt"

def load_watchlists_from_file():
    """Loads watchlists from a text file."""
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, 'r') as f:
                data = json.load(f)
                for key in ['Watchlist 1', 'Watchlist 2', 'Watchlist 3']:
                    data[key] = set(data.get(key, []))
                return data
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    return {'Watchlist 1': set(), 'Watchlist 2': set(), 'Watchlist 3': set()}

def save_watchlists_to_file(watchlists):
    """Saves watchlists to a text file."""
    try:
        with open(WATCHLIST_FILE, 'w') as f:
            serializable_watchlists = {k: list(v) for k, v in watchlists.items()}
            json.dump(serializable_watchlists, f)
    except IOError as e:
        st.error(f"Error saving watchlists to file: {e}")

# --- API Key Configuration ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except (KeyError, FileNotFoundError):
    st.sidebar.warning("Google AI API Key not found. News analysis will be disabled.", icon="⚠️")
    GOOGLE_API_KEY = None

# --- Initialize Session State ---
if 'watchlists' not in st.session_state:
    st.session_state.watchlists = load_watchlists_from_file()

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Watchlist Prediction Parameters")
today = date.today()
start_date_input = st.sidebar.date_input("Start Date", value=today - timedelta(days=5*365), key="wl_start_date")
end_date_input = st.sidebar.date_input("End Date", value=today, key="wl_end_date")

st.sidebar.subheader("Model Parameters")
training_window_size_input = st.sidebar.slider("Time steps (look-back days)", 5, 60, 30, key="wl_ts")
test_set_split_ratio_input = st.sidebar.slider("Test set split ratio", 0.10, 0.50, 0.20, 0.05, key="wl_split")
prediction_horizon_input = st.sidebar.slider("Prediction Horizon (days)", 1, 30, 5, key="wl_horizon")
model_type_input = st.sidebar.radio("Select Model Type", ('LSTM', 'Transformer', 'Hybrid (LSTM + Transformer)'), index=2, key="wl_model_type")

st.sidebar.subheader("Optional Features")
all_tech_indicators = ["SMA_20", "SMA_50", "RSI", "MACD", "OBV", "Bollinger Bands", "ATR", "MFI"]
selected_tech_indicators = st.sidebar.multiselect("Technical Indicators", all_tech_indicators, default=["SMA_20", "RSI", "MACD"], key="wl_tech")

all_macro_indicators = ['S&P 500', 'Crude Oil', 'DXY', '10-Year Yield', 'VIX', 'Nifty 50']
selected_macro_indicators = st.sidebar.multiselect("Macroeconomic Indicators", all_macro_indicators, default=[], key="wl_macro")

all_fundamental_indicators = ['Total Revenue', 'Net Income', 'EBITDA', 'Total Assets', 'Total Liabilities', 'Operating Cash Flow', 'Capital Expenditure']
selected_fundamental_indicators = st.sidebar.multiselect("Fundamental Indicators", all_fundamental_indicators, default=[], key="wl_fundamental")

st.sidebar.subheader("Ensemble Mode")
use_ensemble = st.sidebar.checkbox("Use Ensemble of Models", value=True, key="wl_ensemble")
if use_ensemble:
    num_ensemble_models = st.sidebar.slider("Ensemble Size", 2, 10, 5, key="wl_ensemble_size")
else:
    num_ensemble_models = 1

# For watchlists, we'll stick to manual mode for simplicity and speed.
# Advanced tuning can be done on the main ML Prediction page.
st.sidebar.subheader("Manual Hyperparameters")
epochs = st.sidebar.number_input("Epochs (Manual)", 1, 100, 30, key="wl_epochs")
batch_size = st.sidebar.number_input("Batch Size (Manual)", 1, 256, 32, key="wl_batch_size")
learning_rate = st.sidebar.number_input("Learning Rate (Manual)", 0.0001, 0.1, 0.001, format="%.4f", key="wl_lr")

manual_params = {}
if model_type_input == 'LSTM':
    manual_params['num_lstm_layers'] = st.sidebar.slider("LSTM Layers", 1, 3, 2, key="wl_lstm_layers")
    manual_params['lstm_units_1'] = st.sidebar.slider("LSTM Layer 1 Units", 32, 256, 100, 32, key="wl_lstm_u1")
    manual_params['dropout_1'] = st.sidebar.slider("LSTM Layer 1 Dropout", 0.0, 0.5, 0.2, 0.05, key="wl_lstm_d1")
elif model_type_input == 'Transformer':
    manual_params['embed_dim'] = st.sidebar.slider("Embedding Dim", 32, 128, 64, 16, key="wl_trans_embed")
    manual_params['num_transformer_blocks'] = st.sidebar.slider("Transformer Blocks", 1, 4, 2, key="wl_trans_blocks")
    manual_params['num_heads'] = st.sidebar.slider("Attention Heads", 1, 8, 4, key="wl_trans_heads")
    manual_params['ff_dim'] = st.sidebar.slider("Feed Forward Dim", 16, 64, 32, 16, key="wl_trans_ff")
elif model_type_input == 'Hybrid (LSTM + Transformer)':
    manual_params['lstm_units'] = st.sidebar.slider("LSTM Units (Hybrid)", 32, 256, 64, 32, key="wl_hy_lstm_u")
    manual_params['lstm_dropout'] = st.sidebar.slider("LSTM Dropout (Hybrid)", 0.0, 0.5, 0.2, 0.05, key="wl_hy_lstm_d")
    manual_params['num_transformer_blocks'] = st.sidebar.slider("Transformer Blocks (Hybrid)", 1, 4, 1, key="wl_hy_trans_b")
    manual_params['num_heads'] = st.sidebar.slider("Attention Heads (Hybrid)", 1, 8, 2, key="wl_hy_trans_h")


# --- UI Display Logic ---
st.title("📈 My Stock Watchlists")
st.markdown("Create and manage your stock watchlists. Use the sidebar to configure prediction parameters and run analysis on all tickers in a list.")

tab1, tab2, tab3 = st.tabs(["Watchlist 1", "Watchlist 2", "Watchlist 3"])

def display_watchlist_tab(tab_name, tab_content):
    with tab_content:
        st.header(f"{tab_name}")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            new_ticker = st.text_input("Add Ticker", key=f"add_{tab_name}").upper()
        with col2:
            st.write("") 
            if st.button(f"Add to {tab_name}", use_container_width=True, key=f"add_btn_{tab_name}"):
                if new_ticker:
                    st.session_state.watchlists[tab_name].add(new_ticker)
                    save_watchlists_to_file(st.session_state.watchlists)
                    st.success(f"Added {new_ticker}!")
                    st.rerun()

        st.markdown("---")
        if not st.session_state.watchlists[tab_name]:
            st.info("This watchlist is empty.")
        else:
            sorted_tickers = sorted(list(st.session_state.watchlists[tab_name]))
            for i in range(0, len(sorted_tickers), 4):
                cols = st.columns(4)
                for j in range(4):
                    if i + j < len(sorted_tickers):
                        ticker = sorted_tickers[i+j]
                        with cols[j]:
                            with st.container(border=True):
                                st.subheader(ticker)
                                if st.button("Remove", key=f"rem_{tab_name}_{ticker}", use_container_width=True):
                                    st.session_state.watchlists[tab_name].remove(ticker)
                                    save_watchlists_to_file(st.session_state.watchlists)
                                    st.rerun()
        
        st.markdown("---")
        if st.button(f"🚀 Run Full Analysis for {tab_name}", use_container_width=True, key=f"run_{tab_name}"):
            watchlist_tickers = st.session_state.watchlists[tab_name]
            if not watchlist_tickers:
                st.warning("Cannot run analysis on an empty watchlist.")
                return

            st.info(f"Running analysis for {len(watchlist_tickers)} tickers...")
            progress_bar = st.progress(0, text="Initializing...")
            all_results = {}

            for i, ticker in enumerate(watchlist_tickers):
                progress_bar.progress((i + 1) / len(watchlist_tickers), text=f"Analyzing {ticker}...")
                
                # --- IMPROVEMENT: Call the centralized pipeline ---
                results = run_prediction_pipeline(
                    ticker, start_date_input, end_date_input, selected_tech_indicators,
                    selected_macro_indicators, selected_fundamental_indicators,
                    apply_differencing=False, enable_pca=False, n_components_manual=0, # Simplified for watchlist
                    training_window_size=training_window_size_input,
                    test_set_split_ratio=test_set_split_ratio_input,
                    prediction_horizon=prediction_horizon_input,
                    model_type=model_type_input,
                    hp_mode="Manual", # Always manual for watchlist
                    force_retune=False, epochs=epochs, num_trials=0, executions_per_trial=0,
                    manual_params=manual_params, batch_size=batch_size, learning_rate=learning_rate,
                    num_ensemble_models=num_ensemble_models
                )

                if 'error' in results:
                    st.error(f"Could not analyze {ticker}: {results['error']}")
                else:
                    all_results[ticker] = results
            
            st.session_state[f'results_{tab_name}'] = all_results
            progress_bar.empty()

        if f'results_{tab_name}' in st.session_state:
            st.success("Analysis complete!")
            for ticker, result in st.session_state[f'results_{tab_name}'].items():
                st.subheader(f"🔮 Results for {ticker}")
                
                res_col1, res_col2 = st.columns([1,1])
                with res_col1:
                    st.markdown("**Future Price Predictions**")
                    if result['future_predicted_prices'] is not None:
                        predictions_df = result['future_predicted_prices'].reset_index()
                        predictions_df.columns = ['Date', 'Prediction']
                        predictions_df['Date'] = predictions_df['Date'].dt.strftime('%Y-%m-%d')
                        predictions_df['Prediction'] = predictions_df['Prediction'].map('${:,.2f}'.format)
                        st.dataframe(predictions_df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("No future predictions were generated.")

                with res_col2:
                    st.markdown("**AI News Summary**")
                    with st.container(height=300):
                         st.markdown(result['news_summary'])

                st.markdown("---")

# --- Render Tabs ---
display_watchlist_tab("Watchlist 1", tab1)
display_watchlist_tab("Watchlist 2", tab2)
display_watchlist_tab("Watchlist 3", tab3)

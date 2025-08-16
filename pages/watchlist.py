# pages/watchlist.py - A Streamlit page for a stock watchlist with multiple tabs.

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import warnings
import os
import json

# --- Import Core Functions from other modules ---
from data import fetch_stock_data, add_technical_indicators, add_macro_indicators, add_fundamental_indicators
from utils import preprocess_data, create_sequences, calculate_metrics
from model import build_lstm_model, build_transformer_model, build_hybrid_model, train_model, predict_prices

# Suppress warnings to keep the UI clean
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

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

# --- Page Configuration ---
st.set_page_config(page_title="Stock Watchlists", page_icon="📈", layout="wide")

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

st.sidebar.subheader("Hyperparameter & Ensemble Mode")
use_ensemble = st.sidebar.checkbox("Use Ensemble of Models", value=True, key="wl_ensemble")
if use_ensemble:
    num_ensemble_models = st.sidebar.slider("Ensemble Size", 2, 10, 5, key="wl_ensemble_size")
else:
    num_ensemble_models = 1

manual_params = {}
st.sidebar.subheader("Manual Hyperparameters")
epochs = st.sidebar.number_input("Epochs (Manual)", 1, 100, 30, key="wl_epochs")
batch_size = st.sidebar.number_input("Batch Size (Manual)", 1, 256, 32, key="wl_batch_size")
learning_rate = st.sidebar.number_input("Learning Rate (Manual)", 0.0001, 0.1, 0.001, format="%.4f", key="wl_lr")

if model_type_input == 'LSTM':
    manual_params['num_lstm_layers'] = st.sidebar.slider("LSTM Layers", 1, 3, 2, key="wl_lstm_layers")
    manual_params['lstm_units_1'] = st.sidebar.slider("LSTM Layer 1 Units", 32, 256, 100, 32, key="wl_lstm_u1")
    manual_params['dropout_1'] = st.sidebar.slider("LSTM Layer 1 Dropout", 0.0, 0.5, 0.2, 0.05, key="wl_lstm_d1")
elif model_type_input == 'Transformer':
    manual_params['num_transformer_blocks'] = st.sidebar.slider("Transformer Blocks", 1, 4, 2, key="wl_trans_blocks")
    manual_params['num_heads'] = st.sidebar.slider("Attention Heads", 1, 8, 4, key="wl_trans_heads")
    manual_params['ff_dim'] = st.sidebar.slider("Feed Forward Dim", 16, 64, 32, 16, key="wl_trans_ff")
elif model_type_input == 'Hybrid (LSTM + Transformer)':
    manual_params['lstm_units'] = st.sidebar.slider("LSTM Units (Hybrid)", 32, 256, 64, 32, key="wl_hy_lstm_u")
    manual_params['lstm_dropout'] = st.sidebar.slider("LSTM Dropout (Hybrid)", 0.0, 0.5, 0.2, 0.05, key="wl_hy_lstm_d")
    manual_params['num_transformer_blocks'] = st.sidebar.slider("Transformer Blocks (Hybrid)", 1, 4, 1, key="wl_hy_trans_b")
    manual_params['num_heads'] = st.sidebar.slider("Attention Heads (Hybrid)", 1, 8, 2, key="wl_hy_trans_h")

# --- Main Prediction Logic for a Single Ticker ---
def run_prediction_for_ticker(ticker, start_date, end_date, selected_tech, selected_macro, selected_fund, window_size, split_ratio, horizon, model_type, manual_params, epochs, batch_size, learning_rate, num_models):
    """
    Runs the full prediction pipeline for a single ticker, including ensembling.
    """
    # 1. Fetch and prepare data
    data = fetch_stock_data(ticker, start_date, end_date)
    if data.empty: return None, None, f"Data fetching failed for {ticker}."
    if selected_tech: data = add_technical_indicators(data, selected_tech)
    if selected_macro: data = add_macro_indicators(data, selected_macro)
    if selected_fund: data = add_fundamental_indicators(data, ticker, selected_fund)
    if data.empty: return None, None, f"Data became empty after adding features for {ticker}."

    processed_data, scaler, close_col_index, last_actuals, _ = preprocess_data(data.copy())
    if processed_data.empty: return None, None, f"Data preprocessing failed for {ticker}."

    target_idx = processed_data.columns.get_loc('Close_scaled')
    X, y = create_sequences(processed_data.values, target_idx, window_size, horizon)
    if X.size == 0: return None, None, f"Failed to create sequences for {ticker}."

    test_size = int(len(X) * split_ratio)
    X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]

    all_future_preds = []
    all_test_preds = []
    
    # 2. Build, Train, and Predict for each model in the ensemble
    for i in range(num_models):
        model_builder = {'LSTM': build_lstm_model, 'Transformer': build_transformer_model, 'Hybrid (LSTM + Transformer)': build_hybrid_model}
        model = model_builder[model_type]((X_train.shape[1], X_train.shape[2]), horizon, manual_params=manual_params, learning_rate=learning_rate)
        train_model(model, X_train, y_train, epochs, batch_size, learning_rate, X_val=X_test, y_val=y_test)
        
        future_p = predict_prices(model, processed_data, scaler, close_col_index, window_size, horizon, last_actuals)
        all_future_preds.append(future_p)

        test_p_scaled = model.predict(X_test)[:, 0]
        dummy_pred = np.zeros((len(test_p_scaled), scaler.n_features_in_))
        dummy_pred[:, close_col_index] = test_p_scaled
        all_test_preds.append(scaler.inverse_transform(dummy_pred)[:, close_col_index])

    # 3. Process Ensemble Results
    if not all_future_preds:
        return None, None, "All models in the ensemble failed to produce predictions."

    # Future predictions
    avg_future_preds = np.mean(all_future_preds, axis=0)
    std_future_preds = np.std(all_future_preds, axis=0)
    last_date = data.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=horizon)
    
    results_df = pd.DataFrame({
        'Date': future_dates,
        'Ensemble Prediction': avg_future_preds,
        'Confidence Lower': avg_future_preds - 1.96 * std_future_preds,
        'Confidence Upper': avg_future_preds + 1.96 * std_future_preds
    })

    # Metrics calculation
    y_test_first_step = y_test[:, 0]
    dummy_actual = np.zeros((len(y_test_first_step), scaler.n_features_in_))
    dummy_actual[:, close_col_index] = y_test_first_step
    actual_prices = scaler.inverse_transform(dummy_actual)[:, close_col_index]
    
    avg_test_preds = np.mean(all_test_preds, axis=0)
    rmse, mae, r2 = calculate_metrics(actual_prices, avg_test_preds)
    metrics = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    
    return results_df, metrics, None

# --- UI Display Logic ---
st.title("📈 My Stock Watchlists")
st.markdown("Create and manage your stock watchlists. Use the sidebar to configure prediction parameters and run analysis on all tickers in a list.")

tab1, tab_2, tab3 = st.tabs(["Watchlist 1", "Watchlist 2", "Watchlist 3"])

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
            for ticker in sorted(list(st.session_state.watchlists[tab_name])):
                c1, c2 = st.columns([4, 1])
                c1.subheader(ticker)
                if c2.button("Remove", key=f"rem_{tab_name}_{ticker}", use_container_width=True):
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
            progress_bar = st.progress(0)
            all_results = {}

            for i, ticker in enumerate(watchlist_tickers):
                with st.spinner(f"Analyzing {ticker}..."):
                    predictions_df, metrics, error = run_prediction_for_ticker(
                        ticker, start_date_input, end_date_input, selected_tech_indicators,
                        selected_macro_indicators, selected_fundamental_indicators,
                        training_window_size_input, test_set_split_ratio_input,
                        prediction_horizon_input, model_type_input, manual_params,
                        epochs, batch_size, learning_rate, num_ensemble_models
                    )
                    if error:
                        st.error(f"Could not analyze {ticker}: {error}")
                    else:
                        all_results[ticker] = {'predictions': predictions_df, 'metrics': metrics}
                progress_bar.progress((i + 1) / len(watchlist_tickers))
            
            if all_results:
                st.success("Analysis complete!")
                for ticker, result in all_results.items():
                    st.subheader(f"🔮 Results for {ticker}")
                    
                    if result['metrics']:
                        m_col1, m_col2, m_col3 = st.columns(3)
                        m_col1.metric("RMSE", f"{result['metrics']['RMSE']:.2f}")
                        m_col2.metric("MAE", f"{result['metrics']['MAE']:.2f}")
                        m_col3.metric("R² Score", f"{result['metrics']['R2']:.2f}")

                    df_display = result['predictions'].copy()
                    df_display['Date'] = df_display['Date'].dt.strftime('%Y-%m-%d')
                    for col in ['Ensemble Prediction', 'Confidence Lower', 'Confidence Upper']:
                        df_display[col] = df_display[col].map('${:,.2f}'.format)
                    
                    st.dataframe(df_display, use_container_width=True, hide_index=True)
                    st.markdown("---")

# --- Render Tabs ---
display_watchlist_tab("Watchlist 1", tab1)
display_watchlist_tab("Watchlist 2", tab_2)
display_watchlist_tab("Watchlist 3", tab3)

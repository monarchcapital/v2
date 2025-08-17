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

from data import fetch_stock_data, add_technical_indicators, add_macro_indicators, add_fundamental_indicators
from utils import preprocess_data, apply_pca, create_sequences, plot_predictions, calculate_metrics, display_log
from model import build_lstm_model, build_transformer_model, build_hybrid_model, train_model, predict_prices, run_hyperparameter_tuning, KERAS_TUNER_AVAILABLE, load_best_tuner_model

warnings.filterwarnings('ignore', category=FutureWarning)
tf.get_logger().setLevel('ERROR')

st.set_page_config(
    page_title="Stock Price Predictor (Deep Learning Hybrid)",
    page_icon="✅",
    layout="wide"
)

st.title("✅ Stock Price Predictor (Deep Learning Hybrid)")

# --- Paths for saving models and history ---
BEST_MODELS_DIR = "best_models"
MANUAL_RUNS_DIR = "manual_runs"

# --- Refactored State Management ---
def initialize_session_state():
    """Initializes all required session state keys with default values."""
    defaults = {
        'run_analysis': False, 'run_rolling_forecast': False, 'clear_analysis_cache': False,
        'data': None, 'y_test_len': 0,
        'processed_data': None, 'scaler': None, 'close_col_index': -1,
        'trained_models': [], 'test_actual_prices': None, 'test_predicted_prices': None,
        'future_predicted_prices': None, 'forecast_dates': [], 'model_metrics': {},
        'ensemble_confidence_score': None, 'log_history': [], 'individual_model_metrics': [],
        'current_step_count': 0, 'total_steps': 1, 'confidence_upper_inv': None,
        'confidence_lower_inv': None, 'last_actual_values_before_diff': pd.Series(),
        'was_differenced': False, 'rolling_forecast_metrics': {}, 'rolling_forecast_plot': None,
        'pca_object': None, 'original_feature_names': None
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

all_fundamental_indicators = [
    'Total Revenue', 'Net Income', 'EBITDA', 'Total Assets', 
    'Total Liabilities', 'Operating Cash Flow', 'Capital Expenditure'
]
selected_fundamental_indicators = st.sidebar.multiselect(
    "Fundamental Indicators", 
    all_fundamental_indicators, 
    default=[]
)

apply_differencing = st.sidebar.checkbox("Apply Differencing for Stationarity", value=False)

# --- NEW: Enhanced PCA Controls ---
enable_pca = st.sidebar.checkbox("Enable PCA", value=False)
if enable_pca:
    pca_mode = st.sidebar.radio("PCA Mode", ["Automatic (95% variance)", "Manual"], index=0)
    if pca_mode == "Manual":
        n_components_manual = st.sidebar.slider("Number of Components", 2, 20, 5, 1)
    else:
        n_components_manual = 0 # Signal for automatic mode in apply_pca

st.sidebar.subheader("Hyperparameter Mode")
hp_mode = st.sidebar.radio("Select Mode", ["Automatic (Tuner)", "Manual"])
force_retune = st.sidebar.checkbox("Force Hyperparameter Retune (if Automatic)", value=False)

manual_params = {}
if hp_mode == "Automatic (Tuner)":
    st.sidebar.warning("⚠️ Automatic tuning is very time-consuming.")
    epochs = st.sidebar.slider("Max Epochs for Tuning", 10, 100, 30, 10)
    num_trials = st.sidebar.slider("Number of Tuning Trials", 5, 100, 30, 5)
    executions_per_trial = st.sidebar.number_input("Executions per Trial", 1, 3, 1)
    # Set default values for batch_size and learning_rate, which will be overwritten by tuner
    batch_size = 32
    learning_rate = 0.001
else: # Manual
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
        manual_params['num_transformer_blocks'] = st.sidebar.slider("Transformer Blocks", 1, 4, 2)
        manual_params['num_heads'] = st.sidebar.slider("Attention Heads", 1, 8, 4)
        manual_params['ff_dim'] = st.sidebar.slider("Feed Forward Dim", 16, 64, 32, 16)
    elif model_type == 'Hybrid (LSTM + Transformer)':
        manual_params['lstm_units'] = st.sidebar.slider("LSTM Units (Hybrid)", 32, 256, 64, 32)
        manual_params['lstm_dropout'] = st.sidebar.slider("LSTM Dropout (Hybrid)", 0.0, 0.5, 0.2, 0.05)
        manual_params['num_transformer_blocks'] = st.sidebar.slider("Transformer Blocks (Hybrid)", 1, 4, 1)
        manual_params['num_heads'] = st.sidebar.slider("Attention Heads (Hybrid)", 1, 8, 2)


use_ensemble = st.sidebar.checkbox("Use Ensemble of Models", value=True)
if use_ensemble:
    num_ensemble_models = st.sidebar.slider("Ensemble Size", 2, 10, 5)
    confidence_factor = st.sidebar.slider("Confidence Interval Multiplier", 0.5, 3.0, 1.96, 0.01)
else:
    num_ensemble_models = 1

if st.sidebar.button("🚀 Run Prediction"):
    reset_analysis_state()
    st.session_state['run_analysis'] = True
    st.rerun()

st.sidebar.subheader("Advanced Evaluation")
st.sidebar.warning("⚠️ Rolling forecast is a deep backtest and can take several minutes.")
rolling_n_splits = st.sidebar.slider("Rolling Forecast Splits", 3, 15, 5, 1)
rolling_test_size = st.sidebar.slider("Rolling Forecast Test Size (days)", 30, 180, 60, 10)
if st.sidebar.button("📊 Run Rolling Forecast Evaluation"):
    reset_analysis_state()
    st.session_state['run_rolling_forecast'] = True
    st.rerun()

if st.sidebar.button("🧹 Clear Cache"):
    reset_analysis_state()
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Cache cleared!")
    st.rerun()

# --- Main Content Area ---
st.header("📊 Analysis & Results")
with st.expander("See Logs", expanded=False):
    log_container = st.container()

progress_bar = st.empty()

# --- Main Prediction Logic ---
if st.session_state['run_analysis']:
    with st.spinner(f"Analyzing: {ticker_symbol}..."):
        # 1. Fetch Data
        data = fetch_stock_data(ticker_symbol, start_date, end_date)
        if data.empty:
            display_log("❌ Data fetching failed. Halting analysis.", "error")
            st.stop()
        
        # 2. Add Features
        if selected_indicators: data = add_technical_indicators(data, selected_indicators)
        if selected_macro_indicators: data = add_macro_indicators(data, selected_macro_indicators)
        if selected_fundamental_indicators: data = add_fundamental_indicators(data, ticker_symbol, selected_fundamental_indicators)
        
        if data.empty:
            display_log("❌ Data became empty after adding features. Halting analysis.", "error")
            st.stop()
        st.session_state['data'] = data

        # 3. Preprocess Data
        processed_data, scaler, close_col_index, last_actual_values, _ = preprocess_data(data.copy(), apply_differencing=apply_differencing)
        
        if processed_data.empty or scaler is None:
            display_log("❌ Data preprocessing failed. Halting analysis.", "error")
            st.stop()
        st.session_state.update(scaler=scaler, close_col_index=close_col_index, last_actual_values_before_diff=last_actual_values, was_differenced=apply_differencing)
        
        # 4. PCA (Optional)
        data_for_seq = processed_data
        if enable_pca and 'Close_scaled' in processed_data.columns:
            features_for_pca = processed_data.drop(columns=['Close_scaled'])
            if not features_for_pca.empty:
                pca_result, pca_obj = apply_pca(features_for_pca, n_components=n_components_manual)
                if not pca_result.empty and pca_obj is not None:
                    st.session_state['pca_object'] = pca_obj
                    st.session_state['original_feature_names'] = features_for_pca.columns.tolist()
                    
                    pca_result.reset_index(drop=True, inplace=True)
                    close_scaled_series = processed_data['Close_scaled'].reset_index(drop=True)
                    data_for_seq = pd.concat([pca_result, close_scaled_series], axis=1)
                    display_log(f"✅ PCA applied. New data shape for sequencing: {data_for_seq.shape}", "info")

        # 5. Create Sequences
        if 'Close_scaled' not in data_for_seq.columns:
            display_log("❌ Target column 'Close_scaled' not found after PCA. Halting.", "error")
            st.stop()
        target_idx = data_for_seq.columns.get_loc('Close_scaled')
        X, y = create_sequences(data_for_seq.values, target_idx, training_window_size, prediction_horizon)
        if X.size == 0:
            display_log("❌ Failed to create sequences. Halting analysis.", "error")
            st.stop()
        test_size = int(len(X) * test_set_split_ratio)
        X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]
        st.session_state['y_test_len'] = len(y_test)

        # FIX: ADDED HYPERPARAMETER TUNING LOGIC
        # 6. Hyperparameter Tuning (Optional) or Manual Model Building
        input_shape = (X_train.shape[1], X_train.shape[2])
        output_dim = prediction_horizon
        best_hps = None
        tuner_results_summary = None

        if hp_mode == "Automatic (Tuner)" and KERAS_TUNER_AVAILABLE:
            display_log("🚀 Starting Hyperparameter Tuning...", "info")
            tuned_model, best_hps, tuner_params, tuner_results_summary = run_hyperparameter_tuning(
                model_type=model_type,
                input_shape=input_shape,
                output_dim=output_dim,
                X_train=X_train,
                y_train=y_train,
                num_trials=num_trials,
                executions_per_trial=executions_per_trial,
                best_models_dir=BEST_MODELS_DIR,
                force_retune=force_retune,
                epochs=epochs
            )
            if tuner_params:
                learning_rate = tuner_params.get('learning_rate', learning_rate)
                batch_size = tuner_params.get('batch_size', batch_size)
                display_log(f"✅ Tuner finished. Using LR: {learning_rate}, Batch Size: {batch_size}", "info")
        else:
            display_log("🛠️ Using Manual Hyperparameters.", "info")

        # 7. Build, Train, Predict (Loop for Ensemble)
        all_future_preds, all_test_preds_scaled, trained_models_list = [], [], []
        model_builder = {'LSTM': build_lstm_model, 'Transformer': build_transformer_model, 'Hybrid (LSTM + Transformer)': build_hybrid_model}

        for i in range(num_ensemble_models):
            display_log(f"Training model {i+1}/{num_ensemble_models}...")
            
            # Use the tuner's best HPs to build each model in the ensemble
            if best_hps:
                model = model_builder[model_type](input_shape, output_dim, hp=best_hps)
            else: # Fallback to manual parameters
                model = model_builder[model_type](input_shape, output_dim, manual_params=manual_params, learning_rate=learning_rate)
                
            if model:
                train_model(model, X_train, y_train, epochs, batch_size, learning_rate, X_val=X_test, y_val=y_test)
                future_preds = predict_prices(model, data_for_seq, scaler, close_col_index, training_window_size, prediction_horizon, last_actual_values, was_differenced=apply_differencing)
                if future_preds.size > 0: all_future_preds.append(future_preds)
                if X_test.size > 0:
                    all_test_preds_scaled.append(model.predict(X_test, verbose=0))
                trained_models_list.append(model)
        st.session_state['trained_models'] = trained_models_list

        # 8. Process Results
        if all_future_preds:
            avg_future_preds = np.mean(all_future_preds, axis=0)
            last_date = data.index[-1]
            future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=prediction_horizon)
            st.session_state['future_predicted_prices'] = pd.Series(avg_future_preds, index=future_dates)
        
        if all_test_preds_scaled and y_test.size > 0:
            avg_test_preds_scaled = np.mean(all_test_preds_scaled, axis=0)[:, 0]
            y_test_first_step_scaled = y_test[:, 0]
            
            actual_dummy = np.zeros((len(y_test_first_step_scaled), scaler.n_features_in_))
            predicted_dummy = np.zeros((len(avg_test_preds_scaled), scaler.n_features_in_))
            actual_dummy[:, close_col_index] = y_test_first_step_scaled
            predicted_dummy[:, close_col_index] = avg_test_preds_scaled
            actual_prices_unscaled = scaler.inverse_transform(actual_dummy)[:, close_col_index]
            predicted_prices_unscaled = scaler.inverse_transform(predicted_dummy)[:, close_col_index]

            if apply_differencing:
                last_actual_price = data['Close'].iloc[-len(y_test)-1]
                actual_prices_inv = last_actual_price + np.cumsum(actual_prices_unscaled)
                predicted_prices_inv = last_actual_price + np.cumsum(predicted_prices_unscaled)
            else:
                actual_prices_inv, predicted_prices_inv = actual_prices_unscaled, predicted_prices_unscaled
            
            test_dates = data.index[-len(y_test):]
            st.session_state['test_actual_prices'] = pd.Series(actual_prices_inv, index=test_dates)
            st.session_state['test_predicted_prices'] = pd.Series(predicted_prices_inv, index=test_dates)
            rmse, mae, r2 = calculate_metrics(actual_prices_inv, predicted_prices_inv)
            st.session_state['model_metrics'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# --- Display Logs (Moved Down to Update Live) ---
with log_container:
    for log in st.session_state.get('log_history', []):
        st.info(log['message']) if log['level'] == 'info' else (st.warning(log['message']) if log['level'] == 'warning' else st.error(log['message']))

# --- Display Results ---
if st.session_state['model_metrics']:
    st.subheader("📈 Performance Metrics (Test Set - Day 1 Forecast)")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{st.session_state['model_metrics']['MAE']:.2f}")
    col2.metric("RMSE", f"{st.session_state['model_metrics']['RMSE']:.2f}")
    col3.metric("R²", f"{st.session_state['model_metrics']['R2']:.2f}")

if st.session_state['future_predicted_prices'] is not None:
    st.subheader("🔮 Future Predicted Prices")
    
    plot_data = st.session_state.get('data')
    y_test_len = st.session_state.get('y_test_len', 0)

    if plot_data is not None and y_test_len > 0 and st.session_state['test_predicted_prices'] is not None:
        fig = plot_predictions(
            actual_prices=plot_data['Close'][-y_test_len - training_window_size:],
            predicted_prices=st.session_state['test_predicted_prices'],
            ticker=ticker_symbol,
            future_predictions=st.session_state['future_predicted_prices']
        )
        st.plotly_chart(fig, use_container_width=True)

# --- NEW: Display PCA Results ---
if st.session_state.get('pca_object') is not None:
    st.markdown("---")
    st.subheader("🔬 Principal Component Analysis (PCA) Insights")
    
    pca = st.session_state['pca_object']
    feature_names = st.session_state['original_feature_names']
    
    # 1. Explained Variance Plot
    exp_var_ratio = pca.explained_variance_ratio_
    cum_exp_var = np.cumsum(exp_var_ratio)
    
    pca_df = pd.DataFrame({
        'Principal Component': [f'PC_{i+1}' for i in range(len(exp_var_ratio))],
        'Explained Variance': exp_var_ratio,
        'Cumulative Explained Variance': cum_exp_var
    })
    
    fig_var = px.bar(pca_df, x='Principal Component', y='Explained Variance', 
                     text=[f'{x:.1%}' for x in exp_var_ratio],
                     title="Explained Variance by Principal Component")
    fig_var.add_trace(go.Scatter(x=pca_df['Principal Component'], y=cum_exp_var,
                                 name='Cumulative Variance', mode='lines+markers'))
    st.plotly_chart(fig_var, use_container_width=True)

    # 2. Component Loadings Heatmap
    st.markdown("##### Component Loadings")
    st.markdown("This heatmap shows how original features contribute to each principal component. Bright or dark colors indicate a strong influence (positive or negative).")
    
    loadings_df = pd.DataFrame(pca.components_.T, columns=[f'PC_{i+1}' for i in range(pca.n_components_)], index=feature_names)
    fig_loadings = px.imshow(loadings_df, text_auto='.2f', aspect="auto",
                             title="Heatmap of PCA Component Loadings")
    st.plotly_chart(fig_loadings, use_container_width=True)
    
    # 3. Interpretation
    st.markdown("##### Interpretation of Principal Components")
    for i in range(min(5, pca.n_components_)):
        loadings = loadings_df[f'PC_{i+1}'].sort_values(ascending=False)
        top_positive = loadings.head(3)
        top_negative = loadings.tail(3).sort_values()
        
        st.markdown(f"**Principal Component {i+1} (explains {exp_var_ratio[i]:.1%} of variance):**")
        st.markdown(f"This component primarily represents a contrast between:")
        st.markdown(f"- **Positively correlated features:** `{', '.join(top_positive.index.tolist())}`")
        st.markdown(f"- **Negatively correlated features:** `{', '.join(top_negative.index.tolist())}`")
        st.markdown("---")


elif st.session_state['run_analysis'] and not st.session_state.get('future_predicted_prices'):
    st.warning("Analysis complete, but no results to display. Please check the logs for potential errors.")

elif not st.session_state['run_analysis']:
    st.info("Configure parameters in the sidebar and click 'Run Prediction' to start.")
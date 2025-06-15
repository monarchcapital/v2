# pages/Home.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta, date, datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

# Import functions and global variables from utils.py and config.py
import config
from utils import ( 
    download_data, create_features, get_model, _train_single_model,
    train_models_pipeline, generate_predictions_pipeline,
    make_ensemble_prediction, perform_walk_forward_backtesting,
    _predict_single_model_raw,
    calculate_pivot_points, save_prediction, load_past_predictions,
    PREDICTION_LOG_COLUMNS, training_messages_log,
    parse_int_list # Imported parse_int_list from utils
)

st.set_page_config(page_title="Monarch: Stock Price Predictor", layout="wide")

# --- Helper function for rgba color conversion ---
def hex_to_rgba(hex_color, alpha):
    """Converts a hex color string to an rgba string with a specified alpha value."""
    hex_color = hex_color.lstrip('#')
    hlen = len(hex_color)
    if hlen not in (6, 8):
        raise ValueError("Invalid hex color string. Must be 6 or 8 characters (RRGGBB or RRGGBBAA).")
    
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    # If hex_color includes alpha, combine it with the new alpha
    if hlen == 8:
        current_alpha = int(hex_color[6:8], 16) / 255.0
        alpha = alpha * current_alpha # Multiply for combined transparency
        
    return f'rgba({r},{g},{b},{alpha:.2f})'

# --- Styling function for DataFrame ---
def highlight_end_color_home(row):
    # Create a list of empty strings for default styles
    styles = [''] * len(row)
    
    # Get the index of the 'End_Color' column
    try:
        end_color_idx = row.index.get_loc('End_Color')
    except KeyError:
        # If 'End_Color' column is not found (e.g., initial empty df), return empty styles
        return styles

    # Apply color based on the value in 'End_Color'
    if row['End_Color'] == 'Green (Up)':
        styles[end_color_idx] = 'background-color: #d4edda;' # Light green
    elif row['End_Color'] == 'Red (Down)':
        styles[end_color_idx] = 'background-color: #f8d7da;' # Light red
    elif row['End_Color'] == 'Flat (Neutral)':
        styles[end_color_idx] = 'background-color: #fff3cd;' # Light yellow/neutral
    
    return styles

# --- UI Elements ---
st.header("📈 Stock Price Predictor")
st.markdown("""
Analyze a single stock, visualize its historical data with technical indicators,
and forecast future prices using advanced machine learning models.
""")

# Sidebar inputs
st.sidebar.header("🛠️ Configuration Panel")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, use .NS for Indian stocks):", value="AAPL").upper()

today = date.today()
default_end_bt = today - timedelta(days=1)
default_start_bt = default_end_bt - timedelta(days=5*365) # Default to 5 years for more data

st.sidebar.subheader("🗓️ Training Period")
start_bt = st.sidebar.date_input("Training Start Date (t1):", value=default_start_bt, help="Start date for model training data.")
end_bt = st.sidebar.date_input("Training End Date (t2):", value=default_end_bt, help="End date for model training data. Predictions will start from t2+1.")

if start_bt >= end_bt:
    st.sidebar.error("Training Start Date (t1) must be before Training End Date (t2)")
    st.stop()

st.sidebar.subheader("🤖 Model Selection")
# Filter out LSTM from model choices since it's no longer supported
model_choice = st.sidebar.selectbox("Select Main Model (for Close Price):", [m for m in config.MODEL_CHOICES if m != 'LSTM'], help="The primary model used for predictions.")
perform_tuning = st.sidebar.checkbox("Perform Hyperparameter Tuning", value=False, help="May significantly increase training time but can improve model accuracy. Includes Train/Validation split.")

# Confidence Interval Parameter
enable_confidence_interval = st.sidebar.checkbox("Enable Prediction Confidence Intervals", value=True, help="Display prediction intervals based on model residuals.")
confidence_level_pct = 90
if enable_confidence_interval:
    confidence_level_pct = st.sidebar.slider("Confidence Level (%)", min_value=70, max_value=99, value=90, step=1, help="The confidence level for the prediction interval (e.g., 90% means 5th-95th percentile).")

st.sidebar.subheader("🤝 Ensemble Prediction (Close Price)")
enable_ensemble = st.sidebar.checkbox("Enable Ensemble Prediction (Close)", value=False, help="Combines predictions from multiple models for Close Price.")
ensemble_models = []
if enable_ensemble:
    ensemble_models = st.sidebar.multiselect(
        "Select Models for Ensemble:", 
        [m for m in config.MODEL_CHOICES if m != 'LSTM'], # Filter out LSTM
        default=[m for m in config.MODEL_CHOICES[:3] if m != 'LSTM'], # Default to a few models
        help="Choose models whose 'Close' price predictions will be averaged for the ensemble."
    )
    # Ensure the main model (if not 'Ensemble' itself) is part of the ensemble list for prediction consistency
    if model_choice != 'Ensemble' and model_choice not in ensemble_models and enable_ensemble: 
        ensemble_models.insert(0, model_choice)


n_future = st.sidebar.slider("Predict Future Days (after t2):", min_value=1, max_value=90, value=15, help="Number of future trading days to forecast.")

st.sidebar.subheader("📊 Model Comparison")
# Ensure the default uses models from config.MODEL_CHOICES
compare_models = st.sidebar.multiselect("Select Models to Compare:", [m for m in config.MODEL_CHOICES if m != 'LSTM'], default=[m for m in config.MODEL_CHOICES[:3] if m != 'LSTM'], help="Additional models to compare against the main model on recent data.")
train_days_comparison = st.sidebar.slider("Recent Data for Comparison (days):", min_value=30, max_value=1000, value=config.DEFAULT_RECENT_DATA_FOR_COMPARISON, step=10, help="How many recent days of data to use for the model comparison chart.")

st.sidebar.subheader("🔄 Walk-Forward Backtesting")
enable_walk_forward = st.sidebar.checkbox("Enable Walk-Forward Backtesting", value=False, help="Perform more realistic backtesting by iteratively retraining and predicting.")
if enable_walk_forward:
    initial_train_days_wf = st.sidebar.slider("Initial Training Period (days):", min_value=90, max_value=730, value=180, step=30, help="Number of days for the very first training window.")
    step_forward_days_wf = st.sidebar.slider("Prediction Step (days):", min_value=1, max_value=30, value=5, step=1, help="Number of days to predict forward in each step.")
    wf_models_to_test = st.sidebar.multiselect("Models for Walk-Forward:", [m for m in config.MODEL_CHOICES if m != 'LSTM'], default=[model_choice], help="Select models to include in the walk-forward backtest.")
    
st.sidebar.subheader("⚙️ Technical Indicator Settings")
st.sidebar.markdown("Uncheck an indicator to disable it and remove its features from model training.")

# Dictionary to store selected indicator parameters
selected_indicator_params = {}

# Function to provide error callback to parse_int_list for Streamlit display
def sidebar_error_callback(message):
    st.sidebar.error(message)

# Loop through TECHNICAL_INDICATORS_DEFAULTS to create UI for each
for indicator_name, (default_value, default_enabled) in config.TECHNICAL_INDICATORS_DEFAULTS.items():
    if indicator_name == 'PARSAR_ACCELERATION' or indicator_name == 'PARSAR_MAX_ACCELERATION':
        # These are handled inside the PARSAR_ENABLED block
        continue
    
    # Use a unique key for each checkbox to avoid Streamlit warnings
    checkbox_key = f"enable_{indicator_name.lower()}_home" # Added _home for uniqueness
    param_key_prefix = indicator_name.replace('_', ' ')

    if indicator_name.endswith('_ENABLED'): # For indicators like OBV_ENABLED, PARSAR_ENABLED
        enabled = st.sidebar.checkbox(f"Enable {param_key_prefix.replace('_enabled', '')}:", value=default_enabled, key=checkbox_key)
        selected_indicator_params[indicator_name] = enabled
        if indicator_name == 'PARSAR_ENABLED' and enabled:
            # Show Parabolic SAR specific parameters only if enabled
            selected_indicator_params['PARSAR_ACCELERATION'] = st.sidebar.number_input(f"  Parabolic SAR Acceleration:", min_value=0.01, max_value=0.5, value=config.TECHNICAL_INDICATORS_DEFAULTS['PARSAR_ACCELERATION'][0], step=0.01, key=f"input_parsar_accel_home")
            selected_indicator_params['PARSAR_MAX_ACCELERATION'] = st.sidebar.number_input(f"  Parabolic SAR Max Acceleration:", min_value=0.01, max_value=0.5, value=config.TECHNICAL_INDICATORS_DEFAULTS['PARSAR_MAX_ACCELERATION'][0], step=0.01, key=f"input_parsar_max_accel_home")

    else: # For indicators with windows/values
        enabled = st.sidebar.checkbox(f"Enable {param_key_prefix}:", value=default_enabled, key=checkbox_key)
        
        if enabled:
            if isinstance(default_value, list):
                # List inputs (Lag, MA, STD windows)
                parsed_list = parse_int_list(
                    st.sidebar.text_input(f"  {param_key_prefix.replace('_', ' ')} (comma-separated days, e.g., {','.join(map(str, default_value))}):", 
                                       value=", ".join(map(str, default_value)), key=f"input_{indicator_name.lower()}_home"),
                    default_value,
                    sidebar_error_callback # Pass the error callback
                )
                selected_indicator_params[indicator_name] = parsed_list if parsed_list else None # Store None if empty after parsing
            elif isinstance(default_value, (int, float)):
                # Single value inputs (RSI, MACD, BB, ATR, Stoch, CCI, ROC, ADX, CMF)
                if indicator_name == 'BB_STD_DEV':
                    selected_indicator_params[indicator_name] = st.sidebar.number_input(f"  {param_key_prefix.replace('_', ' ')} Multiplier:", min_value=0.1, value=default_value, step=0.1, key=f"input_{indicator_name.lower()}_home")
                else:
                    selected_indicator_params[indicator_name] = st.sidebar.number_input(f"  {param_key_prefix.replace('_', ' ')}:", min_value=1, value=default_value, step=1, key=f"input_{indicator_name.lower()}_home")
            else:
                selected_indicator_params[indicator_name] = None # Should not happen with current config, but for safety
        else:
            selected_indicator_params[indicator_name] = None # Indicator is disabled


# --- Training Log Display Area ---
log_expander = st.sidebar.expander("📜 Training Log & Messages", expanded=False)
log_placeholder = log_expander.empty()

# Function to update the log in the sidebar
def update_log(message):
    """Appends a message to the training log displayed in the sidebar."""
    training_messages_log.append(message)
    log_placeholder.text_area("Log:", "".join(f"{msg}\n" for msg in training_messages_log), height=300, disabled=True)

# --- Main Application Flow ---
with st.spinner(f"Downloading data for {ticker}..."):
    data = download_data(ticker)

if not data.empty:
    # Clear previous logs for this run
    training_messages_log.clear()
    update_log(f"Data loaded for {ticker}: {len(data)} rows.")

    with st.spinner("Creating features and processing data..."):
        # Pass the selected_indicator_params dictionary to create_features
        df_features_full = create_features(data.copy(), selected_indicator_params)
    update_log(f"Features created. Rows after NaN drop: {len(df_features_full)}")

    # Ensure enough data for feature calculation based on *enabled* indicators
    min_data_required = 0
    
    # Calculate min_data_required from enabled indicators
    # Iterate through the actual values in selected_indicator_params, not config.TECHNICAL_INDICATORS_DEFAULTS
    for param_name, (default_value, default_enabled) in config.TECHNICAL_INDICATORS_DEFAULTS.items():
        if param_name in selected_indicator_params and selected_indicator_params[param_name] is not None:
            param_value = selected_indicator_params[param_name] # Use the *selected* value
            if isinstance(param_value, list) and param_value: # For LAG_FEATURES, MA_WINDOWS, STD_WINDOWS
                min_data_required = max(min_data_required, max(param_value))
            elif isinstance(param_value, (int, float)): # For single-window indicators
                min_data_required = max(min_data_required, int(param_value)) # Convert float to int for max()
    
    # Fallback if no indicators are enabled or small windows (to ensure minimal training data)
    if min_data_required < 50: 
        min_data_required = 50 

    if len(df_features_full) < min_data_required:
        st.warning(f"Not enough data after feature creation for selected parameters ({len(df_features_full)} rows). Need at least {min_data_required} rows for enabled indicators. Adjust dates or parameters.")
        update_log("❌ Not enough data after feature creation.")
        st.stop()


    df_train_period = df_features_full[(df_features_full['Date'] >= pd.to_datetime(start_bt)) & (df_features_full['Date'] <= pd.to_datetime(end_bt))].copy()

    if df_train_period.empty:
        st.warning("No data in selected training range after feature creation. Adjust dates or check ticker.")
        update_log("❌ No data in training range.")
        st.stop()
    
    update_log(f"Training data period: {len(df_train_period)} rows from {start_bt} to {end_bt}.")
    
    # Train main models for Close, Open, High, Low, Volatility
    st.markdown("---")
    st.subheader(f"📈 Main Model Training: {model_choice}")
    
    trained_models_main = {}
    main_model_val_metrics = {}
    with st.spinner(f"Training main models for {model_choice}..."):
        trained_models_main, main_model_val_metrics = train_models_pipeline(df_train_period.copy(), model_choice, perform_tuning, update_log, selected_indicator_params) 
    
    # Check if models were trained for all *expected* targets given the enabled indicators
    expected_targets = ['Close', 'Open', 'High', 'Low']
    if selected_indicator_params.get('STD_WINDOWS') and selected_indicator_params['STD_WINDOWS']:
         volatility_target_col_name_check = f'Volatility_{max(selected_indicator_params["STD_WINDOWS"])}'
         if volatility_target_col_name_check in df_train_period.columns: # Check if the column exists after feature creation
             expected_targets.append(volatility_target_col_name_check)
         
    # Filter expected_targets to only include those that are actually present in df_train_period columns
    actual_expected_targets = [t for t in expected_targets if t in df_train_period.columns]

    if not trained_models_main or not all(target in trained_models_main and trained_models_main[target].get('model') is not None for target in actual_expected_targets):
        st.error("One or more main models failed to train. Check logs in sidebar for details. Ensure you have enough data for the selected indicators.")
        st.stop()

    # Display validation metrics for the main models
    if main_model_val_metrics:
        st.markdown("**Validation Metrics (on hold-out validation set):**")
        val_metrics_df = pd.DataFrame.from_dict(main_model_val_metrics, orient='index')
        st.dataframe(val_metrics_df.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}"}))


    # --- Output 1: Predicted prices for t2 + 1 (Next Trading Day) ---
    st.markdown("---")
    st.subheader(f"🔮 Predicted Values for the Next Trading Day")

    last_day_features_df = df_features_full.tail(1).copy() # Use full features for the last day to predict next
    
    if not last_day_features_df.empty:
        last_known_date = df_features_full['Date'].iloc[-1]
        next_trading_day = last_known_date + timedelta(days=1)
        while next_trading_day.weekday() >= 5: next_trading_day += timedelta(days=1)

        st.markdown(f"**Forecast for {next_trading_day.strftime('%Y-%m-%d')}**")
        
        next_day_predictions_list = []
        
        # Get last actuals from the downloaded data for logging comparison
        last_actual_close_full_data = data['Close'].iloc[-1] if not data.empty else np.nan
        last_actual_open_full_data = data['Open'].iloc[-1] if not data.empty else np.nan
        last_actual_high_full_data = data['High'].iloc[-1] if not data.empty else np.nan
        last_actual_low_full_data = data['Low'].iloc[-1] if not data.empty else np.nan
        
        # Get the target column name for volatility from the trained model info
        volatility_target_col_name = ''
        if selected_indicator_params.get('STD_WINDOWS') and selected_indicator_params['STD_WINDOWS']:
             # Use the actual column name from df_features_full to ensure it exists
            volatility_target_col_name = f'Volatility_{max(selected_indicator_params["STD_WINDOWS"])}'
            if volatility_target_col_name not in df_features_full.columns:
                volatility_target_col_name = '' # Reset if not found


        last_actual_volatility_full_data = df_features_full[volatility_target_col_name].iloc[-1] if volatility_target_col_name and not df_features_full.empty and volatility_target_col_name in df_features_full.columns else np.nan

        with st.spinner("Generating next day predictions..."):
            # Handle Main Model Prediction or Ensemble Prediction for Close
            pred_close_main, pred_close_lower, pred_close_upper = np.nan, np.nan, np.nan
            model_display_name = model_choice # Default
            if enable_ensemble and ensemble_models:
                # Train individual models for ensemble
                ensemble_trained_models = []
                for m_name in ensemble_models:
                    current_model_trained, _ = train_models_pipeline(df_train_period.copy(), m_name, perform_tuning, update_log, selected_indicator_params) 
                    if current_model_trained.get('Close'): # Only add if Close model was successfully trained
                        ensemble_trained_models.append(current_model_trained)
                
                if ensemble_trained_models:
                    # make_ensemble_prediction now returns (point_array, lower_array, upper_array)
                    # For a single prediction (last_day_features_df is 1 row), take the first element
                    point_arr, lower_arr, upper_arr = make_ensemble_prediction(ensemble_trained_models, last_day_features_df.copy(), update_log, confidence_level_pct if enable_confidence_interval else None)
                    pred_close_main = point_arr[0] if point_arr.size > 0 else np.nan
                    pred_close_lower = lower_arr[0] if lower_arr.size > 0 else np.nan
                    pred_close_upper = upper_arr[0] if upper_arr.size > 0 else np.nan
                    model_display_name = f"Ensemble ({', '.join(ensemble_models)})"
                else:
                    model_display_name = "Ensemble (Failed)"
            else: # Regular main model prediction
                if 'Close' in trained_models_main and not trained_models_main['Close'].get('model') is None:
                    # _predict_single_model_raw now returns (point_array, lower_array, upper_array)
                    point_arr, lower_arr, upper_arr = _predict_single_model_raw(trained_models_main['Close'], last_day_features_df.copy(), confidence_level_pct if enable_confidence_interval else None)
                    pred_close_main = point_arr[0] if point_arr.size > 0 else np.nan
                    pred_close_lower = lower_arr[0] if lower_arr.size > 0 else np.nan
                    pred_close_upper = upper_arr[0] if upper_arr.size > 0 else np.nan
                model_display_name = model_choice

            # For Open, High, Low, Volatility, always use the main selected model
            # This part assumes a single model for OHLV for simplicity, not ensemble
            ohlv_models_to_predict = {k: v for k,v in trained_models_main.items() if k != 'Close'}
            
            # Create a dictionary to hold the predictions for OHLV, including bounds
            next_day_ohlv_preds_formatted = {}
            for target, model_info in ohlv_models_to_predict.items():
                if model_info and model_info.get('model'):
                    # Call _predict_single_model_raw for each OHLV target
                    # It returns arrays, so take the first element [0]
                    point_p_arr, lower_p_arr, upper_p_arr = _predict_single_model_raw(model_info, last_day_features_df.copy(), confidence_level_pct if enable_confidence_interval else None)
                    point_p = point_p_arr[0] if point_p_arr.size > 0 else np.nan
                    lower_p = lower_p_arr[0] if lower_p_arr.size > 0 else np.nan
                    upper_p = upper_p_arr[0] if upper_p_arr.size > 0 else np.nan

                    next_day_ohlv_preds_formatted[target] = {'point': point_p, 'lower': lower_p, 'upper': upper_p}
                else:
                    next_day_ohlv_preds_formatted[target] = {'point': np.nan, 'lower': np.nan, 'upper': np.nan}
            
            pred_open_main = next_day_ohlv_preds_formatted.get('Open', {}).get('point', np.nan)
            pred_open_lower = next_day_ohlv_preds_formatted.get('Open', {}).get('lower', np.nan)
            pred_open_upper = next_day_ohlv_preds_formatted.get('Open', {}).get('upper', np.nan)

            pred_high_main = next_day_ohlv_preds_formatted.get('High', {}).get('point', np.nan)
            pred_high_lower = next_day_ohlv_preds_formatted.get('High', {}).get('lower', np.nan)
            pred_high_upper = next_day_ohlv_preds_formatted.get('High', {}).get('upper', np.nan)

            pred_low_main = next_day_ohlv_preds_formatted.get('Low', {}).get('point', np.nan)
            pred_low_lower = next_day_ohlv_preds_formatted.get('Low', {}).get('lower', np.nan)
            pred_low_upper = next_day_ohlv_preds_formatted.get('Low', {}).get('upper', np.nan)

            pred_vol_main = next_day_ohlv_preds_formatted.get(volatility_target_col_name, {}).get('point', np.nan)
            pred_vol_lower = next_day_ohlv_preds_formatted.get(volatility_target_col_name, {}).get('lower', np.nan)
            pred_vol_upper = next_day_ohlv_preds_formatted.get(volatility_target_col_name, {}).get('upper', np.nan)

            # Determine End_Color for the main model's next day close prediction
            end_color_main = 'N/A'
            if pd.notna(pred_close_main) and pd.notna(last_actual_close_full_data):
                if pred_close_main > last_actual_close_full_data:
                    end_color_main = 'Green (Up)'
                elif pred_close_main < last_actual_close_full_data:
                    end_color_main = 'Red (Down)'
                else:
                    end_color_main = 'Flat (Neutral)'

            next_day_predictions_list.append({
                'Model': model_display_name,
                'Last Actual Close': f"{last_actual_close_full_data:.2f}" if pd.notna(last_actual_close_full_data) else "N/A",
                'End_Color': end_color_main,
                'Predicted Close': pred_close_main,
                'Predicted Close Lower': pred_close_lower,
                'Predicted Close Upper': pred_close_upper,
                'Predicted Open': pred_open_main,
                'Predicted Open Lower': pred_open_lower,
                'Predicted Open Upper': pred_open_upper,
                'Predicted High': pred_high_main,
                'Predicted High Lower': pred_high_lower,
                'Predicted High Upper': pred_high_upper,
                'Predicted Low': pred_low_main,
                'Predicted Low Lower': pred_low_lower,
                'Predicted Low Upper': pred_low_upper,
                'Predicted Volatility': pred_vol_main,
                'Predicted Volatility Lower': pred_vol_lower,
                'Predicted Volatility Upper': pred_vol_upper
            })
            # Save predictions for all targets
            for target_type, predicted_val, actual_val, lower_b, upper_b in [
                ('Close', pred_close_main, last_actual_close_full_data, pred_close_lower, pred_close_upper),
                ('Open', pred_open_main, last_actual_open_full_data, pred_open_lower, pred_open_upper),
                ('High', pred_high_main, last_actual_high_full_data, pred_high_lower, pred_high_upper),
                ('Low', pred_low_main, last_actual_low_full_data, pred_low_lower, pred_low_upper),
                (volatility_target_col_name, pred_vol_main, last_actual_volatility_full_data, pred_vol_lower, pred_vol_upper) if volatility_target_col_name else (None, None, None, None, None)
            ]:
                if target_type and pd.notna(predicted_val): # Only save if target_type is not None
                    save_prediction(ticker, next_trading_day, predicted_val, actual_val, model_display_name, datetime.now(), end_bt, predicted_type=target_type, predicted_lower_bound=lower_b, predicted_upper_bound=upper_b)

            # Comparison Models Next Day Prediction (only if not doing ensemble for main)
            if not enable_ensemble: # Only compare other models if not using ensemble as primary
                for comp_model_name in compare_models:
                    if comp_model_name == model_choice: continue
                    
                    comp_model_last_day_features_df = df_features_full.tail(1).copy() # Use just the last day for traditional models

                    trained_comp_model_dict, _ = train_models_pipeline(df_train_period.copy(), comp_model_name, perform_tuning, update_log, selected_indicator_params) 
                    
                    if trained_comp_model_dict and all(model_info.get('model') is not None for model_info in trained_comp_model_dict.values() if model_info):
                        # Get predictions for all targets
                        comp_preds_formatted = {}
                        for target, model_info in trained_comp_model_dict.items():
                             if model_info and model_info.get('model'):
                                # It returns arrays, so take the first element [0]
                                point_p_arr, lower_p_arr, upper_p_arr = _predict_single_model_raw(model_info, comp_model_last_day_features_df.copy(), confidence_level_pct if enable_confidence_interval else None)
                                point_p = point_p_arr[0] if point_p_arr.size > 0 else np.nan
                                lower_p = lower_p_arr[0] if lower_p_arr.size > 0 else np.nan
                                upper_p = upper_p_arr[0] if upper_p_arr.size > 0 else np.nan

                                comp_preds_formatted[target] = {'point': point_p, 'lower': lower_p, 'upper': upper_p}
                             else:
                                comp_preds_formatted[target] = {'point': np.nan, 'lower': np.nan, 'upper': np.nan}


                        pred_close_comp = comp_preds_formatted.get('Close', {}).get('point', np.nan)
                        pred_close_lower_comp = comp_preds_formatted.get('Close', {}).get('lower', np.nan)
                        pred_close_upper_comp = comp_preds_formatted.get('Close', {}).get('upper', np.nan)

                        pred_open_comp = comp_preds_formatted.get('Open', {}).get('point', np.nan)
                        pred_open_lower_comp = comp_preds_formatted.get('Open', {}).get('lower', np.nan)
                        pred_open_upper_comp = comp_preds_formatted.get('Open', {}).get('upper', np.nan)
                        
                        pred_high_comp = comp_preds_formatted.get('High', {}).get('point', np.nan)
                        pred_high_lower_comp = comp_preds_formatted.get('High', {}).get('lower', np.nan)
                        pred_high_upper_comp = comp_preds_formatted.get('High', {}).get('upper', np.nan)

                        pred_low_comp = comp_preds_formatted.get('Low', {}).get('point', np.nan)
                        pred_low_lower_comp = comp_preds_formatted.get('Low', {}).get('lower', np.nan)
                        pred_low_upper_comp = comp_preds_formatted.get('Low', {}).get('upper', np.nan)

                        pred_vol_comp = comp_preds_formatted.get(volatility_target_col_name, {}).get('point', np.nan)
                        pred_vol_lower_comp = comp_preds_formatted.get(volatility_target_col_name, {}).get('lower', np.nan)
                        pred_vol_upper_comp = comp_preds_formatted.get(volatility_target_col_name, {}).get('upper', np.nan)


                        end_color_comp = 'N/A'
                        if pd.notna(pred_close_comp) and pd.notna(last_actual_close_full_data):
                            if pred_close_comp > last_actual_close_full_data:
                                end_color_comp = 'Green (Up)'
                            elif pred_close_comp < last_actual_close_full_data:
                                end_color_comp = 'Red (Down)'
                            else:
                                end_color_comp = 'Flat (Neutral)'

                        next_day_predictions_list.append({
                            'Model': comp_model_name,
                            'Last Actual Close': f"{last_actual_close_full_data:.2f}" if pd.notna(last_actual_close_full_data) else "N/A",
                            'End_Color': end_color_comp,
                            'Predicted Close': pred_close_comp,
                            'Predicted Close Lower': pred_close_lower_comp,
                            'Predicted Close Upper': pred_close_upper_comp,
                            'Predicted Open': pred_open_comp,
                            'Predicted Open Lower': pred_open_lower_comp,
                            'Predicted Open Upper': pred_open_upper_comp,
                            'Predicted High': pred_high_comp,
                            'Predicted High Lower': pred_high_lower_comp,
                            'Predicted High Upper': pred_high_upper_comp,
                            'Predicted Low': pred_low_comp,
                            'Predicted Low Lower': pred_low_lower_comp,
                            'Predicted Low Upper': pred_low_upper_comp,
                            'Predicted Volatility': pred_vol_comp,
                            'Predicted Volatility Lower': pred_vol_lower_comp,
                            'Predicted Volatility Upper': pred_vol_upper_comp
                        })
                        for target_type, predicted_val, actual_val, lower_b, upper_b in [
                            ('Close', pred_close_comp, last_actual_close_full_data, pred_close_lower_comp, pred_close_upper_comp),
                            ('Open', pred_open_comp, last_actual_open_full_data, pred_open_lower_comp, pred_open_upper_comp),
                            ('High', pred_high_comp, last_actual_high_full_data, pred_high_lower_comp, pred_high_upper_comp),
                            ('Low', pred_low_comp, last_actual_low_full_data, pred_low_lower_comp, pred_low_upper_comp),
                            (volatility_target_col_name, pred_vol_comp, last_actual_volatility_full_data, pred_vol_lower_comp, pred_vol_upper_comp) if volatility_target_col_name else (None, None, None, None, None)
                        ]:
                            if target_type and pd.notna(predicted_val):
                                save_prediction(ticker, next_trading_day, predicted_val, actual_val, comp_model_name, datetime.now(), end_bt, predicted_type=target_type, predicted_lower_bound=lower_b, predicted_upper_bound=upper_b)
            
            df_next_day_preds = pd.DataFrame(next_day_predictions_list)
            if not df_next_day_preds.empty:
                df_next_day_preds['Date'] = next_trading_day
                # Updated display columns to include bounds
                display_cols = [
                    'Date', 'Model', 'Last Actual Close', 'End_Color', 
                    'Predicted Close', 'Predicted Close Lower', 'Predicted Close Upper',
                    'Predicted Open', 'Predicted Open Lower', 'Predicted Open Upper',
                    'Predicted High', 'Predicted High Lower', 'Predicted High Upper',
                    'Predicted Low', 'Predicted Low Lower', 'Predicted Low Upper',
                    'Predicted Volatility', 'Predicted Volatility Lower', 'Predicted Volatility Upper'
                ]
                df_next_day_preds = df_next_day_preds[display_cols].sort_values('Model').reset_index(drop=True)
                
                # Apply styling
                st.dataframe(df_next_day_preds.style.apply(highlight_end_color_home, axis=1).format({
                    "Predicted Close": "{:.2f}", "Predicted Close Lower": "{:.2f}", "Predicted Close Upper": "{:.2f}",
                    "Predicted Open": "{:.2f}", "Predicted Open Lower": "{:.2f}", "Predicted Open Upper": "{:.2f}",
                    "Predicted High": "{:.2f}", "Predicted High Lower": "{:.2f}", "Predicted High Upper": "{:.2f}",
                    "Predicted Low": "{:.2f}", "Predicted Low Lower": "{:.2f}", "Predicted Low Upper": "{:.2f}",
                    "Predicted Volatility": "{:.4f}", "Predicted Volatility Lower": "{:.4f}", "Predicted Volatility Upper": "{:.4f}"
                }), hide_index=True)
                
                if pd.notna(pred_close_main) and pd.notna(last_actual_close_full_data):
                    st.markdown(f"**Difference (Last Actual Close - Predicted Next Day Close) for Main Model ({model_display_name}):** {(last_actual_close_full_data - pred_close_main):.2f}")
            else:
                st.warning("Could not generate next day predictions.")

    # --- Output 6: Resistance, Support, High, Low from last trading day ---
    st.markdown("---")
    st.subheader(f"📊 Key Levels from Last Trading Day ({ticker})")
    if not data.empty:
        last_trading_day_data = data.iloc[-1]
        st.markdown(f"**Last Actual Trading Day:** {last_trading_day_data['Date'].strftime('%Y-%m-%d')}")
        st.markdown(f"**Last Actual High:** {last_trading_day_data['High']:.2f}")
        st.markdown(f"**Last Actual Low:** {last_trading_day_data['Low']:.2f}")
        st.markdown(f"**Last Actual Close:** {last_trading_day_data['Close']:.2f}")

        pivot_points = calculate_pivot_points(data.tail(1)) # Calculate for the last complete day
        st.markdown(f"**Pivot Point (PP):** {pivot_points['PP']:.2f}")
        st.markdown(f"**Resistance Levels:** R1: {pivot_points['R1']:.2f}, R2: {pivot_points['R2']:.2f}, R3: {pivot_points['R3']:.2f}")
        st.markdown(f"**Support Levels:** S1: {pivot_points['S1']:.2f}, S2: {pivot_points['S2']:.2f}, S3: {pivot_points['S3']:.2f}")
        
        st.info("""
        **About Trendlines, Resistance & Support:**
        * **Trendlines** are typically drawn manually by traders to connect significant highs or lows. While not automatically generated by this model, observing the Moving Averages (MA) on the chart can provide a sense of trend direction.
        * **Resistance and Support levels** (like Pivot Points above) are calculated based on historical price action. These are areas where price tends to pause or reverse. Future resistance/support might relate to predicted Highs/Lows, but their exact levels are still best derived from historical patterns.
        """)
    else:
        st.info("No data available to calculate and display key levels.")

    # --- Historical Chart with Technical Indicators and Trendlines ---
    st.markdown("---")
    st.subheader(f"📈 Historical Price Chart for {ticker} with Technical Indicators")
    if not df_features_full.empty:
        fig_hist_chart = go.Figure(data=[go.Candlestick(x=df_features_full['Date'],
                                                        open=df_features_full['Open'],
                                                        high=df_features_full['High'],
                                                        low=df_features_full['Low'],
                                                        close=df_features_full['Close'],
                                                        name='Candlestick')])
        
        # Add Moving Averages (only if enabled)
        if selected_indicator_params.get('MA_WINDOWS') and selected_indicator_params['MA_WINDOWS']:
            for win in selected_indicator_params['MA_WINDOWS']:
                if f'MA_{win}' in df_features_full.columns: # Check if column exists
                    fig_hist_chart.add_trace(go.Scatter(x=df_features_full['Date'], y=df_features_full[f'MA_{win}'],
                                                        mode='lines', name=f'MA {win}', line=dict(width=1)))
        
        # Add Bollinger Bands (only if enabled)
        if selected_indicator_params.get('BB_WINDOW') and selected_indicator_params.get('BB_STD_DEV') is not None and \
           'BB_Upper' in df_features_full.columns and 'BB_Lower' in df_features_full.columns and 'BB_Middle' in df_features_full.columns:
            fig_hist_chart.add_trace(go.Scatter(x=df_features_full['Date'], y=df_features_full['BB_Upper'],
                                                mode='lines', name='BB Upper', line=dict(color='gray', dash='dash', width=1)))
            fig_hist_chart.add_trace(go.Scatter(x=df_features_full['Date'], y=df_features_full['BB_Middle'],
                                                mode='lines', name='BB Middle', line=dict(color='gray', width=1)))
            fig_hist_chart.add_trace(go.Scatter(x=df_features_full['Date'], y=df_features_full['BB_Lower'],
                                                mode='lines', name='BB Lower', line=dict(color='gray', dash='dash', width=1)))

        # Add Parabolic SAR (only if enabled)
        if selected_indicator_params.get('PARSAR_ENABLED') and 'SAR' in df_features_full.columns and not df_features_full['SAR'].isnull().all():
            fig_hist_chart.add_trace(go.Scatter(x=df_features_full['Date'], y=df_features_full['SAR'],
                                                mode='markers', name='Parabolic SAR',
                                                marker=dict(symbol='diamond', size=4, color='darkorange')))

        # Add Pivot Points (Resistance and Support) - for the most recent day
        pivot_points_last = calculate_pivot_points(data.tail(1))
        last_date_data = df_features_full['Date'].iloc[-1]
        
        # Draw horizontal lines for resistance and support from the latest full day
        if pd.notna(pivot_points_last['PP']):
            fig_hist_chart.add_shape(type="line", x0=df_features_full['Date'].min(), y0=pivot_points_last['PP'],
                                     x1=df_features_full['Date'].max(), y1=pivot_points_last['PP'],
                                     line=dict(color="purple", width=1, dash="dot"),
                                     name="Pivot Point")
            fig_hist_chart.add_annotation(x=last_date_data, y=pivot_points_last['PP'], text=f"PP:{pivot_points_last['PP']:.2f}", showarrow=False, yshift=10)

            for level, color in zip(['R1', 'R2', 'R3'], ['red', 'darkred', 'crimson']):
                if pd.notna(pivot_points_last[level]):
                    fig_hist_chart.add_shape(type="line", x0=df_features_full['Date'].min(), y0=pivot_points_last[level],
                                             x1=df_features_full['Date'].max(), y1=pivot_points_last[level],
                                             line=dict(color=color, width=1, dash="dash"),
                                             name=level)
                    fig_hist_chart.add_annotation(x=last_date_data, y=pivot_points_last[level], text=f"{level}:{pivot_points_last[level]:.2f}", showarrow=False, yshift=10)

            for level, color in zip(['S1', 'S2', 'S3'], ['green', 'darkgreen', 'forestgreen']):
                if pd.notna(pivot_points_last[level]):
                    fig_hist_chart.add_shape(type="line", x0=df_features_full['Date'].min(), y0=pivot_points_last[level],
                                             x1=df_features_full['Date'].max(), y1=pivot_points_last[level],
                                             line=dict(color=color, width=1, dash="dash"),
                                             name=level)
                    fig_hist_chart.add_annotation(x=last_date_data, y=pivot_points_last[level], text=f"{level}:{pivot_points_last[level]:.2f}", showarrow=False, yshift=10)
        
        fig_hist_chart.update_layout(title=f'Historical Price with Indicators for {ticker}',
                                     xaxis_title="Date",
                                     yaxis_title="Price",
                                     xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_hist_chart, use_container_width=True)

        # Plotting other indicators like RSI, MACD, ATR, Stochastic, CCI, ROC, ADX, OBV, CMF below the main chart
        st.markdown("---")
        st.subheader("📊 Supplementary Technical Indicators")
        
        # RSI Chart
        if selected_indicator_params.get('RSI_WINDOW') and 'RSI' in df_features_full.columns and not df_features_full['RSI'].isnull().all():
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df_features_full['Date'], y=df_features_full['RSI'], mode='lines', name='RSI', line=dict(color='darkblue')))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)", annotation_position="top right")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)", annotation_position="bottom right")
            fig_rsi.update_layout(title='Relative Strength Index (RSI)', xaxis_title="Date", yaxis_title="RSI Value", height=300)
            st.plotly_chart(fig_rsi, use_container_width=True)

        # MACD Chart
        if selected_indicator_params.get('MACD_LONG_WINDOW') and \
           'MACD' in df_features_full.columns and 'MACD_Signal' in df_features_full.columns and 'MACD_Hist' in df_features_full.columns and \
            not df_features_full['MACD'].isnull().all():
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=df_features_full['Date'], y=df_features_full['MACD'], mode='lines', name='MACD Line', line=dict(color='blue')))
            fig_macd.add_trace(go.Scatter(x=df_features_full['Date'], y=df_features_full['MACD_Signal'], mode='lines', name='Signal Line', line=dict(color='orange')))
            colors_hist = np.where(df_features_full['MACD_Hist'] > 0, 'green', 'red')
            fig_macd.add_trace(go.Bar(x=df_features_full['Date'], y=df_features_full['MACD_Hist'], name='Histogram', marker_color=colors_hist))
            fig_macd.update_layout(title='MACD', xaxis_title="Date", yaxis_title="Value", height=300)
            st.plotly_chart(fig_macd, use_container_width=True)
            
        # ATR Chart
        if selected_indicator_params.get('ATR_WINDOW') and 'ATR' in df_features_full.columns and not df_features_full['ATR'].isnull().all():
            fig_atr = go.Figure()
            fig_atr.add_trace(go.Scatter(x=df_features_full['Date'], y=df_features_full['ATR'], mode='lines', name='ATR', line=dict(color='purple')))
            fig_atr.update_layout(title='Average True Range (ATR)', xaxis_title="Date", yaxis_title="ATR Value", height=300)
            st.plotly_chart(fig_atr, use_container_width=True)

        # Stochastic Oscillator Chart
        if selected_indicator_params.get('STOCH_WINDOW') and \
           '%K' in df_features_full.columns and '%D' in df_features_full.columns and not df_features_full['%K'].isnull().all():
            fig_stoch = go.Figure()
            fig_stoch.add_trace(go.Scatter(x=df_features_full['Date'], y=df_features_full['%K'], mode='lines', name='%K', line=dict(color='darkcyan')))
            fig_stoch.add_trace(go.Scatter(x=df_features_full['Date'], y=df_features_full['%D'], mode='lines', name='%D', line=dict(color='lightcoral')))
            fig_stoch.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overbought (80)", annotation_position="top right")
            fig_stoch.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Oversold (20)", annotation_position="bottom right")
            fig_stoch.update_layout(title='Stochastic Oscillator', xaxis_title="Date", yaxis_title="Value", height=300)
            st.plotly_chart(fig_stoch, use_container_width=True)

        # CCI Chart
        if selected_indicator_params.get('CCI_WINDOW') and 'CCI' in df_features_full.columns and not df_features_full['CCI'].isnull().all():
            fig_cci = go.Figure()
            fig_cci.add_trace(go.Scatter(x=df_features_full['Date'], y=df_features_full['CCI'], mode='lines', name='CCI', line=dict(color='darkgreen')))
            fig_cci.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Overbought (+100)", annotation_position="top right")
            fig_cci.add_hline(y=-100, line_dash="dash", line_color="green", annotation_text="Oversold (-100)", annotation_position="bottom right")
            fig_cci.update_layout(title='Commodity Channel Index (CCI)', xaxis_title="Date", yaxis_title="CCI Value", height=300)
            st.plotly_chart(fig_cci, use_container_width=True)

        # ROC Chart
        if selected_indicator_params.get('ROC_WINDOW') and 'ROC' in df_features_full.columns and not df_features_full['ROC'].isnull().all():
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=df_features_full['Date'], y=df_features_full['ROC'], mode='lines', name='ROC', line=dict(color='orange')))
            fig_roc.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Zero Line", annotation_position="top left")
            fig_roc.update_layout(title='Rate of Change (ROC)', xaxis_title="Date", yaxis_title="Percentage Change", height=300)
            st.plotly_chart(fig_roc, use_container_width=True)

        # ADX Chart
        if selected_indicator_params.get('ADX_WINDOW') and \
           'ADX' in df_features_full.columns and '+DI' in df_features_full.columns and '-DI' in df_features_full.columns and \
           not df_features_full['ADX'].isnull().all():
            fig_adx = go.Figure()
            fig_adx.add_trace(go.Scatter(x=df_features_full['Date'], y=df_features_full['ADX'], mode='lines', name='ADX', line=dict(color='red', width=2)))
            fig_adx.add_trace(go.Scatter(x=df_features_full['Date'], y=df_features_full['+DI'], mode='lines', name='+DI', line=dict(color='green', dash='dot')))
            fig_adx.add_trace(go.Scatter(x=df_features_full['Date'], y=df_features_full['-DI'], mode='lines', name='-DI', line=dict(color='blue', dash='dot')))
            fig_adx.add_hline(y=25, line_dash="dash", line_color="purple", annotation_text="Strong Trend (25)", annotation_position="top left")
            fig_adx.update_layout(title='Average Directional Index (ADX)', xaxis_title="Date", yaxis_title="Value", height=300)
            st.plotly_chart(fig_adx, use_container_width=True)

        # OBV Chart
        if selected_indicator_params.get('OBV_ENABLED') and 'OBV' in df_features_full.columns and not df_features_full['OBV'].isnull().all():
            fig_obv = go.Figure()
            fig_obv.add_trace(go.Scatter(x=df_features_full['Date'], y=df_features_full['OBV'], mode='lines', name='OBV', line=dict(color='brown')))
            fig_obv.update_layout(title='On-Balance Volume (OBV)', xaxis_title="Date", yaxis_title="Volume", height=300)
            st.plotly_chart(fig_obv, use_container_width=True)

        # CMF Chart
        if selected_indicator_params.get('CMF_WINDOW') and 'CMF' in df_features_full.columns and not df_features_full['CMF'].isnull().all():
            fig_cmf = go.Figure()
            fig_cmf.add_trace(go.Scatter(x=df_features_full['Date'], y=df_features_full['CMF'], mode='lines', name='CMF', line=dict(color='teal')))
            fig_cmf.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Zero Line", annotation_position="top left")
            fig_cmf.add_hline(y=0.2, line_dash="dash", line_color="green", annotation_text="Bullish (+0.2)", annotation_position="top right")
            fig_cmf.add_hline(y=-0.2, line_dash="dash", line_color="red", annotation_text="Bearish (-0.2)", annotation_position="bottom right")
            fig_cmf.update_layout(title='Chaikin Money Flow (CMF)', xaxis_title="Date", yaxis_title="Value", height=300)
            st.plotly_chart(fig_cmf, use_container_width=True)

    else:
        st.info("No feature data available to plot historical chart with indicators.")


    # --- Output 5: Model Comparison on Recent Data ---
    st.markdown("---")
    st.subheader("📊 Selected Models Comparison on Recent Data")
    compare_days_actual = min(train_days_comparison, len(df_features_full))
    df_compare_data = df_features_full.tail(compare_days_actual).copy()
    
    comparison_results_list = []
    best_comp_model_name, best_comp_rmse, best_comp_pct_rmse = "N/A", float('inf'), float('inf')

    if df_compare_data.empty:
        st.warning("Not enough data for model comparison chart.")
    else:
        avg_actual_compare = df_compare_data['Close'].mean()
        fig_compare_chart = go.Figure()
        fig_compare_chart.add_trace(go.Scatter(x=df_compare_data['Date'], y=df_compare_data['Close'], mode='lines', name='Actual Close', line=dict(color='black', width=2)))
        
        colors = px.colors.qualitative.Plotly
        
        models_for_chart = compare_models[:]
        if model_choice not in models_for_chart and not enable_ensemble: # Add main model if not ensemble, and not already in comparison
            models_for_chart.insert(0, model_choice)
        elif enable_ensemble and ensemble_models: # Add a placeholder for ensemble if enabled
            models_for_chart.insert(0, f"Ensemble ({', '.join(ensemble_models)})")
            # Ensure ensemble is not duplicated if model_choice is part of it.
            if model_choice in models_for_chart[1:]: models_for_chart.remove(model_choice)

        with st.spinner("Comparing models on recent data..."):
            for i, model_name_iter in enumerate(models_for_chart):
                y_pred_iter = np.array([])
                y_lower_iter = np.array([])
                y_upper_iter = np.array([])

                # Determine which model to use for this iteration in comparison
                current_model_for_comparison_info = None
                
                if "Ensemble" in model_name_iter:
                    ensemble_trained_models_comp = []
                    for m_name in ensemble_models:
                        comp_model_trained_dict, _ = train_models_pipeline(df_train_period.copy(), m_name, perform_tuning, update_log, selected_indicator_params) 
                        if comp_model_trained_dict.get('Close'):
                            ensemble_trained_models_comp.append(comp_model_trained_dict)
                    
                    if ensemble_trained_models_comp:
                        # make_ensemble_prediction now returns (point_array, lower_array, upper_array)
                        y_pred_iter, y_lower_iter, y_upper_iter = make_ensemble_prediction(ensemble_trained_models_comp, df_compare_data.copy(), update_log, confidence_level_pct if enable_confidence_interval else None)
                        model_display_for_chart = model_name_iter
                    else:
                        update_log(f"Ensemble model failed for comparison.")
                        continue
                else: # Individual model for comparison chart
                    # If this is the main model, use the already trained model from trained_models_main['Close']
                    if model_name_iter == model_choice:
                        current_model_for_comparison_info = trained_models_main.get('Close')
                    else: # Otherwise, train the comparison model for 'Close' target
                        comp_trained_dict, _ = train_models_pipeline(df_train_period.copy(), model_name_iter, perform_tuning, update_log, selected_indicator_params)
                        current_model_for_comparison_info = comp_trained_dict.get('Close')
                    
                    if current_model_for_comparison_info and current_model_for_comparison_info.get('model'):
                        # Now, generate predictions using this specific model info for the 'Close' target
                        # The generate_predictions_pipeline returns a dict of dfs, each with bounds
                        preds_dict_iter = generate_predictions_pipeline(df_compare_data.copy(), {'Close': current_model_for_comparison_info}, update_log, confidence_level_pct if enable_confidence_interval else None)
                        
                        if 'Close' in preds_dict_iter and not preds_dict_iter['Close'].empty:
                            y_pred_iter = preds_dict_iter['Close'][f'Predicted Close'].values
                            y_lower_iter = preds_dict_iter['Close'][f'Predicted Close Lower'].values
                            y_upper_iter = preds_dict_iter['Close'][f'Predicted Close Upper'].values
                            model_display_for_chart = model_name_iter
                        else:
                            update_log(f"Model {model_name_iter} failed for comparison.")
                            continue
                    else:
                        update_log(f"Model {model_name_iter} (Close target) not available for comparison.")
                        continue
                
                # Ensure actuals are aligned to the length of predictions
                # Use df_compare_data['Close'] for actuals, ensuring it matches the prediction length
                y_actual_iter = df_compare_data['Close'].iloc[-len(y_pred_iter):].values
                
                if y_pred_iter.size > 0 and len(y_actual_iter) == len(y_pred_iter):
                    # Filter out NaN predictions and actuals for metric calculation
                    valid_mask_comp = ~np.isnan(y_pred_iter) & ~np.isnan(y_actual_iter)
                    y_pred_iter_valid = y_pred_iter[valid_mask_comp]
                    y_actual_iter_valid = y_actual_iter[valid_mask_comp]

                    if len(y_actual_iter_valid) > 0:
                        mae_iter = mean_absolute_error(y_actual_iter_valid, y_pred_iter_valid)
                        rmse_iter = np.sqrt(mean_squared_error(y_actual_iter_valid, y_pred_iter_valid))
                        pct_mae_iter = (mae_iter / avg_actual_compare) * 100 if avg_actual_compare > 0 else np.nan
                        pct_rmse_iter = (rmse_iter / avg_actual_compare) * 100 if avg_actual_compare > 0 else np.nan
                        
                        # Calculate Directional Accuracy for comparison chart
                        # Adjust actuals and predictions for directional accuracy calculation
                        # Need the last actual close BEFORE the comparison period for the first directional comparison
                        last_actual_before_compare = df_features_full['Close'].iloc[-(compare_days_actual + 1)] if compare_days_actual > 0 and len(df_features_full) > compare_days_actual else np.nan
                        
                        if pd.notna(last_actual_before_compare):
                            # Combine the last actual before the period with actuals in the period
                            full_actual_for_dir = np.concatenate(([last_actual_before_compare], y_actual_iter))
                            true_directions_comp = np.sign(np.diff(full_actual_for_dir))
                        else:
                            true_directions_comp = np.array([])
                        
                        predicted_directions_comp = []
                        if y_pred_iter.size > 0 and pd.notna(last_actual_before_compare) and pd.notna(y_pred_iter[0]):
                             # First prediction's direction against the last known actual before the comparison period
                            if y_pred_iter[0] > last_actual_before_compare: predicted_directions_comp.append(1)
                            elif y_pred_iter[0] < last_actual_before_compare: predicted_directions_comp.append(-1)
                            else: predicted_directions_comp.append(0)
                        else:
                            predicted_directions_comp.append(np.nan)

                        for k in range(1, len(y_pred_iter)):
                            # Subsequent predictions against the actual value of the *previous* day in the comparison period
                            if pd.notna(y_pred_iter[k]) and pd.notna(y_actual_iter[k-1]):
                                if y_pred_iter[k] > y_actual_iter[k-1]: predicted_directions_comp.append(1)
                                elif y_pred_iter[k] < y_actual_iter[k-1]: predicted_directions_comp.append(-1)
                                else: predicted_directions_comp.append(0)
                            else:
                                predicted_directions_comp.append(np.nan)
                        
                        predicted_directions_comp = np.array(predicted_directions_comp)

                        # Align lengths and remove NaNs for accuracy calculation
                        min_len_dir = min(len(true_directions_comp), len(predicted_directions_comp))
                        if min_len_dir > 0:
                            valid_dir_mask = ~np.isnan(true_directions_comp[:min_len_dir]) & ~np.isnan(predicted_directions_comp[:min_len_dir])
                            if np.sum(valid_dir_mask) > 0:
                                directional_accuracy_comp = accuracy_score(true_directions_comp[:min_len_dir][valid_dir_mask], predicted_directions_comp[:min_len_dir][valid_dir_mask])
                            else:
                                directional_accuracy_comp = np.nan
                                update_log(f"Warning: No valid data points for directional accuracy in comparison for {model_display_for_chart}.")
                        else:
                            directional_accuracy_comp = np.nan
                        

                        comparison_results_list.append({'Model': model_display_for_chart, 'MAE': mae_iter, 'RMSE': rmse_iter, '%-MAE': pct_mae_iter, '%-RMSE': pct_rmse_iter, 'Directional Accuracy': directional_accuracy_comp})
                        
                        if rmse_iter < best_comp_rmse: best_comp_rmse, best_comp_model_name = rmse_iter, model_display_for_chart
                        if pct_rmse_iter < best_comp_pct_rmse: best_comp_pct_rmse = pct_rmse_iter
                            
                        # Plot predictions
                        fig_compare_chart.add_trace(go.Scatter(x=df_compare_data['Date'].iloc[-len(y_pred_iter):], y=y_pred_iter, mode='lines', name=f"{model_display_for_chart} Pred.", line=dict(color=colors[i % len(colors)], dash='dot')))
                        
                        # Plot confidence intervals if enabled and available
                        if enable_confidence_interval and pd.notna(y_lower_iter).all() and pd.notna(y_upper_iter).all():
                            fig_compare_chart.add_trace(go.Scatter(
                                x=df_compare_data['Date'].iloc[-len(y_pred_iter):].tolist() + df_compare_data['Date'].iloc[-len(y_pred_iter):].tolist()[::-1], # x, then x reversed
                                y=y_upper_iter.tolist() + y_lower_iter.tolist()[::-1], # upper, then lower reversed
                                fill='toself',
                                fillcolor=hex_to_rgba(colors[i % len(colors)], 0.2), # Use helper function
                                line=dict(color='rgba(255,255,255,0)'),
                                hoverinfo="skip",
                                showlegend=False,
                                name=f"{model_display_for_chart} {confidence_level_pct}% CI"
                            ))


        fig_compare_chart.update_layout(title=f"Model Comparison: Actual vs Predicted Close ({compare_days_actual} days)", xaxis_title="Date", yaxis_title="Price", legend_title="Legend")
        st.plotly_chart(fig_compare_chart, use_container_width=True)
        
        if comparison_results_list:
            df_comparison_tbl = pd.DataFrame(comparison_results_list).sort_values(['RMSE', '%-RMSE']).dropna(subset=['RMSE'])
            st.dataframe(df_comparison_tbl.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "%-MAE": "{:.2f}%", "%-RMSE": "{:.2f}%", "Directional Accuracy": "{:.2%}"})) # Added Directional Accuracy format
            if best_comp_model_name != "N/A":
                 st.markdown(f"🏆 **Best performing in comparison (lowest RMSE): {best_comp_model_name}** (RMSE: {best_comp_rmse:.4f}, %-RMSE: {best_comp_pct_rmse:.2f}%)")


    # --- Output 2: Training period actual vs predicted (Main Model Close Price or Ensemble) ---
    st.markdown("---")
    st.subheader(f"🎯 Training Period Performance: {model_display_name} (Close Price)") # Adjusted title for ensemble
    
    # Generate predictions for the training period
    # Use the full df_train_period for predictions to match actuals
    # generate_predictions_pipeline now returns a dict of dfs, each with bounds
    
    y_pred_train_df = pd.DataFrame() # Initialize an empty DataFrame
    
    with st.spinner("Generating training period predictions..."):
        if enable_ensemble and ensemble_models:
            # Use the ensemble trained models from earlier in the script for the training period
            ensemble_trained_models_train_perf = []
            for m_name in ensemble_models:
                current_model_trained, _ = train_models_pipeline(df_train_period.copy(), m_name, perform_tuning, update_log, selected_indicator_params) 
                if current_model_trained.get('Close'):
                    ensemble_trained_models_train_perf.append(current_model_trained)
            
            if ensemble_trained_models_train_perf:
                # make_ensemble_prediction now returns (point_array, lower_array, upper_array)
                point_arr, lower_arr, upper_arr = make_ensemble_prediction(ensemble_trained_models_train_perf, df_train_period.copy(), update_log, confidence_level_pct if enable_confidence_interval else None)
                y_pred_train_df = pd.DataFrame({
                    'Date': df_train_period['Date'].values,
                    'Predicted Close': point_arr,
                    'Predicted Close Lower': lower_arr,
                    'Predicted Close Upper': upper_arr
                })
                # Join with actuals for plotting and metrics
                y_pred_train_df = pd.merge(df_train_period[['Date', 'Close']], y_pred_train_df, on='Date', how='inner')
                y_pred_train_df.rename(columns={'Close': 'Actual Close'}, inplace=True)

        else: # Single model for training performance
            if 'Close' in trained_models_main and trained_models_main['Close'].get('model'):
                # generate_predictions_pipeline returns a dict of dfs, access the 'Close' target's DF
                train_preds_dict = generate_predictions_pipeline(df_train_period.copy(), {'Close': trained_models_main['Close']}, update_log, confidence_level_pct if enable_confidence_interval else None)
                if 'Close' in train_preds_dict and not train_preds_dict['Close'].empty:
                    y_pred_train_df = train_preds_dict['Close']
                    y_pred_train_df.rename(columns={'Actual Close': 'Actual Close'}, inplace=True) # Ensure correct column name if it was from original
                else:
                    update_log("No valid training predictions found for plotting.")
            else:
                update_log("Main Close model not available for training period performance plot.")

    if not y_pred_train_df.empty and 'Predicted Close' in y_pred_train_df.columns:
        # Align actuals and predictions
        y_actual_train_aligned = y_pred_train_df['Actual Close'].values
        y_pred_train_values = y_pred_train_df['Predicted Close'].values

        # Filter out NaN predictions and actuals for metric calculation
        valid_mask_train = ~np.isnan(y_pred_train_values) & ~np.isnan(y_actual_train_aligned)
        y_pred_train_valid = y_pred_train_values[valid_mask_train]
        y_actual_train_valid = y_actual_train_aligned[valid_mask_train]

        if len(y_actual_train_valid) > 0:
            mae_train_main = mean_absolute_error(y_actual_train_valid, y_pred_train_valid)
            rmse_train_main = np.sqrt(mean_squared_error(y_actual_train_valid, y_pred_train_valid))
            avg_actual_train_main = y_actual_train_valid.mean()
            pct_mae_train_main = (mae_train_main / avg_actual_train_main) * 100 if avg_actual_train_main > 0 else np.nan
            pct_rmse_train_main = (rmse_train_main / avg_actual_train_main) * 100 if avg_actual_train_main > 0 else np.nan

            # Directional Accuracy for training period
            # Need actuals from the day before the first predicted day in `y_pred_train_df`
            # `df_train_period` is the original training data.
            # Get the index of the first date in `y_pred_train_df` within `df_train_period`
            first_pred_date_idx = df_train_period[df_train_period['Date'] == y_pred_train_df['Date'].iloc[0]].index
            
            last_actual_before_train_pred = np.nan
            if first_pred_date_idx.size > 0 and first_pred_date_idx[0] > 0:
                last_actual_before_train_pred = df_train_period['Close'].iloc[first_pred_date_idx[0] - 1]
            
            # Combine the last actual before the predicted training period with actuals in the period
            full_actual_for_dir_train = np.array([])
            if pd.notna(last_actual_before_train_pred):
                full_actual_for_dir_train = np.concatenate(([last_actual_before_train_pred], y_actual_train_aligned))
            else:
                full_actual_for_dir_train = y_actual_train_aligned # Fallback if no prior day for comparison

            if full_actual_for_dir_train.size > 1:
                true_directions_train = np.sign(np.diff(full_actual_for_dir_train))
            else:
                true_directions_train = np.array([])

            predicted_directions_train = []
            # First prediction's direction against the last actual before the training period
            if y_pred_train_values.size > 0 and pd.notna(last_actual_before_train_pred) and pd.notna(y_pred_train_values[0]):
                if y_pred_train_values[0] > last_actual_before_train_pred: predicted_directions_train.append(1)
                elif y_pred_train_values[0] < last_actual_before_train_pred: predicted_directions_train.append(-1)
                else: predicted_directions_train.append(0)
            else:
                predicted_directions_train.append(np.nan) # Cannot calculate if no prior actual or prediction

            for k in range(1, len(y_pred_train_values)):
                # Subsequent predictions against the *actual* value of the previous day within the training period
                if pd.notna(y_pred_train_values[k]) and pd.notna(y_actual_train_aligned[k-1]):
                    if y_pred_train_values[k] > y_actual_train_aligned[k-1]: predicted_directions_train.append(1)
                    elif y_pred_train_values[k] < y_actual_train_aligned[k-1]: predicted_directions_train.append(-1)
                    else: predicted_directions_train.append(0)
                else:
                    predicted_directions_train.append(np.nan)

            predicted_directions_train = np.array(predicted_directions_train)

            min_len_dir_train = min(len(true_directions_train), len(predicted_directions_train))
            if min_len_dir_train > 0:
                valid_dir_mask_train = ~np.isnan(true_directions_train[:min_len_dir_train]) & ~np.isnan(predicted_directions_train[:min_len_dir_train])
                if np.sum(valid_dir_mask_train) > 0:
                    directional_accuracy_train = accuracy_score(true_directions_train[:min_len_dir_train][valid_dir_mask_train], predicted_directions_train[:min_len_dir_train][valid_dir_mask_train])
                else:
                    directional_accuracy_train = np.nan
                    update_log(f"Warning: No valid data points for directional accuracy in training period for {model_display_name}.")
            else:
                directional_accuracy_train = np.nan
            
            st.markdown(f"**Metrics on Training Data ({start_bt.strftime('%Y-%m-%d')} to {end_bt.strftime('%Y-%m-%d')}):**")
            st.markdown(f"MAE: {mae_train_main:.4f} ({pct_mae_train_main:.2f}%) | RMSE: {rmse_train_main:.4f} ({pct_rmse_train_main:.2f}%) | Directional Accuracy: {directional_accuracy_train:.2%}")
        else:
            st.info("No valid training data after filtering for metrics calculation.")
        
        fig_train_perf = go.Figure()
        fig_train_perf.add_trace(go.Scatter(x=y_pred_train_df['Date'], y=y_pred_train_df['Actual Close'], mode='lines', name='Actual Close', line=dict(color='royalblue')))
        fig_train_perf.add_trace(go.Scatter(x=y_pred_train_df['Date'], y=y_pred_train_df['Predicted Close'], mode='lines', name='Predicted Close', line=dict(color='orangered', dash='dash')))
        
        # Plot confidence intervals if enabled and available
        if enable_confidence_interval and 'Predicted Close Lower' in y_pred_train_df.columns and 'Predicted Close Upper' in y_pred_train_df.columns and \
           pd.notna(y_pred_train_df['Predicted Close Lower']).all() and pd.notna(y_pred_train_df['Predicted Close Upper']).all():
            fig_train_perf.add_trace(go.Scatter(
                x=y_pred_train_df['Date'].tolist() + y_pred_train_df['Date'].tolist()[::-1],
                y=y_pred_train_df['Predicted Close Upper'].tolist() + y_pred_train_df['Predicted Close Lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255,165,0,0.2)', # Orange with transparency
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False,
                name=f"{model_display_name} {confidence_level_pct}% CI"
            ))

        fig_train_perf.update_layout(title=f"Training: Actual vs Predicted Close ({model_display_name})", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_train_perf, use_container_width=True)
    else: st.warning("Could not generate training period predictions for the main Close model or ensemble.")


    # --- Output 3: Walk-Forward Backtesting ---
    st.markdown("---")
    st.subheader(f"🔄 Walk-Forward Backtesting Results")
    if enable_walk_forward:
        if not wf_models_to_test:
            st.warning("Please select at least one model for Walk-Forward Backtesting.")
        else:
            # Determine the starting point for walk-forward data (from the full features)
            # This should be the earliest date that still allows 'initial_train_days_wf' and subsequent steps.
            # For simplicity, let's take the last 2 years of data for walk-forward, if available
            wf_start_date = pd.to_datetime(end_bt) - timedelta(days=2*365) # Approx 2 years before end_bt
            df_walk_forward_data = df_features_full[df_features_full['Date'] >= wf_start_date].copy()
            df_walk_forward_data.reset_index(drop=True, inplace=True) # Reset index after filtering

            # Ensure enough historical data is available for feature calculation
            min_data_for_wf = initial_train_days_wf + step_forward_days 
            
            if len(df_walk_forward_data) < min_data_for_wf:
                st.warning(f"Not enough data for walk-forward backtesting with current settings ({len(df_walk_forward_data)} rows). Need at least {min_data_for_wf} rows. Consider reducing initial training period or step size, or extending historical data period.")
            else:
                with st.spinner("Performing walk-forward backtesting... This may take a while."):
                    wf_predictions_df, wf_metrics_df = perform_walk_forward_backtesting(
                        df_walk_forward_data.copy(),
                        wf_models_to_test,
                        perform_tuning,
                        initial_train_days_wf,
                        step_forward_days, 
                        selected_indicator_params, # Pass the selected indicator parameters
                        update_log
                    )
                
                if not wf_predictions_df.empty:
                    st.markdown("**Walk-Forward Predictions (Sample)**")
                    st.dataframe(wf_predictions_df.head().style.format({"Actual Close": "{:.2f}", "Predicted Close": "{:.2f}"})) # Format the numbers
                    
                    st.markdown("**Walk-Forward Evaluation Metrics (Aggregated)**")
                    # Aggregate metrics across all periods for each model
                    agg_metrics = wf_metrics_df.groupby('Model').agg(
                        Avg_MAE=('MAE', 'mean'),
                        Avg_RMSE=('RMSE', 'mean'),
                        Avg_Directional_Accuracy=('Directional Accuracy', 'mean')
                    ).reset_index()
                    st.dataframe(agg_metrics.style.format({
                        "Avg_MAE": "{:.4f}", 
                        "Avg_RMSE": "{:.4f}", 
                        "Avg_Directional_Accuracy": "{:.2%}"
                    }))

                    # Plotting Walk-Forward Results
                    fig_wf = go.Figure()
                    # Plot actuals (deduplicate if multiple models predicted for same date)
                    actuals_to_plot = wf_predictions_df[['Date', 'Actual Close']].drop_duplicates(subset=['Date'])
                    fig_wf.add_trace(go.Scatter(x=actuals_to_plot['Date'], y=actuals_to_plot['Actual Close'], mode='lines', name='Actual Close', line=dict(color='black', width=2)))
                    
                    # Plot predictions for each model
                    wf_colors = px.colors.qualitative.Dark24
                    for idx, model_name_wf in enumerate(wf_models_to_test):
                        model_wf_preds = wf_predictions_df[wf_predictions_df['Model'] == model_name_wf]
                        fig_wf.add_trace(go.Scatter(x=model_wf_preds['Date'], y=model_wf_preds['Predicted Close'], mode='lines', name=f'{model_name_wf} Pred', line=dict(color=wf_colors[idx % len(wf_colors)], dash='dash')))
                    
                    fig_wf.update_layout(title='Walk-Forward Backtesting: Actual vs Predicted Close', xaxis_title='Date', yaxis_title='Price')
                    st.plotly_chart(fig_wf, use_container_width=True)
                    
                    st.info("""
                    **About Walk-Forward Backtesting:**
                    This method simulates real-world trading more closely by:
                    * Periodically retraining the model using all available data up to that point.
                    * Making predictions for a short, fixed future period (the "step forward").
                    * Evaluating performance on these out-of-sample predictions.
                    This process helps assess the model's robustness and adaptability over time.
                    """)

                else:
                    st.info("No walk-forward backtesting results generated. Check log for issues.")
    else:
        st.info("Walk-Forward Backtesting is disabled. Enable it in the sidebar to view results.")


    # --- Output 4: Future Predictions (Iterative, Main Model Close Price or Ensemble) ---
    st.markdown("---")
    st.subheader(f"🚀 Future {n_future} Days Predicted Close Prices ({model_display_name})") # Adjusted title for ensemble
    st.info("""
    **Iterative Forecasting Explained:** Monarch predicts future prices one day at a time. 
    The prediction for Day 1 is made. Then, this Day 1 prediction is used as an input (as if it were actual data) to help predict Day 2, and so on. 
    This process repeats for the number of future days you select. 
    *Remember: These are model-based projections and not financial advice.*
    """)

    future_predictions_output_list = []
    
    # Determine which model(s) to use for future predictions: main model or ensemble
    models_for_future_prediction = []
    if enable_ensemble and ensemble_models:
        for m_name in ensemble_models:
            current_model_trained, _ = train_models_pipeline(df_train_period.copy(), m_name, perform_tuning, update_log, selected_indicator_params) 
            if current_model_trained.get('Close'):
                models_for_future_prediction.append(current_model_trained)
        if not models_for_future_prediction:
            st.warning("No valid models for ensemble future prediction. Check training logs.")
    else:
        if 'Close' in trained_models_main and trained_models_main['Close'].get('model'):
            models_for_future_prediction.append({'Close': trained_models_main['Close']})
        else:
            st.warning("Main Close model not available for future predictions.")

    if not models_for_future_prediction:
        st.info("No models configured or trained for future predictions.")
    else:
        # Clone df_features_full for iterative predictions to avoid modifying original
        df_hist_context_fut = df_features_full[df_features_full['Date'] <= pd.to_datetime(end_bt)].copy()

        # Determine max historical window needed for feature calculation for any *enabled* feature
        max_hist_window_fut = 0
        for param_name, (default_value, default_enabled) in config.TECHNICAL_INDICATORS_DEFAULTS.items():
            if param_name in selected_indicator_params and selected_indicator_params[param_name] is not None:
                param_value = selected_indicator_params[param_name]
                if isinstance(param_value, list) and param_value: 
                    max_hist_window_fut = max(max_hist_window_fut, max(param_value))
                elif isinstance(param_value, (int, float)):
                    max_hist_window_fut = max(max_hist_window_fut, int(param_value))
        
        max_hist_window_fut += 1 # Account for window calculation requiring previous data points
        if max_hist_window_fut < 50: 
            max_hist_window_fut = 50 


        if len(df_hist_context_fut) < max_hist_window_fut:
            st.warning(f"Not enough historical data (before {end_bt}) to reliably generate features for future predictions. Need at least {max_hist_window_fut} data points for enabled indicators. Consider extending the 'Training Period'.")
        
        current_iter_date = pd.to_datetime(end_bt)
        # Use the actual last day's data as the starting point for iterative prediction
        last_day_data_for_iter_start = data.iloc[-1] if not data.empty else pd.Series()
        current_iter_close = last_day_data_for_iter_start.get('Close', np.nan)
        current_iter_open = last_day_data_for_iter_start.get('Open', np.nan)
        current_iter_high = last_day_data_for_iter_start.get('High', np.nan)
        current_iter_low = last_day_data_for_iter_start.get('Low', np.nan)
        current_iter_volume = last_day_data_for_iter_start.get('Volume', np.nan)
        
        if df_hist_context_fut.empty or df_hist_context_fut.tail(1).isnull().values.any():
            st.warning("No valid historical data available to start iterative prediction.")
            st.stop()
        
        # Use a spinner for the future prediction loop
        with st.spinner(f"Generating {n_future} days of future predictions..."):
            for i in range(n_future):
                next_pred_date = current_iter_date + timedelta(days=1)
                while next_pred_date.weekday() >= 5: next_pred_date += timedelta(days=1)

                # Simulate a new row with the previous day's predicted/actual close, etc.
                # This new row will be used to generate features for the current prediction.
                simulated_current_day_data = pd.DataFrame([{
                    'Date': current_iter_date,
                    'Open': current_iter_open, 
                    'High': current_iter_high,
                    'Low': current_iter_low,
                    'Close': current_iter_close,
                    'Volume': current_iter_volume
                }])

                # Concatenate the simulated row to the historical context for feature generation
                df_hist_context_for_features = pd.concat([df_hist_context_fut, simulated_current_day_data], ignore_index=True)

                # Ensure df_hist_context_for_features has enough rows for creating features
                required_rows_for_features = max_hist_window_fut
                
                if len(df_hist_context_for_features) < required_rows_for_features:
                    update_log(f"Not enough historical context for feature creation in future prediction at step {i+1}. Stopping.")
                    break
                
                # Slice df_hist_context_for_features to provide only the relevant part for features
                temp_df_for_feature_creation_slice = df_hist_context_for_features.tail(required_rows_for_features).copy()
                
                df_features_for_pred = create_features(
                    temp_df_for_feature_creation_slice, 
                    selected_indicator_params
                )
                
                if df_features_for_pred.empty:
                    update_log(f"⚠️ Empty features at future step {i+1}. Stopping iterative prediction.")
                    break
                
                # For prediction, get the last row
                predict_input_df = df_features_for_pred.tail(1)


                predicted_price_fut, predicted_price_lower_fut, predicted_price_upper_fut = np.nan, np.nan, np.nan

                if enable_ensemble and models_for_future_prediction:
                    # make_ensemble_prediction returns arrays, so take the first element for single row prediction
                    point_arr, lower_arr, upper_arr = make_ensemble_prediction(models_for_future_prediction, predict_input_df.copy(), update_log, confidence_level_pct if enable_confidence_interval else None)
                    predicted_price_fut = point_arr[0] if point_arr.size > 0 else np.nan
                    predicted_price_lower_fut = lower_arr[0] if lower_arr.size > 0 else np.nan
                    predicted_price_upper_fut = upper_arr[0] if upper_arr.size > 0 else np.nan
                elif models_for_future_prediction: # Single model for future prediction
                    single_model_info = models_for_future_prediction[0].get('Close')
                    if single_model_info and single_model_info.get('model'):
                        # _predict_single_model_raw returns arrays, so take the first element for single row prediction
                        point_arr, lower_arr, upper_arr = _predict_single_model_raw(single_model_info, predict_input_df.copy(), confidence_level_pct if enable_confidence_interval else None)
                        predicted_price_fut = point_arr[0] if point_arr.size > 0 else np.nan
                        predicted_price_lower_fut = lower_arr[0] if lower_arr.size > 0 else np.nan
                        predicted_price_upper_fut = upper_arr[0] if upper_arr.size > 0 else np.nan
                    else:
                        update_log(f"Main Close model not found for future prediction in step {i+1}.")
                
                if pd.isna(predicted_price_fut):
                    update_log(f"❌ Failed to predict future price at step {i+1}. Stopping iterative prediction.")
                    break

                future_predictions_output_list.append({
                    'Date': next_pred_date, 
                    'Predicted Close': predicted_price_fut,
                    'Predicted Close Lower': predicted_price_lower_fut,
                    'Predicted Close Upper': predicted_price_upper_fut
                })
                
                # Update current_iter_ variables for the next loop, assuming the predicted values become the 'actuals'
                current_iter_date = next_pred_date
                current_iter_close = predicted_price_fut
                # Simple estimation for Open, High, Low, Volume based on predicted close for next iteration's features
                # These are heuristic and can be refined.
                current_iter_open = predicted_price_fut * (1 + np.random.uniform(-0.005, 0.005)) 
                current_iter_high = max(predicted_price_fut, current_iter_open) * (1 + np.random.uniform(0.001, 0.01))
                current_iter_low = min(predicted_price_fut, current_iter_open) * (1 - np.random.uniform(0.001, 0.01))
                current_iter_volume = max(current_iter_volume * (1 + np.random.uniform(-0.05, 0.05)), 1) # Ensure volume is at least 1
                
                # Append the "new actual" (which is the predicted value from this step) to df_hist_context_fut
                # This ensures the context grows for the next iteration's feature creation.
                new_row_for_context = pd.DataFrame([{
                    'Date': current_iter_date,
                    'Open': current_iter_open,
                    'High': current_iter_high,
                    'Low': current_iter_low,
                    'Close': current_iter_close,
                    'Volume': current_iter_volume
                }])
                df_hist_context_fut = pd.concat([df_hist_context_fut, new_row_for_context], ignore_index=True)


        if future_predictions_output_list:
            df_future_preds_tbl = pd.DataFrame(future_predictions_output_list)
            st.dataframe(df_future_preds_tbl.style.format({
                "Predicted Close": "{:.2f}",
                "Predicted Close Lower": "{:.2f}",
                "Predicted Close Upper": "{:.2f}"
            }))
            fig_future_chart = px.line(df_future_preds_tbl, x='Date', y='Predicted Close', title=f"Future {n_future} Days Predicted Close ({model_display_name})")
            fig_future_chart.update_traces(line=dict(color='mediumseagreen'))

            # Add confidence interval bands for future predictions
            if enable_confidence_interval and pd.notna(df_future_preds_tbl['Predicted Close Lower']).all() and pd.notna(df_future_preds_tbl['Predicted Close Upper']).all():
                fig_future_chart.add_trace(go.Scatter(
                    x=df_future_preds_tbl['Date'].tolist() + df_future_preds_tbl['Date'].tolist()[::-1],
                    y=df_future_preds_tbl['Predicted Close Upper'].tolist() + df_future_preds_tbl['Predicted Close Lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(60,179,113,0.2)', # Mediumseagreen with transparency
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"{model_display_name} {confidence_level_pct}% CI"
                ))

            st.plotly_chart(fig_future_chart, use_container_width=True)
            
            for pred_item in future_predictions_output_list:
                save_prediction(ticker, pred_item['Date'], pred_item['Predicted Close'], np.nan, model_display_name, datetime.now(), end_bt, predicted_type='Close', predicted_lower_bound=pred_item['Predicted Close Lower'], predicted_upper_bound=pred_item['Predicted Close Upper'])
        else:
            st.warning("Could not generate future predictions. Check logs.")
    


    # --- Historical Prediction Log Display ---
    st.markdown("---")
    st.header("🕰️ Historical Prediction Log")
    st.markdown("""
    This log shows all predictions made by Monarch for this ticker. 
    `prediction_generation_date`: When the prediction was made.
    `prediction_for_date`: The date the prediction was *for*.
    `training_end_date_used`: The 'Training End Date (t2)' used for the model that made this prediction.
    `predicted_type`: What specific value was predicted (e.g., Close, Open, High, Low, Volatility).
    This helps track how forecasts for a specific date changed as more data became available.
    """)
    past_preds_df = load_past_predictions(ticker)
    if not past_preds_df.empty:
        # Ensure all expected columns exist, adding them with NaT/NaN if not (for backward compatibility with old logs)
        # Using PREDICTION_LOG_COLUMNS which now includes 'predicted_type' and bounds
        for col in PREDICTION_LOG_COLUMNS:
            if col not in past_preds_df.columns:
                if 'date' in col: past_preds_df[col] = pd.NaT
                elif col == 'predicted_type': past_preds_df[col] = 'Close' # Default for older entries
                else: past_preds_df[col] = np.nan
        
        # Sort and display
        past_preds_df_display = past_preds_df.sort_values(by=['prediction_for_date', 'prediction_generation_date', 'predicted_type'], ascending=[False, False, True]).reset_index(drop=True)
        
        # Select and reorder columns for display
        display_cols = [
            'prediction_for_date', 'predicted_type', 'predicted_value', 
            'predicted_lower_bound', 'predicted_upper_bound', # Added bounds
            'actual_close', 'model_used', 'prediction_generation_date', 'training_end_date_used'
        ]
        past_preds_df_display = past_preds_df_display[display_cols]

        st.dataframe(past_preds_df_display.style.format({
            "predicted_value": "{:.2f}", 
            "predicted_lower_bound": "{:.2f}",
            "predicted_upper_bound": "{:.2f}",
            "actual_close": "{:.2f}",
            "prediction_generation_date": lambda x: pd.to_datetime(x).strftime('%Y-%m-%d %H:%M') if pd.notnull(x) else 'N/A',
            "prediction_for_date": lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if pd.notnull(x) else 'N/A',
            "training_end_date_used": lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if pd.notnull(x) else 'N/A'
        }))

        st.subheader("Visualizing Past Predictions vs. Actuals (Close Price)")
        # Filter for 'Close' predictions to avoid mixing scales
        plot_df_past_close = past_preds_df_display[past_preds_df_display['predicted_type'] == 'Close'].dropna(subset=['actual_close', 'predicted_value', 'prediction_for_date']).copy()
        
        if not plot_df_past_close.empty:
            fig_past_log = go.Figure()
            actual_data_log_plot = plot_df_past_close.groupby('prediction_for_date')['actual_close'].first().reset_index()
            fig_past_log.add_trace(go.Scatter(x=actual_data_log_plot['prediction_for_date'], y=actual_data_log_plot['actual_close'], mode='lines+markers', name='Actual Close', line=dict(color='navy', width=2)))

            unique_gen_dates = plot_df_past_close['prediction_generation_date'].dt.date.unique()
            # Show predictions from the last 5 distinct generation dates for clarity
            limited_gen_dates = sorted(unique_gen_dates, reverse=True)[:5] 

            for gen_d in limited_gen_dates:
                gen_df_subset = plot_df_past_close[plot_df_past_close['prediction_generation_date'].dt.date == gen_d]
                if not gen_df_subset.empty:
                    for model_log_name in gen_df_subset['model_used'].unique():
                        model_gen_subset = gen_df_subset[gen_df_subset['model_used'] == model_log_name]
                        fig_past_log.add_trace(go.Scatter(
                            x=model_gen_subset['prediction_for_date'], y=model_gen_subset['predicted_value'],
                            mode='lines+markers', name=f'Pred ({model_log_name}) Gen: {gen_d.strftime("%Y-%m-%d")}',
                            line=dict(dash='dot') 
                        ))
                        # Add confidence intervals if available for historical log
                        if enable_confidence_interval and pd.notna(model_gen_subset['predicted_lower_bound']).all() and pd.notna(model_gen_subset['predicted_upper_bound']).all():
                            fig_past_log.add_trace(go.Scatter(
                                x=model_gen_subset['prediction_for_date'].tolist() + model_gen_subset['prediction_for_date'].tolist()[::-1],
                                y=model_gen_subset['predicted_upper_bound'].tolist() + model_gen_subset['predicted_lower_bound'].tolist()[::-1],
                                fill='toself',
                                fillcolor=hex_to_rgba('000080', 0.1), # Navy hex color with transparency
                                line=dict(color='rgba(255,255,255,0)'),
                                hoverinfo="skip",
                                showlegend=False,
                                name=f"Pred ({model_log_name}) {confidence_level_pct}% CI"
                            ))

            fig_past_log.update_layout(title=f'Historical Log: Predicted vs. Actual Close for {ticker} (Recent Generations)', xaxis_title="Date of Stock Price", yaxis_title="Price", hovermode="x unified")
            st.plotly_chart(fig_past_log, use_container_width=True)
        else: st.info("No past 'Close' predictions with known actuals to plot from log.")
    else: st.info(f"No past prediction logs found for {ticker}.")


    # --- Feature Importance (Main Close Model) ---
    st.markdown("---")
    st.subheader(f"🧠 Feature Importance / Coefficients ({model_display_name} - Close Price Model)") # Adjusted title for ensemble
    
    # Feature importance only makes sense for a single model, not an ensemble average or LSTM
    if enable_ensemble:
        st.info("Feature importance is not directly applicable to ensemble models (which average predictions). Please select a single model for 'Main Model' to view its feature importance.")
    else: # LSTM is now removed, so no specific check needed
        main_close_model_fi = trained_models_main.get('Close')
        if main_close_model_fi and main_close_model_fi.get('model') and main_close_model_fi.get('features'):
            model_fi = main_close_model_fi['model']
            feature_names_fi = main_close_model_fi['features']
            
            fi_data = None
            if hasattr(model_fi, 'feature_importances_'):
                fi_data = pd.DataFrame({'Feature': feature_names_fi, 'Importance': model_fi.feature_importances_}).sort_values('Importance', ascending=False)
                chart_title_fi = f'Feature Importances for {model_choice} (Close Price)'
                bar_x_fi, bar_y_fi = 'Importance', 'Feature'
            elif hasattr(model_fi, 'coef_'):
                fi_data = pd.DataFrame({'Feature': feature_names_fi, 'Coefficient': model_fi.coef_}).sort_values('Coefficient', ascending=False)
                chart_title_fi = f'Coefficients for {model_choice} (Close Price)'
                bar_x_fi, bar_y_fi = 'Coefficient', 'Feature'
            
            if fi_data is not None:
                st.dataframe(fi_data.head(15)) # Show top 15
                fig_fi_chart = px.bar(fi_data.head(15), x=bar_x_fi, y=bar_y_fi, orientation='h', title=chart_title_fi)
                st.plotly_chart(fig_fi_chart, use_container_width=True)
            else: st.info(f"Feature importance/coefficients not available for {model_choice}.")
        else: st.info("Main Close model details not available for feature importance.")

else:
    st.warning(f"Could not load data for {ticker}. Please check the ticker symbol and your internet connection.")

st.markdown("---")
st.caption("Monarch Stock Price Predictor | Disclaimer: For educational and informational purposes only. Not financial advice")

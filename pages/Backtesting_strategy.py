# pages/Backtesting_Strategy.py
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
    download_data, create_features,
    train_models_pipeline, generate_predictions_pipeline,
    parse_int_list # For parsing indicator inputs
)

st.set_page_config(page_title="Monarch: Backtesting & Strategy", layout="wide")

st.header("📈 Backtesting & Strategy Page")
st.markdown("""
This page allows you to rigorously test the prediction performance of your chosen models and technical indicators on past data,
and to simulate a basic trading strategy.

**Backtesting Prediction (t1 to t2):** Select a historical training period (`t1` to `t2`). The application will predict the stock's closing price
for the day immediately after `t2` (`t2+1`), using *only* data up to `t2` for model training. The actual closing price for `t2+1`
will then be provided separately, allowing you to directly calculate and analyze the prediction error for historical forecasts.

**Trading Strategy Simulation (s1 to s2):** Define a historical period to simulate trades. Based on your model's predictions,
the system will generate buy/sell signals and track a hypothetical portfolio's performance.
""")

# --- Sidebar for Inputs ---
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, RELIANCE.NS)", value="AAPL").upper()

today = date.today()

# --- Section for Single-Point Backtesting ---
st.sidebar.subheader("Backtesting Prediction (t1 to t2)")
default_t2_end_date = today - timedelta(days=30) # End training 30 days ago
default_t1_start_date = default_t2_end_date - timedelta(days=365 * 2) # Start training 2 years before end

t1_start_date = st.sidebar.date_input("Training Start Date (t1)", value=default_t1_start_date, help="The beginning of the historical data used for model training for the single-point prediction.")
t2_end_date = st.sidebar.date_input("Training End Date (t2)", value=default_t2_end_date, help="The end of the historical data used for model training for the single-point prediction. Predictions will be made for the day after this date (t2+1).")

if t1_start_date >= t2_end_date:
    st.sidebar.error("Training Start Date (t1) must be before Training End Date (t2).")
    st.stop()

# --- Hyperparameter Tuning Option for Backtesting ---
perform_tuning_backtest = st.sidebar.checkbox("Perform Hyperparameter Tuning (Backtesting)", value=False, help="Enable this to optimize model parameters during training. Can increase processing time.")

# --- Confidence Interval Parameter for Backtesting ---
enable_confidence_interval_backtest = st.sidebar.checkbox("Enable Prediction Confidence Intervals (Backtesting)", value=False, help="Display prediction intervals based on model residuals.")
confidence_level_pct_backtest = 90
if enable_confidence_interval_backtest:
    confidence_level_pct_backtest = st.sidebar.slider("Confidence Level (%) (Backtesting)", min_value=70, max_value=99, value=90, step=1, help="The confidence level for the prediction interval (e.g., 90% means 5th-95th percentile).")


# --- Section for Strategy Simulation ---
st.sidebar.subheader("Trading Strategy Simulation (s1 to s2)")
default_s2_end_date = today - timedelta(days=7) # End strategy 7 days ago

# Dictionary to store selected indicator parameters for backtesting (moved up for min_data_required calculation)
selected_indicator_params = {}

# Function to provide error callback to parse_int_list for Streamlit display (moved up)
def sidebar_error_callback(message):
    st.sidebar.error(message)

# Loop through TECHNICAL_INDICATORS_DEFAULTS to create UI for each (moved up)
for indicator_name, (default_value, default_enabled) in config.TECHNICAL_INDICATORS_DEFAULTS.items():
    if indicator_name == 'PARSAR_ACCELERATION' or indicator_name == 'PARSAR_MAX_ACCELERATION':
        # These are handled inside the PARSAR_ENABLED block
        continue
    
    checkbox_key = f"enable_{indicator_name.lower()}_backtest" # Unique key
    param_key_prefix = indicator_name.replace('_', ' ')

    if indicator_name.endswith('_ENABLED'):
        enabled = st.sidebar.checkbox(f"Enable {param_key_prefix.replace('_enabled', '')}:", value=default_enabled, key=checkbox_key)
        selected_indicator_params[indicator_name] = enabled
        if indicator_name == 'PARSAR_ENABLED' and enabled:
            selected_indicator_params['PARSAR_ACCELERATION'] = st.sidebar.number_input(f"  Parabolic SAR Acceleration:", min_value=0.01, max_value=0.5, value=config.TECHNICAL_INDICATORS_DEFAULTS['PARSAR_ACCELERATION'][0], step=0.01, key=f"input_parsar_accel_backtest")
            selected_indicator_params['PARSAR_MAX_ACCELERATION'] = st.sidebar.number_input(f"  Parabolic SAR Max Acceleration:", min_value=0.01, max_value=0.5, value=config.TECHNICAL_INDICATORS_DEFAULTS['PARSAR_MAX_ACCELERATION'][0], step=0.01, key=f"input_parsar_max_accel_backtest")

    else:
        enabled = st.sidebar.checkbox(f"Enable {param_key_prefix}:", value=default_enabled, key=checkbox_key)
        
        if enabled:
            if isinstance(default_value, list):
                parsed_list = parse_int_list(
                    st.sidebar.text_input(f"  {param_key_prefix.replace('_', ' ')} (comma-separated days, e.g., {','.join(map(str, default_value))}):", 
                                       value=", ".join(map(str, default_value)), key=f"input_{indicator_name.lower()}_backtest"),
                    default_value,
                    sidebar_error_callback
                )
                selected_indicator_params[indicator_name] = parsed_list if parsed_list else None
            elif isinstance(default_value, (int, float)):
                if indicator_name == 'BB_STD_DEV':
                    selected_indicator_params[indicator_name] = st.sidebar.number_input(f"  {param_key_prefix.replace('_', ' ')} Multiplier:", min_value=0.1, value=default_value, step=0.1, key=f"input_{indicator_name.lower()}_backtest")
                else:
                    selected_indicator_params[indicator_name] = st.sidebar.number_input(f"  {param_key_prefix.replace('_', ' ')}:", min_value=1, value=default_value, step=1, key=f"input_{indicator_name.lower()}_backtest")
            else:
                selected_indicator_params[indicator_name] = None
        else:
            selected_indicator_params[indicator_name] = None

# Calculate min_data_required after all indicator parameters are set (Moved to global scope)
min_data_required = 0
for param_name, (default_value, default_enabled) in config.TECHNICAL_INDICATORS_DEFAULTS.items():
    if param_name in selected_indicator_params and selected_indicator_params[param_name] is not None:
        param_value = selected_indicator_params[param_name]
        if isinstance(param_value, list) and param_value:
            min_data_required = max(min_data_required, max(param_value))
        elif isinstance(param_value, (int, float)):
            min_data_required = max(min_data_required, int(param_value))
if min_data_required < 50: # Fallback to a reasonable minimum
    min_data_required = 50 

# Adjust default s1_start_date to ensure sufficient historical data for initial training
# We need to fetch data for min_data_required + a buffer (e.g., 50 trading days for NaNs, etc.) before s1_start_date
# Let's target fetching at least min_data_required * 2 in calendar days prior to s1_start_date to be safe.
# This ensures that even after feature creation and dropna, we have enough data *before* the first day of simulation.
default_s1_start_date = default_s2_end_date - timedelta(days=(min_data_required * 2) + 60) # min_data_required * 2 to account for sparse data, +60 for more buffer

s1_start_date = st.sidebar.date_input("Strategy Start Date (s1)", value=default_s1_start_date, help=f"The beginning of the period for simulating trading strategy. Ensure this date allows for at least {min_data_required} days of historical data for initial model training (consider setting it sufficiently far back based on your indicators).")
s2_end_date = st.sidebar.date_input("Strategy End Date (s2)", value=default_s2_end_date, help="The end of the period for simulating trading strategy.")

if s1_start_date >= s2_end_date:
    st.sidebar.error("Strategy Start Date (s1) must be before Strategy End Date (s2).")
    st.stop()
if s1_start_date < t2_end_date:
    st.sidebar.warning("It's recommended for the strategy period (s1-s2) to be after the backtesting training period (t1-t2) to avoid data overlap for distinct analyses.")


st.sidebar.subheader("🤖 Model Selection")
selected_models = st.sidebar.multiselect(
    "Choose Models for Close Price Prediction:",
    options=[m for m in config.MODEL_CHOICES if m != 'LSTM'], # Filter out LSTM
    default=[config.MODEL_CHOICES[0]] if config.MODEL_CHOICES else [], # Default to the first model
    help="Select one or more machine learning models to use for predicting the Close price for both backtesting and strategy."
)


st.sidebar.markdown("---")
# Add a warning if min_data_required is high and default s1_start_date might be too recent
if min_data_required > 100: # Threshold for warning, e.g., if a 100-day MA is active
    st.sidebar.warning(f"**Data Requirement Alert:** Your selected indicators require at least **{min_data_required}** days of historical data for proper training. For strategy simulation, ensure 'Strategy Start Date (s1)' is set sufficiently far in the past to allow models to 'warm up' with enough data.")


# For this page, we'll just use a simple print for logs or pass a dummy function
def dummy_log(message):
    pass # No extensive logging needed for this single-point backtest


# --- Main Content Area - Single-Point Backtest ---
if st.button("Run Single-Point Backtest Prediction", type="primary"):
    if not ticker or not selected_models:
        st.warning("Please enter a stock ticker and select at least one model to run the backtest.")
        st.stop()
    
    st.divider() # Visual separator
    st.subheader("Results: Single-Point Backtest Prediction")

    with st.spinner(f"Downloading data for {ticker}..."):
        # Download data from t1 up to t2 + ~5 trading days to ensure we capture t2+1's actual close.
        # Use np.ceil for years to ensure sufficient data is fetched.
        num_years_to_fetch = max(1, int(np.ceil((t2_end_date + timedelta(days=10) - t1_start_date).days / 365.25)))
        full_data = download_data(ticker, period=f"{num_years_to_fetch}y") 
        
        # Ensure we filter by exact dates, as period="Xy" might give slightly more data
        full_data_filtered = full_data[(full_data['Date'] >= pd.to_datetime(t1_start_date)) & (full_data['Date'] <= pd.to_datetime(t2_end_date + timedelta(days=10)))].copy()
        full_data_filtered.set_index('Date', inplace=True) # Set Date as index for easier slicing

    if full_data_filtered.empty:
        st.error(f"Could not download data for {ticker} for the specified range. Please check the ticker symbol, dates, and your internet connection.")
        st.stop()

    # Define the training data slice (t1 to t2)
    train_data_raw = full_data_filtered[full_data_filtered.index.date <= t2_end_date].copy()

    if train_data_raw.empty:
        st.error(f"No sufficient data available for training within the period {t1_start_date} to {t2_end_date}.")
        st.stop()

    st.markdown(f"**Training Period (t1 to t2):** `{train_data_raw.index.min().strftime('%Y-%m-%d')}` to `{train_data_raw.index.max().strftime('%Y-%m-%d')}`")

    # Determine the actual prediction date (t2+1, skipping weekends/holidays)
    prediction_date_t2_plus_1 = t2_end_date + timedelta(days=1)
    while prediction_date_t2_plus_1.weekday() >= 5: # Skip Saturday (5) and Sunday (6)
        prediction_date_t2_plus_1 += timedelta(days=1)
    
    # Ensure prediction_date_t2_plus_1 is within the downloaded data for actual comparison
    initial_prediction_date_check = prediction_date_t2_plus_1
    while prediction_date_t2_plus_1.strftime('%Y-%m-%d') not in full_data_filtered.index.strftime('%Y-%m-%d').tolist() and prediction_date_t2_plus_1 <= full_data_filtered.index.max().date():
        prediction_date_t2_plus_1 += timedelta(days=1)
        if prediction_date_t2_plus_1.weekday() >= 5: # Skip weekends again
             prediction_date_t2_plus_1 += timedelta(days=1)
    
    if prediction_date_t2_plus_1 > full_data_filtered.index.max().date():
        st.error(f"Cannot find a valid trading day for prediction after {t2_end_date} within the downloaded data. Please extend your training end date or reduce its recency.")
        st.stop()

    st.markdown(f"**Prediction Target Date (t2+1):** `{prediction_date_t2_plus_1.strftime('%Y-%m-%d')}`")

    # Get the actual close price for t2+1
    actual_close_t2_plus_1 = np.nan
    actual_row_t2_plus_1 = full_data_filtered[full_data_filtered.index.date == prediction_date_t2_plus_1]
    if not actual_row_t2_plus_1.empty:
        actual_close_t2_plus_1 = actual_row_t2_plus_1['Close'].iloc[0]
    else:
        st.warning(f"Actual close price for {prediction_date_t2_plus_1.strftime('%Y-%m-%d')} is not available in the downloaded data. This date might be in the future, a holiday, or data is missing for this specific date in Yahoo Finance.")

    st.info("Creating features and processing training data...")
    # Create features only for the training data (t1 to t2)
    train_features_df = create_features(train_data_raw.reset_index().copy(), selected_indicator_params)
    train_features_df.dropna(inplace=True)

    if len(train_features_df) < min_data_required: # Use the now globally defined min_data_required
        st.error(f"Not enough valid data after feature creation for the selected indicators. Needed at least {min_data_required} rows for training, but only got {len(train_features_df)}. Please adjust training dates or the selected technical indicators.")
        st.stop()
    
    st.markdown(f"Training data (with features) available: {len(train_features_df)} rows from {train_features_df['Date'].min().strftime('%Y-%m-%d')} to {train_features_df['Date'].max().strftime('%Y-%m-%d')}.")

    prediction_results = []
    plot_data_points = []

    plot_data_points.append({'Date': pd.to_datetime(prediction_date_t2_plus_1), 'Price': actual_close_t2_plus_1, 'Type': 'Actual Close'})

    for model_choice in selected_models:
        with st.spinner(f"Training {model_choice} model and predicting for {prediction_date_t2_plus_1}..."):
            trained_models_for_close, _ = train_models_pipeline(
                train_features_df.copy(),
                model_choice,
                perform_tuning_backtest, # Use the backtest-specific tuning flag
                dummy_log,
                selected_indicator_params
            )

            close_model_info = trained_models_for_close.get('Close')
            if not close_model_info or not close_model_info.get('model'):
                st.warning(f"Model '{model_choice}' for Close price could not be trained. Skipping prediction.")
                prediction_results.append({
                    'Model': model_choice,
                    'Predicted Close (t2+1)': np.nan,
                    'Predicted Lower Bound': np.nan,
                    'Predicted Upper Bound': np.nan,
                    'Actual Close (t2+1)': actual_close_t2_plus_1,
                    'Absolute Error': np.nan,
                    'Percentage Error': np.nan,
                    'Directional Accuracy': np.nan
                })
                continue

            input_for_prediction = train_features_df.tail(1).copy()

            single_prediction_dict = generate_predictions_pipeline(
                input_for_prediction,
                {'Close': close_model_info},
                dummy_log,
                confidence_level_pct_backtest if enable_confidence_interval_backtest else None # Use backtest-specific CI
            )

            # Retrieve predicted values and bounds. Use .get() and default to np.nan for safety.
            predicted_close = np.nan
            predicted_lower = np.nan
            predicted_upper = np.nan

            if 'Close' in single_prediction_dict and not single_prediction_dict['Close'].empty:
                predicted_close = single_prediction_dict['Close']['Predicted Close'].iloc[0]
                # Check for column existence before accessing, important if CI is disabled
                if 'Predicted Close Lower' in single_prediction_dict['Close'].columns:
                    predicted_lower = single_prediction_dict['Close']['Predicted Close Lower'].iloc[0]
                if 'Predicted Close Upper' in single_prediction_dict['Close'].columns:
                    predicted_upper = single_prediction_dict['Close']['Predicted Close Upper'].iloc[0]

            abs_error = np.nan
            pct_error = np.nan
            directional_accuracy = np.nan

            # Calculate metrics ONLY if both predicted_close AND actual_close_t2_plus_1 are valid numbers
            if pd.notna(predicted_close) and pd.notna(actual_close_t2_plus_1):
                abs_error = abs(predicted_close - actual_close_t2_plus_1)
                pct_error = (abs_error / actual_close_t2_plus_1) * 100 if actual_close_t2_plus_1 != 0 else np.nan
                
                close_at_t2 = train_data_raw['Close'].iloc[-1]
                
                # Ensure close_at_t2 is also not NaN for directional accuracy
                if pd.notna(close_at_t2):
                    true_direction = np.sign(actual_close_t2_plus_1 - close_at_t2)
                    predicted_direction = np.sign(predicted_close - close_at_t2)

                    if pd.notna(true_direction) and pd.notna(predicted_direction):
                        directional_accuracy = 1.0 if true_direction == predicted_direction else 0.0
                    else:
                        directional_accuracy = np.nan
                else: # close_at_t2 is NaN, cannot determine direction
                    directional_accuracy = np.nan
            else: # If predicted_close or actual_close_t2_plus_1 is NaN, set errors to NaN
                abs_error = np.nan
                pct_error = np.nan
                directional_accuracy = np.nan

            prediction_results.append({
                'Model': model_choice,
                'Predicted Close (t2+1)': predicted_close,
                'Predicted Lower Bound': predicted_lower,
                'Predicted Upper Bound': predicted_upper,
                'Actual Close (t2+1)': actual_close_t2_plus_1,
                'Absolute Error': abs_error,
                'Percentage Error': pct_error,
                'Directional Accuracy': directional_accuracy
            })

            plot_data_points.append({'Date': pd.to_datetime(prediction_date_t2_plus_1), 'Price': predicted_close, 'Type': f'Predicted {model_choice}'})
            if enable_confidence_interval_backtest and pd.notna(predicted_lower) and pd.notna(predicted_upper): # Use backtest-specific CI
                 plot_data_points.append({'Date': pd.to_datetime(prediction_date_t2_plus_1), 'Price': predicted_lower, 'Type': f'Lower Bound {model_choice}'})
                 plot_data_points.append({'Date': pd.to_datetime(prediction_date_t2_plus_1), 'Price': predicted_upper, 'Type': f'Upper Bound {model_choice}'})

    
    if prediction_results:
        st.subheader("Prediction Results Table")
        results_df = pd.DataFrame(prediction_results)
        
        format_dict = {
            "Predicted Close (t2+1)": "{:,.2f}",
            "Predicted Lower Bound": "{:,.2f}",
            "Predicted Upper Bound": "{:,.2f}",
            "Actual Close (t2+1)": "{:,.2f}",
            "Absolute Error": "{:,.2f}",
            "Percentage Error": "{:,.2f}%",
            "Directional Accuracy": "{:.2%}"
        }
        # Apply formatting. NaN values will automatically appear as empty or "None" with this format, which is desired.
        st.dataframe(results_df.style.format(format_dict), hide_index=True)

        st.subheader("Prediction Visualization")
        plot_hist_df = train_data_raw.reset_index().copy()
        plot_hist_df = plot_hist_df[['Date', 'Close']].rename(columns={'Close': 'Price'})
        plot_hist_df['Type'] = 'Historical Close'

        combined_plot_df = pd.concat([plot_hist_df, pd.DataFrame(plot_data_points)], ignore_index=True)
        combined_plot_df.sort_values('Date', inplace=True)

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=combined_plot_df[combined_plot_df['Type'] == 'Historical Close']['Date'],
                                 y=combined_plot_df[combined_plot_df['Type'] == 'Historical Close']['Price'],
                                 mode='lines', name='Historical Close', line=dict(color='blue', width=2)))

        if pd.notna(actual_close_t2_plus_1):
            fig.add_trace(go.Scatter(x=[pd.to_datetime(prediction_date_t2_plus_1)], y=[actual_close_t2_plus_1],
                                     mode='markers', marker=dict(size=10, color='green', symbol='star-diamond'),
                                     name=f'Actual Close ({prediction_date_t2_plus_1.strftime("%Y-%m-%d")})'))
        
        colors = px.colors.qualitative.Plotly

        for i, model_choice in enumerate(selected_models):
            pred_point = next((item['Price'] for item in plot_data_points if item['Type'] == f'Predicted {model_choice}'), np.nan)
            if pd.notna(pred_point):
                fig.add_trace(go.Scatter(x=[pd.to_datetime(prediction_date_t2_plus_1)], y=[pred_point],
                                         mode='markers', marker=dict(size=10, color=colors[i % len(colors)], symbol='circle-open'),
                                         name=f'Predicted {model_choice}'))
                
                if enable_confidence_interval_backtest: # Use backtest-specific CI
                    lower_bound = next((item['Price'] for item in plot_data_points if item['Type'] == f'Lower Bound {model_choice}'), np.nan)
                    upper_bound = next((item['Price'] for item in plot_data_points if item['Type'] == f'Upper Bound {model_choice}'), np.nan)
                    
                    if pd.notna(lower_bound) and pd.notna(upper_bound):
                        fig.add_trace(go.Scatter(
                            x=[pd.to_datetime(prediction_date_t2_plus_1), pd.to_datetime(prediction_date_t2_plus_1)],
                            y=[lower_bound, upper_bound],
                            mode='lines',
                            line=dict(color=colors[i % len(colors)], width=1, dash='dot'),
                            name=f'{model_choice} CI',
                            showlegend=False
                        ))

        fig.update_layout(
            title=f'{ticker} Single-Point Backtesting: Historical Close & Forecast for {prediction_date_t2_plus_1.strftime("%Y-%m-%d")}',
            xaxis_title='Date',
            yaxis_title='Close Price',
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No predictions were generated for the single-point backtest. Please check your selections and data availability.")


# --- Main Content Area - Trading Strategy Simulation ---
st.markdown("---")
st.header("📈 Trading Strategy Simulation")

# Strategy parameters
st.subheader("Strategy Parameters")
signal_threshold_pct = st.slider(
    "Prediction Signal Threshold (%)",
    min_value=0.0,
    max_value=2.0,
    value=0.5,
    step=0.1,
    format="%.1f%%",
    help="If next day's predicted close is X% above current close, generate BUY. If X% below, generate SELL. Otherwise, HOLD."
)

initial_capital = st.number_input("Initial Capital for Simulation ($)", min_value=100.0, value=10000.0, step=100.0, format="%.2f") # Removed '$' from format

if st.button("Run Trading Strategy Simulation", type="primary"):
    if not ticker or not selected_models:
        st.warning("Please enter a stock ticker and select at least one model to run the strategy simulation.")
        st.stop()

    st.divider()

    with st.spinner(f"Downloading data for {ticker} for strategy simulation..."):
        # We need data from (s1 - max_indicator_window) to s2
        max_indicator_window_strategy = min_data_required # Use the globally defined min_data_required

        # Ensure enough data to calculate all indicators and for the strategy period
        data_fetch_start_strategy = s1_start_date - timedelta(days=max_indicator_window_strategy * 2) # Give ample buffer
        num_years_for_strategy = max(1, int(np.ceil((s2_end_date + timedelta(days=10) - data_fetch_start_strategy).days / 365.25)))
        
        full_strategy_data = download_data(ticker, period=f"{num_years_for_strategy}y")
        full_strategy_data_filtered = full_strategy_data[(full_strategy_data['Date'] >= pd.to_datetime(data_fetch_start_strategy)) & (full_strategy_data['Date'] <= pd.to_datetime(s2_end_date))].copy()
        full_strategy_data_filtered.set_index('Date', inplace=True)

    if full_strategy_data_filtered.empty:
        st.error(f"Could not download sufficient data for strategy simulation for {ticker} in the range {data_fetch_start_strategy} to {s2_end_date}. Please check the ticker or dates.")
        st.stop()

    st.markdown(f"**Strategy Simulation Period (s1 to s2):** `{s1_start_date.strftime('%Y-%m-%d')}` to `{s2_end_date.strftime('%Y-%m-%d')}`")
    st.markdown(f"Data fetched for strategy: {full_strategy_data_filtered.index.min().strftime('%Y-%m-%d')} to {full_strategy_data_filtered.index.max().strftime('%Y-%m-%d')}")

    strategy_performance_results = []
    trade_log = []
    
    # Counters for skipped days
    skipped_insufficient_data = 0
    skipped_model_training_failed = 0
    skipped_no_prediction = 0

    # Use the first selected model for strategy simulation for simplicity
    if not selected_models:
        st.warning("Please select at least one model to run the strategy simulation.")
        st.stop()
        
    main_strategy_model = selected_models[0]
    st.info(f"Simulating strategy using model: **{main_strategy_model}**")

    # Prepare historical data for strategy simulation, including a buffer for indicators
    strategy_hist_data = full_strategy_data_filtered[full_strategy_data_filtered.index.date <= s2_end_date].copy()
    
    if strategy_hist_data.empty:
        st.error(f"No historical data available for strategy simulation within the defined period.")
        st.stop()

    # Create features for the entire historical data needed for strategy
    with st.spinner("Creating features for strategy data..."):
        strategy_features_df = create_features(strategy_hist_data.reset_index().copy(), selected_indicator_params)
        strategy_features_df.dropna(inplace=True) # Drop NaNs introduced by indicators
        strategy_features_df.set_index('Date', inplace=True) # Re-set Date as index

    if strategy_features_df.empty:
        st.error("Not enough data after feature engineering for strategy simulation. Try a longer historical period or fewer indicators.")
        st.stop()
    
    # Filter strategy_features_df to the actual s1 to s2 range for simulation
    # Ensure this range uses trading days only
    actual_strategy_period_df = strategy_features_df[(strategy_features_df.index.date >= s1_start_date) & (strategy_features_df.index.date <= s2_end_date)].copy()
    
    if actual_strategy_period_df.empty:
        st.error(f"No trading days found in the strategy simulation period {s1_start_date} to {s2_end_date} after feature engineering. Adjust dates or select fewer indicators.")
        st.stop()

    st.markdown(f"Effective strategy simulation data (with features) from: `{actual_strategy_period_df.index.min().strftime('%Y-%m-%d')}` to `{actual_strategy_period_df.index.max().strftime('%Y-%m-%d')}` ({len(actual_strategy_period_df)} trading days)")


    # Initialize strategy variables
    portfolio_value = initial_capital
    shares_held = 0
    
    # Create a placeholder for the daily log inside the loop for live updates
    daily_log_placeholder = st.empty()
    daily_log_messages = []

    # Simulate day by day
    # Iterate from the first valid day in the strategy period up to the second-to-last day (to predict for the next day)
    # The 'index' here refers to the actual date in the DataFrame index
    for i in range(len(actual_strategy_period_df)):
        current_day_data = actual_strategy_period_df.iloc[i]
        current_date = current_day_data.name.date() # Get the date from the index
        current_close = current_day_data['Close']

        # Ensure we have enough data BEFORE the current day to train the model
        # The training data for day `D` should be all data up to `D-1`.
        # This requires `strategy_features_df` to contain data before `s1_start_date`
        # We need to explicitly define the training slice for each prediction.
        
        # Training data for predicting (current_date + 1)
        # Use data from the beginning of `strategy_features_df` up to `current_date`
        # This ensures no future data leakage for the specific prediction.
        train_data_for_prediction = strategy_features_df[strategy_features_df.index.date <= current_date].copy()
        
        daily_message = f"**{current_date.strftime('%Y-%m-%d')}** (Close: {current_close:,.2f}): "

        # Check if there is sufficient data for training *for this specific day*
        if len(train_data_for_prediction) < min_data_required:
            daily_message += f"Insufficient data for training ({len(train_data_for_prediction)} rows, need {min_data_required}). Action: INSUFFICIENT_DATA."
            current_portfolio_value_plot = portfolio_value + (shares_held * current_close)
            strategy_performance_results.append({
                'Date': current_date,
                'Current Close': current_close,
                f'Predicted Next Close ({main_strategy_model})': np.nan,
                'Portfolio Value': current_portfolio_value_plot,
                'Shares Held': shares_held,
                'Action': "INSUFFICIENT_DATA"
            })
            skipped_insufficient_data += 1
            daily_log_messages.append(daily_message)
            daily_log_placeholder.markdown("\n".join(daily_log_messages[-5:])) # Show last 5 messages
            continue # Skip to next day if not enough data for training

        # Train model for this specific daily prediction
        trained_models_daily, _ = train_models_pipeline(
            train_data_for_prediction.copy(), # Use all available data up to current_date for training
            main_strategy_model,
            perform_tuning_backtest, # Use backtest-specific tuning
            dummy_log,
            selected_indicator_params
        )
        
        model_info_daily = trained_models_daily.get('Close')

        if not model_info_daily or not model_info_daily.get('model'):
            daily_message += f"Model training failed for {main_strategy_model}. Action: MODEL_TRAINING_FAILED."
            current_portfolio_value_plot = portfolio_value + (shares_held * current_close)
            strategy_performance_results.append({
                'Date': current_date,
                'Current Close': current_close,
                f'Predicted Next Close ({main_strategy_model})': np.nan,
                'Portfolio Value': current_portfolio_value_plot,
                'Shares Held': shares_held,
                'Action': "MODEL_TRAINING_FAILED"
            })
            skipped_model_training_failed += 1
            daily_log_messages.append(daily_message)
            daily_log_placeholder.markdown("\n".join(daily_log_messages[-5:])) # Show last 5 messages
            continue

        # Prepare input for prediction (the last row of the *training* data)
        input_for_daily_prediction = train_data_for_prediction.tail(1).copy()
        
        daily_prediction_dict = generate_predictions_pipeline(
            input_for_daily_prediction,
            {'Close': model_info_daily},
            dummy_log,
            None # No confidence intervals needed for strategy simulation
        )

        predicted_next_close = daily_prediction_dict['Close']['Predicted Close'].iloc[0] if 'Close' in daily_prediction_dict and not daily_prediction_dict['Close'].empty else np.nan

        if pd.notna(predicted_next_close):
            # Trading logic
            # Buy signal if predicted close is significantly higher than current close
            buy_signal_price = current_close * (1 + signal_threshold_pct / 100)
            # Sell signal if predicted close is significantly lower than current close
            sell_signal_price = current_close * (1 - signal_threshold_pct / 100)

            action = "HOLD"
            trade_price = np.nan
            shares_traded = 0

            # Find the actual next day's data from full_strategy_data_filtered to execute the trade
            next_day_date = current_date + timedelta(days=1)
            # Find the actual next trading day that exists in our filtered data
            next_trading_day_data = full_strategy_data_filtered[full_strategy_data_filtered.index.date == next_day_date]
            
            # If the directly next day is not a trading day, find the next available one within s2_end_date
            # Limit the look-ahead to avoid infinite loops if data completely missing
            look_ahead_days = 0
            while next_trading_day_data.empty and next_day_date <= s2_end_date + timedelta(days=5) and look_ahead_days < 7: # Check max 7 days ahead (e.g., across a long weekend)
                next_day_date += timedelta(days=1)
                next_trading_day_data = full_strategy_data_filtered[full_strategy_data_filtered.index.date == next_day_date]
                look_ahead_days += 1


            if not next_trading_day_data.empty: # Only attempt trade if next trading day data is found
                # Trade at the opening price of the next trading day
                trade_execution_price = next_trading_day_data['Open'].iloc[0]

                if predicted_next_close > buy_signal_price and shares_held == 0:
                    # Buy
                    shares_traded = int(portfolio_value / trade_execution_price)
                    if shares_traded > 0:
                        portfolio_value -= shares_traded * trade_execution_price
                        shares_held += shares_traded
                        action = "BUY"
                        trade_price = trade_execution_price
                        trade_log.append({'Date': next_day_date, 'Action': action, 'Shares': shares_traded, 'Price': trade_price, 'Portfolio Value': portfolio_value, 'Shares Held': shares_held})
                        daily_message += f"Predicted: {predicted_next_close:,.2f}. BUY signal! Shares: {shares_traded}, New Value: {portfolio_value + (shares_held * current_close):,.2f}"
                    else:
                        daily_message += f"Predicted: {predicted_next_close:,.2f}. BUY signal but insufficient capital ({portfolio_value:,.2f}) for trade at {trade_execution_price:,.2f}. Action: HOLD."
                elif predicted_next_close < sell_signal_price and shares_held > 0:
                    # Sell
                    portfolio_value += shares_held * trade_execution_price
                    action = "SELL"
                    trade_price = trade_execution_price
                    shares_traded = shares_held
                    shares_held = 0 # All shares sold
                    trade_log.append({'Date': next_day_date, 'Action': action, 'Shares': shares_traded, 'Price': trade_price, 'Portfolio Value': portfolio_value, 'Shares Held': shares_held})
                    daily_message += f"Predicted: {predicted_next_close:,.2f}. SELL signal! Shares: {shares_traded}, New Value: {portfolio_value + (shares_held * current_close):,.2f}"
                else:
                    daily_message += f"Predicted: {predicted_next_close:,.2f}. Within threshold. Action: HOLD. Current Value: {portfolio_value + (shares_held * current_close):,.2f}"
                
                # If it was a hold or no trade was possible due to shares_traded == 0 for BUY, the portfolio value is updated
                # below the if/elif
            else:
                daily_message += f"No next trading day data found for {next_day_date.strftime('%Y-%m-%d')} to execute trade. Action: DATA_GAP_HOLD."
            
            # Always update portfolio value to reflect current holdings value for plotting
            current_portfolio_value_plot = portfolio_value + (shares_held * current_close)

            strategy_performance_results.append({
                'Date': current_date,
                'Current Close': current_close,
                f'Predicted Next Close ({main_strategy_model})': predicted_next_close,
                'Portfolio Value': current_portfolio_value_plot,
                'Shares Held': shares_held,
                'Action': action
            })
        else: # If predicted_next_close is NaN
            daily_message += f"Prediction for next close is NaN. Action: NO_PREDICTION_GENERATED."
            current_portfolio_value_plot = portfolio_value + (shares_held * current_close)
            strategy_performance_results.append({
                'Date': current_date,
                'Current Close': current_close,
                f'Predicted Next Close ({main_strategy_model})': np.nan,
                'Portfolio Value': current_portfolio_value_plot,
                'Shares Held': shares_held,
                'Action': "NO_PREDICTION_GENERATED"
            })
            skipped_no_prediction += 1
        
        daily_log_messages.append(daily_message)
        daily_log_placeholder.markdown("\n".join(daily_log_messages[-5:])) # Show last 5 messages

    if strategy_performance_results:
        strategy_df = pd.DataFrame(strategy_performance_results)
        strategy_df['Date'] = pd.to_datetime(strategy_df['Date'])
        strategy_df.set_index('Date', inplace=True)

        st.subheader("Strategy Performance Over Time")
        fig_strat = go.Figure()
        fig_strat.add_trace(go.Scatter(x=strategy_df.index, y=strategy_df['Current Close'], mode='lines', name='Actual Close Price', line=dict(color='blue')))
        fig_strat.add_trace(go.Scatter(x=strategy_df.index, y=strategy_df['Portfolio Value'], mode='lines', name='Portfolio Value', line=dict(color='orange')))
        
        # Mark Buy/Sell points
        buy_points = strategy_df[strategy_df['Action'] == 'BUY']
        sell_points = strategy_df[strategy_df['Action'] == 'SELL']
        
        if not buy_points.empty:
            fig_strat.add_trace(go.Scatter(x=buy_points.index, y=buy_points['Current Close'], mode='markers',
                                           marker=dict(symbol='triangle-up', size=10, color='green'),
                                           name='BUY Signal'))
        if not sell_points.empty:
            fig_strat.add_trace(go.Scatter(x=sell_points.index, y=sell_points['Current Close'], mode='markers',
                                           marker=dict(symbol='triangle-down', size=10, color='red'),
                                           name='SELL Signal'))

        fig_strat.update_layout(
            title=f'{ticker} Trading Strategy Simulation ({main_strategy_model})',
            xaxis_title='Date',
            yaxis_title='Value ($)',
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig_strat, use_container_width=True)

        st.subheader("Final Strategy Summary")
        # Final liquidation of any remaining shares at the last available close price in the simulation
        last_close_for_liquidation = strategy_df['Current Close'].iloc[-1] if not strategy_df.empty else 0
        final_portfolio_value = portfolio_value + (shares_held * last_close_for_liquidation)
        total_profit = final_portfolio_value - initial_capital
        percentage_gain = (total_profit / initial_capital) * 100 if initial_capital != 0 else 0

        summary_data = {
            "Metric": [
                "Initial Capital",
                "Final Portfolio Value (Liquidated)",
                "Total Profit/Loss",
                "Percentage Gain/Loss",
                "Total Trades Executed",
                "Days Skipped (Insufficient Data)",
                "Days Skipped (Model Training Failed)",
                "Days Skipped (No Valid Prediction)"
            ],
            "Value": [
                f"${initial_capital:,.2f}",
                f"${final_portfolio_value:,.2f}",
                f"${total_profit:,.2f}",
                f"{percentage_gain:,.2f}%",
                len(trade_log),
                skipped_insufficient_data,
                skipped_model_training_failed,
                skipped_no_prediction
            ]
        }
        st.table(pd.DataFrame(summary_data).set_index("Metric"))
        
        if trade_log:
            st.subheader("Detailed Trade Log")
            trade_log_df = pd.DataFrame(trade_log)
            st.dataframe(trade_log_df, hide_index=True)
        else:
            st.info("No trades were executed during the simulation period. This can happen if predictions consistently fall within the 'HOLD' threshold, or if there were many skipped days due to data/model issues.")

    else:
        st.info("No strategy simulation results. This can happen if there's insufficient data for training for any day, or if no valid predictions could be generated for the selected strategy period. Please check data availability, selected indicators, and the date range.")

st.markdown("---")
st.caption("Monarch Stock Price Predictor | Disclaimer: For educational and informational purposes only. Not financial advice")

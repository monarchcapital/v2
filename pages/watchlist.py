# pages/Watchlist.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta, date, datetime
import os

# Import functions from utils.py and config.py
import config
from utils import (
    download_data, create_features,
    train_models_pipeline, generate_predictions_pipeline,
    calculate_pivot_points, save_prediction,
    is_morning_star, is_evening_star, get_short_term_trend,
    parse_int_list # Imported parse_int_list from utils
)

st.set_page_config(page_title="Monarch: Stock Price Predictor", layout="wide")

st.header("📋 My Watchlist")
st.markdown("""
Add stocks to your watchlist to get quick insights into their next-day predicted values
for Close, Open, High, Low, Volatility, and calculated Resistance/Support levels.
""")

# --- Watchlist Management (using a simple text file for persistence) ---
WATCHLIST_FILE = "monarch_watchlist.txt"

def load_watchlist():
    """Loads ticker symbols from the watchlist file."""
    if not os.path.exists(WATCHLIST_FILE):
        return []
    with open(WATCHLIST_FILE, "r") as f:
        tickers = [line.strip().upper() for line in f if line.strip()]
    return list(set(tickers)) # Use set to ensure unique tickers

def save_watchlist(tickers):
    """Saves ticker symbols to the watchlist file."""
    with open(WATCHLIST_FILE, "w") as f:
        for ticker in sorted(list(set(tickers))):
            f.write(f"{ticker}\n")

# Initialize watchlist in session state
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = load_watchlist()

st.sidebar.subheader("➕ Manage Watchlist")
new_ticker = st.sidebar.text_input("Add Ticker (e.g., RELIANCE.NS):").upper()
if st.sidebar.button("Add to Watchlist") and new_ticker:
    if new_ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(new_ticker)
        save_watchlist(st.session_state.watchlist)
        st.sidebar.success(f"{new_ticker} added to watchlist!")
    else:
        st.sidebar.info(f"{new_ticker} is already in your watchlist.")

# Allow removal
if st.session_state.watchlist:
    ticker_to_remove = st.sidebar.selectbox("Remove Ticker:", [""] + sorted(st.session_state.watchlist))
    if st.sidebar.button("Remove from Watchlist") and ticker_to_remove:
        st.session_state.watchlist.remove(ticker_to_remove)
        save_watchlist(st.session_state.watchlist)
        st.sidebar.success(f"{ticker_to_remove} removed from watchlist.")
        # Rerun to update the list in the selectbox
        st.rerun()
else:
    st.sidebar.info("Your watchlist is empty.")

# --- Watchlist Model Settings (Now with specific dates) ---
st.sidebar.subheader("⚙️ Watchlist Model Settings")

today = date.today()
default_end_bt = today - timedelta(days=1)
default_start_bt = default_end_bt - timedelta(days=3*365) # Default to 3 years for more data

start_date_watchlist = st.sidebar.date_input(
    "Training Start Date (Watchlist):",
    value=default_start_bt,
    help="Start date for model training data on the watchlist. Set this to match Home page for consistent predictions."
)
end_date_watchlist = st.sidebar.date_input(
    "Training End Date (Watchlist):",
    value=default_end_bt,
    help="End date for model training data on the watchlist. Set this to match Home page for consistent predictions."
)

if start_date_watchlist >= end_date_watchlist:
    st.sidebar.error("Watchlist Training Start Date must be before Watchlist Training End Date.")
    st.stop()

# --- Hyperparameter Tuning Option ---
watchlist_perform_tuning = st.sidebar.checkbox("Perform Hyperparameter Tuning (Watchlist)", value=False, help="May increase processing time for each watchlist item but can improve accuracy.")

# --- Confidence Interval Parameter for Watchlist ---
enable_confidence_interval_watchlist = st.sidebar.checkbox("Enable Prediction Confidence Intervals (Watchlist)", value=False, help="Display prediction intervals based on model residuals for watchlist items.")
confidence_level_pct_watchlist = 90
if enable_confidence_interval_watchlist:
    confidence_level_pct_watchlist = st.sidebar.slider("Confidence Level (%) (Watchlist)", min_value=70, max_value=99, value=90, step=1, help="The confidence level for the prediction interval.")


st.sidebar.markdown("**Technical Indicator Parameters (for Watchlist predictions)**")
st.sidebar.info("These should ideally match settings on the Home page for consistent results.")

# Dictionary to store selected indicator parameters for watchlist
selected_indicator_params_watchlist = {}

# Function to provide error callback for Streamlit display in sidebar
def sidebar_error_callback_watchlist(message):
    st.sidebar.error(message)

# Loop through TECHNICAL_INDICATORS_DEFAULTS to create UI for each
for indicator_name, (default_value, default_enabled) in config.TECHNICAL_INDICATORS_DEFAULTS.items():
    if indicator_name == 'PARSAR_ACCELERATION' or indicator_name == 'PARSAR_MAX_ACCELERATION':
        # These are handled inside the PARSAR_ENABLED block
        continue
    
    # Use a unique key for each checkbox to avoid Streamlit warnings
    checkbox_key = f"enable_{indicator_name.lower()}_wl" # Added _wl for uniqueness
    param_key_prefix = indicator_name.replace('_', ' ')

    if indicator_name.endswith('_ENABLED'): # For indicators like OBV_ENABLED, PARSAR_ENABLED
        enabled = st.sidebar.checkbox(f"Enable {param_key_prefix.replace('_enabled', '')} (WL):", value=default_enabled, key=checkbox_key)
        selected_indicator_params_watchlist[indicator_name] = enabled
        if indicator_name == 'PARSAR_ENABLED' and enabled:
            # Show Parabolic SAR specific parameters only if enabled
            selected_indicator_params_watchlist['PARSAR_ACCELERATION'] = st.sidebar.number_input(f"  Parabolic SAR Acceleration (WL):", min_value=0.01, max_value=0.5, value=config.TECHNICAL_INDICATORS_DEFAULTS['PARSAR_ACCELERATION'][0], step=0.01, key=f"input_parsar_accel_wl")
            selected_indicator_params_watchlist['PARSAR_MAX_ACCELERATION'] = st.sidebar.number_input(f"  Parabolic SAR Max Acceleration (WL):", min_value=0.01, max_value=0.5, value=config.TECHNICAL_INDICATORS_DEFAULTS['PARSAR_MAX_ACCELERATION'][0], step=0.01, key=f"input_parsar_max_accel_wl")

    else: # For indicators with windows/values
        enabled = st.sidebar.checkbox(f"Enable {param_key_prefix} (WL):", value=default_enabled, key=checkbox_key)
        
        if enabled:
            if isinstance(default_value, list):
                # List inputs (Lag, MA, STD windows)
                parsed_list = parse_int_list(
                    st.sidebar.text_input(f"  {param_key_prefix.replace('_', ' ')} (comma-separated days, e.g., {','.join(map(str, default_value))}):", 
                                       value=", ".join(map(str, default_value)), key=f"input_{indicator_name.lower()}_wl"),
                    default_value,
                    sidebar_error_callback_watchlist # Pass the error callback
                )
                selected_indicator_params_watchlist[indicator_name] = parsed_list if parsed_list else None # Store None if empty after parsing
            elif isinstance(default_value, (int, float)):
                # Single value inputs (RSI, MACD, BB, ATR, Stoch, CCI, ROC, ADX, CMF)
                if indicator_name == 'BB_STD_DEV':
                    selected_indicator_params_watchlist[indicator_name] = st.sidebar.number_input(f"  {param_key_prefix.replace('_', ' ')} Multiplier (WL):", min_value=0.1, value=default_value, step=0.1, key=f"input_{indicator_name.lower()}_wl")
                else:
                    selected_indicator_params_watchlist[indicator_name] = st.sidebar.number_input(f"  {param_key_prefix.replace('_', ' ')} (WL):", min_value=1, value=default_value, step=1, key=f"input_{indicator_name.lower()}_wl")
            else:
                selected_indicator_params_watchlist[indicator_name] = None # Should not happen with current config, but for safety
        else:
            selected_indicator_params_watchlist[indicator_name] = None # Indicator is disabled


# Calculate the number of days for display
training_duration_days = (end_date_watchlist - start_date_watchlist).days
st.sidebar.info(f"Models on the watchlist are currently trained using data from **{start_date_watchlist.strftime('%Y-%m-%d')}** to **{end_date_watchlist.strftime('%Y-%m-%d')}** ({training_duration_days} days). Please set these dates to match your 'Training Period' on the Home page for consistent predictions.")


# --- Function to calculate Heikin Ashi candles ---
def calculate_heikin_ashi(df):
    ha_df = pd.DataFrame(index=df.index)
    ha_df['Date'] = df['Date'] # Ensure Date is carried over
    ha_df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4

    # Initialize HA_Open
    ha_df['HA_Open'] = 0.0
    # The first HA_Open is the same as the first regular Open
    ha_df.iloc[0, ha_df.columns.get_loc('HA_Open')] = df['Open'].iloc[0]

    for i in range(1, len(df)):
        # HA_Open for current bar = (Previous HA_Open + Previous HA_Close) / 2
        ha_df.iloc[i, ha_df.columns.get_loc('HA_Open')] = (ha_df['HA_Open'].iloc[i-1] + ha_df['HA_Close'].iloc[i-1]) / 2

    # HA_High = Max of (High, HA_Open, HA_Close) for the current bar
    ha_df['HA_High'] = df['High'].combine(ha_df['HA_Open'], max).combine(ha_df['HA_Close'], max)
    # HA_Low = Min of (Low, HA_Open, HA_Close) for the current bar
    ha_df['HA_Low'] = df['Low'].combine(ha_df['HA_Open'], min).combine(ha_df['HA_Close'], min)

    return ha_df[['Date', 'HA_Open', 'HA_High', 'HA_Low', 'HA_Close']]


# --- Function to generate a small chart ---
def create_small_chart(df, ticker_symbol, chart_type='heikin_ashi'):
    if chart_type == 'heikin_ashi':
        # Calculate Heikin Ashi data
        ha_df = calculate_heikin_ashi(df)

        # Create Heikin Ashi Candlestick chart
        fig = go.Figure(data=[go.Candlestick(x=ha_df['Date'],
                                            open=ha_df['HA_Open'],
                                            high=ha_df['HA_High'],
                                            low=ha_df['HA_Low'],
                                            close=ha_df['HA_Close'],
                                            name='Heikin Ashi Candlestick',
                                            increasing_line_color= 'green',
                                            decreasing_line_color= 'red')])
    else: # Default to standard candlestick if type is not recognized
        fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                                            open=df['Open'],
                                            high=df['High'],
                                            low=df['Low'],
                                            close=df['Close'],
                                            name='Standard Candlestick')])

    fig.update_layout(
        title=f'{ticker_symbol} Price ({chart_type.replace("_", " ").title()})',
        title_x=0.5, # Center title
        xaxis_rangeslider_visible=False,
        xaxis_title="",
        yaxis_title="",
        height=200, # Small height for concise view
        margin=dict(l=20, r=20, t=40, b=20), # Adjust margins
        showlegend=False
    )
    return fig

# --- Watchlist Display and Prediction ---
st.markdown("---")
st.subheader("Next-Day Forecasts for Your Watchlist")

if not st.session_state.watchlist:
    st.info("Your watchlist is empty. Add stocks using the sidebar to see forecasts here.")
else:
    watchlist_results = []

    # Define common model choices for the watchlist (now all models from config)
    models_for_watchlist = config.MODEL_CHOICES # Use all models from config

    # Define a simple logger for watchlist page to avoid cluttering main log
    def watchlist_log(message):
        pass # In this context, we don't need detailed logging for each watchlist item

    progress_text = "Processing watchlist..."
    my_bar = st.progress(0, text=progress_text)

    # Prepare a list to hold data for the main predictions table
    predictions_table_data = []

    for i, ticker_item in enumerate(st.session_state.watchlist):
        my_bar.progress((i + 1) / len(st.session_state.watchlist), text=f"Processing {ticker_item} ({i+1}/{len(st.session_state.watchlist)})...")

        item_data = {'Ticker': ticker_item}

        df_raw = download_data(ticker_item)

        if df_raw.empty:
            item_data['Status'] = f"No data for {ticker_item}"
            # Initialize all model-specific prediction columns to N/A
            for model_name in models_for_watchlist:
                item_data[f'Close ({model_name})'] = np.nan # Store as NaN
                item_data[f'Open ({model_name})'] = np.nan # Store as NaN
                item_data[f'Close Lower ({model_name})'] = np.nan
                item_data[f'Close Upper ({model_name})'] = np.nan
                item_data[f'Open Lower ({model_name})'] = np.nan
                item_data[f'Open Upper ({model_name})'] = np.nan
            item_data['Detected_Pattern'] = 'N/A'
            item_data['Last Actual Close'] = np.nan # Store as NaN
            item_data['R1 (Resistance)'] = np.nan # Store as NaN
            item_data['S1 (Support)'] = np.nan # Store as NaN
            item_data['chart_fig'] = None # Store None for chart
            predictions_table_data.append(item_data.copy()) # Add to table data
            continue

        # Pass selected_indicator_params_watchlist dictionary to create_features
        df_features = create_features(df_raw.copy(), selected_indicator_params_watchlist)

        # Ensure enough data for feature calculation (based on max window from ALL configured indicators)
        min_data_required_wl = 0
        for param_name, (default_value, default_enabled) in config.TECHNICAL_INDICATORS_DEFAULTS.items():
            if param_name in selected_indicator_params_watchlist and selected_indicator_params_watchlist[param_name] is not None:
                param_value = selected_indicator_params_watchlist[param_name]
                if isinstance(param_value, list) and param_value:
                    min_data_required_wl = max(min_data_required_wl, max(param_value))
                elif isinstance(param_value, (int, float)):
                    min_data_required_wl = max(min_data_required_wl, int(param_value))

        if min_data_required_wl < 50: # Fallback to a reasonable minimum
            min_data_required_wl = 50 
        
        if len(df_features) < min_data_required_wl:
            item_data['Status'] = f"Insufficient data for features ({len(df_features)} rows). Need at least {min_data_required_wl} rows for enabled indicators."
            for model_name in models_for_watchlist:
                item_data[f'Close ({model_name})'] = np.nan
                item_data[f'Open ({model_name})'] = np.nan
                item_data[f'Close Lower ({model_name})'] = np.nan
                item_data[f'Close Upper ({model_name})'] = np.nan
                item_data[f'Open Lower ({model_name})'] = np.nan
                item_data[f'Open Upper ({model_name})'] = np.nan
            item_data['Detected_Pattern'] = 'N/A'
            item_data['Last Actual Close'] = np.nan
            item_data['R1 (Resistance)'] = np.nan
            item_data['S1 (Support)'] = np.nan
            item_data['chart_fig'] = None # Store None for chart
            predictions_table_data.append(item_data.copy()) # Add to table data
            continue

        # Filter features data based on the new watchlist training dates
        df_train_period_watchlist = df_features[(df_features['Date'] >= pd.to_datetime(start_date_watchlist)) & (df_features['Date'] <= pd.to_datetime(end_date_watchlist))].copy()

        if df_train_period_watchlist.empty:
            item_data['Status'] = "No data in watchlist training range."
            for model_name in models_for_watchlist:
                item_data[f'Close ({model_name})'] = np.nan
                item_data[f'Open ({model_name})'] = np.nan
                item_data[f'Close Lower ({model_name})'] = np.nan
                item_data[f'Close Upper ({model_name})'] = np.nan
                item_data[f'Open Lower ({model_name})'] = np.nan
                item_data[f'Open Upper ({model_name})'] = np.nan
            item_data['Detected_Pattern'] = 'N/A'
            item_data['Last Actual Close'] = np.nan
            item_data['R1 (Resistance)'] = np.nan
            item_data['S1 (Support)'] = np.nan
            item_data['chart_fig'] = None
            predictions_table_data.append(item_data.copy())
            continue


        # Get last trading day's features to predict the next day
        last_day_features = df_features.tail(1).copy()
        last_known_date_item = df_features['Date'].iloc[-1]
        next_trading_day_item = last_known_date_item + timedelta(days=1)
        while next_trading_day_item.weekday() >= 5: next_trading_day_item += timedelta(days=1)


        # Calculate Pivot Points based on the most recent full day's data
        pivot_points = calculate_pivot_points(df_raw.tail(1)) # Use raw data for PP calculation as it needs true HLC

        # Add Pivot Points to item_data
        item_data['R1 (Resistance)'] = pivot_points['R1'] # Store as number
        item_data['S1 (Support)'] = pivot_points['S1'] # Store as number
        item_data['Status'] = 'OK'

        # Determine next day's close price for general info
        last_actual_close = df_raw['Close'].iloc[-1] if len(df_raw) > 0 else np.nan
        item_data['Last Actual Close'] = last_actual_close # Store as number
        
        # Train and predict with each selected model
        for model_name in models_for_watchlist:
            # Train models for this ticker using the user-defined training period
            if df_train_period_watchlist.empty:
                 item_data[f'Close ({model_name})'] = np.nan
                 item_data[f'Open ({model_name})'] = np.nan
                 item_data[f'Close Lower ({model_name})'] = np.nan
                 item_data[f'Close Upper ({model_name})'] = np.nan
                 item_data[f'Open Lower ({model_name})'] = np.nan
                 item_data[f'Open Upper ({model_name})'] = np.nan
                 item_data['Status'] = "Partial Data"
                 continue

            trained_models_for_item, _ = train_models_pipeline(
                df_train_period_watchlist.copy(), # Use the date-filtered training data
                model_name,
                watchlist_perform_tuning, # Pass the tuning preference
                watchlist_log,
                selected_indicator_params_watchlist # Pass the indicator parameters
            )

            # Define the targets to predict for watchlist (Close and Open)
            targets_to_predict = {}
            if 'Close' in trained_models_for_item and trained_models_for_item['Close'].get('model'):
                targets_to_predict['Close'] = trained_models_for_item['Close']
            if 'Open' in trained_models_for_item and trained_models_for_item['Open'].get('model'):
                targets_to_predict['Open'] = trained_models_for_item['Open']


            if targets_to_predict: # If any target model was successfully trained
                next_day_preds_dict = generate_predictions_pipeline(last_day_features.copy(), targets_to_predict, watchlist_log, confidence_level_pct_watchlist if enable_confidence_interval_watchlist else None)

                # Get Close predictions
                pred_close = np.nan
                pred_close_lower = np.nan
                pred_close_upper = np.nan
                if 'Close' in next_day_preds_dict and not next_day_preds_dict['Close'].empty:
                    pred_close = next_day_preds_dict['Close']['Predicted Close'].iloc[-1]
                    # Access bounds only if they exist in the DataFrame
                    if f'Predicted Close Lower' in next_day_preds_dict['Close'].columns:
                        pred_close_lower = next_day_preds_dict['Close']['Predicted Close Lower'].iloc[-1]
                    if f'Predicted Close Upper' in next_day_preds_dict['Close'].columns:
                        pred_close_upper = next_day_preds_dict['Close']['Predicted Close Upper'].iloc[-1]


                # Get Open predictions
                pred_open = np.nan
                pred_open_lower = np.nan
                pred_open_upper = np.nan
                if 'Open' in next_day_preds_dict and not next_day_preds_dict['Open'].empty:
                    pred_open = next_day_preds_dict['Open']['Predicted Open'].iloc[-1]
                    # Access bounds only if they exist in the DataFrame
                    if f'Predicted Open Lower' in next_day_preds_dict['Open'].columns:
                        pred_open_lower = next_day_preds_dict['Open']['Predicted Open Lower'].iloc[-1]
                    if f'Predicted Open Upper' in next_day_preds_dict['Open'].columns:
                        pred_open_upper = next_day_preds_dict['Open']['Predicted Open Upper'].iloc[-1]
                
                item_data[f'Close ({model_name})'] = pred_close
                item_data[f'Close Lower ({model_name})'] = pred_close_lower
                item_data[f'Close Upper ({model_name})'] = pred_close_upper
                item_data[f'Open ({model_name})'] = pred_open
                item_data[f'Open Lower ({model_name})'] = pred_open_lower
                item_data[f'Open Upper ({model_name})'] = pred_open_upper

                # Save predictions
                for target_type_key, pred_val, actual_val, lower_b, upper_b in [
                    ('Close', pred_close, last_actual_close, pred_close_lower, pred_close_upper),
                    ('Open', pred_open, df_raw['Open'].iloc[-1] if len(df_raw)>0 else np.nan, pred_open_lower, pred_open_upper) # Use actual open for last day
                ]:
                    if pd.notna(pred_val):
                        save_prediction(
                            ticker_item, 
                            next_trading_day_item, 
                            pred_val, 
                            actual_val, 
                            model_name, 
                            datetime.now(), 
                            end_date_watchlist, 
                            predicted_type=target_type_key,
                            predicted_lower_bound=lower_b,
                            predicted_upper_bound=upper_b
                        )
            else: # If no target model was successfully trained for this item/model
                item_data[f'Close ({model_name})'] = np.nan
                item_data[f'Open ({model_name})'] = np.nan
                item_data[f'Close Lower ({model_name})'] = np.nan
                item_data[f'Close Upper ({model_name})'] = np.nan
                item_data[f'Open Lower ({model_name})'] = np.nan
                item_data[f'Open Upper ({model_name})'] = np.nan
                item_data['Status'] = "Partial Data"


        # Detect Patterns
        pattern_found = []
        # Use a longer period for pattern detection if available, e.g., last 30 days of raw data
        df_for_patterns = df_raw.tail(30).copy()
        if len(df_for_patterns) >= 3:
            if is_morning_star(df_for_patterns):
                pattern_found.append("Morning Star")
            elif is_evening_star(df_for_patterns):
                pattern_found.append("Evening Star")

        # Add short-term trend (using first two MA windows from configured watchlist MAs)
        # Pass the full indicator_params dictionary to get_short_term_trend
        pattern_found.append(get_short_term_trend(df_features, selected_indicator_params_watchlist))

        item_data['Detected_Pattern'] = ", ".join(pattern_found) if pattern_found else "No specific pattern"

        # Generate chart for the ticker (e.g., last 60 days)
        chart_df = df_raw.tail(60).copy() # Use raw data for chart
        if not chart_df.empty:
            item_data['chart_fig'] = create_small_chart(chart_df, ticker_item, chart_type='heikin_ashi') # Use Heikin Ashi
        else:
            item_data['chart_fig'] = None

        predictions_table_data.append(item_data.copy()) # Add to table data

    my_bar.empty() # Clear the progress bar

    if predictions_table_data:
        df_predictions_table = pd.DataFrame(predictions_table_data)

        # Dynamically create display columns for the main predictions table
        display_cols_predictions_table = ['Ticker', 'Status', 'Last Actual Close', 'Detected_Pattern'] 
        for model_name in models_for_watchlist: # Loop through ALL models
            display_cols_predictions_table.extend([
                f'Close ({model_name})',
            ])
            if enable_confidence_interval_watchlist:
                 display_cols_predictions_table.extend([
                    f'Close Lower ({model_name})',
                    f'Close Upper ({model_name})'
                 ])
            display_cols_predictions_table.extend([
                f'Open ({model_name})' # Only display Close and Open
            ])
            if enable_confidence_interval_watchlist:
                 display_cols_predictions_table.extend([
                    f'Open Lower ({model_name})',
                    f'Open Upper ({model_name})'
                 ])
        display_cols_predictions_table.extend(['R1 (Resistance)', 'S1 (Support)'])

        # Filter columns to only include those actually generated for the table
        df_predictions_table = df_predictions_table[[col for col in display_cols_predictions_table if col in df_predictions_table.columns]]

        # Define formatting dictionary for numerical columns
        format_dict = {
            "Last Actual Close": "{:.2f}",
            "R1 (Resistance)": "{:.2f}",
            "S1 (Support)": "{:.2f}",
        }
        for model_name in models_for_watchlist:
            format_dict[f"Close ({model_name})"] = "{:.2f}"
            format_dict[f"Open ({model_name})"] = "{:.2f}"
            if enable_confidence_interval_watchlist:
                format_dict[f"Close Lower ({model_name})"] = "{:.2f}"
                format_dict[f"Close Upper ({model_name})"] = "{:.2f}"
                format_dict[f"Open Lower ({model_name})"] = "{:.2f}"
                format_dict[f"Open Upper ({model_name})"] = "{:.2f}"

        st.dataframe(df_predictions_table.style.format(format_dict), hide_index=True)


        st.markdown("---")
        st.subheader("Charts for Watchlist Stocks") # Updated subheader

        # Display charts for each ticker (news display loop removed)
        for item in predictions_table_data: # Use predictions_table_data which includes chart_fig
            st.markdown(f"#### {item['Ticker']}")

            # Use st.expander for the chart
            with st.expander(f"View Chart for {item['Ticker']}", expanded=False):
                if item.get('chart_fig'): # Use .get() for safety
                    st.plotly_chart(item['chart_fig'], use_container_width=True)
                else:
                    st.info(f"Chart not available for {item['Ticker']}.")

            st.markdown("---") # Separator between tickers

        st.info("""
        **About Watchlist Predictions & Patterns:**
        * **Last Actual Close:** The actual closing price from the most recent trading day.
        * **Detected Pattern:** Currently identifies "Morning Star" and "Evening Star" candlestick patterns (3-day patterns) and a "Short-term Trend" based on moving average crossovers.
        * **Limitations on Complex Patterns:** Chart patterns like "ascending triangles" or "head and shoulders" require advanced geometric analysis of multiple price swings, trendline validation, and often volume confirmation. Implementing robust and accurate detection for such complex patterns typically requires specialized technical analysis libraries (e.g., `TA-Lib`) or highly sophisticated custom algorithms, which are beyond the scope of this simplified implementation. The current patterns are based on straightforward candlestick definitions and moving average crossovers.
        * Predictions on the watchlist page now use **all available models** (as configured in `config.py`). Hyperparameter tuning can be enabled in the sidebar.
        """)
    else:
        st.info("No forecast data available for your watchlist items.")

st.markdown("---")
st.caption("Monarch Stock Price Predictor | Disclaimer: For educational and informational purposes only. Not financial advice")

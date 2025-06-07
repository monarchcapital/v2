# pages/Watchlist.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta, date, datetime
import os
# Removed yfinance import as it's only for news fetching
# import yfinance as yf # Import yfinance for news fetching

# Import functions from utils.py
from utils import (
    download_data, create_features,
    train_models_pipeline, generate_predictions_pipeline,
    calculate_pivot_points,
    # No direct save/load prediction here, as watchlist just displays next-day predictions
    # but the underlying models still use the save_prediction from utils
)

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

# Calculate the number of days for display
training_duration_days = (end_date_watchlist - start_date_watchlist).days
st.sidebar.info(f"Models on the watchlist are currently trained using data from **{start_date_watchlist.strftime('%Y-%m-%d')}** to **{end_date_watchlist.strftime('%Y-%m-%d')}** ({training_duration_days} days). Please set these dates to match your 'Training Period' on the Home page for consistent predictions.")


# --- Candlestick Pattern Detection Functions (Simplified) ---
def is_morning_star(df_recent):
    # Requires at least 3 candles:
    # 1. Long bearish candle
    # 2. Small-bodied candle (could be bullish or bearish, typically a doji or spinning top)
    # 3. Long bullish candle that closes well into the body of the first candle
    if len(df_recent) < 3:
        return False

    c1, c2, c3 = df_recent.iloc[-3], df_recent.iloc[-2], df_recent.iloc[-1]

    # Conditions for Morning Star (simplified)
    # C1: Long bearish (Close < Open, large body)
    cond1 = c1['Close'] < c1['Open'] and (c1['Open'] - c1['Close']) / c1['Open'] > 0.015

    # C2: Small body (Close close to Open)
    cond2 = abs(c2['Close'] - c2['Open']) / c2['Open'] < 0.005 # Small body

    # C3: Long bullish (Close > Open, large body)
    cond3 = c3['Close'] > c3['Open'] and (c3['Close'] - c3['Open']) / c3['Open'] > 0.015

    # C3 opens above or near C2 close, and closes into C1's body
    cond4 = c3['Open'] > c2['Close'] and c3['Close'] > (c1['Open'] + c1['Close']) / 2

    return cond1 and cond2 and cond3 and cond4

def is_evening_star(df_recent):
    # Requires at least 3 candles:
    # 1. Long bullish candle
    # 2. Small-bodied candle (could be bullish or bearish, typically a doji or spinning top)
    # 3. Long bearish candle that closes well into the body of the first candle
    if len(df_recent) < 3:
        return False

    c1, c2, c3 = df_recent.iloc[-3], df_recent.iloc[-2], df_recent.iloc[-1]

    # Conditions for Evening Star (simplified)
    # C1: Long bullish (Close > Open, large body)
    cond1 = c1['Close'] > c1['Open'] and (c1['Close'] - c1['Open']) / c1['Open'] > 0.015

    # C2: Small body (Close close to Open)
    cond2 = abs(c2['Close'] - c2['Open']) / c2['Open'] < 0.005 # Small body

    # C3: Long bearish (Close < Open, large body)
    cond3 = c3['Close'] < c3['Open'] and (c3['Open'] - c3['Close']) / c3['Open'] > 0.015

    # C3 opens below or near C2 close, and closes into C1's body
    cond4 = c3['Open'] < c2['Close'] and c3['Close'] < (c1['Open'] + c1['Close']) / 2

    return cond1 and cond2 and cond3 and cond4

def get_short_term_trend(df_features, short_ma_window=5, long_ma_window=20):
    if len(df_features) < max(short_ma_window, long_ma_window):
        return "Not Enough Data"

    # Recalculate MAs on the fly if needed for just the latest data point
    current_short_ma = df_features['Close'].iloc[-short_ma_window:].mean()
    current_long_ma = df_features['Close'].iloc[-long_ma_window:].mean()

    if current_short_ma > current_long_ma:
        return "Uptrend (Short-term)"
    elif current_short_ma < current_long_ma:
        return "Downtrend (Short-term)"
    else:
        return "Neutral (Short-term)"

# --- Styling function for DataFrame ---
def highlight_end_color(row):
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

# --- Removed fetch_stock_news function ---
# def fetch_stock_news(ticker_symbol, limit=10):
#     try:
#         ticker_obj = yf.Ticker(ticker_symbol)
#         news = ticker_obj.news

#         if not news: # Check if news list is empty
#             print(f"DEBUG: No news data returned from yfinance for {ticker_symbol}.") # Debug print
#             return []

#         # Sort news by publish time (newest to oldest) and limit
#         # Ensure 'providerPublishTime' exists before using it for sorting
#         news_list = sorted([item for item in news if 'providerPublishTime' in item],
#                            key=lambda x: x['providerPublishTime'], reverse=True)[:limit]

#         if not news_list: # If filtering removed all news (e.g., missing providerPublishTime)
#             print(f"DEBUG: News filtered out due to missing 'providerPublishTime' for {ticker_symbol}.") # Debug print
#             return []

#         formatted_news = []
#         for item in news_list:
#             title = item.get('title', 'No Title')
#             link = item.get('link', '#')
#             # Convert Unix timestamp to datetime, then format
#             publish_time_unix = item.get('providerPublishTime')
#             publish_date = datetime.fromtimestamp(publish_time_unix).strftime('%Y-%m-%d %H:%M') if publish_time_unix else 'N/A'
#             formatted_news.append({
#                 'title': title,
#                 'link': link,
#                 'date': publish_date
#             })
#         return formatted_news
#     except Exception as e:
#         print(f"DEBUG: Error fetching news for {ticker_symbol}: {e}") # Log to console
#         return []

# --- Watchlist Display and Prediction ---
st.markdown("---")
st.subheader("Next-Day Forecasts for Your Watchlist")

if not st.session_state.watchlist:
    st.info("Your watchlist is empty. Add stocks using the sidebar to see forecasts here.")
else:
    watchlist_results = []

    # Define common model choices for the watchlist
    # Tuning is set to False for faster processing on the watchlist page
    models_for_watchlist = ['XGBoost', 'Random Forest', 'Linear Regression']
    watchlist_perform_tuning = False

    # These should be consistent with Home.py for feature generation
    lag_features_list = [1, 2, 3, 5, 10]
    ma_windows_list = [10, 20, 50]
    std_windows_list = [10, 20]
    rsi_window = 14
    macd_short_window = 12
    macd_long_window = 26
    macd_signal_window = 9
    bb_window = 20
    bb_std_dev = 2.0
    atr_window = 14
    stoch_window = 14
    stoch_smooth_window = 3

    # Define a simple logger for watchlist page to avoid cluttering main log
    def watchlist_log(message):
        pass # In this context, we don't need detailed logging for each watchlist item

    progress_text = "Processing watchlist... This may take a moment."
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
                item_data[f'Close ({model_name})'] = 'N/A'
                item_data[f'Open ({model_name})'] = 'N/A'
                item_data[f'High ({model_name})'] = 'N/A'
                item_data[f'Low ({model_name})'] = 'N/A'
                item_data[f'Volatility ({model_name})'] = 'N/A'
            item_data['End_Color'] = 'N/A'
            item_data['Detected_Pattern'] = 'N/A'
            item_data['Last Actual Close'] = 'N/A'
            item_data['R1 (Resistance)'] = 'N/A'
            item_data['S1 (Support)'] = 'N/A'
            item_data['chart_fig'] = None # Store None for chart
            # Removed item_data['news'] = []
            watchlist_results.append(item_data)
            predictions_table_data.append(item_data.copy()) # Add to table data
            continue

        df_features = create_features(df_raw.copy(), lag_features_list, ma_windows_list, std_windows_list, rsi_window, macd_short_window, macd_long_window, macd_signal_window, bb_window, bb_std_dev, atr_window, stoch_window, stoch_smooth_window)

        # Ensure enough data for feature calculation after creation
        min_data_required = max(lag_features_list + ma_windows_list + std_windows_list + [rsi_window, macd_long_window, bb_window, atr_window, stoch_window, stoch_smooth_window]) + 1
        if len(df_features) < min_data_required:
            item_data['Status'] = f"Insufficient data for features ({len(df_features)} rows)"
            for model_name in models_for_watchlist:
                item_data[f'Close ({model_name})'] = 'N/A'
                item_data[f'Open ({model_name})'] = 'N/A'
                item_data[f'High ({model_name})'] = 'N/A'
                item_data[f'Low ({model_name})'] = 'N/A'
                item_data[f'Volatility ({model_name})'] = 'N/A'
            item_data['End_Color'] = 'N/A'
            item_data['Detected_Pattern'] = 'N/A'
            item_data['Last Actual Close'] = 'N/A'
            item_data['R1 (Resistance)'] = 'N/A'
            item_data['S1 (Support)'] = 'N/A'
            item_data['chart_fig'] = None # Store None for chart
            # Removed item_data['news'] = []
            watchlist_results.append(item_data)
            predictions_table_data.append(item_data.copy()) # Add to table data
            continue

        # Filter features data based on the new watchlist training dates
        df_train_period_watchlist = df_features[(df_features['Date'] >= pd.to_datetime(start_date_watchlist)) & (df_features['Date'] <= pd.to_datetime(end_date_watchlist))].copy()

        if df_train_period_watchlist.empty:
            item_data['Status'] = "No data in watchlist training range."
            for model_name in models_for_watchlist:
                item_data[f'Close ({model_name})'] = 'N/A (No Train Data)'
                item_data[f'Open ({model_name})'] = 'N/A (No Train Data)'
                item_data[f'High ({model_name})'] = 'N/A (No Train Data)'
                item_data[f'Low ({model_name})'] = 'N/A (No Train Data)'
                item_data[f'Volatility ({model_name})'] = 'N/A (No Train Data)'
            item_data['End_Color'] = 'N/A'
            item_data['Detected_Pattern'] = 'N/A'
            item_data['Last Actual Close'] = 'N/A'
            item_data['R1 (Resistance)'] = 'N/A'
            item_data['S1 (Support)'] = 'N/A'
            item_data['chart_fig'] = None
            # Removed item_data['news'] = []
            watchlist_results.append(item_data)
            predictions_table_data.append(item_data.copy())
            continue


        # Get last trading day's features to predict the next day
        last_day_features = df_features.tail(1).copy()

        # Calculate Pivot Points based on the most recent full day's data
        pivot_points = calculate_pivot_points(df_raw.tail(1)) # Use raw data for PP calculation as it needs true HLC

        # Add Pivot Points to item_data
        item_data['R1 (Resistance)'] = f"{pivot_points['R1']:.2f}" if pd.notna(pivot_points['R1']) else "N/A"
        item_data['S1 (Support)'] = f"{pivot_points['S1']:.2f}" if pd.notna(pivot_points['S1']) else "N/A"
        item_data['Status'] = 'OK'

        # Determine End_Color for the next day's close price
        last_actual_close = df_raw['Close'].iloc[-1] if len(df_raw) > 0 else np.nan
        item_data['Last Actual Close'] = f"{last_actual_close:.2f}" if pd.notna(last_actual_close) else "N/A"
        predicted_close_main_model = np.nan # Will store main model's close for color comparison

        # Train and predict with each selected model
        for model_name in models_for_watchlist:
            # Train models for this ticker using the user-defined training period
            # Use df_train_period_watchlist instead of tail(watchlist_training_days_param)

            if df_train_period_watchlist.empty: # Double check if df_train_period_watchlist became empty after filtering
                 item_data[f'Close ({model_name})'] = 'N/A (No Train Data)'
                 item_data[f'Open ({model_name})'] = 'N/A (No Train Data)'
                 item_data[f'High ({model_name})'] = 'N/A (No Train Data)'
                 item_data[f'Low ({model_name})'] = 'N/A (No Train Data)'
                 item_data[f'Volatility ({model_name})'] = 'N/A (No Train Data)'
                 item_data['Status'] = "Partial Data"
                 continue


            trained_models_for_item = train_models_pipeline(
                df_train_period_watchlist.copy(), # Use the date-filtered training data
                model_name,
                watchlist_perform_tuning,
                std_windows_list,
                watchlist_log
            )

            if trained_models_for_item and all(mi.get('model') is not None for mi in trained_models_for_item.values() if mi):
                next_day_preds_dict = generate_predictions_pipeline(last_day_features.copy(), trained_models_for_item, watchlist_log)

                pred_close = next_day_preds_dict.get('Close', {}).get('Predicted Close', pd.Series()).iloc[-1] if not next_day_preds_dict.get('Close', {}).empty else np.nan
                pred_open = next_day_preds_dict.get('Open', {}).get('Predicted Open', pd.Series()).iloc[-1] if not next_day_preds_dict.get('Open', {}).empty else np.nan
                pred_high = next_day_preds_dict.get('High', {}).get('Predicted High', pd.Series()).iloc[-1] if not next_day_preds_dict.get('High', {}).empty else np.nan
                pred_low = next_day_preds_dict.get('Low', {}).get('Predicted Low', pd.Series()).iloc[-1] if not next_day_preds_dict.get('Low', {}).empty else np.nan
                pred_vol = next_day_preds_dict.get('Volatility', {}).get('Predicted Volatility', pd.Series()).iloc[-1] if not next_day_preds_dict.get('Volatility', {}).empty else np.nan

                item_data[f'Close ({model_name})'] = f"{pred_close:.2f}" if pd.notna(pred_close) else "N/A"
                item_data[f'Open ({model_name})'] = f"{pred_open:.2f}" if pd.notna(pred_open) else "N/A"
                item_data[f'High ({model_name})'] = f"{pred_high:.2f}" if pd.notna(pred_high) else "N/A"
                item_data[f'Low ({model_name})'] = f"{pred_low:.2f}" if pd.notna(pred_low) else "N/A"
                item_data[f'Volatility ({model_name})'] = f"{pred_vol:.4f}" if pd.notna(pred_vol) else "N/A"

                if model_name == models_for_watchlist[0] and pd.notna(pred_close): # Use the first model's close for color
                    predicted_close_main_model = pred_close
            else:
                item_data[f'Close ({model_name})'] = 'N/A (Model Failed)'
                item_data[f'Open ({model_name})'] = 'N/A (Model Failed)'
                item_data[f'High ({model_name})'] = 'N/A (Model Failed)'
                item_data[f'Low ({model_name})'] = 'N/A (Model Failed)'
                item_data[f'Volatility ({model_name})'] = 'N/A (Model Failed)'
                item_data['Status'] = "Partial Data"

        # Determine End_Color based on the first model's predicted close vs last actual close
        if pd.notna(predicted_close_main_model) and pd.notna(last_actual_close):
            if predicted_close_main_model > last_actual_close:
                item_data['End_Color'] = 'Green (Up)'
            elif predicted_close_main_model < last_actual_close:
                item_data['End_Color'] = 'Red (Down)'
            else:
                item_data['End_Color'] = 'Flat (Neutral)'
        else:
            item_data['End_Color'] = 'N/A'

        # Detect Patterns
        pattern_found = []
        # Use a longer period for pattern detection if available, e.g., last 30 days of raw data
        df_for_patterns = df_raw.tail(30).copy()
        if len(df_for_patterns) >= 3:
            if is_morning_star(df_for_patterns):
                pattern_found.append("Morning Star")
            elif is_evening_star(df_for_patterns):
                pattern_found.append("Evening Star")

        # Add short-term trend
        if len(df_features) >= max(5, 20):
            pattern_found.append(get_short_term_trend(df_features))

        item_data['Detected_Pattern'] = ", ".join(pattern_found) if pattern_found else "No specific pattern"

        # Generate chart for the ticker (e.g., last 60 days)
        chart_df = df_raw.tail(60).copy() # Use raw data for chart
        if not chart_df.empty:
            item_data['chart_fig'] = create_small_chart(chart_df, ticker_item, chart_type='heikin_ashi') # Use Heikin Ashi
        else:
            item_data['chart_fig'] = None

        # Removed news fetching:
        # item_data['news'] = fetch_stock_news(ticker_item)

        watchlist_results.append(item_data)
        predictions_table_data.append(item_data.copy()) # Add to table data

    my_bar.empty() # Clear the progress bar

    if predictions_table_data:
        df_predictions_table = pd.DataFrame(predictions_table_data)

        # Dynamically create display columns for the main predictions table
        display_cols_predictions_table = ['Ticker', 'Status', 'Last Actual Close', 'End_Color', 'Detected_Pattern']
        for model_name in models_for_watchlist:
            display_cols_predictions_table.extend([
                f'Close ({model_name})',
                f'Open ({model_name})',
                f'High ({model_name})',
                f'Low ({model_name})',
                f'Volatility ({model_name})'
            ])
        display_cols_predictions_table.extend(['R1 (Resistance)', 'S1 (Support)'])

        # Filter columns to only include those actually generated for the table
        df_predictions_table = df_predictions_table[[col for col in display_cols_predictions_table if col in df_predictions_table.columns]]

        st.dataframe(df_predictions_table.style.apply(highlight_end_color, axis=1), hide_index=True)

        st.markdown("---")
        st.subheader("Charts for Watchlist Stocks") # Updated subheader

        # Display charts for each ticker (news display loop removed)
        for item in watchlist_results:
            st.markdown(f"#### {item['Ticker']}")

            # Use st.expander for the chart
            with st.expander(f"View Chart for {item['Ticker']}", expanded=False):
                if item['chart_fig']:
                    st.plotly_chart(item['chart_fig'], use_container_width=True)
                else:
                    st.info(f"Chart not available for {item['Ticker']}.")

            # Removed news display
            # st.markdown("##### Latest News:")
            # if item['news']:
            #     for news_item in item['news']:
            #         st.markdown(f"- **{news_item['date']}**: [{news_item['title']}]({news_item['link']})")
            # else:
            #     st.info(f"No recent news found for {item['Ticker']}.")
            st.markdown("---") # Separator between tickers

        st.info("""
        **About Watchlist Predictions & Patterns:**
        * **End Color:** Indicates if the *predicted* next-day close price (from the first model in the list) is higher (Green), lower (Red), or same (Flat) than the *last actual* close price. This cell is now color-coded for quick visual cues.
        * **Last Actual Close:** The actual closing price from the most recent trading day.
        * **Detected Pattern:** Currently identifies "Morning Star" and "Evening Star" candlestick patterns (3-day patterns) and a "Short-term Trend" based on 5-day vs 20-day Moving Averages.
        * **Limitations on Complex Patterns:** Chart patterns like "ascending triangles" or "head and shoulders" require advanced geometric analysis of multiple price swings, trendline validation, and often volume confirmation. Implementing robust and accurate detection for such complex patterns typically requires specialized technical analysis libraries (e.g., `TA-Lib`) or highly sophisticated custom algorithms, which are beyond the scope of this simplified implementation. The current patterns are based on straightforward candlestick definitions and moving average crossovers.
        * Predictions on the watchlist page use a fixed set of common models (XGBoost, Random Forest, Linear Regression) without hyperparameter tuning for faster processing.
        """)
    else:
        st.info("No forecast data available for your watchlist items.")

st.markdown("---")
st.caption("Monarch Stock Price Predictor | Disclaimer: For educational and informational purposes only. Not financial advice")
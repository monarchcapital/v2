# pages/Home.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta, date, datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import functions and global variables from utils.py
from utils import ( 
    download_data, create_features, get_model, _train_single_model,
    train_models_pipeline, generate_predictions_pipeline,
    calculate_pivot_points, save_prediction, load_past_predictions,
    PREDICTION_LOG_COLUMNS, training_messages_log 
)

st.set_page_config(page_title="Monarch: Stock Price Predictor", layout="wide")

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
and forecast its future price movements using advanced machine learning models.
""")

# Sidebar inputs
st.sidebar.header("🛠️ Configuration Panel")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, use .NS for Indian stocks):", value="AAPL").upper()

today = date.today()
default_end_bt = today - timedelta(days=1)
default_start_bt = default_end_bt - timedelta(days=3*365) # Default to 3 years for more data

st.sidebar.subheader("🗓️ Training Period")
start_bt = st.sidebar.date_input("Training Start Date (t1):", value=default_start_bt, help="Start date for model training data.")
end_bt = st.sidebar.date_input("Training End Date (t2):", value=default_end_bt, help="End date for model training data. Predictions will start from t2+1.")

if start_bt >= end_bt:
    st.sidebar.error("Training Start Date (t1) must be before Training End Date (t2)")
    st.stop()

st.sidebar.subheader("🤖 Model Selection")
model_choices = ['Random Forest', 'XGBoost', 'Gradient Boosting', 'Linear Regression', 'SVR', 'KNN', 'Decision Tree']
model_choice = st.sidebar.selectbox("Select Main Model (for Close Price):", model_choices, help="The primary model used for predictions.")
perform_tuning = st.sidebar.checkbox("Perform Hyperparameter Tuning", value=False, help="May significantly increase training time but can improve model accuracy.")
n_future = st.sidebar.slider("Predict Future Days (after t2):", min_value=1, max_value=90, value=15, help="Number of future trading days to forecast.")

st.sidebar.subheader("📊 Model Comparison")
compare_models = st.sidebar.multiselect("Select Models to Compare:", model_choices, default=model_choices[:3], help="Additional models to compare against the main model on recent data.")
train_days_comparison = st.sidebar.slider("Recent Data for Comparison (days):", min_value=30, max_value=1000, value=180, step=10, help="How many recent days of data to use for the model comparison chart.")

st.sidebar.subheader("⚙️ Technical Indicator Settings")
st.sidebar.markdown("---")
st.sidebar.markdown("**Moving Average (MA):** *Common: 10, 20, 50, 200 days.*")
ma_input = st.sidebar.text_input("MA Windows (comma-separated):", value="10,20,50")
try:
    ma_windows_list = [int(x.strip()) for x in ma_input.split(',') if x.strip()]
    if not ma_windows_list: ma_windows_list = [10, 20, 50]
    if any(w <= 0 for w in ma_windows_list):
         st.sidebar.error("MA windows must be positive.")
         st.stop()
except ValueError:
    st.sidebar.error("Invalid MA input.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("**Volatility (Std Dev):** *Common: 10, 20 days.*")
std_input = st.sidebar.text_input("Volatility Windows (comma-separated):", value="10,20")
try:
    std_windows_list = [int(x.strip()) for x in std_input.split(',') if x.strip()]
    if not std_windows_list: std_windows_list = [10, 20]
    if any(w <= 0 for w in std_windows_list):
         st.sidebar.error("Volatility windows must be positive.")
         st.stop()
except ValueError:
    st.sidebar.error("Invalid Volatility input.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("**Relative Strength Index (RSI):** *Common: 14 days.*")
rsi_window = st.sidebar.number_input("RSI Window:", min_value=1, value=14)

st.sidebar.markdown("---")
st.sidebar.markdown("**MACD:** *Common: 12, 26, 9 days.*")
macd_short_window = st.sidebar.number_input("MACD Short Window:", min_value=1, value=12)
macd_long_window = st.sidebar.number_input("MACD Long Window:", min_value=1, value=26)
macd_signal_window = st.sidebar.number_input("MACD Signal Window:", min_value=1, value=9)

st.sidebar.markdown("---")
st.sidebar.markdown("**Bollinger Bands (BB):** *Common: 20 day window, 2.0 std dev.*")
bb_window = st.sidebar.number_input("BB Window:", min_value=1, value=20)
bb_std_dev = st.sidebar.number_input("BB Std Dev Multiplier:", min_value=0.1, value=2.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Average True Range (ATR):** *Common: 14 days.*")
atr_window = st.sidebar.number_input("ATR Window:", min_value=1, value=14)

st.sidebar.markdown("---")
st.sidebar.markdown("**Stochastic Oscillator:** *Common: 14 day %K, 3 day %D.*")
stoch_window = st.sidebar.number_input("Stochastic %K Window:", min_value=1, value=14)
stoch_smooth_window = st.sidebar.number_input("Stochastic %D Window:", min_value=1, value=3)
st.sidebar.markdown("---")

lag_features_list = [1, 2, 3, 5, 10] # Fixed lag features

# --- Training Log Display Area ---
log_expander = st.sidebar.expander("📜 Training Log & Messages", expanded=False)
log_placeholder = log_expander.empty()

# Function to update the log in the sidebar
def update_log(message):
    training_messages_log.append(message)
    log_placeholder.text_area("Log:", "".join(f"{msg}\n" for msg in training_messages_log), height=300, disabled=True)

# --- Main Application Flow ---
data = download_data(ticker)

if not data.empty:
    # Clear previous logs for this run
    training_messages_log.clear()
    update_log(f"Data loaded for {ticker}: {len(data)} rows.")

    df_features_full = create_features(data.copy(), lag_features_list, ma_windows_list, std_windows_list, rsi_window, macd_short_window, macd_long_window, macd_signal_window, bb_window, bb_std_dev, atr_window, stoch_window, stoch_smooth_window)
    update_log(f"Features created. Rows after NaN drop: {len(df_features_full)}")

    # Ensure enough data for feature calculation
    min_data_required = max(lag_features_list + ma_windows_list + std_windows_list + [rsi_window, macd_long_window, bb_window, atr_window, stoch_window, stoch_smooth_window]) + 1
    if len(df_features_full) < min_data_required:
        st.warning(f"Not enough data after feature creation for selected parameters ({len(df_features_full)} rows). Need at least {min_data_required} rows. Adjust dates or parameters.")
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
    trained_models_main = train_models_pipeline(df_train_period.copy(), model_choice, perform_tuning, std_windows_list, update_log)
    
    if not trained_models_main or not all(model_info.get('model') is not None for model_info in trained_models_main.values() if model_info): # Check if models were trained for all requested targets
        st.error("One or more main models failed to train. Check logs in sidebar for details.")
        st.stop()

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
        last_actual_volatility_full_data = df_features_full[trained_models_main.get('Volatility',{}).get('target_col', '')].iloc[-1] if 'Volatility' in trained_models_main and not df_features_full.empty and trained_models_main.get('Volatility',{}).get('target_col', '') in df_features_full.columns else np.nan

        # Main Model Next Day Prediction
        next_day_preds_main_dict = generate_predictions_pipeline(last_day_features_df.copy(), trained_models_main, update_log)
        
        pred_close_main = next_day_preds_main_dict.get('Close', {}).get('Predicted Close', pd.Series()).iloc[-1] if not next_day_preds_main_dict.get('Close', {}).empty else np.nan
        pred_open_main = next_day_preds_main_dict.get('Open', {}).get('Predicted Open', pd.Series()).iloc[-1] if not next_day_preds_main_dict.get('Open', {}).empty else np.nan
        pred_high_main = next_day_preds_main_dict.get('High', {}).get('Predicted High', pd.Series()).iloc[-1] if not next_day_preds_main_dict.get('High', {}).empty else np.nan
        pred_low_main = next_day_preds_main_dict.get('Low', {}).get('Predicted Low', pd.Series()).iloc[-1] if not next_day_preds_main_dict.get('Low', {}).empty else np.nan
        pred_vol_main = next_day_preds_main_dict.get('Volatility', {}).get('Predicted Volatility', pd.Series()).iloc[-1] if not next_day_preds_main_dict.get('Volatility', {}).empty else np.nan

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
            'Model': model_choice,
            'Last Actual Close': f"{last_actual_close_full_data:.2f}" if pd.notna(last_actual_close_full_data) else "N/A", # New
            'End_Color': end_color_main, # New
            'Predicted Close': pred_close_main,
            'Predicted Open': pred_open_main,
            'Predicted High': pred_high_main,
            'Predicted Low': pred_low_main,
            'Predicted Volatility': pred_vol_main
        })
        # Save predictions for all targets
        for target_type, predicted_val, actual_val in [
            ('Close', pred_close_main, last_actual_close_full_data),
            ('Open', pred_open_main, last_actual_open_full_data),
            ('High', pred_high_main, last_actual_high_full_data),
            ('Low', pred_low_main, last_actual_low_full_data),
            ('Volatility', pred_vol_main, last_actual_volatility_full_data)
        ]:
            if pd.notna(predicted_val):
                save_prediction(ticker, next_trading_day, predicted_val, actual_val, model_choice, datetime.now(), end_bt, predicted_type=target_type)

        # Comparison Models Next Day Prediction
        for comp_model_name in compare_models:
            if comp_model_name == model_choice: continue
            
            trained_comp_model_dict = train_models_pipeline(df_train_period.copy(), comp_model_name, perform_tuning, std_windows_list, update_log)
            
            if trained_comp_model_dict and all(model_info.get('model') is not None for model_info in trained_comp_model_dict.values() if model_info):
                next_day_preds_comp_dict = generate_predictions_pipeline(last_day_features_df.copy(), trained_comp_model_dict, update_log)
                pred_close_comp = next_day_preds_comp_dict.get('Close', {}).get('Predicted Close', pd.Series()).iloc[-1] if not next_day_preds_comp_dict.get('Close', {}).empty else np.nan
                pred_open_comp = next_day_preds_comp_dict.get('Open', {}).get('Predicted Open', pd.Series()).iloc[-1] if not next_day_preds_comp_dict.get('Open', {}).empty else np.nan
                pred_high_comp = next_day_preds_comp_dict.get('High', {}).get('Predicted High', pd.Series()).iloc[-1] if not next_day_preds_comp_dict.get('High', {}).empty else np.nan
                pred_low_comp = next_day_preds_comp_dict.get('Low', {}).get('Predicted Low', pd.Series()).iloc[-1] if not next_day_preds_comp_dict.get('Low', {}).empty else np.nan
                pred_vol_comp = next_day_preds_comp_dict.get('Volatility', {}).get('Predicted Volatility', pd.Series()).iloc[-1] if not next_day_preds_comp_dict.get('Volatility', {}).empty else np.nan

                # Determine End_Color for comparison model
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
                    'Last Actual Close': f"{last_actual_close_full_data:.2f}" if pd.notna(last_actual_close_full_data) else "N/A", # New
                    'End_Color': end_color_comp, # New
                    'Predicted Close': pred_close_comp,
                    'Predicted Open': pred_open_comp,
                    'Predicted High': pred_high_comp,
                    'Predicted Low': pred_low_comp,
                    'Predicted Volatility': pred_vol_comp
                })
                for target_type, predicted_val, actual_val in [
                    ('Close', pred_close_comp, last_actual_close_full_data),
                    ('Open', pred_open_comp, last_actual_open_full_data),
                    ('High', pred_high_comp, last_actual_high_full_data),
                    ('Low', pred_low_comp, last_actual_low_full_data),
                    ('Volatility', pred_vol_comp, last_actual_volatility_full_data)
                ]:
                    if pd.notna(predicted_val):
                         save_prediction(ticker, next_trading_day, predicted_val, actual_val, comp_model_name, datetime.now(), end_bt, predicted_type=target_type)
        
        df_next_day_preds = pd.DataFrame(next_day_predictions_list)
        if not df_next_day_preds.empty:
            df_next_day_preds['Date'] = next_trading_day
            # Updated display columns
            display_cols = ['Date', 'Model', 'Last Actual Close', 'End_Color', 'Predicted Close', 'Predicted Open', 'Predicted High', 'Predicted Low', 'Predicted Volatility']
            df_next_day_preds = df_next_day_preds[display_cols].sort_values('Model').reset_index(drop=True)
            
            # Apply styling
            st.dataframe(df_next_day_preds.style.apply(highlight_end_color_home, axis=1).format({
                "Predicted Close": "{:.2f}", 
                "Predicted Open": "{:.2f}", 
                "Predicted High": "{:.2f}", 
                "Predicted Low": "{:.2f}", 
                "Predicted Volatility": "{:.4f}"
            }), hide_index=True) # Added hide_index for cleaner look
            
            if pd.notna(pred_close_main) and pd.notna(last_actual_close_full_data):
                 st.markdown(f"**Difference (Last Actual Close - Predicted Next Day Close) for Main Model ({model_choice}):** {(last_actual_close_full_data - pred_close_main):.2f}")
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

    # --- Detected Candlestick Patterns & Short-Term Trend (New Section) ---
    st.markdown("---")
    st.subheader("💡 Detected Candlestick Patterns & Short-Term Trend")
    pattern_found = []
    if len(data) >= 3: # Need at least 3 days for Morning/Evening Star
        df_recent_ohlc = data.tail(3).copy() # Use raw data for OHLc for pattern detection
        if is_morning_star(df_recent_ohlc):
            pattern_found.append("Morning Star")
        elif is_evening_star(df_recent_ohlc):
            pattern_found.append("Evening Star")
    
    # Add short-term trend
    if len(df_features_full) >= max(5, 20): # Need at least 20 days for MA calculation
        pattern_found.append(get_short_term_trend(df_features_full))
    
    if pattern_found:
        st.markdown(f"**Detected Patterns:** {', '.join(pattern_found)}")
    else:
        st.info("No specific candlestick patterns or short-term trends detected in recent data.")
    
    st.info("""
    **Notes on Pattern Detection:**
    * Currently identifies common candlestick patterns like "Morning Star" and "Evening Star" (3-day patterns).
    * Also indicates the "Short-term Trend" based on a simple 5-day vs 20-day Moving Average crossover.
    * More complex chart patterns (e.g., ascending triangles, head and shoulders) require advanced geometric analysis and are not included in this simplified implementation.
    """)

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
        
        # Add Moving Averages
        for win in ma_windows_list:
            if f'MA_{win}' in df_features_full.columns:
                fig_hist_chart.add_trace(go.Scatter(x=df_features_full['Date'], y=df_features_full[f'MA_{win}'],
                                                    mode='lines', name=f'MA {win}', line=dict(width=1)))
        
        # Add Bollinger Bands
        if 'BB_Upper' in df_features_full.columns and 'BB_Lower' in df_features_full.columns and 'BB_Middle' in df_features_full.columns:
            fig_hist_chart.add_trace(go.Scatter(x=df_features_full['Date'], y=df_features_full['BB_Upper'],
                                                mode='lines', name='BB Upper', line=dict(color='gray', dash='dash', width=1)))
            fig_hist_chart.add_trace(go.Scatter(x=df_features_full['Date'], y=df_features_full['BB_Middle'],
                                                mode='lines', name='BB Middle', line=dict(color='gray', width=1)))
            fig_hist_chart.add_trace(go.Scatter(x=df_features_full['Date'], y=df_features_full['BB_Lower'],
                                                mode='lines', name='BB Lower', line=dict(color='gray', dash='dash', width=1)))

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

        # Plotting other indicators like RSI, MACD, ATR, Stochastic below the main chart
        st.markdown("---")
        st.subheader("📊 Supplementary Technical Indicators")
        
        # RSI Chart
        if 'RSI' in df_features_full.columns:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df_features_full['Date'], y=df_features_full['RSI'], mode='lines', name='RSI', line=dict(color='darkblue')))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)", annotation_position="top right")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)", annotation_position="bottom right")
            fig_rsi.update_layout(title='Relative Strength Index (RSI)', xaxis_title="Date", yaxis_title="RSI Value", height=300)
            st.plotly_chart(fig_rsi, use_container_width=True)

        # MACD Chart
        if 'MACD' in df_features_full.columns and 'MACD_Signal' in df_features_full.columns and 'MACD_Hist' in df_features_full.columns:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=df_features_full['Date'], y=df_features_full['MACD'], mode='lines', name='MACD Line', line=dict(color='blue')))
            fig_macd.add_trace(go.Scatter(x=df_features_full['Date'], y=df_features_full['MACD_Signal'], mode='lines', name='Signal Line', line=dict(color='orange')))
            colors_hist = np.where(df_features_full['MACD_Hist'] > 0, 'green', 'red')
            fig_macd.add_trace(go.Bar(x=df_features_full['Date'], y=df_features_full['MACD_Hist'], name='Histogram', marker_color=colors_hist))
            fig_macd.update_layout(title='MACD', xaxis_title="Date", yaxis_title="Value", height=300)
            st.plotly_chart(fig_macd, use_container_width=True)
            
        # ATR Chart
        if 'ATR' in df_features_full.columns:
            fig_atr = go.Figure()
            fig_atr.add_trace(go.Scatter(x=df_features_full['Date'], y=df_features_full['ATR'], mode='lines', name='ATR', line=dict(color='purple')))
            fig_atr.update_layout(title='Average True Range (ATR)', xaxis_title="Date", yaxis_title="ATR Value", height=300)
            st.plotly_chart(fig_atr, use_container_width=True)

        # Stochastic Oscillator Chart
        if '%K' in df_features_full.columns and '%D' in df_features_full.columns:
            fig_stoch = go.Figure()
            fig_stoch.add_trace(go.Scatter(x=df_features_full['Date'], y=df_features_full['%K'], mode='lines', name='%K', line=dict(color='darkcyan')))
            fig_stoch.add_trace(go.Scatter(x=df_features_full['Date'], y=df_features_full['%D'], mode='lines', name='%D', line=dict(color='lightcoral')))
            fig_stoch.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overbought (80)", annotation_position="top right")
            fig_stoch.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Oversold (20)", annotation_position="bottom right")
            fig_stoch.update_layout(title='Stochastic Oscillator', xaxis_title="Date", yaxis_title="Value", height=300)
            st.plotly_chart(fig_stoch, use_container_width=True)

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
        if model_choice not in models_for_chart:
            models_for_chart.insert(0, model_choice)

        for i, model_name_iter in enumerate(models_for_chart):
            if model_name_iter == model_choice and trained_models_main and trained_models_main.get('Close', {}).get('model'):
                model_info_iter = {'Close': trained_models_main['Close']}
            else:
                model_info_iter = train_models_pipeline(df_compare_data.copy(), model_name_iter, perform_tuning, std_windows_list, update_log)

            if model_info_iter and model_info_iter.get('Close', {}).get('model'):
                preds_dict_iter = generate_predictions_pipeline(df_compare_data.copy(), model_info_iter, update_log)
                if 'Close' in preds_dict_iter and not preds_dict_iter['Close'].empty:
                    pred_df_iter = preds_dict_iter['Close']
                    if not pred_df_iter.empty and 'Actual Close' in pred_df_iter and 'Predicted Close' in pred_df_iter:
                        pred_df_iter.dropna(subset=['Actual Close', 'Predicted Close'], inplace=True)
                        if not pred_df_iter.empty:
                            y_actual_iter = pred_df_iter['Actual Close']
                            y_pred_iter = pred_df_iter['Predicted Close']
                            mae_iter = mean_absolute_error(y_actual_iter, y_pred_iter)
                            rmse_iter = np.sqrt(mean_squared_error(y_actual_iter, y_pred_iter))
                            pct_mae_iter = (mae_iter / avg_actual_compare) * 100 if avg_actual_compare > 0 else np.nan
                            pct_rmse_iter = (rmse_iter / avg_actual_compare) * 100 if avg_actual_compare > 0 else np.nan
                            
                            comparison_results_list.append({'Model': model_name_iter, 'MAE': mae_iter, 'RMSE': rmse_iter, '%-MAE': pct_mae_iter, '%-RMSE': pct_rmse_iter})
                            if rmse_iter < best_comp_rmse: best_comp_rmse, best_comp_model_name = rmse_iter, model_name_iter
                            if pct_rmse_iter < best_comp_pct_rmse: best_comp_pct_rmse = pct_rmse_iter
                                
                            fig_compare_chart.add_trace(go.Scatter(x=pred_df_iter['Date'], y=y_pred_iter, mode='lines', name=f"{model_name_iter} Pred.", line=dict(color=colors[i % len(colors)], dash='dot')))
        
        fig_compare_chart.update_layout(title=f"Model Comparison: Actual vs Predicted Close ({compare_days_actual} days)", xaxis_title="Date", yaxis_title="Price", legend_title="Legend")
        st.plotly_chart(fig_compare_chart, use_container_width=True)
        
        if comparison_results_list:
            df_comparison_tbl = pd.DataFrame(comparison_results_list).sort_values(['RMSE', '%-RMSE']).dropna(subset=['RMSE'])
            st.dataframe(df_comparison_tbl.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "%-MAE": "{:.2f}%", "%-RMSE": "{:.2f}%"}))
            if best_comp_model_name != "N/A":
                 st.markdown(f"🏆 **Best performing in comparison (lowest RMSE): {best_comp_model_name}** (RMSE: {best_comp_rmse:.4f}, %-RMSE: {best_comp_pct_rmse:.2f}%)")


    # --- Output 2: Training period actual vs predicted (Main Model Close Price) ---
    st.markdown("---")
    st.subheader(f"🎯 Training Period Performance: {model_choice} (Close Price)")
    train_preds_dict = generate_predictions_pipeline(df_train_period.copy(), {'Close': trained_models_main.get('Close')}, update_log)

    if 'Close' in train_preds_dict and not train_preds_dict['Close'].empty:
        train_pred_df_main = train_preds_dict['Close']
        if not train_pred_df_main.empty and 'Actual Close' in train_pred_df_main:
            actual_train_main = train_pred_df_main['Actual Close']
            predicted_train_main = train_pred_df_main['Predicted Close']
            mae_train_main = mean_absolute_error(actual_train_main, predicted_train_main)
            rmse_train_main = np.sqrt(mean_squared_error(actual_train_main, predicted_train_main))
            avg_actual_train_main = actual_train_main.mean()
            pct_mae_train_main = (mae_train_main / avg_actual_train_main) * 100 if avg_actual_train_main > 0 else np.nan
            pct_rmse_train_main = (rmse_train_main / avg_actual_train_main) * 100 if avg_actual_train_main > 0 else np.nan
            st.markdown(f"**Metrics on Training Data ({start_bt.strftime('%Y-%m-%d')} to {end_bt.strftime('%Y-%m-%d')}):**")
            st.markdown(f"MAE: {mae_train_main:.4f} ({pct_mae_train_main:.2f}%) | RMSE: {rmse_train_main:.4f} ({pct_rmse_train_main:.2f}%)")

            fig_train_perf = go.Figure()
            fig_train_perf.add_trace(go.Scatter(x=train_pred_df_main['Date'], y=actual_train_main, mode='lines', name='Actual Close', line=dict(color='royalblue')))
            fig_train_perf.add_trace(go.Scatter(x=train_pred_df_main['Date'], y=predicted_train_main, mode='lines', name='Predicted Close', line=dict(color='orangered', dash='dash')))
            fig_train_perf.update_layout(title=f"Training: Actual vs Predicted Close ({model_choice})", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig_train_perf, use_container_width=True)
        else: st.info("Not enough data in training predictions to plot/calculate metrics.")
    else: st.warning("Could not generate training period predictions for the main Close model.")

    # --- Output 3: Backtesting on last 30 days of training (Main Model) ---
    st.markdown("---")
    st.subheader(f"📉 Backtesting (Last 30 Trading Days of Training Period): {model_choice} (Close Price)")
    backtest_start_date_30d = pd.to_datetime(end_bt) - timedelta(days=45) # Fetch a bit more to ensure 30 trading days
    df_backtest_data_30d = df_features_full[(df_features_full['Date'] > backtest_start_date_30d) & (df_features_full['Date'] <= pd.to_datetime(end_bt))].copy()
    df_backtest_data_30d = df_backtest_data_30d.tail(30) # Get last 30 available trading days

    if not df_backtest_data_30d.empty:
        bt_preds_dict = generate_predictions_pipeline(df_backtest_data_30d.copy(), {'Close': trained_models_main.get('Close')}, update_log)
        if 'Close' in bt_preds_dict and not bt_preds_dict['Close'].empty:
            bt_pred_df_main = bt_preds_dict['Close']
            if not bt_pred_df_main.empty and 'Actual Close' in bt_pred_df_main:
                actual_bt_main = bt_pred_df_main['Actual Close']
                predicted_bt_main = bt_pred_df_main['Predicted Close']
                mae_bt_main = mean_absolute_error(actual_bt_main, predicted_bt_main)
                rmse_bt_main = np.sqrt(mean_squared_error(actual_bt_main, predicted_bt_main))
                avg_actual_bt_main = actual_bt_main.mean()
                pct_mae_bt_main = (mae_bt_main / avg_actual_bt_main) * 100 if avg_actual_bt_main > 0 else np.nan
                pct_rmse_bt_main = (rmse_bt_main / avg_actual_bt_main) * 100 if avg_actual_bt_main > 0 else np.nan
                st.markdown(f"**Backtest Metrics (approx last 30 days of training):**")
                st.markdown(f"MAE: {mae_bt_main:.4f} ({pct_mae_bt_main:.2f}%) | RMSE: {rmse_bt_main:.4f} ({pct_rmse_bt_main:.2f}%)")
                st.dataframe(bt_pred_df_main[['Date', 'Actual Close', 'Predicted Close', 'Difference']].style.format({"Actual Close": "{:.2f}", "Predicted Close": "{:.2f}", "Difference": "{:.2f}"}))
                
                for _, row in bt_pred_df_main.iterrows():
                    save_prediction(ticker, row['Date'], row['Predicted Close'], row['Actual Close'], model_choice, datetime.now(), end_bt, predicted_type='Close')
            else: st.info("Not enough data in backtest predictions to display.")
        else: st.warning("Could not generate backtest predictions for the main Close model.")
    else: st.info("Not enough data for the 30-day backtest period.")


    # --- Output 4: Future Predictions (Iterative, Main Model Close Price) ---
    st.markdown("---")
    st.subheader(f"🚀 Future {n_future} Days Predicted Close Prices ({model_choice})")
    st.info("""
    **Iterative Forecasting Explained:** Monarch predicts future prices one day at a time. 
    The prediction for Day 1 is made. Then, this Day 1 prediction is used as an input (as if it were actual data) to help predict Day 2, and so on. 
    This process repeats for the number of future days you select. 
    *Remember: These are model-based projections and not financial advice.*
    """)

    future_predictions_output_list = []
    main_close_model_info = trained_models_main.get('Close')

    if main_close_model_info and main_close_model_info['model']:
        model_fut = main_close_model_info['model']
        scaler_fut = main_close_model_info['scaler']
        feature_cols_fut_train = main_close_model_info['features']
        
        # Determine max historical window needed for feature calculation for any feature
        # Using a conservative estimate to ensure all features can be calculated.
        max_hist_window = max(max(lag_features_list), max(ma_windows_list), max(std_windows_list), rsi_window, macd_long_window, bb_window, atr_window, stoch_window, stoch_smooth_window) if all([lag_features_list, ma_windows_list, std_windows_list]) else 200 # Default if lists are empty
        
        # Get historical context up to end_bt to start iterative prediction
        df_hist_context_fut = df_features_full[df_features_full['Date'] <= pd.to_datetime(end_bt)].copy()
        
        # Ensure enough data in context for feature generation for next step
        if len(df_hist_context_fut) < max_hist_window:
            st.warning(f"Not enough historical data (before {end_bt}) to reliably generate features for future predictions. Need at least {max_hist_window} data points.")
        
        current_iter_date = pd.to_datetime(end_bt)
        current_iter_close = df_hist_context_fut['Close'].iloc[-1] if not df_hist_context_fut.empty else np.nan
        current_iter_open = df_hist_context_fut['Open'].iloc[-1] if not df_hist_context_fut.empty else np.nan
        current_iter_high = df_hist_context_fut['High'].iloc[-1] if not df_hist_context_fut.empty else np.nan
        current_iter_low = df_hist_context_fut['Low'].iloc[-1] if not df_hist_context_fut.empty else np.nan
        current_iter_volume = df_hist_context_fut['Volume'].iloc[-1] if not df_hist_context_fut.empty else np.nan


        for i in range(n_future):
            next_pred_date = current_iter_date + timedelta(days=1)
            while next_pred_date.weekday() >= 5: next_pred_date += timedelta(days=1) # Skip weekends

            # To generate features for next_pred_date, we need data up to current_iter_date
            # The df_hist_context_fut already contains this
            
            # Re-create features for the updated historical context (including last predicted day)
            # This is crucial for lag features and moving averages to update
            df_features_for_pred = create_features(df_hist_context_fut.copy(), lag_features_list, ma_windows_list, std_windows_list, rsi_window, macd_short_window, macd_long_window, macd_signal_window, bb_window, bb_std_dev, atr_window, stoch_window, stoch_smooth_window)
            
            if df_features_for_pred.empty:
                update_log(f"⚠️ Empty features at future step {i+1}. Stopping iterative prediction.")
                break
            
            last_feature_row = df_features_for_pred.tail(1)
            
            X_fut_predict = last_feature_row.reindex(columns=feature_cols_fut_train, fill_value=0)
            if X_fut_predict.empty or X_fut_predict.shape[1] != len(feature_cols_fut_train):
                update_log(f"⚠️ Feature mismatch at future step {i+1}. Stopping iterative prediction.")
                break
            
            X_fut_scaled = scaler_fut.transform(X_fut_predict)
            try:
                predicted_price_fut = model_fut.predict(X_fut_scaled)[0]
            except Exception as e_fut:
                update_log(f"❌ Error at future step {i+1} during prediction: {e_fut}. Stopping iterative prediction.")
                break

            future_predictions_output_list.append({'Date': next_pred_date, 'Predicted Close': predicted_price_fut})
            
            # Append simulated row to historical context for next iteration
            # We use the predicted close as the new actual close for the next iteration's feature calculation
            # For Open, High, Low, Volume, we can use simple assumptions or previous values
            new_row_simulated = pd.DataFrame([{
                'Date': next_pred_date,
                'Open': current_iter_open, # For simplicity, use previous Open. Could be more complex.
                'High': predicted_price_fut * 1.01, # Simple assumption: High slightly above predicted Close
                'Low': predicted_price_fut * 0.99,  # Simple assumption: Low slightly below predicted Close
                'Close': predicted_price_fut,
                'Volume': current_iter_volume # For simplicity, use previous Volume
            }])
            df_hist_context_fut = pd.concat([df_hist_context_fut, new_row_simulated], ignore_index=True)
            current_iter_date = next_pred_date
            current_iter_close = predicted_price_fut
            # Update OHLV to reflect latest simulated data for next iteration's 'new_row_simulated'
            current_iter_open = new_row_simulated['Open'].iloc[-1]
            current_iter_high = new_row_simulated['High'].iloc[-1]
            current_iter_low = new_row_simulated['Low'].iloc[-1]
            current_iter_volume = new_row_simulated['Volume'].iloc[-1]


        if future_predictions_output_list:
            df_future_preds_tbl = pd.DataFrame(future_predictions_output_list)
            st.dataframe(df_future_preds_tbl.style.format({"Predicted Close": "{:.2f}"}))
            fig_future_chart = px.line(df_future_preds_tbl, x='Date', y='Predicted Close', title=f"Future {n_future} Days Predicted Close ({model_choice})")
            fig_future_chart.update_traces(line=dict(color='mediumseagreen'))
            st.plotly_chart(fig_future_chart, use_container_width=True)
            
            for pred_item in future_predictions_output_list:
                save_prediction(ticker, pred_item['Date'], pred_item['Predicted Close'], np.nan, model_choice, datetime.now(), end_bt, predicted_type='Close')
        else:
            st.warning("Could not generate future predictions. Check logs.")
    else:
        st.warning("Main Close model not available for future predictions.")


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
        # Using PREDICTION_LOG_COLUMNS which now includes 'predicted_type'
        for col in PREDICTION_LOG_COLUMNS:
            if col not in past_preds_df.columns:
                if 'date' in col: past_preds_df[col] = pd.NaT
                elif col == 'predicted_type': past_preds_df[col] = 'Close' # Default for older entries
                else: past_preds_df[col] = np.nan
        
        # Sort and display
        past_preds_df_display = past_preds_df.sort_values(by=['prediction_for_date', 'prediction_generation_date', 'predicted_type'], ascending=[False, False, True]).reset_index(drop=True)
        
        # Select and reorder columns for display
        display_cols = ['prediction_for_date', 'predicted_type', 'predicted_value', 'actual_close', 'model_used', 'prediction_generation_date', 'training_end_date_used']
        past_preds_df_display = past_preds_df_display[display_cols]

        st.dataframe(past_preds_df_display.style.format({
            "predicted_value": "{:.2f}", "actual_close": "{:.2f}",
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
            
            fig_past_log.update_layout(title=f'Historical Log: Predicted vs. Actual Close for {ticker} (Recent Generations)', xaxis_title="Date of Stock Price", yaxis_title="Price", hovermode="x unified")
            st.plotly_chart(fig_past_log, use_container_width=True)
        else: st.info("No past 'Close' predictions with known actuals to plot from log.")
    else: st.info(f"No past prediction logs found for {ticker}.")


    # --- Feature Importance (Main Close Model) ---
    st.markdown("---")
    st.subheader(f"🧠 Feature Importance / Coefficients ({model_choice} - Close Price Model)")
    main_close_model_info_fi = trained_models_main.get('Close')
    if main_close_model_info_fi and main_close_model_info_fi.get('model') and main_close_model_info_fi.get('features'):
        model_fi = main_close_model_info_fi['model']
        feature_names_fi = main_close_model_info_fi['features']
        
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



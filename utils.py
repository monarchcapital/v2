# utils.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint, t # Import t-distribution for confidence intervals
from datetime import timedelta, date, datetime
from sklearn.preprocessing import StandardScaler
import os
import streamlit as st # Import streamlit for caching

# config is now mainly for default values, parameters are passed explicitly
import config

# --- Global list for collecting training messages ---
training_messages_log = []

# Define the expected columns for the prediction log globally
PREDICTION_LOG_COLUMNS = ['prediction_generation_date', 'prediction_for_date', 'training_end_date_used', 'ticker', 'model_used', 'predicted_value', 'actual_close', 'predicted_type', 'predicted_lower_bound', 'predicted_upper_bound']


# --- Helper to parse comma-separated integers ---
def parse_int_list(input_str, default_list, error_callback=None):
    """
    Parses a comma-separated string of integers into a sorted list of unique integers.
    If parsing fails, returns the default list and calls an optional error_callback.
    """
    try:
        parsed = sorted(list(set([int(x.strip()) for x in input_str.split(',') if x.strip()])))
        return parsed if parsed else None # Return None if parsed list is empty
    except ValueError:
        if error_callback:
            error_callback(f"Invalid format for list input: '{input_str}'. Please use comma-separated integers (e.g., '{','.join(map(str, default_list))}'). Using default values.")
        return default_list


# --- Prediction Logging Functions ---
def save_prediction(ticker, prediction_for_date, predicted_value, actual_close_price, model_name, prediction_generation_date, training_end_date_used, predicted_type='Close', predicted_lower_bound=np.nan, predicted_upper_bound=np.nan):
    """
    Saves a single prediction entry to a ticker-specific CSV file.
    Manages appending to existing files and dropping duplicates.
    Now includes lower and upper bounds for predictions.
    """
    predictions_dir = "monarch_predictions_data"
    os.makedirs(predictions_dir, exist_ok=True)
    file_path = os.path.join(predictions_dir, f"{ticker}_predictions.csv")

    # Prepare the new entry as a DataFrame
    new_entry = pd.DataFrame([{
        'prediction_generation_date': prediction_generation_date,
        'prediction_for_date': prediction_for_date,
        'training_end_date_used': training_end_date_used,
        'ticker': ticker,
        'model_used': model_name,
        'predicted_value': predicted_value,
        'actual_close': actual_close_price,
        'predicted_type': predicted_type,
        'predicted_lower_bound': predicted_lower_bound,
        'predicted_upper_bound': predicted_upper_bound
    }])
    
    # Ensure date columns are in datetime format
    for col in ['prediction_generation_date', 'prediction_for_date', 'training_end_date_used']:
        new_entry[col] = pd.to_datetime(new_entry[col])

    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        # Ensure date columns are in datetime format when reading existing
        for col in ['prediction_generation_date', 'prediction_for_date', 'training_end_date_used']:
             existing_df[col] = pd.to_datetime(existing_df[col])
        
        # Add new columns if they don't exist (for backward compatibility)
        for col in ['predicted_type', 'predicted_lower_bound', 'predicted_upper_bound']:
            if col not in existing_df.columns:
                existing_df[col] = 'Close' if col == 'predicted_type' else np.nan # Default value for old entries

        combined_df = pd.concat([existing_df, new_entry], ignore_index=True)
    else:
        combined_df = new_entry
    
    # Sort and drop duplicates, keeping the latest prediction for a given (ticker, date, type, model, training_end_date)
    # The 'predicted_type' is crucial for uniqueness.
    # Keep the latest generation date for unique (prediction_for_date, predicted_type, model_used, training_end_date_used) combination
    combined_df.sort_values(by=['prediction_for_date', 'predicted_type', 'model_used', 'training_end_date_used', 'prediction_generation_date'], ascending=[False, True, True, False, False], inplace=True)
    combined_df.drop_duplicates(subset=['prediction_for_date', 'predicted_type', 'model_used', 'training_end_date_used'], keep='first', inplace=True)
    
    # Save the combined DataFrame
    combined_df.to_csv(file_path, index=False)


def load_past_predictions(ticker):
    """
    Loads past prediction entries for a given ticker from its CSV file.
    Returns an empty DataFrame with correct columns if file does not exist.
    """
    predictions_dir = "monarch_predictions_data"
    file_path = os.path.join(predictions_dir, f"{ticker}_predictions.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        for col in ['prediction_generation_date', 'prediction_for_date', 'training_end_date_used']:
            df[col] = pd.to_datetime(df[col])
        # Add new columns if they don't exist (for backward compatibility)
        for col in ['predicted_type', 'predicted_lower_bound', 'predicted_upper_bound']:
            if col not in df.columns:
                df[col] = 'Close' if col == 'predicted_type' else np.nan
        return df
    return pd.DataFrame(columns=PREDICTION_LOG_COLUMNS) # Return empty DataFrame with correct columns


# --- Data Download and Feature Engineering ---
@st.cache_data # Cache data download for performance
def download_data(ticker_symbol, period="5y"):
    """
    Downloads historical stock data for a given ticker symbol using yfinance.
    Caches the result to avoid repeated downloads.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(period=period)
        if data.empty:
            st.warning(f"No data found for ticker: {ticker_symbol}. Check the symbol or period.")
            return pd.DataFrame()
        data.reset_index(inplace=True)
        # Handle cases where 'Date' might be timezone-aware (e.g., in newer yfinance versions)
        if pd.api.types.is_datetime64_any_dtype(data['Date']):
            if data['Date'].dt.tz is not None:
                data['Date'] = data['Date'].dt.tz_convert(None) # Convert to naive datetime
        return data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        st.error(f"Error downloading data for {ticker_symbol}: {e}. Please check the ticker symbol and your internet connection.")
        return pd.DataFrame()

def create_features(df, indicator_params):
    """
    Generates various technical indicators and features from raw OHLCV data.
    Accepts indicator parameters as a dictionary and conditionally creates features.
    `indicator_params` is expected to be a dictionary where keys are indicator names
    and values are their parameters (e.g., window size, or a list of window sizes).
    If an indicator is disabled in the UI, its corresponding entry in this dict
    might be None or an empty list/value, allowing skipping its calculation.
    """
    df_copy = df.copy()
    
    # Sort by date to ensure correct calculation of time-series features
    df_copy = df_copy.sort_values(by='Date').reset_index(drop=True)

    # Calculate Daily Returns (always included)
    df_copy['Daily_Return'] = df_copy['Close'].pct_change()

    # Lag features
    lag_features_list = indicator_params.get('LAG_FEATURES')
    if lag_features_list:
        for lag in lag_features_list:
            df_copy[f'Close_Lag_{lag}'] = df_copy['Close'].shift(lag)
            df_copy[f'Volume_Lag_{lag}'] = df_copy['Volume'].shift(lag)

    # Moving Averages
    ma_windows_list = indicator_params.get('MA_WINDOWS')
    if ma_windows_list:
        for window in ma_windows_list:
            df_copy[f'MA_{window}'] = df_copy['Close'].rolling(window=window).mean()

    # Volatility (Standard Deviation of Close Price)
    std_windows_list = indicator_params.get('STD_WINDOWS')
    if std_windows_list:
        for window in std_windows_list:
            df_copy[f'Volatility_{window}'] = df_copy['Close'].rolling(window=window).std()

    # Relative Strength Index (RSI)
    rsi_window = indicator_params.get('RSI_WINDOW')
    if rsi_window:
        delta = df_copy['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
        rs = gain / loss
        df_copy['RSI'] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    macd_short_window = indicator_params.get('MACD_SHORT_WINDOW')
    macd_long_window = indicator_params.get('MACD_LONG_WINDOW')
    macd_signal_window = indicator_params.get('MACD_SIGNAL_WINDOW')
    if macd_short_window and macd_long_window and macd_signal_window:
        exp1 = df_copy['Close'].ewm(span=macd_short_window, adjust=False).mean()
        exp2 = df_copy['Close'].ewm(span=macd_long_window, adjust=False).mean()
        df_copy['MACD'] = exp1 - exp2
        df_copy['MACD_Signal'] = df_copy['MACD'].ewm(span=macd_signal_window, adjust=False).mean()
        df_copy['MACD_Hist'] = df_copy['MACD'] - df_copy['MACD_Signal']

    # Bollinger Bands
    bb_window = indicator_params.get('BB_WINDOW')
    bb_std_dev = indicator_params.get('BB_STD_DEV')
    if bb_window and bb_std_dev is not None:
        df_copy['BB_Middle'] = df_copy['Close'].rolling(window=bb_window).mean()
        std_dev = df_copy['Close'].rolling(window=bb_window).std()
        df_copy['BB_Upper'] = df_copy['BB_Middle'] + (std_dev * bb_std_dev)
        df_copy['BB_Lower'] = df_copy['BB_Middle'] - (std_dev * bb_std_dev)

    # Average True Range (ATR)
    atr_window = indicator_params.get('ATR_WINDOW')
    if atr_window:
        high_low = df_copy['High'] - df_copy['Low']
        high_close = np.abs(df_copy['High'] - df_copy['Close'].shift())
        low_close = np.abs(df_copy['Low'] - df_copy['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df_copy['ATR'] = tr.ewm(span=atr_window, adjust=False).mean()

    # Stochastic Oscillator
    stoch_window = indicator_params.get('STOCH_WINDOW')
    stoch_smooth_window = indicator_params.get('STOCH_SMOOTH_WINDOW')
    if stoch_window and stoch_smooth_window:
        lowest_low = df_copy['Low'].rolling(window=stoch_window).min()
        highest_high = df_copy['High'].rolling(window=stoch_window).max()
        df_copy['%K'] = ((df_copy['Close'] - lowest_low) / (highest_high - lowest_low)) * 100
        df_copy['%D'] = df_copy['%K'].rolling(window=stoch_smooth_window).mean()

    # Commodity Channel Index (CCI)
    cci_window = indicator_params.get('CCI_WINDOW')
    if cci_window:
        TP = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3
        MA_TP = TP.rolling(window=cci_window).mean()
        MD_TP = TP.rolling(window=cci_window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        df_copy['CCI'] = (TP - MA_TP) / (0.015 * MD_TP) # 0.015 is a common constant in CCI formula

    # Rate of Change (ROC)
    roc_window = indicator_params.get('ROC_WINDOW')
    if roc_window:
        df_copy['ROC'] = df_copy['Close'].pct_change(periods=roc_window) * 100

    # Average Directional Index (ADX)
    adx_window = indicator_params.get('ADX_WINDOW')
    if adx_window:
        # Calculate Directional Movement (DM)
        df_copy['DMplus'] = (df_copy['High'] - df_copy['High'].shift(1)).clip(lower=0)
        df_copy['DMminus'] = (df_copy['Low'].shift(1) - df_copy['Low']).clip(lower=0)
        
        # If DMplus > DMminus, and DMplus > 0, then +DM = DMplus, else 0
        df_copy['+DM'] = np.where((df_copy['DMplus'] > df_copy['DMminus']) & (df_copy['DMplus'] > 0), df_copy['DMplus'], 0)
        # If DMminus > DMplus, and DMminus > 0, then -DM = DMminus, else 0
        df_copy['-DM'] = np.where((df_copy['DMminus'] > df_copy['DMplus']) & (df_copy['DMminus'] > 0), df_copy['DMminus'], 0)
        
        # Calculate True Range (TR) - same as ATR's TR
        high_low = df_copy['High'] - df_copy['Low'] # Recalculate as a safety measure for this block
        high_close = np.abs(df_copy['High'] - df_copy['Close'].shift())
        low_close = np.abs(df_copy['Low'] - df_copy['Close'].shift())
        df_copy['TR_ADX'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate DI+ and DI- (smoothed DM over adx_window period)
        df_copy['+DI'] = (df_copy['+DM'].ewm(span=adx_window, adjust=False).mean() / df_copy['TR_ADX'].ewm(span=adx_window, adjust=False).mean()) * 100
        df_copy['-DI'] = (df_copy['-DM'].ewm(span=adx_window, adjust=False).mean() / df_copy['TR_ADX'].ewm(span=adx_window, adjust=False).mean()) * 100
        
        # Calculate DX
        df_copy['DX'] = np.abs(df_copy['+DI'] - df_copy['-DI']) / (df_copy['+DI'] + df_copy['-DI']) * 100
        
        # Calculate ADX (smoothed DX over adx_window period)
        df_copy['ADX'] = df_copy['DX'].ewm(span=adx_window, adjust=False).mean()

    # On-Balance Volume (OBV)
    obv_enabled = indicator_params.get('OBV_ENABLED')
    if obv_enabled: # OBV is simply enabled/disabled, no window
        df_copy['OBV'] = (np.sign(df_copy['Close'].diff()) * df_copy['Volume']).fillna(0).cumsum()

    # Chaikin Money Flow (CMF)
    cmf_window = indicator_params.get('CMF_WINDOW')
    if cmf_window:
        # Money Flow Multiplier (MFM)
        df_copy['MFM'] = ((df_copy['Close'] - df_copy['Low']) - (df_copy['High'] - df_copy['Close'])) / (df_copy['High'] - df_copy['Low'])
        df_copy['MFM'] = df_copy['MFM'].fillna(0) # Handle division by zero
        # Money Flow Volume (MFV)
        df_copy['MFV'] = df_copy['MFM'] * df_copy['Volume']
        # CMF
        df_copy['CMF'] = df_copy['MFV'].rolling(window=cmf_window).sum() / df_copy['Volume'].rolling(window=cmf_window).sum()

    # Parabolic SAR
    parsar_enabled = indicator_params.get('PARSAR_ENABLED')
    parsar_acceleration = indicator_params.get('PARSAR_ACCELERATION')
    parsar_max_acceleration = indicator_params.get('PARSAR_MAX_ACCELERATION')
    if parsar_enabled and parsar_acceleration is not None and parsar_max_acceleration is not None:
        # Initialize SAR values
        df_copy['SAR'] = df_copy['Close'].copy()
        df_copy['EP'] = df_copy['High'].copy() # Extreme Point
        df_copy['AF'] = parsar_acceleration # Acceleration Factor
        df_copy['Trend'] = 0 # 1 for uptrend, -1 for downtrend, 0 for initial
        
        for i in range(1, len(df_copy)):
            if i == 1: # Initialize trend based on first two candles
                if df_copy['Close'].iloc[i] > df_copy['Close'].iloc[i-1]:
                    df_copy.loc[i, 'Trend'] = 1
                    df_copy.loc[i, 'EP'] = df_copy['High'].iloc[i]
                    df_copy.loc[i, 'SAR'] = df_copy['Low'].iloc[i-1] # Initial SAR for uptrend
                else:
                    df_copy.loc[i, 'Trend'] = -1
                    df_copy.loc[i, 'EP'] = df_copy['Low'].iloc[i]
                    df_copy.loc[i, 'SAR'] = df_copy['High'].iloc[i-1] # Initial SAR for downtrend
                continue
                
            prev_sar = df_copy['SAR'].iloc[i-1]
            prev_ep = df_copy['EP'].iloc[i-1]
            prev_af = df_copy['AF'].iloc[i-1]
            prev_trend = df_copy['Trend'].iloc[i-1]

            current_high = df_copy['High'].iloc[i]
            current_low = df_copy['Low'].iloc[i]
            current_close = df_copy['Close'].iloc[i]

            new_sar = prev_sar + prev_af * (prev_ep - prev_sar)
            
            new_trend = prev_trend
            new_ep = prev_ep
            new_af = prev_af

            if prev_trend == 1: # Previously uptrend
                # Check for reversal (SAR crosses above price)
                if new_sar > current_low:
                    new_trend = -1 # Reverse to downtrend
                    new_sar = max(current_high, prev_ep) # New SAR is current High or previous EP
                    new_ep = current_low
                    new_af = parsar_acceleration # Reset AF
                else:
                    # Still uptrend, update EP and AF
                    if current_high > prev_ep:
                        new_ep = current_high
                        new_af = min(prev_af + parsar_acceleration, parsar_max_acceleration)
                    # Ensure SAR doesn't go above current/previous Low
                    new_sar = min(new_sar, current_low, df_copy['Low'].iloc[i-1])

            elif prev_trend == -1: # Previously downtrend
                # Check for reversal (SAR crosses below price)
                if new_sar < current_high:
                    new_trend = 1 # Reverse to uptrend
                    new_sar = min(current_low, prev_ep) # New SAR is current Low or previous EP
                    new_ep = current_high
                    new_af = parsar_acceleration # Reset AF
                else:
                    # Still downtrend, update EP and AF
                    if current_low < prev_ep:
                        new_ep = current_low
                        new_af = min(prev_af + parsar_acceleration, parsar_max_acceleration)
                    # Ensure SAR doesn't go below current/previous High
                    new_sar = max(new_sar, current_high, df_copy['High'].iloc[i-1])
            
            df_copy.loc[i, 'SAR'] = new_sar
            df_copy.loc[i, 'EP'] = new_ep
            df_copy.loc[i, 'AF'] = new_af
            df_copy.loc[i, 'Trend'] = new_trend

    # Day of Week, Day of Month, Day of Year (cyclical features can be added, but keeping it simple for now)
    df_copy['DayOfWeek'] = df_copy['Date'].dt.dayofweek
    df_copy['DayOfMonth'] = df_copy['Date'].dt.day
    df_copy['Month'] = df_copy['Date'].dt.month
    df_copy['Year'] = df_copy['Date'].dt.year

    # Drop intermediate columns for ADX, SAR calculation if not needed as features
    # Only drop if they exist to avoid errors if the indicator was disabled
    cols_to_drop = [col for col in ['DMplus', 'DMminus', '+DM', '-DM', 'TR_ADX', 'DX', 'EP', 'AF', 'Trend', 'MFM', 'MFV'] if col in df_copy.columns]
    df_copy.drop(columns=cols_to_drop, errors='ignore', inplace=True)

    # Drop rows with NaN values resulting from feature creation (should be done at the end)
    df_copy.dropna(inplace=True)
    df_copy.reset_index(drop=True, inplace=True)
    return df_copy


# --- Candlestick Pattern Detection Functions (Simplified) ---
def is_morning_star(df_recent):
    """
    Detects the Morning Star candlestick pattern from the most recent 3 candles.
    A bullish reversal pattern.
    """
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
    """
    Detects the Evening Star candlestick pattern from the most recent 3 candles.
    A bearish reversal pattern.
    """
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

def get_short_term_trend(df_features, indicator_params):
    """
    Determines short-term trend based on simple moving average crossover.
    Accepts indicator_params dictionary to get MA window parameters.
    """
    ma_windows = indicator_params.get('MA_WINDOWS')
    if not ma_windows or len(ma_windows) < 2:
        return "Not enough MA windows configured for Short-term Trend detection (requires at least 2)."

    short_ma_window = ma_windows[0]
    long_ma_window = ma_windows[1] # Assumes the first two in the list are short and long

    if len(df_features) < max(short_ma_window, long_ma_window):
        return "Not Enough Data"
    
    # Ensure these MA columns exist from create_features, if not, recalculate them.
    ma_short_col = f'MA_{short_ma_window}'
    ma_long_col = f'MA_{long_ma_window}'

    if ma_short_col in df_features.columns and ma_long_col in df_features.columns:
        current_short_ma = df_features[ma_short_col].iloc[-1]
        current_long_ma = df_features[ma_long_col].iloc[-1]
    else:
        # Fallback to recalculating for the latest data if columns are missing
        if len(df_features) >= long_ma_window: # Ensure enough data to calculate long_ma
            if 'Close' in df_features.columns and not df_features['Close'].isnull().all():
                current_short_ma = df_features['Close'].iloc[-short_ma_window:].mean()
                current_long_ma = df_features['Close'].iloc[-long_ma_window:].mean()
            else:
                return "Not Enough Data (Close price missing)"
        else:
            return "Not Enough Data"

    if current_short_ma > current_long_ma:
        return "Uptrend (Short-term)"
    elif current_short_ma < current_long_ma:
        return "Downtrend (Short-term)"
    else:
        return "Neutral (Short-term)"

# --- Model Management and Training ---
def get_model(model_name):
    """
    Returns an unfitted machine learning model based on the provided name.
    """
    if model_name == 'Random Forest':
        return RandomForestRegressor(random_state=42)
    elif model_name == 'XGBoost':
        return xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
    elif model_name == 'Gradient Boosting':
        return GradientBoostingRegressor(random_state=42)
    elif model_name == 'Linear Regression':
        return LinearRegression()
    elif model_name == 'SVR':
        return SVR() # Can be slow on large datasets without careful tuning
    elif model_name == 'KNN':
        return KNeighborsRegressor()
    elif model_name == 'Decision Tree':
        return DecisionTreeRegressor(random_state=42)
    else:
        raise ValueError("Unknown model name")

def _train_single_model(df_train, target_column, model_name, perform_tuning, update_log_func):
    """
    Trains a single model for a given target column.
    Applies StandardScaler and optional RandomizedSearchCV for tuning.
    """
    # Exclude columns that are not features or are targets themselves
    # The list of excluded columns should be robust to columns not existing if indicators are disabled.
    # We also exclude target_column if it happens to be in the exclude_cols or features already.
    exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return']
    
    # Dynamically build features by taking all columns except common non-features/targets.
    # This automatically filters out columns not created if an indicator was disabled.
    # Specifically handle MACD_Hist, BB_Upper, BB_Lower, BB_Middle, SAR, Trend as they are often derived or intermediate.
    # Only remove if they are explicitly listed as something to exclude and exist in features.
    
    features = [col for col in df_train.columns if col not in exclude_cols and col != target_column and 
                not col.startswith('SAR') and not col.startswith('Trend') and 
                col not in ['MACD_Hist', 'BB_Upper', 'BB_Lower', 'BB_Middle', 'MFM', 'MFV', 'DMplus', 'DMminus', '+DM', '-DM', 'TR_ADX', 'DX', 'EP', 'AF']] # Exclude SAR related temp cols

    X = df_train[features]
    y = df_train[target_column]

    if X.empty or y.empty:
        update_log_func(f"Warning: Empty features or target for {target_column}. Cannot train.")
        return None, None, None, None # Return None for residuals too

    # Handle cases where all feature values might be NaN after dropna, leading to empty X
    if X.isnull().all().all() or X.shape[1] == 0:
        update_log_func(f"Warning: All features are NaN or no features available for {target_column}. Cannot train.")
        return None, None, None, None # Return None for residuals too

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = get_model(model_name)

    # SVR/KNN tuning can be very slow, Linear Regression has no params
    if perform_tuning and model_name not in ['SVR', 'KNN', 'Linear Regression']: 
        update_log_func(f"Performing hyperparameter tuning for {model_name} on {target_column}...")
        param_distributions = {}
        if model_name == 'Random Forest':
            param_distributions = {
                'n_estimators': randint(50, 200),
                'max_features': uniform(0.6, 0.3), # e.g., from 60% to 90%
                'max_depth': randint(5, 15),
                'min_samples_split': randint(2, 10),
                'min_samples_leaf': randint(1, 5)
            }
        elif model_name == 'XGBoost':
            param_distributions = {
                'n_estimators': randint(50, 200),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(3, 10),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4)
            }
        elif model_name == 'Gradient Boosting':
            param_distributions = {
                'n_estimators': randint(50, 200),
                'learning_rate': uniform(0.01, 0.2),
                'max_depth': randint(3, 8),
                'subsample': uniform(0.6, 0.4),
                'min_samples_split': randint(2, 10),
                'min_samples_leaf': randint(1, 5)
            }
        elif model_name == 'Decision Tree':
            param_distributions = {
                'max_depth': randint(5, 20),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10)
            }

        if param_distributions:
            try:
                # Use fewer iterations for speed in Streamlit
                random_search = RandomizedSearchCV(model, param_distributions, n_iter=20, cv=3, verbose=0, random_state=42, n_jobs=-1, scoring='neg_mean_squared_error')
                random_search.fit(X_scaled, y)
                model = random_search.best_estimator_
                update_log_func(f"Tuning complete. Best params for {target_column}: {random_search.best_params_}")
            except Exception as e:
                update_log_func(f"Error during tuning for {target_column}: {e}. Using default model.")
    
    try:
        model.fit(X_scaled, y)
        predictions_train = model.predict(X_scaled)
        residuals = y - predictions_train # Calculate residuals for confidence intervals
    except Exception as e:
        update_log_func(f"Error fitting model {model_name} for {target_column}: {e}")
        return None, None, None, None # Return None for residuals too

    return model, scaler, features, residuals

def train_models_pipeline(df_train_period, model_choice, perform_tuning, update_log_func, indicator_params):
    """
    Trains models for Close, Open, High, Low, and Volatility targets.
    Now returns validation metrics and stores residuals for confidence intervals.
    """
    trained_models = {}
    validation_metrics = {}
    
    target_cols_to_train = ['Close', 'Open', 'High', 'Low']
    
    # Dynamically determine the volatility target column based on available std windows
    if indicator_params.get('STD_WINDOWS') and indicator_params['STD_WINDOWS']: # Use passed indicator_params
        volatility_col_name = f'Volatility_{max(indicator_params["STD_WINDOWS"])}'
        # Check if the generated volatility column actually exists in the DataFrame after create_features
        if volatility_col_name in df_train_period.columns:
            target_cols_to_train.append(volatility_col_name)
        else:
            update_log_func(f"Warning: Volatility column '{volatility_col_name}' not found despite STD_WINDOWS being enabled. Skipping Volatility training.")


    for target_col in target_cols_to_train:
        # Check if the target column actually exists in the DataFrame before attempting to train
        if target_col not in df_train_period.columns:
            update_log_func(f"Warning: Target column '{target_col}' not found in training data. Skipping model training for it. This might happen if its corresponding indicator was disabled.")
            continue

        update_log_func(f"Training {model_choice} for {target_col}...")
        model, scaler, features, residuals = _train_single_model(df_train_period.copy(), target_col, model_choice, perform_tuning, update_log_func)
        if model:
            trained_models[target_col] = {
                'model': model, 
                'scaler': scaler, 
                'features': features, 
                'target_col': target_col,
                'residuals': residuals # Store residuals
            }
            update_log_func(f"✓ {model_choice} model for {target_col} trained.")

            # Calculate validation metrics if there are residuals
            if residuals is not None and len(residuals) > 0:
                mae_val = np.mean(np.abs(residuals))
                rmse_val = np.sqrt(np.mean(residuals**2))
                validation_metrics[target_col] = {'MAE': mae_val, 'RMSE': rmse_val}
                update_log_func(f"Validation MAE for {target_col}: {mae_val:.4f}, RMSE: {rmse_val:.4f}")
            else:
                validation_metrics[target_col] = {'MAE': np.nan, 'RMSE': np.nan}
                update_log_func(f"No valid residuals to calculate validation metrics for {target_col}.")
        else:
            update_log_func(f"✗ Failed to train {model_choice} model for {target_col}.")
            validation_metrics[target_col] = {'MAE': np.nan, 'RMSE': np.nan}


    return trained_models, validation_metrics

def _predict_single_model_raw(model_info, df_features_predict, confidence_level_pct=None):
    """
    Generates raw numerical predictions and optionally confidence intervals from a single model.
    Returns a tuple of numpy arrays: (prediction_points, lower_bounds, upper_bounds)
    """
    model = model_info['model']
    scaler = model_info['scaler']
    features = model_info['features']
    residuals = model_info.get('residuals') # Get stored residuals

    # Handle empty or invalid input immediately
    if model is None or scaler is None or not features or df_features_predict.empty:
        # Return arrays of NaNs with the correct length if prediction input is empty
        num_rows = len(df_features_predict) if not df_features_predict.empty else 0
        return np.full(num_rows, np.nan), np.full(num_rows, np.nan), np.full(num_rows, np.nan)

    # Reindex to ensure order and presence of all features the model was trained on
    X_predict = df_features_predict.reindex(columns=features, fill_value=0).copy()

    # If, after reindexing, there are still NaN in features or no features
    if X_predict.isnull().values.any() or X_predict.shape[1] == 0:
        num_rows = len(df_features_predict)
        return np.full(num_rows, np.nan), np.full(num_rows, np.nan), np.full(num_rows, np.nan)

    try:
        X_scaled_predict = scaler.transform(X_predict)
        predictions = model.predict(X_scaled_predict)
        
        lower_bounds = np.full_like(predictions, np.nan)
        upper_bounds = np.full_like(predictions, np.nan)

        if confidence_level_pct is not None and residuals is not None and len(residuals) > 1:
            # Calculate standard deviation of residuals
            std_err = np.std(residuals)
            
            # Degrees of freedom (n - 1), common for sample standard deviation
            df = len(residuals) - 1 
            if df > 0:
                # Calculate t-score for the desired confidence level
                alpha = 1 - (confidence_level_pct / 100)
                t_score = t.ppf(1 - alpha / 2, df) # Two-tailed t-score

                margin_of_error = t_score * std_err
                lower_bounds = predictions - margin_of_error
                upper_bounds = predictions + margin_of_error

        return predictions, lower_bounds, upper_bounds
    except Exception as e:
        # It's generally good practice to log the actual exception for debugging
        # print(f"Error during prediction in _predict_single_model_raw: {e}") 
        num_rows = len(df_features_predict)
        return np.full(num_rows, np.nan), np.full(num_rows, np.nan), np.full(num_rows, np.nan)


def generate_predictions_pipeline(df_features_input, trained_models_dict, update_log_func, confidence_level_pct=None):
    """
    Generates predictions and confidence intervals for multiple target types using pre-trained models.
    Returns a dictionary of DataFrames, one for each target type, each containing 'Date', 
    'Actual {Target}', 'Predicted {Target}', 'Predicted {Target} Lower', 'Predicted {Target} Upper'.
    
    Removed pred_df.dropna() to allow NaN predictions to propagate for better debugging and display.
    """
    predicted_dfs = {}
    
    # Ensure df_features_input is not empty and has a 'Date' column
    if df_features_input.empty or 'Date' not in df_features_input.columns:
        update_log_func("Input DataFrame for prediction pipeline is empty or missing 'Date' column.")
        return predicted_dfs

    for target_type, model_info in trained_models_dict.items():
        if not model_info or not model_info.get('model'):
            update_log_func(f"Skipping prediction for {target_type}: Model not trained or missing info.")
            # Return a DataFrame with NaNs for this target if the model is not available
            pred_df = pd.DataFrame({
                'Date': df_features_input['Date'].values,
                f'Actual {target_type}': df_features_input[model_info.get('target_col', target_type)].values if model_info and model_info.get('target_col') in df_features_input.columns else np.full(len(df_features_input), np.nan),
                f'Predicted {target_type}': np.full(len(df_features_input), np.nan),
                f'Predicted {target_type} Lower': np.full(len(df_features_input), np.nan),
                f'Predicted {target_type} Upper': np.full(len(df_features_input), np.nan)
            })
            predicted_dfs[target_type] = pred_df
            continue

        predictions_points, lower_bounds, upper_bounds = _predict_single_model_raw(model_info, df_features_input.copy(), confidence_level_pct)

        # Ensure predictions_points, lower_bounds, upper_bounds are arrays of the same length as df_features_input
        # This handles cases where _predict_single_model_raw might return empty arrays or arrays of different lengths
        # due to internal filtering if not properly handled there.
        # However, _predict_single_model_raw is designed to always return arrays of num_rows length.
        num_input_rows = len(df_features_input)
        if predictions_points.size != num_input_rows:
            predictions_points = np.full(num_input_rows, np.nan)
        if lower_bounds.size != num_input_rows:
            lower_bounds = np.full(num_input_rows, np.nan)
        if upper_bounds.size != num_input_rows:
            upper_bounds = np.full(num_input_rows, np.nan)

        pred_df = pd.DataFrame({
            'Date': df_features_input['Date'].values,
            f'Actual {target_type}': df_features_input[model_info['target_col']].values if model_info['target_col'] in df_features_input.columns else np.full_like(predictions_points, np.nan),
            f'Predicted {target_type}': predictions_points,
            f'Predicted {target_type} Lower': lower_bounds,
            f'Predicted {target_type} Upper': upper_bounds
        })
        
        # --- IMPORTANT CHANGE: Removed .dropna(subset=[f'Predicted {target_type}'], inplace=True) ---
        # This allows NaN predictions to propagate, which is necessary for the UI to display them as "N/A"
        # rather than completely skipping the entry.

        if not pred_df.empty:
            # Recalculate difference only on valid actuals
            actual_col = f'Actual {target_type}'
            predicted_col = f'Predicted {target_type}'
            
            # Use .loc for setting values to avoid SettingWithCopyWarning
            # Calculate difference only where both actual and predicted are not NaN
            mask_valid_diff = pd.notna(pred_df[actual_col]) & pd.notna(pred_df[predicted_col])
            pred_df.loc[mask_valid_diff, 'Difference'] = pred_df.loc[mask_valid_diff, actual_col] - pred_df.loc[mask_valid_diff, predicted_col]
            pred_df['Difference'].fillna(np.nan, inplace=True) # Fill NaNs where actual or predicted is NaN
        else:
            pred_df['Difference'] = np.nan # If pred_df is empty from the start

        predicted_dfs[target_type] = pred_df
            
    return predicted_dfs

def make_ensemble_prediction(models_info_list, df_features_predict, update_log_func, confidence_level_pct=None):
    """
    Generates an ensemble prediction (mean) for the 'Close' price from multiple models,
    including averaged confidence intervals.
    Returns a tuple: (ensemble_point_prediction_array, ensemble_lower_bound_array, ensemble_upper_bound_array)
    """
    all_predictions_points_per_model = []
    all_predictions_lower_bounds_per_model = []
    all_predictions_upper_bounds_per_model = []
    
    num_rows = len(df_features_predict) if not df_features_predict.empty else 0
    if num_rows == 0:
        update_log_func("Input DataFrame for ensemble prediction is empty.")
        return np.full(0, np.nan), np.full(0, np.nan), np.full(0, np.nan)

    if not models_info_list:
        update_log_func("No models provided for ensemble prediction.")
        return np.full(num_rows, np.nan), np.full(num_rows, np.nan), np.full(num_rows, np.nan)

    for model_info_dict in models_info_list:
        if 'Close' in model_info_dict and model_info_dict['Close'].get('model'):
            points, lowers, uppers = _predict_single_model_raw(model_info_dict['Close'], df_features_predict.copy(), confidence_level_pct)
            
            # Ensure the returned arrays match the expected input length
            if points.size == num_rows:
                all_predictions_points_per_model.append(points)
                # Only add bounds if they are also the correct size
                if lowers.size == num_rows and uppers.size == num_rows:
                    all_predictions_lower_bounds_per_model.append(lowers)
                    all_predictions_upper_bounds_per_model.append(uppers)
            else:
                update_log_func(f"Skipping a model in ensemble due to prediction size mismatch ({points.size} vs {num_rows}).")
        else:
            update_log_func(f"Skipping a model in ensemble: 'Close' target model not found or not trained for one of the ensemble components.")

    if not all_predictions_points_per_model:
        update_log_func("No valid individual predictions to form an ensemble.")
        return np.full(num_rows, np.nan), np.full(num_rows, np.nan), np.full(num_rows, np.nan)

    # Average only non-NaN predictions. If all are NaN, result will be NaN.
    # Convert list of arrays to a 2D array, then calculate mean ignoring NaNs
    predictions_matrix = np.array(all_predictions_points_per_model)
    ensemble_point_prediction = np.nanmean(predictions_matrix, axis=0) # Use nanmean

    ensemble_lower_bound = np.full_like(ensemble_point_prediction, np.nan)
    ensemble_upper_bound = np.full_like(ensemble_point_prediction, np.nan)

    if all_predictions_lower_bounds_per_model and all_predictions_upper_bounds_per_model and confidence_level_pct is not None:
        lower_bounds_matrix = np.array(all_predictions_lower_bounds_per_model)
        upper_bounds_matrix = np.array(all_predictions_upper_bounds_per_model)
        ensemble_lower_bound = np.nanmean(lower_bounds_matrix, axis=0) # Use nanmean
        ensemble_upper_bound = np.nanmean(upper_bounds_matrix, axis=0) # Use nanmean
    
    return ensemble_point_prediction, ensemble_lower_bound, ensemble_upper_bound


def perform_walk_forward_backtesting(df_full_features, models_to_compare, perform_tuning, 
                                     initial_train_period_days, step_forward_days,
                                     current_indicator_params, # Pass indicator parameters
                                     update_log_func):
    """
    Performs walk-forward backtesting for selected models.
    Trains models on an expanding window and predicts the next 'step_forward_days'.
    Returns a DataFrame of actual vs. predicted values and evaluation metrics.
    """
    backtest_results = []
    evaluation_metrics = []
    
    df_full_features = df_full_features.sort_values('Date').reset_index(drop=True)
    
    if len(df_full_features) < initial_train_period_days + step_forward_days:
        update_log_func(f"Not enough data for walk-forward backtesting. Need at least {initial_train_period_days + step_forward_days} days.")
        return pd.DataFrame(), pd.DataFrame()

    # Iterate through the data for walk-forward validation
    # The training window expands.
    
    # The minimum index to start predicting from
    start_predict_idx = initial_train_period_days

    for i in range(start_predict_idx, len(df_full_features), step_forward_days):
        # Define the training window
        # The training data will always start from the beginning of df_full_features
        # and end at `i - 1` to prevent data leakage.
        train_df = df_full_features.iloc[:i].copy()
        
        # Define the prediction window (actuals for this window)
        # Predict `step_forward_days` ahead, starting from index `i`.
        # Ensure the slice does not exceed the DataFrame bounds.
        predict_df = df_full_features.iloc[i:min(i + step_forward_days, len(df_full_features))].copy()

        if train_df.empty or predict_df.empty:
            update_log_func(f"Skipping iteration at index {i}: empty train or predict data.")
            continue

        current_train_end_date = train_df['Date'].iloc[-1]
        
        update_log_func(f"Walk-forward step: Training from {train_df['Date'].iloc[0].strftime('%Y-%m-%d')} to {current_train_end_date.strftime('%Y-%m-%d')} for prediction starting {predict_df['Date'].iloc[0].strftime('%Y-%m-%d')}")

        for model_name in models_to_compare:
            # Train the model for 'Close' target
            trained_model_info, _ = train_models_pipeline(train_df, model_name, perform_tuning, update_log_func, current_indicator_params) # Pass indicator_params
            
            close_model_info = trained_model_info.get('Close')
            if not close_model_info or not close_model_info.get('model'):
                update_log_func(f"Model {model_name} for Close price not trained in this walk-forward step.")
                continue

            # Generate predictions for the current prediction window
            # Use generate_predictions_pipeline for walk-forward, it's designed for multiple rows
            preds_dict_wf = generate_predictions_pipeline(predict_df.copy(), {'Close': close_model_info}, update_log_func, confidence_level_pct=None)
            
            predictions_raw = np.array([])
            if 'Close' in preds_dict_wf and not preds_dict_wf['Close'].empty:
                predictions_raw = preds_dict_wf['Close']['Predicted Close'].values
            
            if predictions_raw.size > 0:
                actuals = predict_df['Close'].values
                
                # Align predictions and actuals by date
                temp_results_df = pd.DataFrame({
                    'Date': predict_df['Date'],
                    'Actual Close': actuals,
                    'Predicted Close': predictions_raw
                })
                temp_results_df['Model'] = model_name
                temp_results_df['Train End Date'] = current_train_end_date
                
                backtest_results.append(temp_results_df)

                # Calculate metrics for this step
                # Filter out NaN predictions and actuals for metric calculation
                valid_mask_wf = ~np.isnan(predictions_raw) & ~np.isnan(actuals)
                predictions_raw_valid = predictions_raw[valid_mask_wf]
                actuals_valid = actuals[valid_mask_wf]


                if len(actuals_valid) > 0 and len(predictions_raw_valid) == len(actuals_valid):
                    mae = mean_absolute_error(actuals_valid, predictions_raw_valid)
                    rmse = np.sqrt(mean_squared_error(actuals_valid, predictions_raw_valid))
                    
                    # Get the actual closing price just before the first prediction date in `predict_df`
                    last_actual_before_predict_window = train_df['Close'].iloc[-1]
                    
                    # Calculate predicted directions relative to previous day's actual close
                    predicted_directions_for_accuracy = []
                    # Ensure predictions_raw is not empty before attempting to access elements
                    if predictions_raw.size > 0:
                        if pd.notna(predictions_raw[0]) and pd.notna(last_actual_before_predict_window):
                            if predictions_raw[0] > last_actual_before_predict_window:
                                predicted_directions_for_accuracy.append(1) # Up
                            elif predictions_raw[0] < last_actual_before_predict_window:
                                predicted_directions_for_accuracy.append(-1) # Down
                            else:
                                predicted_directions_for_accuracy.append(0) # Flat
                        else:
                            predicted_directions_for_accuracy.append(np.nan)
                        
                        for k in range(1, len(predictions_raw)):
                            if pd.notna(predictions_raw[k]) and pd.notna(actuals[k-1]): # Compare current prediction to previous *actual* in window
                                if predictions_raw[k] > actuals[k-1]:
                                    predicted_directions_for_accuracy.append(1)
                                elif predictions_raw[k] < actuals[k-1]:
                                    predicted_directions_for_accuracy.append(-1)
                                else:
                                    predicted_directions_for_accuracy.append(0)
                            else:
                                predicted_directions_for_accuracy.append(np.nan)
                                
                    predicted_directions_for_accuracy = np.array(predicted_directions_for_accuracy)

                    # Construct a full sequence for actual direction:
                    temp_actual_series = pd.Series([last_actual_before_predict_window] + actuals.tolist())
                    true_directions = np.sign(np.diff(temp_actual_series.values))

                    # Filter out NaN values for comparison
                    valid_dir_indices = ~np.isnan(true_directions) & ~np.isnan(predicted_directions_for_accuracy[:len(true_directions)])
                    
                    if np.sum(valid_dir_indices) > 0:
                        directional_accuracy = accuracy_score(true_directions[valid_dir_indices], predicted_directions_for_accuracy[:len(true_directions)][valid_dir_indices])
                    else:
                        directional_accuracy = np.nan # Mismatch in lengths or no valid points, cannot calculate properly
                        update_log_func(f"Warning: Not enough valid data points for directional accuracy in walk-forward for {model_name}.")


                    evaluation_metrics.append({
                        'Model': model_name,
                        'Period Start Date': predict_df['Date'].iloc[0],
                        'Period End Date': predict_df['Date'].iloc[-1],
                        'MAE': mae,
                        'RMSE': rmse,
                        'Directional Accuracy': directional_accuracy
                    })
                else:
                    update_log_func(f"Could not calculate metrics for {model_name} in this step due to length mismatch or empty valid data.")

    if backtest_results:
        all_backtest_df = pd.concat(backtest_results, ignore_index=True)
    else:
        all_backtest_df = pd.DataFrame(columns=['Date', 'Actual Close', 'Predicted Close', 'Model', 'Train End Date'])

    if evaluation_metrics:
        all_metrics_df = pd.DataFrame(evaluation_metrics)
    else:
        all_metrics_df = pd.DataFrame(columns=['Model', 'Period Start Date', 'Period End Date', 'MAE', 'RMSE', 'Directional Accuracy'])

    return all_backtest_df, all_metrics_df


# --- Technical Indicators / Price Levels ---
def calculate_pivot_points(df_last_day):
    """
    Calculates Pivot Points, Resistance, and Support levels for a given day.
    df_last_day should be a DataFrame with 'High', 'Low', 'Close' for a single day.
    """
    if df_last_day.empty:
        return {'PP': np.nan, 'R1': np.nan, 'R2': np.nan, 'R3': np.nan, 'S1': np.nan, 'S2': np.nan, 'S3': np.nan}

    high = df_last_day['High'].iloc[-1]
    low = df_last_day['Low'].iloc[-1]
    close = df_last_day['Close'].iloc[-1]

    pp = (high + low + close) / 3
    r1 = (2 * pp) - low
    s1 = (2 * pp) - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    r3 = high + (2 * (pp - low))
    s3 = low - (2 * (high - pp))

    return {'PP': pp, 'R1': r1, 'R2': r2, 'R3': r3, 'S1': s1, 'S2': s2, 'S3': s3}

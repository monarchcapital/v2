# utils.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from datetime import timedelta, date, datetime
from sklearn.preprocessing import StandardScaler
import os

# --- Global list for collecting training messages ---
training_messages_log = []

# Define the expected columns for the prediction log globally
PREDICTION_LOG_COLUMNS = ['prediction_generation_date', 'prediction_for_date', 'training_end_date_used', 'ticker', 'model_used', 'predicted_value', 'actual_close', 'predicted_type']

# --- Prediction Logging Functions ---
def save_prediction(ticker, prediction_for_date, predicted_value, actual_close_price, model_name, prediction_generation_date, training_end_date_used, predicted_type='Close'):
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
        'predicted_type': predicted_type
    }])
    
    # Ensure date columns are in datetime format
    for col in ['prediction_generation_date', 'prediction_for_date', 'training_end_date_used']:
        new_entry[col] = pd.to_datetime(new_entry[col])

    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        # Ensure date columns are in datetime format when reading existing
        for col in ['prediction_generation_date', 'prediction_for_date', 'training_end_date_used']:
             existing_df[col] = pd.to_datetime(existing_df[col])
        # Add 'predicted_type' column if it doesn't exist in existing_df (for backward compatibility)
        if 'predicted_type' not in existing_df.columns:
            existing_df['predicted_type'] = 'Close' # Default value for old entries

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
    predictions_dir = "monarch_predictions_data"
    file_path = os.path.join(predictions_dir, f"{ticker}_predictions.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        for col in ['prediction_generation_date', 'prediction_for_date', 'training_end_date_used']:
            df[col] = pd.to_datetime(df[col])
        # Add 'predicted_type' column if it doesn't exist (for backward compatibility)
        if 'predicted_type' not in df.columns:
            df['predicted_type'] = 'Close' # Default value for old entries
        return df
    return pd.DataFrame(columns=PREDICTION_LOG_COLUMNS) # Return empty DataFrame with correct columns


# --- Data Download and Feature Engineering ---
# @st.cache_data # Consider uncommenting this in a real app for performance
def download_data(ticker_symbol, period="5y"):
    try:
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(period=period)
        if data.empty:
            return pd.DataFrame()
        data.reset_index(inplace=True)
        # Handle cases where 'Date' might be timezone-aware (e.g., in newer yfinance versions)
        if pd.api.types.is_datetime64_any_dtype(data['Date']):
            if data['Date'].dt.tz is not None:
                data['Date'] = data['Date'].dt.tz_convert(None) # Convert to naive datetime
        return data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        return pd.DataFrame()

def create_features(df, lag_features_list, ma_windows_list, std_windows_list, rsi_window, macd_short_window, macd_long_window, macd_signal_window, bb_window, bb_std_dev, atr_window, stoch_window, stoch_smooth_window):
    df_copy = df.copy()
    
    # Sort by date to ensure correct calculation of time-series features
    df_copy = df_copy.sort_values(by='Date').reset_index(drop=True)

    # Calculate Daily Returns
    df_copy['Daily_Return'] = df_copy['Close'].pct_change()

    # Lag features
    for lag in lag_features_list:
        df_copy[f'Close_Lag_{lag}'] = df_copy['Close'].shift(lag)
        df_copy[f'Volume_Lag_{lag}'] = df_copy['Volume'].shift(lag)

    # Moving Averages
    for window in ma_windows_list:
        df_copy[f'MA_{window}'] = df_copy['Close'].rolling(window=window).mean()

    # Volatility (Standard Deviation of Close Price)
    for window in std_windows_list:
        df_copy[f'Volatility_{window}'] = df_copy['Close'].rolling(window=window).std()

    # Relative Strength Index (RSI)
    delta = df_copy['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    rs = gain / loss
    df_copy['RSI'] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    exp1 = df_copy['Close'].ewm(span=macd_short_window, adjust=False).mean()
    exp2 = df_copy['Close'].ewm(span=macd_long_window, adjust=False).mean()
    df_copy['MACD'] = exp1 - exp2
    df_copy['MACD_Signal'] = df_copy['MACD'].ewm(span=macd_signal_window, adjust=False).mean()
    df_copy['MACD_Hist'] = df_copy['MACD'] - df_copy['MACD_Signal']

    # Bollinger Bands
    df_copy['BB_Middle'] = df_copy['Close'].rolling(window=bb_window).mean()
    std_dev = df_copy['Close'].rolling(window=bb_window).std()
    df_copy['BB_Upper'] = df_copy['BB_Middle'] + (std_dev * bb_std_dev)
    df_copy['BB_Lower'] = df_copy['BB_Middle'] - (std_dev * bb_std_dev)

    # Average True Range (ATR)
    # TR = max[(High - Low), abs(High - Close_prev), abs(Low - Close_prev)]
    high_low = df_copy['High'] - df_copy['Low']
    high_close = np.abs(df_copy['High'] - df_copy['Close'].shift())
    low_close = np.abs(df_copy['Low'] - df_copy['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_copy['ATR'] = tr.ewm(span=atr_window, adjust=False).mean()

    # Stochastic Oscillator
    # %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    # %D = 3-day SMA of %K
    lowest_low = df_copy['Low'].rolling(window=stoch_window).min()
    highest_high = df_copy['High'].rolling(window=stoch_window).max()
    df_copy['%K'] = ((df_copy['Close'] - lowest_low) / (highest_high - lowest_low)) * 100
    df_copy['%D'] = df_copy['%K'].rolling(window=stoch_smooth_window).mean()

    # Day of Week, Day of Month, Day of Year (cyclical features can be added, but keeping it simple for now)
    df_copy['DayOfWeek'] = df_copy['Date'].dt.dayofweek
    df_copy['DayOfMonth'] = df_copy['Date'].dt.day
    df_copy['Month'] = df_copy['Date'].dt.month
    df_copy['Year'] = df_copy['Date'].dt.year

    # Drop rows with NaN values resulting from feature creation
    df_copy.dropna(inplace=True)
    df_copy.reset_index(drop=True, inplace=True)
    return df_copy


# --- Model Management and Training ---
def get_model(model_name):
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

def _train_single_model(df_train, target_column, model_name, perform_tuning, update_log):
    features = [col for col in df_train.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return'] and not col.startswith('BB_') and not col.startswith('MACD_Hist')]
    
    # Ensure target_column is in features if it's derived like 'Volatility_X'
    if target_column in ['Close', 'Open', 'High', 'Low', 'Volume']:
        X = df_train[features]
    else: # If target is a calculated feature like 'Volatility_X'
        if target_column not in df_train.columns:
            update_log(f"Error: Target column '{target_column}' not found in training data.")
            return None, None, None # Return None for model, scaler, features
        # If target is a feature, ensure it's not also in the features list
        X = df_train[[f for f in features if f != target_column]]
        # Update features list to reflect what was actually used
        features = X.columns.tolist()

    y = df_train[target_column]

    if X.empty or y.empty:
        update_log(f"Warning: Empty features or target for {target_column}. Cannot train.")
        return None, None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = get_model(model_name)

    if perform_tuning and model_name not in ['SVR', 'KNN', 'Linear Regression']: # SVR/KNN tuning can be very slow, Linear Regression has no params
        update_log(f"Performing hyperparameter tuning for {model_name} on {target_column}...")
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
                update_log(f"Tuning complete. Best params for {target_column}: {random_search.best_params_}")
            except Exception as e:
                update_log(f"Error during tuning for {target_column}: {e}. Using default model.")
    
    model.fit(X_scaled, y)
    return model, scaler, features

def train_models_pipeline(df_train_period, model_choice, perform_tuning, std_windows_list, update_log_func):
    trained_models = {}
    
    # Train for Close Price
    update_log_func(f"Training {model_choice} for Close Price...")
    model_close, scaler_close, features_close = _train_single_model(df_train_period.copy(), 'Close', model_choice, perform_tuning, update_log_func)
    if model_close:
        trained_models['Close'] = {'model': model_close, 'scaler': scaler_close, 'features': features_close, 'target_col': 'Close'}
        update_log_func(f"✓ {model_choice} model for Close trained.")
    else:
        update_log_func(f"✗ Failed to train {model_choice} model for Close.")

    # Train for Open Price
    update_log_func(f"Training {model_choice} for Open Price...")
    model_open, scaler_open, features_open = _train_single_model(df_train_period.copy(), 'Open', model_choice, perform_tuning, update_log_func)
    if model_open:
        trained_models['Open'] = {'model': model_open, 'scaler': scaler_open, 'features': features_open, 'target_col': 'Open'}
        update_log_func(f"✓ {model_choice} model for Open trained.")
    else:
        update_log_func(f"✗ Failed to train {model_choice} model for Open.")
    
    # Train for High Price
    update_log_func(f"Training {model_choice} for High Price...")
    model_high, scaler_high, features_high = _train_single_model(df_train_period.copy(), 'High', model_choice, perform_tuning, update_log_func)
    if model_high:
        trained_models['High'] = {'model': model_high, 'scaler': scaler_high, 'features': features_high, 'target_col': 'High'}
        update_log_func(f"✓ {model_choice} model for High trained.")
    else:
        update_log_func(f"✗ Failed to train {model_choice} model for High.")

    # Train for Low Price
    update_log_func(f"Training {model_choice} for Low Price...")
    model_low, scaler_low, features_low = _train_single_model(df_train_period.copy(), 'Low', model_choice, perform_tuning, update_log_func)
    if model_low:
        trained_models['Low'] = {'model': model_low, 'scaler': scaler_low, 'features': features_low, 'target_col': 'Low'}
        update_log_func(f"✓ {model_choice} model for Low trained.")
    else:
        update_log_func(f"✗ Failed to train {model_choice} model for Low.")

    # Train for Volatility (using the largest std window as a target)
    if std_windows_list:
        vol_target_col = f'Volatility_{max(std_windows_list)}'
        update_log_func(f"Training {model_choice} for Volatility ({vol_target_col})...")
        # Ensure the target column exists in df_train_period for Volatility
        if vol_target_col not in df_train_period.columns:
            update_log_func(f"Warning: Volatility target '{vol_target_col}' not found. Skipping Volatility model training.")
        else:
            model_vol, scaler_vol, features_vol = _train_single_model(df_train_period.copy(), vol_target_col, model_choice, perform_tuning, update_log_func)
            if model_vol:
                trained_models['Volatility'] = {'model': model_vol, 'scaler': scaler_vol, 'features': features_vol, 'target_col': vol_target_col}
                update_log_func(f"✓ {model_choice} model for Volatility trained.")
            else:
                update_log_func(f"✗ Failed to train {model_choice} model for Volatility.")
    else:
        update_log_func("No volatility windows defined. Skipping Volatility model training.")

    return trained_models

def generate_predictions_pipeline(df_features, trained_models_dict, update_log_func):
    predicted_dfs = {}
    
    for target_type, model_info in trained_models_dict.items():
        model = model_info['model']
        scaler = model_info['scaler']
        features = model_info['features']
        target_col = model_info['target_col']

        if model is None or scaler is None or not features:
            update_log_func(f"Skipping prediction for {target_type}: Model or scaler not available.")
            continue
        
        # Ensure the dataframe for prediction contains all required features
        X_predict = df_features[features]

        if X_predict.empty or X_predict.isnull().values.any():
            update_log_func(f"Warning: Empty or NaN features for {target_type} prediction. Skipping.")
            continue
        
        try:
            X_scaled_predict = scaler.transform(X_predict)
            predictions = model.predict(X_scaled_predict)

            pred_df = pd.DataFrame({
                'Date': df_features['Date'],
                f'Actual {target_type}': df_features[target_col],
                f'Predicted {target_type}': predictions
            })
            pred_df[f'Difference'] = pred_df[f'Actual {target_type}'] - pred_df[f'Predicted {target_type}']
            predicted_dfs[target_type] = pred_df
        except Exception as e:
            update_log_func(f"Error generating prediction for {target_type}: {e}")
            predicted_dfs[target_type] = pd.DataFrame() # Return empty if error
            
    return predicted_dfs


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

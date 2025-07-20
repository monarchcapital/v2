# utils.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from prophet import Prophet
try:
    import catboost as cb
    CATBOOST_INSTALLED = True
except ImportError:
    CATBOOST_INSTALLED = False
# --- RESTORED IMPORTS ---
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import uniform, randint, t
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
import streamlit as st
import logging
import warnings

warnings.filterwarnings("ignore")

import config

logger = logging.getLogger(__name__)

def parse_int_list(input_str, default_list, error_callback=None):
    """
    Parses a comma-separated string of integers into a sorted list of unique integers.
    
    Args:
        input_str (str): The comma-separated string from a text input.
        default_list (list): The default list to return on parsing failure.
        error_callback (function, optional): A function to call with an error message on failure.
    
    Returns:
        list: A sorted list of unique integers.
    """
    try:
        if not input_str: return default_list
        parsed = sorted(list(set([int(x.strip()) for x in input_str.split(',') if x.strip()])))
        return parsed if parsed else default_list
    except ValueError:
        if error_callback: error_callback(f"Invalid list input: '{input_str}'. Using defaults.")
        return default_list

@st.cache_data
def download_data(ticker_symbol, period="10y"):
    """
    Downloads historical stock data from Yahoo Finance.
    
    Args:
        ticker_symbol (str): The stock ticker symbol.
        period (str, optional): The period for which to download data (e.g., "1y", "5y", "10y"). Defaults to "10y".
    
    Returns:
        pd.DataFrame: A DataFrame with historical stock data, or an empty DataFrame on failure.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(period=period, auto_adjust=True, back_adjust=False)
        if data.empty:
            st.warning(f"No data found for ticker '{ticker_symbol}'. It might be delisted or the symbol is incorrect.")
            return pd.DataFrame()
        data.reset_index(inplace=True)
        # Standardize 'Date' column to be timezone-naive datetime objects
        if pd.api.types.is_datetime64_any_dtype(data['Date']):
            if data['Date'].dt.tz is not None:
                data['Date'] = data['Date'].dt.tz_convert(None)
        data['Date'] = pd.to_datetime(data['Date'])
        return data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        st.error(f"Error downloading data for {ticker_symbol}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600) # Cache for 1 hour
def _download_market_data_cached(tickers_tuple, period="10y"):
    """Cached helper to download data for multiple tickers."""
    if not tickers_tuple: return pd.DataFrame()
    market_data_raw = yf.download(list(tickers_tuple), period=period, progress=False, auto_adjust=True)
    if market_data_raw.empty: return pd.DataFrame()
    # Handle single vs multiple tickers
    if len(tickers_tuple) == 1:
        market_data_close = market_data_raw[['Close']]
        market_data_close.columns = list(tickers_tuple)
        return market_data_close
    return market_data_raw['Close'].dropna(axis=1, how='all')

def add_market_data_features(df, period, update_log_func, selected_tickers=None):
    """
    Adds features from global market indices to the main DataFrame.
    
    Args:
        df (pd.DataFrame): The primary stock data DataFrame.
        period (str): The historical period to fetch for the indices.
        update_log_func (function): Callback function for logging updates.
        selected_tickers (list, optional): List of global index tickers to add.
    
    Returns:
        pd.DataFrame: The DataFrame merged with global market features.
    """
    if df.empty or not selected_tickers: return df
    if update_log_func: update_log_func(f"Fetching data for {len(selected_tickers)} global indices...")
    tickers_tuple = tuple(sorted(selected_tickers)) # Use tuple for caching
    market_data_close = _download_market_data_cached(tickers_tuple=tickers_tuple, period=period)
    
    if market_data_close.empty:
        if update_log_func: update_log_func("⚠️ Could not fetch any global market data.")
        return df

    ticker_to_name_map = {v: k for k, v in config.GLOBAL_MARKET_TICKERS.items()}
    rename_dict = {ticker_col: f"{ticker_to_name_map.get(ticker_col, ticker_col).replace(' ', '_').replace('^', '')}_Close" for ticker_col in market_data_close.columns}
    market_data_renamed = market_data_close.rename(columns=rename_dict)
    
    df['Date'] = pd.to_datetime(df['Date'])
    market_data_renamed.index = pd.to_datetime(market_data_renamed.index)
    if market_data_renamed.index.tz is not None: market_data_renamed.index = market_data_renamed.index.tz_localize(None)

    # Use merge_asof for robust merging of time-series data
    df_merged = pd.merge_asof(df.sort_values('Date'), market_data_renamed.sort_index(), left_on='Date', right_index=True, direction='backward')
    
    for col in market_data_renamed.columns:
        df_merged[f"{col.replace('_Close', '')}_Return"] = df_merged[col].pct_change()
        
    if update_log_func: update_log_func("✓ Global market data features merged.")
    return df_merged

@st.cache_data
def add_fundamental_features(df, ticker, selected_fundamentals, _update_log_func=None):
    """
    Adds selected fundamental metrics as features to the DataFrame,
    including historical ratios derived from financial statements.
    
    Args:
        df (pd.DataFrame): The primary stock data DataFrame.
        ticker (str): The stock ticker to fetch fundamentals for.
        selected_fundamentals (dict): A dictionary mapping friendly names to yfinance keys for static fundamentals
                                     and friendly names to internal column names for derived historical fundamentals.
        _update_log_func (function, optional): Callback for logging.
    
    Returns:
        pd.DataFrame: The DataFrame with added fundamental features.
    """
    if df.empty or not ticker: return df
    
    try:
        if _update_log_func: _update_log_func(f"Fetching fundamental data for {ticker}...")
        stock = yf.Ticker(ticker)
        info = stock.info # Latest info for static fundamentals

        # --- New: Fetch Historical Financials ---
        quarterly_financials = stock.quarterly_financials
        quarterly_balance_sheet = stock.quarterly_balance_sheet
        
        # Prepare a DataFrame for fundamental features
        df_fundamentals_to_merge = pd.DataFrame(index=df['Date'])
        added_count = 0

        # Add static fundamental features from stock.info (as before)
        for friendly_name, yf_key in selected_fundamentals.items():
            if yf_key in config.FUNDAMENTAL_METRICS.values(): # Check if it's a static fundamental
                value = info.get(yf_key)
                if value is not None and isinstance(value, (int, float)):
                    df_fundamentals_to_merge[friendly_name.replace(' ', '_')] = value
                    added_count += 1
                elif _update_log_func: _update_log_func(f"⚠️ Static fundamental '{friendly_name}' not available for {ticker}.")

        # --- Derive Historical Ratios from Financial Statements ---
        if not quarterly_financials.empty and not quarterly_balance_sheet.empty and not df.empty:
            try:
                # Ensure financial statement indices are datetime and timezone-naive
                qf_t = quarterly_financials.T
                qf_t.index = pd.to_datetime(qf_t.index).tz_localize(None) if qf_t.index.tz is not None else pd.to_datetime(qf_t.index)
                qbs_t = quarterly_balance_sheet.T
                qbs_t.index = pd.to_datetime(qbs_t.index).tz_localize(None) if qbs_t.index.tz is not None else pd.to_datetime(qbs_t.index)
                
                # Calculate TTM EPS and Revenue (trailing 4 quarters)
                if 'Diluted EPS' in qf_t.columns:
                    qf_t['TTM_DilutedEPS'] = qf_t['Diluted EPS'].rolling(window=4, min_periods=4).sum()
                if 'Total Revenue' in qf_t.columns:
                    qf_t['TTM_TotalRevenue'] = qf_t['Total Revenue'].rolling(window=4, min_periods=4).sum()
                
                # Merge with our main DataFrame's dates for ratio calculation
                # Use merge_asof to find the latest available quarterly data for each trading day
                df_temp_for_ratios = df[['Date', 'Close']].copy()
                df_temp_for_ratios['Date'] = pd.to_datetime(df_temp_for_ratios['Date']) # Ensure Date is datetime

                # Merge TTM EPS and Revenue
                df_temp_for_ratios = pd.merge_asof(df_temp_for_ratios.sort_values('Date'), 
                                                   qf_t[['TTM_DilutedEPS', 'TTM_TotalRevenue']].sort_index(),
                                                   left_on='Date', right_index=True, direction='backward')
                # Merge Balance Sheet items
                df_temp_for_ratios = pd.merge_asof(df_temp_for_ratios.sort_values('Date'), 
                                                   qbs_t[['Total Debt', 'Total Stockholder Equity']].sort_index(),
                                                   left_on='Date', right_index=True, direction='backward')

                # Calculate historical ratios
                shares_outstanding = info.get('sharesOutstanding', 1) 
                if shares_outstanding == 0: shares_outstanding = 1 # Avoid division by zero

                if 'TTM_DilutedEPS' in df_temp_for_ratios.columns and 'Historical_PE_Ratio' in selected_fundamentals.keys():
                    df_temp_for_ratios['Historical_PE_Ratio'] = df_temp_for_ratios['Close'] / df_temp_for_ratios['TTM_DilutedEPS']
                    df_fundamentals_to_merge['Historical_PE_Ratio'] = df_temp_for_ratios.set_index('Date')['Historical_PE_Ratio']
                    added_count += 1

                if 'TTM_TotalRevenue' in df_temp_for_ratios.columns and 'Historical_PS_Ratio' in selected_fundamentals.keys():
                    df_temp_for_ratios['Historical_PS_Ratio'] = df_temp_for_ratios['Close'] / (df_temp_for_ratios['TTM_TotalRevenue'] / shares_outstanding)
                    df_fundamentals_to_merge['Historical_PS_Ratio'] = df_temp_for_ratios.set_index('Date')['Historical_PS_Ratio']
                    added_count += 1
                
                if 'Total Debt' in df_temp_for_ratios.columns and 'Total Stockholder Equity' in df_temp_for_ratios.columns and 'Historical_Debt_to_Equity' in selected_fundamentals.keys():
                    df_temp_for_ratios['Historical_Debt_to_Equity'] = df_temp_for_ratios['Total Debt'] / df_temp_for_ratios['Total Stockholder Equity']
                    df_fundamentals_to_merge['Historical_Debt_to_Equity'] = df_temp_for_ratios.set_index('Date')['Historical_Debt_to_Equity']
                    added_count += 1

                # Clean up infinite values after ratio calculation
                df_fundamentals_to_merge.replace([np.inf, -np.inf], np.nan, inplace=True)

            except Exception as e:
                if _update_log_func: _update_log_func(f"⚠️ Error deriving historical ratios for {ticker}: {e}")
        
        if df_fundamentals_to_merge.empty:
            if _update_log_func: _update_log_func("No fundamental data could be added.")
            return df

        # Merge the original df with the new fundamental features
        # Use left merge to keep all original dates from df
        df_merged = pd.merge(df, df_fundamentals_to_merge, left_on='Date', right_index=True, how='left')
        
        # Forward fill the newly added historical columns as they are time-series based
        # Static fundamentals are already constants per stock, so ffill after merge is fine for them too.
        newly_added_cols = [col for col in df_fundamentals_to_merge.columns if col not in df.columns]
        df_merged[newly_added_cols] = df_merged[newly_added_cols].ffill()

        if _update_log_func: _update_log_func(f"✓ {added_count} fundamental features merged.")
        return df_merged
    except Exception as e:
        if _update_log_func: _update_log_func(f"❌ Error fetching or processing fundamentals for {ticker}: {e}")
        return df

def create_features(df, indicator_params):
    """
    Creates a rich set of technical indicator features from the base OHLCV data.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data.
        indicator_params (dict): Dictionary with parameters for each technical indicator.
    
    Returns:
        pd.DataFrame: DataFrame enriched with technical features.
    """
    df_copy = df.copy().sort_values(by='Date').reset_index(drop=True)
    
    # Basic Features
    df_copy['Daily_Return'] = df_copy['Close'].pct_change()
    df_copy['Log_Return'] = np.log(df_copy['Close'] / df_copy['Close'].shift(1))
    df_copy['Price_Range'] = df_copy['High'] - df_copy['Low']
    df_copy['Volume_Change'] = df_copy['Volume'].pct_change()
    for window in [5, 10, 20]:
        df_copy[f'Volume_MA_{window}'] = df_copy['Volume'].rolling(window=window).mean()

    # Time-based Features
    df_copy['DayOfWeek'] = df_copy['Date'].dt.dayofweek
    df_copy['Month'] = df_copy['Date'].dt.month
    df_copy['Year'] = df_copy['Date'].dt.year

    # Trend Indicators
    if 'LAG_FEATURES' in indicator_params and indicator_params.get('LAG_FEATURES'):
        for lag in indicator_params['LAG_FEATURES']:
            df_copy[f'Close_Lag_{lag}'] = df_copy['Close'].shift(lag)
    if 'MA_WINDOWS' in indicator_params and indicator_params.get('MA_WINDOWS'):
        for window in indicator_params['MA_WINDOWS']:
            df_copy[f'MA_{window}'] = df_copy['Close'].rolling(window=window).mean()
    macd_params = ['MACD_SHORT_WINDOW', 'MACD_LONG_WINDOW', 'MACD_SIGNAL_WINDOW']
    if all(p in indicator_params and indicator_params.get(p) for p in macd_params):
        exp1 = df_copy['Close'].ewm(span=indicator_params['MACD_SHORT_WINDOW'], adjust=False).mean()
        exp2 = df_copy['Close'].ewm(span=indicator_params['MACD_LONG_WINDOW'], adjust=False).mean()
        df_copy['MACD'] = exp1 - exp2
        df_copy['MACD_Signal'] = df_copy['MACD'].ewm(span=indicator_params['MACD_SIGNAL_WINDOW'], adjust=False).mean()
    if 'ADX_WINDOW' in indicator_params and indicator_params.get('ADX_WINDOW'):
        adx_window = int(indicator_params['ADX_WINDOW'])
        high_diff = df_copy['High'].diff()
        low_diff = df_copy['Low'].diff()
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        tr1 = df_copy['High'] - df_copy['Low']
        tr2 = np.abs(df_copy['High'] - df_copy['Close'].shift())
        tr3 = np.abs(df_copy['Low'] - df_copy['Close'].shift())
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.ewm(span=adx_window, adjust=False).mean()
        plus_di = 100 * (pd.Series(plus_dm).ewm(span=adx_window, adjust=False).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(span=adx_window, adjust=False).mean() / atr)
        dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
        df_copy['ADX'] = dx.ewm(span=adx_window, adjust=False).mean()


    # Momentum Indicators
    if 'RSI_WINDOW' in indicator_params and indicator_params.get('RSI_WINDOW'):
        rsi_window = int(indicator_params['RSI_WINDOW'])
        delta = df_copy['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
        rs = gain / loss
        df_copy['RSI'] = 100 - (100 / (1 + rs))
    if 'ROC_WINDOW' in indicator_params and indicator_params.get('ROC_WINDOW'):
        for window in indicator_params['ROC_WINDOW']:
            df_copy[f'ROC_{window}'] = df_copy['Close'].pct_change(periods=window) * 100
    stoch_params = ['STOCHASTIC_K_WINDOW', 'STOCHASTIC_D_WINDOW']
    if all(p in indicator_params and indicator_params.get(p) for p in stoch_params):
        k_window = int(indicator_params['STOCHASTIC_K_WINDOW'])
        d_window = int(indicator_params['STOCHASTIC_D_WINDOW'])
        low_min = df_copy['Low'].rolling(window=k_window).min()
        high_max = df_copy['High'].rolling(window=k_window).max()
        df_copy['%K'] = 100 * ((df_copy['Close'] - low_min) / (high_max - low_min))
        df_copy['%D'] = df_copy['%K'].rolling(window=d_window).mean()
    if 'WILLIAMS_R_WINDOW' in indicator_params and indicator_params.get('WILLIAMS_R_WINDOW'):
        wr_window = int(indicator_params['WILLIAMS_R_WINDOW'])
        high_max_wr = df_copy['High'].rolling(window=wr_window).max()
        low_min_wr = df_copy['Low'].rolling(window=wr_window).min()
        df_copy['Williams_%R'] = -100 * ((high_max_wr - df_copy['Close']) / (high_max_wr - low_min_wr))


    # Volume Indicators
    if 'OBV' in indicator_params and indicator_params.get('OBV'):
        obv = np.where(df_copy['Close'] > df_copy['Close'].shift(1), df_copy['Volume'],
              np.where(df_copy['Close'] < df_copy['Close'].shift(1), -df_copy['Volume'], 0)).cumsum()
        df_copy['OBV'] = obv
    if 'CMF_WINDOW' in indicator_params and indicator_params.get('CMF_WINDOW'):
        cmf_window = int(indicator_params['CMF_WINDOW'])
        mfm = ((df_copy['Close'] - df_copy['Low']) - (df_copy['High'] - df_copy['Close'])) / (df_copy['High'] - df_copy['Low'])
        mfm = mfm.fillna(0)
        mfv = mfm * df_copy['Volume']
        df_copy['CMF'] = mfv.rolling(window=cmf_window).sum() / df_copy['Volume'].rolling(window=cmf_window).sum()


    # Volatility Indicators
    if 'BB_WINDOW' in indicator_params and 'BB_STD_DEV' in indicator_params:
        bb_window = int(indicator_params['BB_WINDOW'])
        ma = df_copy['Close'].rolling(window=bb_window).mean()
        std = df_copy['Close'].rolling(window=bb_window).std()
        df_copy['BB_Upper'] = ma + (std * indicator_params['BB_STD_DEV'])
        df_copy['BB_Lower'] = ma - (std * indicator_params['BB_STD_DEV'])
    if 'ATR_WINDOW' in indicator_params and indicator_params.get('ATR_WINDOW'):
        atr_window = int(indicator_params['ATR_WINDOW'])
        high_low = df_copy['High'] - df_copy['Low']
        high_close = np.abs(df_copy['High'] - df_copy['Close'].shift())
        low_close = np.abs(df_copy['Low'] - df_copy['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df_copy['ATR'] = tr.ewm(span=atr_window, adjust=False).mean()

    # Final cleanup
    df_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_copy.dropna(inplace=True)
    return df_copy.reset_index(drop=True)

def get_model(model_name):
    """Initializes and returns a machine learning model instance based on its name."""
    models = {
        'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(random_state=42, objective='reg:squarederror', n_jobs=-1),
        'LightGBM': LGBMRegressor(random_state=42, n_jobs=-1),
        'Prophet': Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Linear Regression': LinearRegression(n_jobs=-1),
        'KNN': KNeighborsRegressor(n_jobs=-1),
        'Decision Tree': DecisionTreeRegressor(random_state=42)
    }
    if model_name == 'CatBoost' and CATBOOST_INSTALLED:
        models['CatBoost'] = cb.CatBoostRegressor(random_state=42, verbose=0, thread_count=-1)
    return models.get(model_name)

def _train_single_model(df_train, target_column, model_name, perform_tuning, update_log_func):
    """
    Helper function to train a single model for a specific target column.
    
    Note: For the 'Close' price, the model is trained on daily returns (a stationary series)
    for better performance, while for O-H-L, it's trained on the absolute values.
    """
    # Use 'Daily_Return' as the target for 'Close' price prediction
    is_close_target = (target_column == 'Close')
    actual_target_col = 'Daily_Return' if is_close_target else target_column
    
    # Ensure the actual_target_col exists and is not all NaNs
    if actual_target_col not in df_train.columns or df_train[actual_target_col].isnull().all():
        update_log_func(f"Error: Target column '{actual_target_col}' not available or all NaNs in training data.")
        return None, None, None, None

    market_close_cols = [f"{k.replace(' ', '_').replace('^','')}_Close" for k in config.GLOBAL_MARKET_TICKERS.keys()]
    # Exclude the original 'Close' column if we are predicting 'Daily_Return'
    exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 'Log_Return'] + market_close_cols
    
    # Ensure the actual_target_col is not in features
    features = [col for col in df_train.columns if col not in exclude_cols and col != actual_target_col and pd.api.types.is_numeric_dtype(df_train[col])]
    
    X, y = df_train[features], df_train[actual_target_col]
    
    if X.empty or y.empty: 
        update_log_func(f"Error: Empty features or target for {model_name} on {target_column}.")
        return None, None, None, None

    if model_name == 'Prophet':
        # Prophet requires 'ds' (Date) and 'y' (target)
        prophet_df = df_train[['Date', actual_target_col]].rename(columns={'Date': 'ds', actual_target_col: 'y'})
        # Prophet handles its own scaling internally
        model = get_model(model_name).fit(prophet_df)
        # Residuals for Prophet are y - yhat
        residuals = prophet_df['y'] - model.predict(prophet_df)['yhat'].values
        return model, None, features, residuals

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = get_model(model_name)

    # --- RESTORED: Placeholder for Hyperparameter Tuning ---
    if perform_tuning:
        # NOTE: This section is a placeholder for the tuning logic.
        # A full implementation would define parameter grids (param_dist) for each model
        # and run RandomizedSearchCV. For now, it just trains with default parameters.
        update_log_func(f"Hyperparameter tuning enabled for {model_name}, but using default training for now.")
        # Example:
        # tscv = TimeSeriesSplit(n_splits=5)
        # param_dist = {
        #     'n_estimators': randint(50, 200),
        #     'max_depth': randint(3, 10),
        #     'min_samples_split': randint(2, 10)
        # } # Example for RandomForest
        # random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42)
        # random_search.fit(X_scaled, y)
        # model = random_search.best_estimator_
        model.fit(X_scaled, y)
    else:
        model.fit(X_scaled, y)
    
    # Return residuals for confidence interval calculation
    residuals = y - model.predict(X_scaled)
    return model, scaler, features, residuals

def train_models_pipeline(df_train_period, model_choice, perform_tuning, update_log_func, indicator_params):
    """
    A pipeline to train models for each of the OHLC targets.
    
    Returns:
        dict: A dictionary containing the trained model, scaler, features, and residuals for each target.
    """
    trained_models = {}
    for target_col in ['Close', 'Open', 'High', 'Low']:
        if model_choice == 'Prophet' and target_col != 'Close': continue # Prophet only predicts 'y'
        
        model, scaler, features, residuals = _train_single_model(df_train_period.copy(), target_col, model_choice, perform_tuning, update_log_func)
        if model:
            trained_models[target_col] = {
                'model': model, 
                'scaler': scaler, 
                'features': features, 
                'residuals': residuals
            }
    return trained_models, {}

def _predict_single_model_raw(model_info, df_features_predict, confidence_level_pct=None):
    """Helper function to make a raw prediction and calculate confidence intervals."""
    model, scaler, features, residuals = model_info['model'], model_info['scaler'], model_info['features'], model_info.get('residuals')
    if model is None: return np.full(len(df_features_predict), np.nan), np.full(len(df_features_predict), np.nan), np.full(len(df_features_predict), np.nan)
    
    # Ensure df_features_predict has the required 'Date' column for Prophet
    if isinstance(model, Prophet):
        future = pd.DataFrame({'ds': df_features_predict['Date'].values})
        forecast = model.predict(future)
        predictions = forecast['yhat'].values
        
        lower_bounds, upper_bounds = np.full_like(predictions, np.nan), np.full_like(predictions, np.nan)
        if confidence_level_pct is not None:
            # Prophet provides default confidence intervals (e.g., 80% and 95%)
            # To get custom confidence levels, Prophet needs to be configured with `interval_width`
            # during initialization. For simplicity, we'll use its default CI if available,
            # or fall back to residual method if a custom CI is requested and Prophet's isn't directly matching.
            if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
                # Prophet's default interval_width is 0.95. If a different CI is needed,
                # Prophet model itself would need to be re-initialized with that width.
                # For now, we'll just use what Prophet provides directly.
                lower_bounds = forecast['yhat_lower'].values
                upper_bounds = forecast['yhat_upper'].values
            elif residuals is not None and len(residuals) > 1: # Fallback to residual method
                margin_of_error = t.ppf(1 - (1 - confidence_level_pct / 100) / 2, len(residuals) - 1) * np.std(residuals)
                lower_bounds, upper_bounds = predictions - margin_of_error, predictions + margin_of_error
        
    else: # For non-Prophet models
        X_predict = df_features_predict.reindex(columns=features, fill_value=0)
        predictions = model.predict(scaler.transform(X_predict))
        
        lower_bounds, upper_bounds = np.full_like(predictions, np.nan), np.full_like(predictions, np.nan)
        if confidence_level_pct is not None and residuals is not None and len(residuals) > 1:
            margin_of_error = t.ppf(1 - (1 - confidence_level_pct / 100) / 2, len(residuals) - 1) * np.std(residuals)
            lower_bounds, upper_bounds = predictions - margin_of_error, predictions + margin_of_error
            
    return predictions, lower_bounds, upper_bounds

def generate_predictions_pipeline(df_to_predict, trained_models, log_func, confidence_level_pct=None):
    """
    A pipeline to generate predictions for all OHLC targets from their trained models.
    
    Returns:
        dict: A dictionary of DataFrames, one for each predicted target.
    """
    all_predictions = {}
    for target_type, model_info in trained_models.items():
        if not model_info or not model_info.get('model'): continue
        
        predictions_points_raw, lower_bounds_raw, upper_bounds_raw = _predict_single_model_raw(model_info, df_to_predict.copy(), confidence_level_pct)
        
        # If the target was 'Close', convert the predicted return back to a price level
        if target_type == 'Close':
            # Ensure last_actual_close is a Series to handle multiple prediction points if df_to_predict has multiple rows
            last_actual_close = df_to_predict['Close'].values 
            
            predictions_points = last_actual_close * (1 + predictions_points_raw)
            if pd.notna(lower_bounds_raw).all(): lower_bounds = last_actual_close * (1 + lower_bounds_raw)
            if pd.notna(upper_bounds_raw).all(): upper_bounds = last_actual_close * (1 + upper_bounds_raw)
            else: lower_bounds, upper_bounds = np.full_like(predictions_points, np.nan), np.full_like(predictions_points, np.nan)
        else: # For Open, High, Low, use raw predictions directly
            predictions_points = predictions_points_raw
            lower_bounds = lower_bounds_raw
            upper_bounds = upper_bounds_raw

        all_predictions[target_type] = pd.DataFrame({
            'Date': df_to_predict['Date'].values,
            f'Predicted {target_type}': predictions_points,
            f'Predicted {target_type} Lower': lower_bounds,
            f'Predicted {target_type} Upper': upper_bounds
        })
    return all_predictions

def make_ensemble_prediction(models_info_list, df_features_predict, update_log_func, confidence_level_pct=None):
    """
    Generates an ensemble prediction by averaging the outputs of multiple models.
    
    Returns:
        tuple: The ensemble point prediction, lower bound, and upper bound.
    """
    all_points_raw, all_lowers_raw, all_uppers_raw = [], [], []
    for model_info_dict in models_info_list:
        if 'Close' in model_info_dict and model_info_dict['Close'].get('model'):
            # Get raw predictions (which for 'Close' will be Daily_Return)
            points, lowers, uppers = _predict_single_model_raw(model_info_dict['Close'], df_features_predict.copy(), confidence_level_pct)
            all_points_raw.append(points); all_lowers_raw.append(lowers); all_uppers_raw.append(uppers)
            
    if not all_points_raw: return np.nan, np.nan, np.nan

    # Use nanmean to handle potential NaNs from individual models
    ensemble_point_raw = np.nanmean(np.array(all_points_raw), axis=0)
    ensemble_lower_raw = np.nanmean(np.array(all_lowers_raw), axis=0) if confidence_level_pct and all_lowers_raw else np.full_like(ensemble_point_raw, np.nan)
    ensemble_upper_raw = np.nanmean(np.array(all_uppers_raw), axis=0) if confidence_level_pct and all_uppers_raw else np.full_like(ensemble_point_raw, np.nan)
    
    # Convert raw (Daily_Return) predictions back to absolute Close price
    last_actual_close = df_features_predict['Close'].values
    ensemble_point = last_actual_close * (1 + ensemble_point_raw)
    ensemble_lower = last_actual_close * (1 + ensemble_lower_raw) if pd.notna(ensemble_lower_raw).all() else np.full_like(ensemble_point, np.nan)
    ensemble_upper = last_actual_close * (1 + ensemble_upper_raw) if pd.notna(ensemble_upper_raw).all() else np.full_like(ensemble_point, np.nan)

    return ensemble_point, ensemble_lower, ensemble_upper

def generate_iterative_forecast(raw_data, trained_models, ticker_symbol, n_future, end_date, indicator_params, update_log_func, selected_tickers=None, selected_fundamentals=None):
    """
    Generates a multi-day forecast by iteratively predicting one day at a time
    and feeding the prediction back into the feature set for the next day's prediction.
    
    Args:
        raw_data (pd.DataFrame): The raw historical data.
        trained_models (dict): The dictionary of trained models for OHLC.
        ticker_symbol (str): The stock ticker, needed for fundamental features.
        n_future (int): The number of future days to forecast.
        end_date (date): The last date of the known historical data.
        indicator_params (dict): Parameters for technical indicators.
        update_log_func (function): Callback for logging.
        selected_tickers (list, optional): Global tickers for context.
        selected_fundamentals (dict, optional): Fundamentals to include.
        
    Returns:
        pd.DataFrame: A DataFrame containing the multi-day forecast.
    """
    future_predictions_list = []
    history_for_iteration = raw_data[raw_data['Date'] <= pd.to_datetime(end_date)].copy()
    
    # Add fundamental features once at the start if they are selected
    # Ensure selected_fundamentals is passed correctly (combined static + historical)
    if selected_fundamentals and ticker_symbol:
        history_for_iteration = add_fundamental_features(history_for_iteration, ticker_symbol, selected_fundamentals, _update_log_func=update_log_func)

    for i in range(n_future):
        iter_history = history_for_iteration.copy()
        
        # Add market data features at each step to get the latest available data
        if selected_tickers:
            iter_history = add_market_data_features(iter_history, "10y", update_log_func, selected_tickers=selected_tickers)
        
        # Create features on the current history
        features_for_pred_step = create_features(iter_history, indicator_params)
        if features_for_pred_step.empty: break
        
        last_day_features = features_for_pred_step.tail(1)
        next_day_preds_dict = generate_predictions_pipeline(last_day_features, trained_models, update_log_func)
        
        # Extract the predicted values
        predicted_close = next_day_preds_dict.get('Close', {}).get('Predicted Close', pd.Series([np.nan])).iloc[0]
        if pd.isna(predicted_close): break # Stop if we can't predict a close price
        
        predicted_open = next_day_preds_dict.get('Open', {}).get('Predicted Open', pd.Series([predicted_close])).iloc[0]
        predicted_high = max(predicted_open, predicted_close, next_day_preds_dict.get('High', {}).get('Predicted High', pd.Series([predicted_close])).iloc[0])
        predicted_low = min(predicted_open, predicted_close, next_day_preds_dict.get('Low', {}).get('Predicted Low', pd.Series([predicted_close])).iloc[0])

        # Create the new row for the next day's data
        new_row = {
            'Date': last_day_features['Date'].iloc[0] + timedelta(days=1),
            'Close': predicted_close,
            'Open': predicted_open,
            'High': predicted_high,
            'Low': predicted_low,
            'Volume': last_day_features['Volume'].iloc[0] # Assume volume stays constant
        }
        
        # Carry forward the fundamental features (both static and historical)
        # Identify fundamental columns that were added to last_day_features
        fundamental_cols_in_last_day = [col for col in last_day_features.columns if col.startswith(('Historical_', 'Trailing_', 'Forward_', 'Price_to_', 'Enterprise_to_', 'Profit_Margins', 'Return_on_Equity', 'Debt_to_Equity', 'Dividend_Yield', 'Beta'))]
        for col_name in fundamental_cols_in_last_day:
            new_row[col_name] = last_day_features[col_name].iloc[0]
        
        future_predictions_list.append(new_row)
        # Append the new predicted row to the history for the next iteration
        history_for_iteration = pd.concat([history_for_iteration, pd.DataFrame([new_row])], ignore_index=True)
        
    return pd.DataFrame(future_predictions_list) if future_predictions_list else pd.DataFrame()

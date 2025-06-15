# config.py

# --- Default Feature Engineering Parameters for UI Initialization ---
# These are the default values that will appear in the Streamlit sidebar.
# They are no longer directly used by utils.py for calculation, but passed from Home.py UI.

# Each entry is (default_value, is_enabled_by_default)
# For list-based indicators (e.g., lags, MAs, STDs), default_value is the list.
# For single-value indicators (e.g., RSI, MACD windows), default_value is the number.
# For indicators with no specific window (like OBV), a boolean (True/False) can be used to indicate if it's enabled.
TECHNICAL_INDICATORS_DEFAULTS = {
    'LAG_FEATURES': ([1, 2, 3, 5, 10], True),
    'MA_WINDOWS': ([10, 20, 50, 100, 200], True), # Added 100, 200 for more flexibility
    'STD_WINDOWS': ([10, 20], True),

    # Momentum Indicators
    'RSI_WINDOW': (14, True),
    'MACD_SHORT_WINDOW': (12, True),
    'MACD_LONG_WINDOW': (26, True),
    'MACD_SIGNAL_WINDOW': (9, True),
    'STOCH_WINDOW': (14, True),
    'STOCH_SMOOTH_WINDOW': (3, True),
    'CCI_WINDOW': (20, True),
    'ROC_WINDOW': (12, True), # Rate of Change
    'ADX_WINDOW': (14, True), # Average Directional Index

    # Volatility Indicators
    'ATR_WINDOW': (14, True), # Average True Range

    # Trend-Following Indicators
    'BB_WINDOW': (20, True), # Bollinger Bands
    'BB_STD_DEV': (2.0, True),
    'PARSAR_ENABLED': (True, True), # Parabolic SAR is complex, simplify to just enabled/disabled initially
    'PARSAR_ACCELERATION': (0.02, True), # Default acceleration factor for Parabolic SAR
    'PARSAR_MAX_ACCELERATION': (0.2, True), # Default maximum acceleration factor for Parabolic SAR

    # Volume Indicators
    'OBV_ENABLED': (True, True), # On-Balance Volume
    'CMF_WINDOW': (20, True), # Chaikin Money Flow
}


# --- Model Choices ---
# This list defines the machine learning models available for selection.
MODEL_CHOICES = [
    'Random Forest',
    'XGBoost',
    'Gradient Boosting',
    'Linear Regression',
    'SVR', # Support Vector Regressor (can be slow)
    'KNN', # K-Nearest Neighbors
    'Decision Tree'
]

# --- Default Prediction Horizon ---
DEFAULT_N_FUTURE_DAYS = 15

# --- Default Comparison Period ---
DEFAULT_RECENT_DATA_FOR_COMPARISON = 90 # Number of days for model comparison chart

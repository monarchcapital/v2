# config.py - Central configuration for the application

# Defines the technical indicators available and their default parameters.
# Each key is the indicator name, and the value is a tuple: (default_parameter, default_enabled_state)
TECHNICAL_INDICATORS_DEFAULTS = {
    # Trend Indicators
    'LAG_FEATURES': ([1, 2, 3, 5, 10], True),
    'MA_WINDOWS': ([10, 20, 50, 100], True),
    'MACD_SHORT_WINDOW': (12, True),
    'MACD_LONG_WINDOW': (26, True),
    'MACD_SIGNAL_WINDOW': (9, True),
    'ADX_WINDOW': (14, True),

    # Momentum Indicators
    'RSI_WINDOW': (14, True),
    'ROC_WINDOW': ([12, 24], True),
    'STOCHASTIC_K_WINDOW': (14, True),
    'STOCHASTIC_D_WINDOW': (3, True),
    'WILLIAMS_R_WINDOW': (14, True),

    # Volume Indicators
    'OBV': (True, True),
    'CMF_WINDOW': (20, True),

    # Volatility Indicators
    'BB_WINDOW': (20, True),
    'BB_STD_DEV': (2.0, True),
    'ATR_WINDOW': (14, True)
}

# Defines the fundamental metrics available for feature selection.
# The key is the user-friendly name, and the value is the corresponding key from the yfinance 'info' dictionary.
FUNDAMENTAL_METRICS = {
    'Trailing P/E': 'trailingPE',
    'Forward P/E': 'forwardPE',
    'Price to Sales': 'priceToSalesTrailing12Months',
    'Price to Book': 'priceToBook',
    'Enterprise to Revenue': 'enterpriseToRevenue',
    'Enterprise to EBITDA': 'enterpriseToEbitda',
    'Profit Margins': 'profitMargins',
    'Return on Equity': 'returnOnEquity',
    'Debt to Equity': 'debtToEquity',
    'Dividend Yield': 'dividendYield',
    'Beta': 'beta'
}

# Defines the machine learning models available for prediction throughout the application.
MODEL_CHOICES = [
    'Random Forest',
    'XGBoost',
    'LightGBM',
    'Gradient Boosting',
    'CatBoost',
    'Decision Tree',
    'KNN',
    'Linear Regression',
    'Prophet',
]

# Defines the global market indices available for contextual analysis.
# The key is the user-friendly name, and the value is the Yahoo Finance ticker symbol.
GLOBAL_MARKET_TICKERS = {
    'S&P 500': '^GSPC',
    'Crude Oil': 'CL=F',
    'Gold': 'GC=F',
    'US Dollar Index': 'DX-Y.NYB',
    'VIX Volatility': '^VIX',
    'Bitcoin': 'BTC-USD',
    'US 10Y Treasury': '^TNX'
}

# Default settings for prediction and comparison pages.
DEFAULT_N_FUTURE_DAYS = 15
DEFAULT_RECENT_DATA_FOR_COMPARISON = 90

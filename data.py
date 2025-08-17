# data.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta # Technical Analysis library
from datetime import datetime
import traceback

# Import display_log from utils for consistent logging
from utils import display_log

def fetch_stock_data(ticker: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """
    Fetches historical stock data from Yahoo Finance using yfinance,
    with robust column handling.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL').
        start_date (datetime.date): Start date for data fetching.
        end_date (datetime.date): End date for data fetching.

    Returns:
        pd.DataFrame: Historical stock data with 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'.
                      Returns an empty DataFrame if data fetching fails or 'Close' column is missing/invalid.
    """
    display_log(f"🔄 Data Fetching Started for {ticker}...", "info")
    df = pd.DataFrame() # Initialize empty DataFrame

    try:
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        display_log(f"Attempting to download data for {ticker} from {start_str} to {end_str}...", "info")
        
        df = yf.download(ticker, start=start_str, end=end_str, progress=False, auto_adjust=False)

        if df.empty:
            display_log(f"❗ No data found for {ticker} from {start_str} to {end_str} after yfinance download. Check ticker or date range.", "warning")
            return pd.DataFrame()

        display_log(f"✅ Raw data fetched for {ticker}. Initial Columns (before processing): {df.columns.tolist()}", "info")
        display_log(f"Raw df head:\n{df.head().to_string()}", "info") # Added debug log

        # --- CRITICAL REVISION 3.0: Direct Column Standardization ---
        
        # Step 1: Flatten MultiIndex columns.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            display_log(f"Flattened MultiIndex columns. Intermediate columns: {df.columns.tolist()}", "info")
            display_log(f"df head after flattening MultiIndex:\n{df.head().to_string()}", "info") # Added debug log

        # Step 2: Handle potential duplicate 'Close' columns (Adj Close vs Close)
        if 'Adj Close' in df.columns and 'Close' in df.columns:
            df['Close'] = df['Adj Close']
            df.drop(columns=['Adj Close'], inplace=True)
            display_log("✅ Replaced 'Close' with 'Adj Close' to standardize.", "info")
        elif 'Adj Close' in df.columns and 'Close' not in df.columns:
            df.rename(columns={'Adj Close': 'Close'}, inplace=True)
            display_log("✅ Renamed 'Adj Close' to 'Close'.", "info")
        
        # Step 3: Ensure all required columns are present and in a consistent order.
        required_financial_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        missing_cols = [col for col in required_financial_cols if col not in df.columns]
        if missing_cols:
            display_log(f"❌ Critical: Missing essential financial columns {missing_cols} after processing. Cannot proceed.", "error")
            return pd.DataFrame()

        df = df[required_financial_cols]
        display_log(f"Final selected columns: {df.columns.tolist()}", "info")
        display_log(f"df head after final selection:\n{df.head().to_string()}", "info")
        
        for col in required_financial_cols:
            df[col] = pd.to_numeric(df[col].squeeze(), errors='coerce') 

        display_log(f"After converting to numeric: Nulls in required columns: {df[required_financial_cols].isnull().sum().sum()}", "info")

        initial_rows = df.shape[0]
        df.dropna(inplace=True)
        if df.empty:
            display_log(f"❗ DataFrame became empty after dropping rows with NaNs. Original rows: {initial_rows}.", "warning")
            return pd.DataFrame()
        elif df.shape[0] < initial_rows:
            display_log(f"🧹 Dropped {initial_rows - df.shape[0]} rows due to NaNs. Remaining rows: {df.shape[0]}.", "info")

        display_log(f"After dropping NaNs: df.shape={df.shape}", "info")

        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        if df.isnull().sum().sum() > 0:
            display_log(f"❗ Warning: {df.isnull().sum().sum()} nulls still found after final ffill/bfill in fetch_stock_data. Returning empty DataFrame.", "warning")
            return pd.DataFrame()


        display_log(f"📈 Data fetched and cleaned for {ticker} → final shape: {df.shape}", "info")
        display_log(f"✅ Data fetched successfully for {ticker}. Final Columns: {df.columns.tolist()}", "info")
        return df
    except Exception as e:
        display_log(f"❌ Error in fetch_stock_data for {ticker}: {e}. Traceback: {traceback.format_exc()}", "error")
        st.exception(e)
        display_log(f"❗ Suggestion: Check ticker symbol, date range, or network connectivity. Traceback: {traceback.format_exc()}", "error")
        return pd.DataFrame()

def add_technical_indicators(df: pd.DataFrame, selected_indicators: list, 
                             sma_window: int = 20, rsi_window: int = 14, atr_window: int = 14, mfi_window: int = 14) -> pd.DataFrame:
    """
    Adds selected technical indicators to the DataFrame using the 'ta' library.
    """
    display_log("📊 Adding Technical Indicators...", "info")
    df_copy = df.copy() 
    
    if df_copy.empty:
        display_log("❗ Input DataFrame is empty for technical indicator addition. Skipping TA.", "warning")
        return df 
    if 'Close' not in df_copy.columns:
        display_log("❌ Critical Error: 'Close' column missing in DataFrame for technical indicator addition.", "error")
        return df 

    try:
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df_copy.columns:
                display_log(f"❗ Missing required column for TA: '{col}'. Skipping TA calculation.", "warning")
                return df 
            df_copy[col] = pd.to_numeric(df_copy[col].squeeze(), errors='coerce')
            df_copy[col].fillna(method='ffill', inplace=True)
            df_copy[col].fillna(method='bfill', inplace=True)

        close_series = df_copy['Close'].squeeze()
        volume_series = df_copy['Volume'].squeeze()
        high_series = df_copy['High'].squeeze()
        low_series = df_copy['Low'].squeeze()

        if 'SMA_20' in selected_indicators:
            df_copy['SMA_20'] = ta.trend.sma_indicator(close_series, window=sma_window)
        if 'SMA_50' in selected_indicators:
            df_copy['SMA_50'] = ta.trend.sma_indicator(close_series, window=50)
        if 'RSI' in selected_indicators:
            df_copy['RSI'] = ta.momentum.rsi(close_series, window=rsi_window)
        if 'MACD' in selected_indicators:
            macd = ta.trend.MACD(close_series)
            df_copy['MACD'] = macd.macd()
            df_copy['MACD_Signal'] = macd.macd_signal()
            df_copy['MACD_Diff'] = macd.macd_diff()
        if 'OBV' in selected_indicators:
            df_copy['OBV'] = ta.volume.on_balance_volume(close_series, volume_series)
        if 'Bollinger Bands' in selected_indicators:
            bb = ta.volatility.BollingerBands(close_series)
            df_copy['BBL'] = bb.bollinger_lband()
            df_copy['BBM'] = bb.bollinger_mavg()
            df_copy['BBU'] = bb.bollinger_hband()
        if 'ATR' in selected_indicators:
            df_copy['ATR'] = ta.volatility.average_true_range(high_series, low_series, close_series, window=atr_window)
        if 'MFI' in selected_indicators:
            df_copy['MFI'] = ta.volume.money_flow_index(high_series, low_series, close_series, volume_series, window=mfi_window)

        df_copy.fillna(method='ffill', inplace=True)
        df_copy.fillna(method='bfill', inplace=True)

        display_log(f"✅ Technical indicators added. Current shape: {df_copy.shape}", "info")
        return df_copy
    except Exception as e:
        display_log(f"❌ Error adding technical indicators: {e}. Traceback: {traceback.format_exc()}", "error")
        st.exception(e)
        return df

def add_macro_indicators(df: pd.DataFrame, global_factors: list) -> pd.DataFrame:
    """
    Fetches and adds real macroeconomic indicators to the DataFrame using yfinance.
    """
    display_log("🌍 Adding Real Macroeconomic Indicators...", "info")
    df_copy = df.copy()
    
    macro_tickers_map = {
        'S&P 500': '^GSPC',
        'Crude Oil': 'CL=F',
        'DXY': 'DX-Y.NYB',
        '10-Year Yield': '^TNX',
        'VIX': '^VIX',
        'Nifty 50': '^NSEI'
    }

    selected_macro_tickers = {k: v for k, v in macro_tickers_map.items() if k in global_factors}

    start_date = df_copy.index.min().strftime('%Y-%m-%d')
    end_date = df_copy.index.max().strftime('%Y-%m-%d')

    for factor_name, yf_ticker in selected_macro_tickers.items():
        col_name = f"{factor_name.replace(' ', '_').replace('-', '_')}_Close"
        try:
            macro_df = yf.download(yf_ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)

            if macro_df.empty:
                display_log(f"❗ No data found for {yf_ticker}. Skipping this indicator.", "warning")
                continue

            if isinstance(macro_df.columns, pd.MultiIndex):
                macro_df.columns = macro_df.columns.map(lambda x: x[0] if isinstance(x, tuple) else x)

            if 'Adj Close' in macro_df.columns:
                macro_series = macro_df['Adj Close'].squeeze()
            elif 'Close' in macro_df.columns:
                macro_series = macro_df['Close'].squeeze()
            else:
                display_log(f"❗ Neither 'Close' nor 'Adj Close' found for {yf_ticker}. Skipping.", "warning")
                continue
            
            macro_series = pd.to_numeric(macro_series, errors='coerce').dropna()

            if macro_series.empty:
                display_log(f"❗ {yf_ticker} data became empty after cleaning. Skipping.", "warning")
                continue

            macro_series.name = col_name
            df_copy = df_copy.merge(macro_series, left_index=True, right_index=True, how='left')
            display_log(f"✅ Added {col_name} to DataFrame.", "info")

        except Exception as e:
            display_log(f"❌ Error fetching/adding {yf_ticker} ({factor_name}): {e}", "error")

    df_copy.fillna(method='ffill', inplace=True)
    df_copy.fillna(method='bfill', inplace=True)
    
    if df_copy.isnull().sum().sum() > 0:
        df_copy.dropna(inplace=True)

    if df_copy.empty:
        display_log("❌ Critical Error: DataFrame became empty after adding macro indicators!", "error")
        return df

    display_log(f"✅ Real macroeconomic indicators added → new shape: {df_copy.shape}", "info")
    return df_copy

def add_fundamental_indicators(df: pd.DataFrame, ticker: str, selected_fund_indicators: list) -> pd.DataFrame:
    """
    Fetches and adds selected fundamental indicators to the DataFrame.
    Fundamental data is quarterly, so it will be forward-filled.
    """
    display_log("📈 Adding Fundamental Indicators...", "info")
    df_copy = df.copy()
    
    if not selected_fund_indicators:
        return df_copy

    try:
        stock = yf.Ticker(ticker)
        
        # Fetch quarterly data and transpose it
        financials = stock.quarterly_financials.T
        balance_sheet = stock.quarterly_balance_sheet.T
        cashflow = stock.quarterly_cashflow.T
        
        # Mapping from user-friendly names to yfinance column names and their source DataFrame
        fund_map = {
            'Total Revenue': (financials, 'Total Revenue'),
            'Net Income': (financials, 'Net Income'),
            'EBITDA': (financials, 'EBITDA'),
            'Total Assets': (balance_sheet, 'Total Assets'),
            'Total Liabilities': (balance_sheet, 'Total Liabilities Net Minority Interest'),
            'Operating Cash Flow': (cashflow, 'Cash Flow From Continuing Operating Activities'),
            'Capital Expenditure': (cashflow, 'Capital Expenditure')
        }
        
        fund_data_to_merge = {}
        for indicator_name in selected_fund_indicators:
            if indicator_name in fund_map:
                source_df, col_name = fund_map[indicator_name]
                if not source_df.empty and col_name in source_df.columns:
                    fund_data_to_merge[indicator_name] = source_df[col_name]
                else:
                    display_log(f"❗ Could not find '{indicator_name}' data for {ticker}.", "warning")
        
        if not fund_data_to_merge:
            display_log(f"❗ No fundamental data could be extracted for {ticker} based on selection.", "warning")
            return df_copy

        fund_df = pd.DataFrame(fund_data_to_merge)
        
        # Convert index to datetime to match the main df
        fund_df.index = pd.to_datetime(fund_df.index).tz_localize(None)
        
        # Merge with the main dataframe
        df_copy = df_copy.merge(fund_df, left_index=True, right_index=True, how='left')
        
        # Forward fill the quarterly data
        df_copy.fillna(method='ffill', inplace=True)
        df_copy.fillna(method='bfill', inplace=True) # Also backfill for initial NaNs

        display_log(f"✅ Fundamental indicators added. Current shape: {df_copy.shape}", "info")
        return df_copy
        
    except Exception as e:
        display_log(f"❌ Error adding fundamental indicators for {ticker}: {e}. Traceback: {traceback.format_exc()}", "error")
        st.exception(e)
        return df
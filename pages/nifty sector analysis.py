# pages/Nifty_Sector_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import timedelta, date, datetime
import time
import plotly.express as px

import config
from utils import (
    download_data, create_features,
    add_fundamental_features
)

st.set_page_config(page_title="Monarch: NIFTY 50 Sector Analysis", layout="wide")

st.header("🏢 NIFTY 50 Sector-wise Analysis & Trading Signals")
st.markdown("""
This page provides a comprehensive, sector-wise overview of NIFTY 50 constituent stocks.
It fetches live market data, key fundamental ratios, and generates illustrative
**Buy/Sell/Neutral signals** based on a simple combination of fundamental and technical indicators.

**Disclaimer:** The trading signals generated here are **highly simplistic and for educational purposes only**.
They do not constitute financial advice and should not be used for actual trading decisions.
Real-world trading requires in-depth research, complex strategies, and robust risk management.
""")

# A hardcoded fallback list of all 50 NIFTY constituents in case the live fetch fails.
# This list is updated to be complete and accurate.
FALLBACK_NIFTY50_STOCKS = {
    'HDFCBANK.NS': 11.59, 'RELIANCE.NS': 9.69, 'ICICIBANK.NS': 7.87, 'INFY.NS': 5.02,
    'Lt.NS': 4.50, 'TCS.NS': 4.00, 'BHARTIARTL.NS': 3.70, 'ITC.NS': 3.50,
    'KOTAKBANK.NS': 3.00, 'HINDUNILVR.NS': 2.53, 'AXISBANK.NS': 2.40, 'BAJFINANCE.NS': 2.20,
    'MARUTI.NS': 2.16, 'M&M.NS': 2.10, 'SBIN.NS': 1.98, 'SUNPHARMA.NS': 1.88,
    'TATAMOTORS.NS': 1.85, 'NTPC.NS': 1.62, 'POWERGRID.NS': 1.50, 'TATASTEEL.NS': 1.35,
    'ADANIENT.NS': 1.29, 'ULTRACEMCO.NS': 1.25, 'ASIANPAINT.NS': 1.19, 'COALINDIA.NS': 1.17,
    'BAJAJFINSV.NS': 1.13, 'HCLTECH.NS': 1.11, 'NESTLEIND.NS': 0.95, 'JSWSTEEL.NS': 0.94,
    'INDUSINDBK.NS': 0.92, 'ADANIPORTS.NS': 0.89, 'GRASIM.NS': 0.81, 'HINDALCO.NS': 0.79,
    'EICHERMOT.NS': 0.75, 'DRREDDY.NS': 0.73, 'SBILIFE.NS': 0.71, 'TITAN.NS': 0.69,
    'CIPLA.NS': 0.68, 'TECHM.NS': 0.65, 'BAJAJ-AUTO.NS': 0.61, 'WIPRO.NS': 0.59,
    'SHREECEM.NS': 0.57, 'HEROMOTOCO.NS': 0.55, 'DIVISLAB.NS': 0.53, 'APOLLOHOSP.NS': 0.51,
    'LTIM.NS': 0.49, 'BRITANNIA.NS': 0.47, 'ONGC.NS': 0.45, 'BPCL.NS': 0.43,
    'HDFCLIFE.NS': 0.41, 'SHRIRAMFIN.NS': 0.38
}


@st.cache_data(ttl=86400) # Cache for one day
def get_nifty50_constituents():
    """
    Fetches NIFTY 50 constituents and weights from a reliable source (ETF holdings).
    Falls back to a hardcoded list on failure.
    """
    try:
        etf_ticker = yf.Ticker("NIFTYBEES.NS")
        holdings = etf_ticker.info.get('holdings')
        if not holdings:
            raise ValueError("Could not retrieve holdings from NIFTYBEES.NS ETF info.")
        
        tickers_list = []
        for stock in holdings:
            symbol = stock.get('symbol')
            if symbol:
                if not symbol.endswith(('.NS', '.BO')):
                     symbol += '.NS'
                tickers_list.append(symbol)
        
        if not tickers_list or len(tickers_list) < 45:
            return list(FALLBACK_NIFTY50_STOCKS.keys())
        
        return tickers_list
    except Exception as e:
        return list(FALLBACK_NIFTY50_STOCKS.keys())

# --- Trading Strategy Parameters (in Sidebar) ---
with st.sidebar:
    st.header("⚙️ Trading Signal Configuration")
    st.subheader("Fundamental Filters")
    min_roe = st.slider("Min. Return on Equity (%)", 0, 30, 15)
    max_pe = st.slider("Max. Trailing P/E Ratio", 10.0, 50.0, 25.0, step=1.0)
    
    st.subheader("Technical Indicators")
    ma_window = st.slider("Moving Average Window (days)", 5, 50, 20)
    rsi_overbought = st.slider("RSI Overbought Threshold", 60, 90, 70)
    rsi_oversold = st.slider("RSI Oversold Threshold", 10, 40, 30)

    st.subheader("Data Period for Technicals")
    tech_data_period_days = st.slider("Days of data for Technicals", 30, 180, 90)

# --- Define Trading Signal Logic ---
def generate_trading_signal(stock_data, info, min_roe, max_pe, ma_window, rsi_overbought, rsi_oversold):
    """
    Generates a trading signal (Buy/Sell/Neutral) based on fundamental and technical rules.
    
    Args:
        stock_data (pd.DataFrame): Historical OHLCV data for the stock.
        info (dict): Yahoo Finance info dictionary for the stock.
        min_roe (int): Minimum Return on Equity (%) for Buy signal.
        max_pe (float): Maximum Trailing P/E Ratio for Buy signal.
        ma_window (int): Moving Average window for technical signal.
        rsi_overbought (int): RSI threshold for Overbought (Sell signal).
        rsi_oversold (int): RSI threshold for Oversold (Buy signal).
        
    Returns:
        tuple: (signal_text, signal_emoji, signal_color)
    """
    signal_text = "Neutral"
    signal_emoji = "⚪"
    signal_color = "gray"

    # --- Fundamental Checks ---
    trailing_pe = info.get('trailingPE')
    roe = info.get('returnOnEquity')
    
    fundamentally_strong = False
    if trailing_pe is not None and roe is not None:
        # Ensure they are numbers before comparison
        if isinstance(trailing_pe, (int, float)) and isinstance(roe, (int, float)):
            if trailing_pe > 0 and trailing_pe <= max_pe and roe >= (min_roe / 100):
                fundamentally_strong = True
    
    # --- Technical Checks ---
    if not stock_data.empty and len(stock_data) > ma_window:
        # Ensure 'Close' column exists and is numeric
        if 'Close' not in stock_data.columns or not pd.api.types.is_numeric_dtype(stock_data['Close']):
            return signal_text, signal_emoji, signal_color

        # Calculate MA
        stock_data['MA'] = stock_data['Close'].rolling(window=ma_window).mean()
        
        # Calculate RSI (using utils.create_features for consistency)
        temp_indicator_params = {'RSI_WINDOW': 14} 
        stock_data_with_rsi = create_features(stock_data.copy(), temp_indicator_params)
        
        if 'RSI' in stock_data_with_rsi.columns and not stock_data_with_rsi['RSI'].empty:
            current_rsi = stock_data_with_rsi['RSI'].iloc[-1]
            
            current_price = stock_data['Close'].iloc[-1]
            ma_value = stock_data['MA'].iloc[-1]

            # Buy Conditions
            if (fundamentally_strong and
                not np.isnan(current_price) and not np.isnan(ma_value) and not np.isnan(current_rsi) and
                current_price > ma_value and # Price above MA (uptrend)
                current_rsi < rsi_oversold): # RSI oversold (potential bounce)
                signal_text = "BUY"
                signal_emoji = "🟢"
                signal_color = "green"
            
            # Sell Conditions
            elif (not np.isnan(current_price) and not np.isnan(ma_value) and not np.isnan(current_rsi) and
                  current_price < ma_value and # Price below MA (downtrend)
                  current_rsi > rsi_overbought): # RSI overbought (potential reversal)
                signal_text = "SELL"
                signal_emoji = "🔴"
                signal_color = "red"

    return signal_text, signal_emoji, signal_color

# --- Main Application Flow ---
if st.button("Analyze NIFTY 50 Stocks", type="primary"):
    nifty_tickers = get_nifty50_constituents()
    if not nifty_tickers:
        st.error("Could not retrieve NIFTY 50 tickers. Please try again later.")
        st.stop()

    # --- Log Box Setup ---
    log_messages = []
    processing_status_placeholder = st.empty()

    def update_log_box(message, message_type='info', current_progress=None):
        timestamp = datetime.now().strftime('%H:%M:%S')
        if message_type == 'error':
            log_messages.insert(0, f"🔴 [{timestamp}] ERROR: {message}")
        elif message_type == 'warning':
            log_messages.insert(0, f"🟠 [{timestamp}] WARNING: {message}")
        elif message_type == 'success':
            log_messages.insert(0, f"✅ [{timestamp}] SUCCESS: {message}")
        else:
            log_messages.insert(0, f"🔵 [{timestamp}] INFO: {message}")
        
        with processing_status_placeholder.container():
            st.markdown(f"**Current Status:** {log_messages[0]}")
            if len(log_messages) > 1:
                with st.expander("Show Full Processing Log"):
                    st.code("\n".join(log_messages), language=None)
            
            if current_progress is not None:
                st.progress(current_progress)

    update_log_box(f"Starting analysis for {len(nifty_tickers)} NIFTY 50 stocks...", 'info', current_progress=0.0)
    
    all_stock_data = []
    
    fundamentals_to_fetch = {
        'Trailing P/E': 'trailingPE',
        'Forward P/E': 'forwardPE',
        'Dividend Yield': 'dividendYield',
        'Return on Equity': 'returnOnEquity',
        'Market Cap': 'marketCap',
        'Sector': 'sector',
        'Industry': 'industry'
    }
    historical_derived_fundamentals = ['Historical P/E Ratio', 'Historical P/S Ratio', 'Historical Debt to Equity']
    combined_fundamentals_for_add_func = {**fundamentals_to_fetch, **{name: name for name in historical_derived_fundamentals}}


    for i, ticker_symbol in enumerate(nifty_tickers):
        update_log_box(f"Processing ({i+1}/{len(nifty_tickers)}): {ticker_symbol}", 'info', current_progress=(i + 1) / len(nifty_tickers))
        
        try:
            stock_yf = yf.Ticker(ticker_symbol)
            info = stock_yf.info

            current_price = info.get('currentPrice')
            if current_price is None:
                current_price = np.nan
            
            required_hist_days = max(ma_window, 14, tech_data_period_days)
            historical_data = download_data(ticker_symbol, period=f"{required_hist_days}d") 
            
            data_with_fundamentals = add_fundamental_features(historical_data.copy(), ticker_symbol, combined_fundamentals_for_add_func, _update_log_func=(lambda x: None))

            signal_text, signal_emoji, signal_color = generate_trading_signal(
                data_with_fundamentals, info, min_roe, max_pe, ma_window, rsi_overbought, rsi_oversold
            )

            stock_entry = {
                'Ticker': ticker_symbol,
                'Company': info.get('longName', 'N/A'),
                'Sector': info.get('sector', 'N/A'),
                'Industry': info.get('industry', 'N/A'),
                'Live Price': current_price,
                
                # Assign np.nan if data is not available or not numeric
                'Trailing P/E': float(info.get('trailingPE')) if isinstance(info.get('trailingPE'), (int, float)) else np.nan,
                'Forward P/E': float(info.get('forwardPE')) if isinstance(info.get('forwardPE'), (int, float)) else np.nan,
                'Return on Equity': float(info.get('returnOnEquity')) * 100 if isinstance(info.get('returnOnEquity'), (int, float)) else np.nan,
                'Dividend Yield': float(info.get('dividendYield')) * 100 if isinstance(info.get('dividendYield'), (int, float)) else np.nan,
                'Market Cap (Bn)': float(info.get('marketCap', 0)) / 1e9 if isinstance(info.get('marketCap'), (int, float)) else np.nan,
                
                'Signal': signal_text,
                'Signal_Emoji': signal_emoji,
                'Signal_Color': signal_color
            }
            all_stock_data.append(stock_entry)
            
        except Exception as e:
            update_log_box(f"Could not process {ticker_symbol}: {e}", 'error', current_progress=(i + 1) / len(nifty_tickers))
            all_stock_data.append({
                'Ticker': ticker_symbol, 'Company': 'N/A', 'Sector': 'N/A', 'Industry': 'N/A',
                'Live Price': np.nan,
                'Trailing P/E': np.nan,
                'Forward P/E': np.nan,
                'Return on Equity': np.nan,
                'Dividend Yield': np.nan,
                'Market Cap (Bn)': np.nan,
                'Signal': 'Error', 'Signal_Emoji': '❌', 'Signal_Color': 'red'
            })
        
        time.sleep(0.1)

    update_log_box("Analysis complete!", 'success', current_progress=1.0)
    processing_status_placeholder.empty()
    
    if all_stock_data:
        results_df = pd.DataFrame(all_stock_data)
        
        st.subheader("📊 NIFTY 50 Stock Overview & Signals")
        def color_main_signal(val):
            if val == "BUY": return 'background-color: #e6ffe6'
            elif val == "SELL": return 'background-color: #ffe6e6'
            else: return ''

        st.dataframe(
            results_df.style.map(color_main_signal, subset=['Signal']).format({
                'Live Price': '{:,.2f}',
                'Trailing P/E': '{:,.2f}',
                'Forward P/E': '{:,.2f}',
                'Return on Equity': '{:,.2f}%',
                'Dividend Yield': '{:,.2f}%',
                'Market Cap (Bn)': '{:,.2f}'
            }),
            hide_index=True,
            use_container_width=True
        )

        st.subheader("📈 Sector-wise Performance & Signals")
        sector_summary = results_df.groupby('Sector').agg(
            Total_Stocks=('Ticker', 'count'),
            Buy_Signals=('Signal', lambda x: (x == 'BUY').sum()),
            Sell_Signals=('Signal', lambda x: (x == 'SELL').sum()),
            Avg_PE=('Trailing P/E', 'mean'),
            Avg_ROE=('Return on Equity', 'mean'),
            Avg_Market_Cap_Bn=('Market Cap (Bn)', 'mean')
        ).reset_index()
        
        st.dataframe(sector_summary.style.format({
            'Avg_PE': '{:,.2f}',
            'Avg_ROE': '{:,.2f}%',
            'Avg_Market_Cap_Bn': '{:,.2f}'
        }), use_container_width=True)

        fig_sector_signals = px.bar(
            sector_summary,
            x='Sector',
            y=['Buy_Signals', 'Sell_Signals'],
            title='Main Buy/Sell Signals by Sector',
            labels={'value': 'Number of Signals', 'variable': 'Signal Type'},
            color_discrete_map={'Buy_Signals': 'green', 'Sell_Signals': 'red'}
        )
        fig_sector_signals.update_layout(barmode='group')
        st.plotly_chart(fig_sector_signals, use_container_width=True)

    else:
        st.info("No stock data processed. Adjust parameters or try again.")

else:
    st.info("Click 'Analyze NIFTY 50 Stocks' to get started.")
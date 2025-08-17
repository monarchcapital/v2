import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Global Market Dashboard")

# --- Hide Streamlit Menu & Footer ---
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- Data Configuration ---
SYMBOLS = {
    'Stock Indices': {
        '^GSPC': 'S&P 500', '^IXIC': 'NASDAQ Composite', '^FTSE': 'FTSE 100 (UK)',
        '^N225': 'Nikkei 225 (Japan)', '^GDAXI': 'DAX 30 (Germany)', '^NSEI': 'Nifty 50 (India)',
        '^BVSP': 'Bovespa (Brazil)', '^MXX': 'IPC Mexico', '^HSI': 'Hang Seng (Hong Kong)',
        '^STOXX50E': 'Euro Stoxx 50', '^FCHI': 'CAC 40 (France)',
        '^KS11': 'KOSPI (South Korea)', '^AXJO': 'S&P/ASX 200 (Australia)',
    },
    'Currencies': {
        'DX-Y.NYB': 'US Dollar Index (DXY)', 'EURUSD=X': 'Euro/USD', 'JPY=X': 'USD/JPY',
        'GBPUSD=X': 'British Pound/USD', 'INR=X': 'USD/INR', 'CNY=X': 'USD/CNY',
        'AUDUSD=X': 'Australian Dollar/USD', 'BRL=X': 'USD/BRL',
    },
    'Commodities': {
        'CL=F': 'Crude Oil (WTI)', 'BZ=F': 'Brent Crude Oil', 'NG=F': 'Natural Gas',
        'GC=F': 'Gold', 'SI=F': 'Silver', 'PL=F': 'Platinum', 'HG=F': 'Copper',
    },
    'Government Yields': {
        '^TNX': 'US 10-Year Yield', '^FVX': 'US 5-Year Yield', '^TYX': 'US 30-Year Yield',
        '^IN10Y': 'India 10-Year Yield',
    }
}

# --- Helper Functions ---

@st.cache_data(ttl=3600)
def fetch_last_two_trading_days():
    """Fetches the last two valid trading days from Yahoo Finance data."""
    try:
        ticker = yf.Ticker('^GSPC') 
        hist = ticker.history(period='5d')
        valid_data = hist.dropna(subset=['Close'])
        if len(valid_data) >= 2:
            return valid_data.index[-2].date(), valid_data.index[-1].date()
        return None, None
    except Exception as e:
        st.error(f"Error fetching trading dates: {e}")
        return None, None

@st.cache_data(ttl=3600)
def fetch_all_data(date_1, date_2):
    """
    Fetches market data for all symbols in a single batch call for efficiency and robustness.
    """
    all_symbols = [symbol for category in SYMBOLS.values() for symbol in category.keys()]
    
    # Fetch data for a broader range to handle weekends/holidays
    start_fetch = date_1 - timedelta(days=7)
    end_fetch = date_2 + timedelta(days=1)
    
    try:
        data = yf.download(all_symbols, start=start_fetch, end=end_fetch, progress=False)
        if data.empty:
            return {}
            
        # Forward-fill data to handle non-trading days
        full_date_range = pd.date_range(start=start_fetch, end=end_fetch)
        data = data.reindex(full_date_range).ffill()

        # Get the data for the exact dates requested
        data_on_date_1 = data.loc[pd.to_datetime(date_1)]
        data_on_date_2 = data.loc[pd.to_datetime(date_2)]

        category_dataframes = {}
        for category_name, category_symbols in SYMBOLS.items():
            data_list = []
            for symbol, name in category_symbols.items():
                try:
                    last_close = data_on_date_2['Close'][symbol]
                    previous_close = data_on_date_1['Close'][symbol]
                    
                    if pd.notna(last_close) and pd.notna(previous_close):
                        change = last_close - previous_close
                        percent_change = (change / previous_close) * 100 if previous_close != 0 else 0
                        
                        data_list.append({
                            'Indicator': name, 'Last Close': last_close, 'Previous Close': previous_close,
                            'Open': data_on_date_2['Open'][symbol], 'High': data_on_date_2['High'][symbol],
                            'Low': data_on_date_2['Low'][symbol], 'Change ($)': change, 'Change (%)': percent_change
                        })
                except KeyError:
                    # This ticker might not have been returned by the API
                    continue
            if data_list:
                category_dataframes[category_name] = pd.DataFrame(data_list)
        return category_dataframes

    except Exception:
        return {}


def color_change(val):
    """Applies color to a value based on whether it is positive or negative."""
    color = '#2ECC71' if val > 0 else '#E74C3C' if val < 0 else 'white'
    return f'color: {color};'

def generate_heatmap_grid(df, title):
    """Generates a color-coded grid of market indicators."""
    st.markdown(f"<h4 style='text-align: center;'>{title} Heatmap</h4>", unsafe_allow_html=True)
    if not df.empty:
        sorted_df = df.dropna(subset=["Change (%)"]).sort_values("Change (%)", ascending=False)
        
        with st.container(border=True):
            cols = st.columns(8)
            for i, (_, row) in enumerate(sorted_df.iterrows()):
                pct = row["Change (%)"]
                # More vibrant colors and stronger opacity scaling
                alpha = min(1, abs(pct) / 2.5) # Scale opacity more aggressively
                if pct > 0:
                    color = f"rgba(46, 204, 113, {alpha + 0.2})" # Brighter green
                else:
                    color = f"rgba(231, 76, 60, {alpha + 0.2})" # Brighter red
                
                cols[i % 8].markdown(
                    f"""
                    <div class='heatmap-item' style='background-color: {color};'>
                        {row["Indicator"]}<br>{pct:+.2f}%
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

# --- UI Layout ---

# Sidebar for Navigation
st.sidebar.title("Navigation")
st.sidebar.info("Select an analysis page below. Streamlit automatically lists files from the 'pages' directory.")

# Main Page Content
st.markdown("""
<style>
    body {
        color: #EAECEE;
    }
    .main-header { 
        font-size: 2.5em; 
        font-weight: bold; 
        text-align: center; 
        color: #EAECEE;
    }
    .subheader { 
        font-size: 1.2em; 
        text-align: center; 
        margin-bottom: 1.5em; 
        color: #AAB7B8;
    }
    .stDataFrame { 
        border-radius: 10px; 
        box-shadow: 0 4px 8px rgba(0,0,0,0.1); 
    }
    .heatmap-item {
        padding: 10px; 
        border-radius: 8px; 
        text-align: center;
        margin-bottom: 8px; 
        font-size: 0.85rem; 
        font-weight: bold;
        color: white; 
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 2px 4px rgba(0,0,0,0.4);
    }
    h4 {
        color: #EAECEE;
    }
</style>
<div class="main-header">Global Market Dashboard</div>
<div class="subheader">Daily changes for key stocks, commodities, currencies, and yields</div>
""", unsafe_allow_html=True)

# --- Date Selection ---
if 'end_date' not in st.session_state:
    previous_day, last_day = fetch_last_two_trading_days()
    if previous_day and last_day:
        st.session_state.start_date = previous_day
        st.session_state.end_date = last_day
    else:
        st.session_state.start_date = datetime.now().date() - timedelta(days=2)
        st.session_state.end_date = datetime.now().date() - timedelta(days=1)

col1, col2 = st.columns(2)
date_1 = col1.date_input("Select Start Date", value=st.session_state.start_date)
date_2 = col2.date_input("Select End Date", value=st.session_state.end_date)

# --- View Mode Buttons ---
button_cols = st.columns([1, 1, 1, 1])
if button_cols[0].button('Refresh Data'):
    st.cache_data.clear()
    st.session_state.clear()
    st.rerun()
if button_cols[1].button("Show Heatmap"):
    st.session_state.view_mode = 'heatmap'
    st.rerun()
if button_cols[2].button("Show Tables"):
    st.session_state.view_mode = 'tables'
    st.rerun()

if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'heatmap' # Default view is now heatmap

# --- Data Fetching and Display ---
with st.spinner('Fetching market data...'):
    market_data_dfs = fetch_all_data(date_1, date_2)

st.header(f"Market Performance: {date_1.strftime('%Y-%m-%d')} vs {date_2.strftime('%Y-%m-%d')}")

if not market_data_dfs:
    st.error("Could not fetch any market data. Please check your network connection or try again later.")
else:
    if st.session_state.view_mode == 'tables':
        for category in SYMBOLS.keys():
            st.markdown(f"<h4 style='text-align: center;'>{category} as of {date_2.strftime('%Y-%m-%d')}</h4>", unsafe_allow_html=True)
            if category in market_data_dfs and not market_data_dfs[category].empty:
                df_display = market_data_dfs[category].sort_values("Change (%)", ascending=False).reset_index(drop=True)
                format_dict = {
                    'Last Close': "${:,.2f}", 'Previous Close': "${:,.2f}", 'Open': "${:,.2f}",
                    'High': "${:,.2f}", 'Low': "${:,.2f}", 'Change ($)': "{:+.2f}", 'Change (%)': "{:+.2f}%"
                }
                styled_df = df_display.style.applymap(color_change, subset=['Change ($)', 'Change (%)']).format(format_dict, na_rep='N/A')
                st.dataframe(styled_df, use_container_width=True)
            else:
                st.warning(f"No data available for {category} for the selected dates.")
    
    elif st.session_state.view_mode == 'heatmap':
        for category in SYMBOLS.keys():
            if category in market_data_dfs:
                generate_heatmap_grid(market_data_dfs[category], category)
            else:
                st.warning(f"No data available for {category} heatmap.")

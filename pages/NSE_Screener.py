# pages/multibagger.py - A Streamlit page to automatically identify potential multibagger and intraday momentum stocks from the NSE.

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
import concurrent.futures
from functools import partial
import threading

# --- Page Configuration ---
st.set_page_config(page_title="Automated NSE Screener", page_icon="🤖", layout="wide")

# --- Hide Streamlit Menu & Footer ---
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- Sidebar for User-defined Criteria ---
st.sidebar.header("⚙️ Screener Parameters")

st.sidebar.subheader("Potential Multibagger Criteria")
min_rev_cagr = st.sidebar.slider("Min Revenue CAGR (%)", 0, 100, 15, 1) / 100.0
min_ni_cagr = st.sidebar.slider("Min Net Income CAGR (%)", 0, 100, 18, 1) / 100.0
min_roe = st.sidebar.slider("Min Average ROE (%)", 0, 100, 15, 1) / 100.0
max_d_e = st.sidebar.slider("Max Debt to Equity", 0.0, 5.0, 0.6, 0.1)
max_pe = st.sidebar.slider("Max P/E Ratio", 0, 200, 50, 1)

# Store user-defined multibagger criteria in a dictionary
user_multibagger_criteria = {
    'min_rev_cagr': min_rev_cagr,
    'min_ni_cagr': min_ni_cagr,
    'min_roe': min_roe,
    'max_d_e': max_d_e,
    'max_pe': max_pe
}

st.sidebar.subheader("Intraday Momentum Criteria")
st.sidebar.markdown("**High Momentum**")
high_min_price_change = st.sidebar.slider("Min Price Change (%)", 0.0, 10.0, 2.0, 0.1)
high_min_volume_ratio = st.sidebar.slider("Min Volume Ratio (vs 20-day avg)", 1.0, 10.0, 1.5, 0.1)

st.sidebar.markdown("**Moderate Momentum**")
mod_min_price_change = st.sidebar.slider("Min Price Change (%) ", 0.0, 10.0, 1.0, 0.1)
mod_min_volume_ratio = st.sidebar.slider("Min Volume Ratio (vs 20-day avg) ", 1.0, 10.0, 1.2, 0.1)

# Store user-defined intraday criteria in a dictionary
user_intraday_criteria = {
    'high_price_change': high_min_price_change,
    'high_volume_ratio': high_min_volume_ratio,
    'mod_price_change': mod_min_price_change,
    'mod_volume_ratio': mod_min_volume_ratio,
}

# --- Session State Initialization ---
if 'scan_running' not in st.session_state:
    st.session_state.scan_running = False
if 'stop_scan' not in st.session_state:
    st.session_state.stop_scan = False
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = []
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = None

# --- Helper Functions ---

@st.cache_data
def get_nse_tickers():
    """Fetches the list of all equity tickers from the NSE archives."""
    try:
        url = 'https://archives.nseindia.com/content/equities/EQUITY_L.csv'
        df = pd.read_csv(url)
        tickers = (df['SYMBOL'] + '.NS').tolist()
        return tickers
    except Exception as e:
        st.error(f"Failed to fetch NSE ticker list: {e}")
        return []

@st.cache_data
def get_screening_data(ticker):
    """Fetches fundamental and recent price data for analysis."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or 'currentPrice' not in info:
            return None, f"Incomplete market data for '{ticker}'."
        
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        if financials.empty or balance_sheet.empty:
            return None, f"Missing financial statements for '{ticker}'."

        history = stock.history(period="60d")
        if history.empty:
            return None, f"Could not fetch price history for {ticker}."

        return {
            "info": info,
            "financials": financials,
            "balance_sheet": balance_sheet,
            "history": history
        }, None
    except Exception:
        return None, f"An error occurred while fetching data for {ticker}."

def analyze_multibagger_potential(data, criteria):
    """Analyzes a stock's fundamental data against user-defined criteria."""
    info, financials, balance_sheet = data['info'], data['financials'], data['balance_sheet']
    results = {}
    score = 0
    try:
        revenue = financials.loc['Total Revenue']
        net_income = financials.loc['Net Income']
        total_equity = balance_sheet.loc['Total Stockholder Equity']
        
        num_years_rev = len(revenue) - 1
        if num_years_rev > 0 and revenue.iloc[-1] > 0:
            results['Revenue CAGR'] = (revenue.iloc[0] / revenue.iloc[-1])**(1/num_years_rev) - 1
        else:
            results['Revenue CAGR'] = 0

        num_years_ni = len(net_income) - 1
        if num_years_ni > 0 and net_income.iloc[-1] > 0:
            results['Net Income CAGR'] = (net_income.iloc[0] / net_income.iloc[-1])**(1/num_years_ni) - 1
        else:
            results['Net Income CAGR'] = 0

        results['Average ROE'] = (net_income / total_equity.shift(-1)).mean()
        results['Debt to Equity'] = info.get('debtToEquity', np.inf) / 100 if info.get('debtToEquity') else 0
        results['P/E Ratio'] = info.get('trailingPE', np.inf)
        
        if results.get('Revenue CAGR', 0) > criteria['min_rev_cagr']: score += 1
        if results.get('Net Income CAGR', 0) > criteria['min_ni_cagr']: score += 1
        if results.get('Average ROE', 0) > criteria['min_roe']: score += 1
        if results.get('Debt to Equity', float('inf')) < criteria['max_d_e']: score += 1
        if results.get('P/E Ratio', float('inf')) < criteria['max_pe']: score += 1
    except (KeyError, IndexError):
        return {}, 0
    return results, score

def analyze_intraday_momentum(history, criteria):
    """Analyzes recent price history for signs of intraday momentum based on user criteria."""
    if len(history) < 2:
        return "N/A"
    
    today = history.iloc[-1]
    yesterday = history.iloc[-2]
    
    avg_volume = history['Volume'].iloc[-21:-1].mean()
    volume_ratio = today['Volume'] / avg_volume if avg_volume > 0 else 0
    price_change = ((today['Close'] - yesterday['Close']) / yesterday['Close']) * 100

    if price_change > criteria['high_price_change'] and volume_ratio > criteria['high_volume_ratio']:
        return "High Momentum"
    if price_change > criteria['mod_price_change'] and volume_ratio > criteria['mod_volume_ratio']:
        return "Moderate Momentum"
    return "Low"

def get_recommendation(score):
    if score == 5: return "Strong Buy"
    elif score == 4: return "Buy"
    elif score == 3: return "Consider"
    return "Avoid"

def process_ticker(ticker, multibagger_criteria, intraday_criteria, stop_event):
    """Encapsulates fetching and analysis for a single ticker for parallel execution."""
    if stop_event.is_set():
        return None

    stock_data, error = get_screening_data(ticker)
    if error:
        return None
    
    multi_analysis, multi_score = analyze_multibagger_potential(stock_data, multibagger_criteria)
    intraday_signal = analyze_intraday_momentum(stock_data['history'], intraday_criteria)

    if multi_score >= 3 or intraday_signal != "Low":
        return {
            "Ticker": ticker,
            "Sector": stock_data['info'].get('sector', 'N/A'),
            "Price": stock_data['info'].get('currentPrice', 'N/A'),
            "Multibagger Score": multi_score,
            "Recommendation": get_recommendation(multi_score),
            "Intraday Signal": intraday_signal,
            "Revenue CAGR": multi_analysis.get('Revenue CAGR'),
            "Net Income CAGR": multi_analysis.get('Net Income CAGR'),
            "Avg ROE": multi_analysis.get('Average ROE'),
        }
    return None

# --- Main UI ---
st.title("🇮🇳 Automated NSE Screener")
st.markdown("This tool scans all stocks on the **National Stock Exchange (NSE)** for two types of opportunities:")
st.markdown("- **Potential Multibaggers**: Fundamentally strong companies with high growth (for long-term investment).")
st.markdown("- **Intraday Momentum**: Stocks showing unusual volume and price action today (for short-term trading).")

# --- Control Buttons ---
col1, col2, col3 = st.columns([1,1,5])
if col1.button("🚀 Start Scan", use_container_width=True, type="primary", disabled=st.session_state.scan_running):
    st.session_state.scan_running = True
    st.session_state.stop_scan = False
    st.session_state.scan_results = []
    st.session_state.stop_event = threading.Event()
    # Store the criteria used for this specific run
    st.session_state.run_multibagger_criteria = user_multibagger_criteria
    st.session_state.run_intraday_criteria = user_intraday_criteria
    st.rerun()

if col2.button("🛑 Stop Scan", use_container_width=True, disabled=not st.session_state.scan_running):
    if st.session_state.stop_event:
        st.session_state.stop_event.set()
    st.session_state.stop_scan = True
    st.session_state.scan_running = False
    st.rerun()

st.markdown("---")

# --- Automated Execution ---
if st.session_state.scan_running:
    tickers = get_nse_tickers()
    if not tickers:
        st.error("Could not retrieve NSE tickers. Halting scan.")
        st.session_state.scan_running = False
        st.stop()

    all_results = []
    placeholder = st.empty()
    progress_bar = st.progress(0, text=f"Initializing scan of {len(tickers)} NSE stocks...")
    processed_count = 0

    task_function = partial(process_ticker, 
                            multibagger_criteria=st.session_state.run_multibagger_criteria, 
                            intraday_criteria=st.session_state.run_intraday_criteria,
                            stop_event=st.session_state.stop_event)

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_ticker = {executor.submit(task_function, ticker): ticker for ticker in tickers}
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            if st.session_state.stop_event.is_set():
                for f in future_to_ticker:
                    f.cancel()
                break

            processed_count += 1
            ticker = future_to_ticker[future]
            progress_bar.progress(processed_count / len(tickers), text=f"Analyzing {ticker} ({processed_count}/{len(tickers)})...")
            
            try:
                result = future.result()
                if result:
                    all_results.append(result)
            except concurrent.futures.CancelledError:
                pass # Ignore cancelled futures

    st.session_state.scan_results = all_results
    st.session_state.scan_running = False
    st.rerun()

# --- Display Results ---
if st.session_state.stop_scan:
    st.warning("Scan stopped by user.")
    st.session_state.stop_scan = False # Reset flag

if st.session_state.scan_results:
    results_df = pd.DataFrame(st.session_state.scan_results).sort_values(by=["Sector", "Multibagger Score"], ascending=[True, False])
    
    st.header("🏆 Screening Results")
    for sector, group in results_df.groupby('Sector'):
        st.subheader(sector)
        
        def style_recommendation(val):
            if val == "Strong Buy": return 'background-color: #1E5631; color: white;'
            if val == "Buy": return 'background-color: #2E8B57; color: white;'
            if val == "Consider": return 'background-color: #DAA520;'
            return ''
        
        def style_intraday(val):
            if val == "High Momentum": return 'background-color: #FF4B4B; color: white;'
            if val == "Moderate Momentum": return 'background-color: #FFA500;'
            return ''

        styled_group = group.style.applymap(style_recommendation, subset=['Recommendation'])\
                                  .applymap(style_intraday, subset=['Intraday Signal'])\
                                  .format({
                                      "Price": "₹{:,.2f}", "Multibagger Score": "{}/5",
                                      "Revenue CAGR": "{:.2%}", "Net Income CAGR": "{:.2%}",
                                      "Avg ROE": "{:.2%}"
                                  }, na_rep="N/A")
        
        st.dataframe(styled_group, use_container_width=True, hide_index=True)
elif not st.session_state.scan_running:
    st.info("Click 'Start Scan' to begin screening NSE stocks.")

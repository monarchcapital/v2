# pages/multibagger.py - A Streamlit page to automatically identify potential multibagger and intraday momentum stocks from the NSE.

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta

# --- Page Configuration ---
st.set_page_config(page_title="Automated NSE Screener", page_icon="🤖", layout="wide")

# --- Pre-defined Screening Criteria ---
AUTOMATED_CRITERIA = {
    'min_rev_cagr': 0.15,   # Minimum 15% revenue growth per year
    'min_ni_cagr': 0.18,    # Minimum 18% net income growth per year
    'min_roe': 0.15,        # Minimum 15% average Return on Equity
    'max_d_e': 0.6,         # Maximum Debt to Equity ratio of 0.6
    'max_pe': 50.0          # Maximum P/E ratio of 50
}

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

        # Fetch last 60 days of history for intraday analysis
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
    """Analyzes a stock's fundamental data against pre-defined criteria."""
    info, financials, balance_sheet = data['info'], data['financials'], data['balance_sheet']
    results = {}
    score = 0
    try:
        revenue = financials.loc['Total Revenue']
        net_income = financials.loc['Net Income']
        total_equity = balance_sheet.loc['Total Stockholder Equity']
        
        results['Revenue CAGR'] = (revenue.iloc[0] / revenue.iloc[-1])**(1/len(revenue)) - 1 if len(revenue) > 1 and revenue.iloc[-1] > 0 else 0
        results['Net Income CAGR'] = (net_income.iloc[0] / net_income.iloc[-1])**(1/len(net_income)) - 1 if len(net_income) > 1 and net_income.iloc[-1] > 0 else 0
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

def analyze_intraday_momentum(history):
    """Analyzes recent price history for signs of intraday momentum."""
    if len(history) < 2:
        return "N/A"
    
    today = history.iloc[-1]
    yesterday = history.iloc[-2]
    
    avg_volume = history['Volume'].iloc[-21:-1].mean() # Avg volume of last 20 days
    volume_ratio = today['Volume'] / avg_volume if avg_volume > 0 else 0
    price_change = ((today['Close'] - yesterday['Close']) / yesterday['Close']) * 100

    if price_change > 2 and volume_ratio > 1.5:
        return "High Momentum"
    if price_change > 1 and volume_ratio > 1.2:
        return "Moderate Momentum"
    return "Low"

def get_recommendation(score):
    if score == 5: return "Strong Buy"
    elif score == 4: return "Buy"
    elif score == 3: return "Consider"
    return "Avoid"

# --- Main UI ---
st.title("🇮🇳 Automated NSE Screener")
st.markdown("This tool scans all stocks on the **National Stock Exchange (NSE)** for two types of opportunities:")
st.markdown("- **Potential Multibaggers**: Fundamentally strong companies with high growth (for long-term investment).")
st.markdown("- **Intraday Momentum**: Stocks showing unusual volume and price action today (for short-term trading).")
st.warning("🕒 **Please be patient.** Scanning the entire NSE can take over an hour. Results will appear below as they are found.", icon="⏳")

# --- Automated Execution ---
tickers = get_nse_tickers()
if not tickers:
    st.stop()

all_results = []
placeholder = st.empty()
progress_bar = st.progress(0, text=f"Initializing scan of {len(tickers)} NSE stocks...")

for i, ticker in enumerate(tickers):
    progress_bar.progress((i + 1) / len(tickers), text=f"Analyzing {ticker} ({i+1}/{len(tickers)})...")
    stock_data, error = get_screening_data(ticker)
    
    if error:
        continue
    
    multi_analysis, multi_score = analyze_multibagger_potential(stock_data, AUTOMATED_CRITERIA)
    intraday_signal = analyze_intraday_momentum(stock_data['history'])

    if multi_score >= 3 or intraday_signal != "Low":
        current_result = {
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
        all_results.append(current_result)

        if all_results:
            results_df = pd.DataFrame(all_results).sort_values(by=["Sector", "Multibagger Score"], ascending=[True, False])
            
            with placeholder.container():
                st.header("🏆 Live Screening Results")
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

progress_bar.empty()
if not all_results:
    with placeholder.container():
        st.info("No stocks on the NSE met the high-scoring criteria today.")

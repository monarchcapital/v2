# pages/Fundamentals.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Monarch: Fundamental Analysis", layout="wide")

st.header("📊 Fundamental Analysis & Valuation")
st.markdown("""
This page provides a detailed fundamental report for each company, including multiple valuation models,
historical ratio analysis, and analyst consensus estimates.
""")

# --- Stock Ticker Input ---
ticker_input = st.text_area(
    "Enter Stock Ticker Symbol(s) (e.g., RELIANCE.NS, AAPL, MSFT - separate by comma or new line):",
    "MSFT, AAPL"
)
tickers = [t.strip().upper() for t in ticker_input.replace('\n', ',').split(',') if t.strip()]

# --- Valuation Parameters in Sidebar ---
with st.sidebar:
    st.header("📈 Valuation Parameters")
    st.subheader("Margin of Safety (MoS)")
    mos_dcf = st.slider("DCF MoS (%)", 0, 50, 20, key='dcf_mos')
    mos_graham = st.slider("Graham Formula MoS (%)", 0, 50, 25, key='graham_mos')
    mos_peg = st.slider("PEG Ratio MoS (%)", 0, 50, 15, key='peg_mos')

    with st.expander("DCF Model Parameters"):
        growth_rate = st.slider("Projected FCF Growth Rate (%)", 5, 30, 10, key='growth_rate') / 100
        terminal_growth = st.slider("Terminal Growth Rate (%)", 1, 10, 3, key='terminal_growth') / 100
        discount_rate = st.slider("Discount Rate (WACC) (%)", 5, 20, 9, key='discount_rate') / 100

    with st.expander("Graham Formula Parameters"):
        graham_multiplier = st.slider("Graham Multiplier", 15.0, 25.0, 22.5, step=0.5, key='graham_multiplier')
        max_pe = st.slider("Max P/E Ratio to Apply", 10, 30, 20, key='max_pe')
    
    with st.expander("PEG Ratio Parameters"):
        peg_ratio = st.slider("Target PEG Ratio", 0.5, 2.0, 1.0, step=0.1, key='peg_ratio')

# --- Data Fetching and Caching ---
@st.cache_data(ttl=3600)
def fetch_data_bundle(ticker_symbol):
    """Fetches a complete bundle of data required for the analysis."""
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        if not info or info.get('trailingEps') is None:
             st.warning(f"Could not fetch complete data for {ticker_symbol}.")
             return {}
        
        data_bundle = {
            "info": info,
            "financials": stock.financials,
            "balance_sheet": stock.balance_sheet,
            "cash_flow": stock.cashflow,
            "quarterly_financials": stock.quarterly_financials,
            "history": stock.history(period="5y"),
            "news": stock.news
        }
        return data_bundle
    except Exception as e:
        st.error(f"Error fetching data bundle for {ticker_symbol}: {e}")
        return {}

# --- Valuation Functions ---
def calculate_dcf_fair_value(info, cash_flow, growth_rate, terminal_growth, discount_rate):
    try:
        op_cash = cash_flow.loc['Total Cash From Operating Activities'].iloc[0]
        cap_ex = cash_flow.loc['Capital Expenditure'].iloc[0]
        fcf = op_cash + cap_ex
        shares = info.get('sharesOutstanding')
        cash, debt = info.get('totalCash'), info.get('totalDebt')
        if not all([fcf, shares, cash, debt]): return None, {}
        
        future_fcf = [fcf * ((1 + growth_rate) ** y) for y in range(1, 6)]
        terminal_value = future_fcf[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
        discounted_fcf = [cf / ((1 + discount_rate) ** (i+1)) for i, cf in enumerate(future_fcf)]
        enterprise_value = sum(discounted_fcf) + (terminal_value / ((1 + discount_rate) ** 5))
        equity_value = enterprise_value - debt + cash
        fair_value = equity_value / shares
        
        details = {"FCF": fcf, "Growth Rate": growth_rate, "Discount Rate": discount_rate, "Terminal Growth": terminal_growth}
        return fair_value, details
    except (KeyError, TypeError, ZeroDivisionError):
        return None, {}

def calculate_graham_fair_value(info, graham_multiplier, max_pe):
    try:
        eps, bvps = info.get('trailingEps'), info.get('bookValue')
        if not eps or not bvps or eps <= 0: return None, {}
        raw_value = np.sqrt(graham_multiplier * eps * bvps)
        details = {"EPS": eps, "Book Value/Share": bvps}
        pe = info.get('trailingPE')
        if pe and pe > max_pe:
            return min(raw_value, eps * max_pe), details
        return raw_value, details
    except (TypeError, ValueError):
        return None, {}

def calculate_peg_fair_value(info, peg_ratio):
    try:
        eps = info.get('trailingEps')
        growth = info.get('earningsGrowth') or info.get('revenueGrowth') or 0.05
        if not eps or not growth or eps <= 0: return None, {}
        fair_value = eps * growth * 100 * peg_ratio
        details = {"EPS": eps, "Growth Rate": growth}
        return fair_value, details
    except (TypeError, ValueError):
        return None, {}

# --- Plotting Function (FIXED) ---
def plot_historical_ratios(hist_price, quarterly_financials, info):
    """Plots historical P/E and P/S ratios with robust calculations."""
    try:
        df = pd.DataFrame(index=hist_price.index)
        df['Close'] = hist_price['Close']
        
        fin_df = quarterly_financials.T
        fin_df.index = pd.to_datetime(fin_df.index)
        
        merged_df = pd.merge_asof(df.sort_index(), fin_df, left_index=True, right_index=True, direction='backward')
        
        merged_df['TTM EPS'] = merged_df['Diluted EPS'].rolling(window=4, min_periods=4).sum()
        merged_df['TTM Revenue'] = merged_df['Total Revenue'].rolling(window=4, min_periods=4).sum()
        
        shares = info.get('sharesOutstanding', 1)
        merged_df['TTM Sales Per Share'] = merged_df['TTM Revenue'] / shares
        
        merged_df['P/E Ratio'] = merged_df['Close'] / merged_df['TTM EPS']
        merged_df['P/S Ratio'] = merged_df['Close'] / merged_df['TTM Sales Per Share']
        
        merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        pe_upper = merged_df['P/E Ratio'].quantile(0.95)
        ps_upper = merged_df['P/S Ratio'].quantile(0.95)
        merged_df.loc[merged_df['P/E Ratio'] > pe_upper, 'P/E Ratio'] = np.nan
        merged_df.loc[merged_df['P/S Ratio'] > ps_upper, 'P/S Ratio'] = np.nan
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df['P/E Ratio'], mode='lines', name='P/E Ratio (TTM)'))
        fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df['P/S Ratio'], mode='lines', name='P/S Ratio (TTM)', yaxis='y2'))
        
        fig.update_layout(
            title='Historical Valuations (5-Year TTM)',
            yaxis=dict(title='P/E Ratio'),
            yaxis2=dict(title='P/S Ratio', overlaying='y', side='right'),
            legend=dict(x=0, y=1.1, orientation='h'), height=400
        )
        return fig
    except Exception as e:
        # st.error(f"Chart error: {e}")
        return None

# --- Main Application Logic ---
if not tickers:
    st.info("ℹ️ Please enter one or more stock ticker symbols to begin the analysis.")
else:
    for ticker in tickers:
        st.markdown(f"---")
        data = fetch_data_bundle(ticker)
        if not data:
            st.error(f"Could not process {ticker}. Please check the symbol and try again.")
            continue

        info = data["info"]
        st.header(f"{info.get('longName', ticker)} ({ticker})")
        st.caption(f"{info.get('sector', 'N/A')} | {info.get('industry', 'N/A')} | {info.get('country', 'N/A')}")
        
        # --- Valuation Calculation ---
        dcf_val, dcf_det = calculate_dcf_fair_value(info, data['cash_flow'], growth_rate, terminal_growth, discount_rate)
        graham_val, graham_det = calculate_graham_fair_value(info, graham_multiplier, max_pe)
        peg_val, peg_det = calculate_peg_fair_value(info, peg_ratio)
        
        dcf_adj = dcf_val * (1 - mos_dcf/100) if dcf_val else None
        graham_adj = graham_val * (1 - mos_graham/100) if graham_val else None
        peg_adj = peg_val * (1 - mos_peg/100) if peg_val else None
        
        fair_values = [v for v in [dcf_adj, graham_adj, peg_adj] if v]
        avg_fair_value = np.mean(fair_values) if fair_values else None
        current_price = info.get('currentPrice')

        # --- Summary Section ---
        st.subheader("Valuation & Analyst Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${current_price:,.2f}", delta=None)
        if avg_fair_value:
            col2.metric("Avg. Fair Value (with MoS)", f"${avg_fair_value:,.2f}", delta=f"{(avg_fair_value - current_price)/current_price:.2%}")
        else:
            col2.metric("Avg. Fair Value (with MoS)", "N/A")
        
        analyst_target = info.get('targetMeanPrice')
        if analyst_target:
            col3.metric("Analyst Mean Target", f"${analyst_target:,.2f}", delta=f"{(analyst_target - current_price)/current_price:.2%}")
        else:
            col3.metric("Analyst Mean Target", "N/A")

        # --- Detailed Valuation Section ---
        st.subheader("Intrinsic Value Breakdown")
        st.markdown("This section details the inputs and outputs of each valuation model.")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"##### Discounted Cash Flow (DCF)")
            st.metric("Fair Value", f"${dcf_val:,.2f}" if dcf_val else "N/A")
            st.metric(f"Value with {mos_dcf}% MoS", f"${dcf_adj:,.2f}" if dcf_adj else "N/A")
            with st.expander("Show Details"):
                st.json(dcf_det)
        with c2:
            st.markdown(f"##### Graham Formula")
            st.metric("Fair Value", f"${graham_val:,.2f}" if graham_val else "N/A")
            st.metric(f"Value with {mos_graham}% MoS", f"${graham_adj:,.2f}" if graham_adj else "N/A")
            with st.expander("Show Details"):
                st.json(graham_det)
        with c3:
            st.markdown(f"##### PEG Ratio")
            st.metric("Fair Value", f"${peg_val:,.2f}" if peg_val else "N/A")
            st.metric(f"Value with {mos_peg}% MoS", f"${peg_adj:,.2f}" if peg_adj else "N/A")
            with st.expander("Show Details"):
                st.json(peg_det)

        # --- Historical Ratios Chart (FIXED) ---
        st.subheader("Historical Valuation Context")
        fig_ratios = plot_historical_ratios(data["history"], data["quarterly_financials"], info)
        if fig_ratios:
            st.plotly_chart(fig_ratios, use_container_width=True)
        else:
            st.info("Could not generate historical ratio charts for this stock.")

        # --- Analyst Ratings & Key Metrics ---
        st.subheader("Analyst Consensus & Key Metrics")
        k_c1, k_c2 = st.columns(2)
        with k_c1:
            st.markdown("##### Analyst Ratings")
            st.metric("Recommendation", info.get('recommendationKey', 'N/A').upper())
            st.metric("Number of Analysts", info.get('numberOfAnalystOpinions', 'N/A'))
            st.markdown(f"**Price Target Range:** ${info.get('targetLowPrice', 0):.2f} - ${info.get('targetHighPrice', 0):.2f}")
        with k_c2:
            st.markdown("##### Key Ratios")
            st.metric("Trailing P/E", f"{info.get('trailingPE', 0):.2f}")
            st.metric("Forward P/E", f"{info.get('forwardPE', 0):.2f}")
            st.metric("Price to Sales (TTM)", f"{info.get('priceToSalesTrailing12Months', 0):.2f}")
            
        # --- Financial Statements Expander ---
        with st.expander("View Full Financial Statements (Annual)"):
            st.markdown("###### Income Statement")
            st.dataframe(data['financials'])
            st.markdown("###### Balance Sheet")
            st.dataframe(data['balance_sheet'])
            st.markdown("###### Cash Flow Statement")
            st.dataframe(data['cash_flow'])

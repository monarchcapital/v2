# pages/fundamentals.py - A Streamlit page for fundamental stock analysis and valuation.

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="Fundamental Analysis", page_icon="🧾", layout="wide")

# --- Helper Functions for Data and Valuation ---

@st.cache_data
def get_fundamental_data(ticker):
    """Fetches all necessary fundamental data for a given stock ticker from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info or 'currentPrice' not in info:
            return None, f"Could not retrieve valid data for ticker '{ticker}'. Please check the symbol."
            
        return {
            "info": info,
            "financials": stock.financials,
            "balance_sheet": stock.balance_sheet,
            "cashflow": stock.cashflow
        }, None
    except Exception as e:
        return None, f"An error occurred while fetching data for {ticker}: {e}"

def calculate_cagr(financials):
    """Calculates the Compound Annual Growth Rate of revenue."""
    try:
        revenue = financials.loc['Total Revenue']
        # Ensure we have at least two years of data
        if len(revenue) > 1:
            start_value = revenue.iloc[-1]
            end_value = revenue.iloc[0]
            num_years = len(revenue) - 1
            # Avoid division by zero and ensure start value is positive
            if start_value > 0:
                cagr = (end_value / start_value) ** (1 / num_years) - 1
                return cagr
    except (KeyError, IndexError):
        return None
    return None

def calculate_dcf_fair_value(fcf, growth_rate, terminal_growth_rate, discount_rate, shares_outstanding):
    """Calculates DCF fair value based on provided inputs."""
    if not all([isinstance(val, (int, float)) for val in [fcf, shares_outstanding] if val is not None]):
        return "N/A"
    
    future_fcf = [fcf * (1 + growth_rate) ** i for i in range(1, 6)]
    terminal_value = (future_fcf[-1] * (1 + terminal_growth_rate)) / (discount_rate - terminal_growth_rate)
    discounted_fcf = [fcf_val / (1 + discount_rate) ** (i + 1) for i, fcf_val in enumerate(future_fcf)]
    discounted_terminal_value = terminal_value / (1 + discount_rate) ** 5
    total_intrinsic_value = sum(discounted_fcf) + discounted_terminal_value
    
    return total_intrinsic_value / shares_outstanding

def calculate_graham_number(eps, book_value_per_share):
    """Calculates the Graham Number."""
    if not all([isinstance(val, (int, float)) and val > 0 for val in [eps, book_value_per_share] if val is not None]):
        return "N/A"
    return np.sqrt(22.5 * eps * book_value_per_share)

def calculate_ddm_fair_value(last_dividend, required_rate, dividend_growth_rate):
    """Calculates DDM fair value."""
    if not all([isinstance(val, (int, float)) for val in [last_dividend, required_rate, dividend_growth_rate] if val is not None]):
        return "N/A"
    if required_rate <= dividend_growth_rate:
        return "Growth > Rate"
    return last_dividend / (required_rate - dividend_growth_rate)

# --- Main UI ---
st.title("🧾 Fundamental Analysis & Stock Valuation")
st.markdown("Enter a stock ticker to get its latest financial data, fair value estimations from various models, and an overall investment signal.")

ticker_input = st.text_input("Enter Stock Ticker", value="AAPL").upper()

if st.button("Analyze Stock", use_container_width=True):
    if not ticker_input:
        st.warning("Please enter a stock ticker.")
    else:
        with st.spinner(f"Fetching and analyzing data for {ticker_input}..."):
            stock_data, error = get_fundamental_data(ticker_input)

            if error:
                st.error(error)
            else:
                info = stock_data['info']
                current_price = info.get('currentPrice', 'N/A')

                st.header(f"Key Metrics for {info.get('longName', ticker_input)}")
                cols = st.columns(4)
                cols[0].metric("Current Price", f"${current_price:,.2f}" if isinstance(current_price, (int, float)) else "N/A")
                cols[1].metric("P/E Ratio", f"{info.get('trailingPE'):.2f}" if info.get('trailingPE') else "N/A")
                cols[2].metric("P/B Ratio", f"{info.get('priceToBook'):.2f}" if info.get('priceToBook') else "N/A")
                cols[3].metric("Debt to Equity", f"{info.get('debtToEquity'):.2f}" if info.get('debtToEquity') else "N/A")

                st.markdown("---")
                
                st.header("Analyst Consensus")
                analyst_cols = st.columns(4)
                analyst_cols[0].metric("Mean Target Price", f"${info.get('targetMeanPrice'):,.2f}" if info.get('targetMeanPrice') else "N/A")
                analyst_cols[1].metric("High Target", f"${info.get('targetHighPrice'):,.2f}" if info.get('targetHighPrice') else "N/A")
                analyst_cols[2].metric("Low Target", f"${info.get('targetLowPrice'):,.2f}" if info.get('targetLowPrice') else "N/A")
                analyst_cols[3].info(f"Recommendation: **{info.get('recommendationKey', 'N/A').upper()}**")

                st.markdown("---")
                st.header("Fair Value Estimation Models")

                # --- DCF Model Expander ---
                with st.expander("Discounted Cash Flow (DCF) Model", expanded=True):
                    st.latex(r'''
                    \text{Fair Value} = \frac{\sum_{t=1}^{n} \frac{FCF_t}{(1+r)^t} + \frac{\text{Terminal Value}}{(1+r)^n}}{\text{Shares Outstanding}}
                    ''')
                    
                    fcf = stock_data['cashflow'].loc['Free Cash Flow'].iloc[0] if 'Free Cash Flow' in stock_data['cashflow'].index else None
                    shares = info.get('sharesOutstanding')
                    
                    # Calculate historical CAGR as a default for the growth rate
                    historical_cagr = calculate_cagr(stock_data['financials'])
                    default_growth_rate = historical_cagr if historical_cagr is not None else 0.05

                    st.write(f"**Latest Free Cash Flow (FCF):** {'${:,.0f}'.format(fcf) if fcf else 'N/A'}")
                    st.write(f"**Shares Outstanding:** {shares:,.0f}" if shares else "N/A")
                    st.info(f"**Calculated Historical Revenue CAGR:** {default_growth_rate:.2%} (Used as default growth rate)")

                    dcf_cols = st.columns(3)
                    g_rate = dcf_cols[0].slider("5-Year Growth Rate (g)", -0.10, 0.50, default_growth_rate, 0.01, key='dcf_g')
                    t_rate = dcf_cols[1].slider("Perpetual Growth Rate", 0.0, 0.10, 0.02, 0.005, key='dcf_t')
                    d_rate = dcf_cols[2].slider("Discount Rate (WACC)", 0.05, 0.20, 0.075, 0.005, key='dcf_d')
                    
                    dcf_value = calculate_dcf_fair_value(fcf, g_rate, t_rate, d_rate, shares)
                    st.metric("Calculated DCF Fair Value", f"${dcf_value:,.2f}" if isinstance(dcf_value, (int, float)) else dcf_value)

                # --- Graham Number Expander ---
                with st.expander("Graham Number Model"):
                    st.latex(r'''
                    \text{Graham Number} = \sqrt{22.5 \times \text{EPS} \times \text{Book Value per Share}}
                    ''')
                    eps = info.get('trailingEps')
                    bvps = info.get('bookValue')
                    
                    st.write(f"**Trailing EPS:** {'${:,.2f}'.format(eps) if eps else 'N/A'}")
                    st.write(f"**Book Value per Share:** {'${:,.2f}'.format(bvps) if bvps else 'N/A'}")
                    
                    graham_value = calculate_graham_number(eps, bvps)
                    st.metric("Calculated Graham Number", f"${graham_value:,.2f}" if isinstance(graham_value, (int, float)) else graham_value)

                # --- DDM Model Expander ---
                with st.expander("Dividend Discount Model (DDM)"):
                    st.latex(r'''
                    \text{Fair Value} = \frac{\text{Dividend per Share}}{(\text{Required Rate of Return} - \text{Dividend Growth Rate})}
                    ''')
                    dividend = info.get('dividendRate')
                    st.write(f"**Annual Dividend per Share:** {'${:,.2f}'.format(dividend) if dividend else 'Not a dividend stock'}")

                    ddm_cols = st.columns(2)
                    req_rate = ddm_cols[0].slider("Required Rate of Return", 0.05, 0.20, 0.08, 0.005, key='ddm_req')
                    div_growth = ddm_cols[1].slider("Dividend Growth Rate", 0.0, 0.10, 0.02, 0.005, key='ddm_growth')

                    ddm_value = calculate_ddm_fair_value(dividend, req_rate, div_growth)
                    st.metric("Calculated DDM Fair Value", f"${ddm_value:,.2f}" if isinstance(ddm_value, (int, float)) else ddm_value)

                # --- Final Signal ---
                st.markdown("---")
                st.header("Investment Signal")
                valid_values = [v for v in [dcf_value, graham_value, ddm_value] if isinstance(v, (int, float))]
                average_fair_value = np.mean(valid_values) if valid_values else "N/A"
                
                if isinstance(average_fair_value, (int, float)) and isinstance(current_price, (int, float)):
                    upside = ((average_fair_value - current_price) / current_price) * 100
                    if upside > 20: signal, color = "Strong Buy", "success"
                    elif 10 < upside <= 20: signal, color = "Buy", "success"
                    elif -10 <= upside <= 10: signal, color = "Hold", "warning"
                    else: signal, color = "Sell", "error"
                    
                    getattr(st, color)(f"**Signal: {signal}**")
                    st.write(f"The average fair value from the models is **${average_fair_value:,.2f}**, suggesting a potential upside of **{upside:.2f}%** from the current price of **${current_price:,.2f}**.")
                else:
                    st.info("Could not generate a definitive signal due to missing valuation data.")

                st.markdown("---")
                st.header("Financial Statements")
                with st.expander("Income Statement (Annual)"):
                    st.dataframe(stock_data['financials'].style.format('${:,.0f}'))
                with st.expander("Balance Sheet (Annual)"):
                    st.dataframe(stock_data['balance_sheet'].style.format('${:,.0f}'))
                with st.expander("Cash Flow Statement (Annual)"):
                    st.dataframe(stock_data['cashflow'].style.format('${:,.0f}'))

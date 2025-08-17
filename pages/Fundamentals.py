# pages/fundamentals.py - A Streamlit page for fundamental stock analysis and valuation.

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# --- Page Configuration ---
st.set_page_config(page_title="Fundamental Analysis", page_icon="🧾", layout="wide")

# --- Hide Streamlit Menu & Footer ---
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- API Key Configuration ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except (KeyError, FileNotFoundError):
    st.warning("Google AI API Key is not found. News analysis will be disabled.", icon="⚠️")
    GOOGLE_API_KEY = None

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

@st.cache_data(ttl=3600) # Cache news for 60 minutes
def get_synthesized_stock_summary(query):
    """
    Uses the Gemini API to generate a synthesized financial summary for a specific stock.
    """
    if not GOOGLE_API_KEY:
        return "API Key Missing. Cannot generate news summary."
        
    try:
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        prompt = f"""
        As an expert financial market analyst, generate a concise market summary based on the absolute latest news for the stock '{query}'.

        Your summary should be structured for a quick read:
        1.  **Main Headline:** A compelling title that captures the essence of the recent news.
        2.  **Key Takeaways:** Start with 2-3 bullet points summarizing the most critical information (e.g., earnings surprises, new product launches, regulatory news).
        3.  **Brief Analysis:** A short paragraph explaining the potential impact of this news on the stock's future performance.
        4.  **Formatting:** Use Markdown with bold headings for each section.
        """
        response = model.generate_content(prompt)
        return response.text

    except google_exceptions.GoogleAPICallError as e:
        return f"Could not generate summary due to a Google API error: {e}"
    except Exception as e:
        return f"An unexpected error occurred while generating the news summary: {e}"


def calculate_cagr(financials):
    """Calculates the Compound Annual Growth Rate of revenue."""
    try:
        revenue = financials.loc['Total Revenue']
        if len(revenue) > 1:
            start_value = revenue.iloc[-1]
            end_value = revenue.iloc[0]
            # --- BUG FIX: Correct CAGR Calculation ---
            num_years = len(revenue) - 1
            if start_value > 0 and num_years > 0:
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
    
    if shares_outstanding == 0: return "N/A"
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
st.markdown("Enter a stock ticker to get its latest financial data, fair value estimations, and an AI-powered news summary.")

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
            # --- Create two-column layout ---
            col1, col2 = st.columns([2, 1])

            with col1:
                info = stock_data['info']
                current_price = info.get('currentPrice', 'N/A')

                st.header(f"Key Metrics for {info.get('longName', ticker_input)}")
                metric_cols = st.columns(4)
                metric_cols[0].metric("Current Price", f"${current_price:,.2f}" if isinstance(current_price, (int, float)) else "N/A")
                metric_cols[1].metric("P/E Ratio", f"{info.get('trailingPE'):.2f}" if info.get('trailingPE') else "N/A")
                metric_cols[2].metric("P/B Ratio", f"{info.get('priceToBook'):.2f}" if info.get('priceToBook') else "N/A")
                metric_cols[3].metric("Debt to Equity", f"{info.get('debtToEquity'):.2f}" if info.get('debtToEquity') else "N/A")

                st.markdown("---")
                
                st.header("Analyst Consensus")
                analyst_cols = st.columns(4)
                analyst_cols[0].metric("Mean Target Price", f"${info.get('targetMeanPrice'):,.2f}" if info.get('targetMeanPrice') else "N/A")
                analyst_cols[1].metric("High Target", f"${info.get('targetHighPrice'):,.2f}" if info.get('targetHighPrice') else "N/A")
                analyst_cols[2].metric("Low Target", f"${info.get('targetLowPrice'):,.2f}" if info.get('targetLowPrice') else "N/A")
                analyst_cols[3].info(f"Recommendation: **{info.get('recommendationKey', 'N/A').upper()}**")

                st.markdown("---")
                st.header("Fair Value Estimation Models")

                with st.expander("Discounted Cash Flow (DCF) Model", expanded=True):
                    fcf = stock_data['cashflow'].loc['Free Cash Flow'].iloc[0] if 'Free Cash Flow' in stock_data['cashflow'].index else None
                    shares = info.get('sharesOutstanding')
                    historical_cagr = calculate_cagr(stock_data['financials'])
                    default_growth_rate = historical_cagr if historical_cagr is not None else 0.05
                    st.write(f"**Latest Free Cash Flow (FCF):** {'${:,.0f}'.format(fcf) if fcf else 'N/A'}")
                    st.info(f"**Calculated Historical Revenue CAGR:** {default_growth_rate:.2%} (Used as default growth rate)")
                    dcf_cols = st.columns(3)
                    g_rate = dcf_cols[0].slider("5-Year Growth Rate (g)", -0.10, 0.50, default_growth_rate, 0.01, key='dcf_g')
                    t_rate = dcf_cols[1].slider("Perpetual Growth Rate", 0.0, 0.10, 0.02, 0.005, key='dcf_t')
                    d_rate = dcf_cols[2].slider("Discount Rate (WACC)", 0.05, 0.20, 0.075, 0.005, key='dcf_d')
                    dcf_value = calculate_dcf_fair_value(fcf, g_rate, t_rate, d_rate, shares)
                    st.metric("Calculated DCF Fair Value", f"${dcf_value:,.2f}" if isinstance(dcf_value, (int, float)) else dcf_value)

                with st.expander("Graham Number Model"):
                    eps = info.get('trailingEps')
                    bvps = info.get('bookValue')
                    graham_value = calculate_graham_number(eps, bvps)
                    st.metric("Calculated Graham Number", f"${graham_value:,.2f}" if isinstance(graham_value, (int, float)) else graham_value)

                with st.expander("Dividend Discount Model (DDM)"):
                    dividend = info.get('dividendRate')
                    ddm_cols = st.columns(2)
                    req_rate = ddm_cols[0].slider("Required Rate of Return", 0.05, 0.20, 0.08, 0.005, key='ddm_req')
                    div_growth = ddm_cols[1].slider("Dividend Growth Rate", 0.0, 0.10, 0.02, 0.005, key='ddm_growth')
                    ddm_value = calculate_ddm_fair_value(dividend, req_rate, div_growth)
                    st.metric("Calculated DDM Fair Value", f"${ddm_value:,.2f}" if isinstance(ddm_value, (int, float)) else ddm_value)

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
                    st.write(f"The average fair value is **${average_fair_value:,.2f}**, suggesting a potential upside of **{upside:.2f}%** from the current price of **${current_price:,.2f}**.")
                else:
                    st.info("Could not generate a definitive signal due to missing valuation data.")

            with col2:
                st.header("📰 AI News Analysis")
                if GOOGLE_API_KEY:
                    with st.spinner(f"Getting latest news for {ticker_input}..."):
                        news_summary = get_synthesized_stock_summary(ticker_input)
                        with st.container(border=True):
                            st.markdown(news_summary)
                else:
                    st.warning("Enter a Google AI API Key in your secrets to enable this feature.")

            # --- Financial Statements (Full Width Below Columns) ---
            st.markdown("---")
            st.header("Financial Statements")
            with st.expander("Income Statement (Annual)"):
                st.dataframe(stock_data['financials'].style.format('${:,.0f}'))
            with st.expander("Balance Sheet (Annual)"):
                st.dataframe(stock_data['balance_sheet'].style.format('${:,.0f}'))
            with st.expander("Cash Flow Statement (Annual)"):
                st.dataframe(stock_data['cashflow'].style.format('${:,.0f}'))

# pages/Option_Pricing.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm # For cumulative distribution function
import yfinance as yf
from datetime import date, timedelta, datetime
import math # For log and sqrt

import config
from utils import download_data # Reusing download_data for historical volatility

st.set_page_config(page_title="Monarch: Option Pricing", layout="wide")

st.header("📈 Option Pricing Calculator (Black-Scholes Model)")
st.markdown("""
This page allows you to calculate theoretical Call and Put option prices using the **Black-Scholes model**.
You can input various parameters or use the live stock price and calculated historical volatility.

**Understanding the Inputs:**
- **Stock Price (S):** The current market price of the underlying asset.
- **Strike Price (K):** The price at which the option holder can buy (Call) or sell (Put) the underlying asset.
- **Time to Expiration (T):** The remaining time until the option expires, expressed in years.
- **Risk-Free Rate (r):** The theoretical rate of return of an investment with zero risk, expressed as an annual decimal (e.g., 0.05 for 5%).
- **Volatility (σ):** A measure of the expected fluctuation in the stock's price, expressed as an annualized decimal (e.g., 0.20 for 20%). This is the most crucial and often hardest input to estimate. You can use historical volatility as a proxy or input an implied volatility if known.

**Disclaimer:**
The Black-Scholes model relies on several assumptions (e.g., constant volatility, no dividends, European-style options).
This calculator is for **educational and illustrative purposes only** and should not be used for actual trading decisions.
Real-world option pricing is more complex and often uses implied volatility derived from market prices.
""")

# --- Black-Scholes Formula Implementation ---
def black_scholes(S, K, T, r, sigma, option_type):
    """
    Calculates the theoretical price of a European option using the Black-Scholes model.

    Args:
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to expiration (in years).
        r (float): Risk-free interest rate (annual, decimal).
        sigma (float): Volatility (annual, decimal).
        option_type (str): 'call' or 'put'.

    Returns:
        float: Theoretical option price, or NaN if inputs are invalid.
    """
    if not all(isinstance(arg, (int, float)) for arg in [S, K, T, r, sigma]):
        return np.nan
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan # Invalid inputs for log/sqrt

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    return price

# --- Calculate Historical Volatility ---
@st.cache_data(ttl=3600) # Cache historical data for 1 hour
def calculate_historical_volatility(ticker_symbol, period_days=252):
    """
    Calculates the annualized historical volatility (standard deviation of log returns).
    
    Args:
        ticker_symbol (str): The stock ticker symbol.
        period_days (int): Number of trading days to consider for volatility calculation (approx 252 for 1 year).
        
    Returns:
        float: Annualized historical volatility, or None on failure.
    """
    data = download_data(ticker_symbol, period=f"{math.ceil(period_days/252) + 1}y") # Fetch enough data
    if data.empty or len(data) < period_days + 1:
        st.warning(f"Not enough historical data for {ticker_symbol} to calculate volatility over {period_days} days.")
        return None
    
    # Ensure data is sorted by date
    data = data.sort_values(by='Date').tail(period_days + 1)
    
    # Calculate daily log returns
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Calculate daily standard deviation and annualize it
    daily_volatility = data['Log_Return'].std()
    annualized_volatility = daily_volatility * np.sqrt(252) # Assuming 252 trading days in a year
    
    return annualized_volatility

# --- Sidebar for Inputs ---
st.sidebar.header("🛠️ Option Parameters")

ticker = st.sidebar.text_input("Underlying Stock Ticker:", value="AAPL").upper()

# Fetch Live Stock Price
live_price = None
if st.sidebar.button(f"Fetch Live Price for {ticker}"):
    try:
        live_ticker_info = yf.Ticker(ticker).info
        live_price = live_ticker_info.get('currentPrice')
        if live_price is None:
            st.sidebar.warning(f"Could not fetch live price for {ticker}.")
        else:
            st.sidebar.success(f"Live Price for {ticker}: ${live_price:,.2f}")
    except Exception as e:
        st.sidebar.error(f"Error fetching live price for {ticker}: {e}")

S = st.sidebar.number_input("Current Stock Price (S):", value=live_price if live_price is not None else 150.0, min_value=0.01, format="%.2f")
K = st.sidebar.number_input("Strike Price (K):", value=155.0, min_value=0.01, format="%.2f")

st.sidebar.subheader("Time & Rate")
# Time to Expiration
expiry_date = st.sidebar.date_input("Expiration Date:", value=date.today() + timedelta(days=90))
T = (expiry_date - date.today()).days / 365.0
st.sidebar.info(f"Time to Expiration (T): {T:.4f} years")

r = st.sidebar.slider("Risk-Free Rate (r, %):", 0.0, 10.0, 5.0, 0.1) / 100 # Convert to decimal

st.sidebar.subheader("Volatility (σ)")
# Volatility Input
sigma_input = st.sidebar.number_input("Volatility (σ, %):", value=20.0, min_value=0.1, max_value=100.0, step=0.1, format="%.2f") / 100 # Convert to decimal

# Calculate Historical Volatility Option
calc_hist_vol = st.sidebar.checkbox("Calculate Historical Volatility?", value=False)
historical_vol_period = st.sidebar.slider("Historical Volatility Period (Days):", 30, 252, 60)

if calc_hist_vol and ticker:
    if st.sidebar.button(f"Calculate Historical Volatility for {ticker}"):
        hist_vol = calculate_historical_volatility(ticker, historical_vol_period)
        if hist_vol is not None:
            st.session_state['calculated_hist_vol'] = hist_vol * 100 # Store as percentage
            st.sidebar.success(f"Historical Volatility: {st.session_state['calculated_hist_vol']:.2f}%")
        else:
            st.sidebar.error("Could not calculate historical volatility.")

# Auto-populate sigma if historical volatility was calculated
if 'calculated_hist_vol' in st.session_state and calc_hist_vol:
    sigma_input = st.session_state['calculated_hist_vol'] / 100 # Use it as default for the slider

# --- Main Content Area ---
st.subheader("Calculated Option Prices")

if st.button("Calculate Option Prices", type="primary"):
    if S <= 0 or K <= 0 or T <= 0 or r < 0 or sigma_input <= 0:
        st.error("Please ensure all inputs (Stock Price, Strike Price, Time to Expiration, Volatility) are positive and valid.")
    else:
        try:
            call_price = black_scholes(S, K, T, r, sigma_input, 'call')
            put_price = black_scholes(S, K, T, r, sigma_input, 'put')

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Call Option Price", f"${call_price:,.2f}" if not np.isnan(call_price) else "N/A")
            with col2:
                st.metric("Put Option Price", f"${put_price:,.2f}" if not np.isnan(put_price) else "N/A")
            st.markdown("---")

            st.subheader("Input Summary")
            summary_data = {
                "Parameter": ["Stock Price (S)", "Strike Price (K)", "Time to Expiration (T)", "Risk-Free Rate (r)", "Volatility (σ)"],
                "Value": [f"${S:,.2f}", f"${K:,.2f}", f"{T:.4f} years", f"{r*100:.2f}%", f"{sigma_input*100:.2f}%"]
            }
            st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during calculation: {e}")
            st.info("Please check your input values. Ensure they are numerical and appropriate for the Black-Scholes model.")

else:
    st.info("Adjust the parameters in the sidebar and click 'Calculate Option Prices' to see the results.")


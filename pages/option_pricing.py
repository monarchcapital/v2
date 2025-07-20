# pages/option_pricing.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import yfinance as yf
from datetime import date, timedelta
import math
import plotly.graph_objects as go

st.set_page_config(page_title="Monarch: Option Pricing", layout="wide")

st.header("📈 Option Pricing & Greeks Calculator")
st.markdown("""
This tool calculates theoretical option prices and "the Greeks" using the **Black-Scholes model**. 
You can analyze how an option's price and risk profile change with market conditions.
""")

# --- Black-Scholes and Greeks Formulas ---
def calculate_greeks(S, K, T, r, sigma):
    """Calculates the Black-Scholes price and the main Greeks."""
    if not all(isinstance(arg, (int, float)) for arg in [S, K, T, r, sigma]) or T <= 0 or S <= 0 or K <= 0 or sigma <= 0:
        return {greek: np.nan for greek in ['call_price', 'put_price', 'call_delta', 'put_delta', 'gamma', 'vega', 'call_theta', 'put_theta']}

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    # Prices
    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    # Greeks
    call_delta = norm.cdf(d1)
    put_delta = call_delta - 1
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100 # per 1% change in vol
    
    call_theta = (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365 # per day
    put_theta = (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365 # per day
    
    return {
        'call_price': call_price, 'put_price': put_price,
        'call_delta': call_delta, 'put_delta': put_delta,
        'gamma': gamma, 'vega': vega,
        'call_theta': call_theta, 'put_theta': put_theta
    }

@st.cache_data(ttl=3600)
def calculate_historical_volatility(ticker_symbol, period_days=252):
    """Calculates the annualized historical volatility."""
    end_date = date.today()
    start_date = end_date - timedelta(days=period_days * 2) # Fetch more data to ensure we have enough trading days
    
    data = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)
    if data.empty or len(data) < period_days:
        st.warning(f"Not enough historical data for {ticker_symbol} to calculate volatility.")
        return None
    
    data = data.tail(period_days)
    log_returns = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    annualized_volatility = log_returns.std() * np.sqrt(252)
    return annualized_volatility

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("🛠️ Option Parameters")
    ticker = st.text_input("Underlying Stock Ticker:", value="AAPL").upper()

    # Fetch Live Data
    if st.button(f"Fetch Live Data for {ticker}", type="primary"):
        try:
            live_ticker_info = yf.Ticker(ticker).info
            st.session_state.live_price = live_ticker_info.get('currentPrice', live_ticker_info.get('previousClose'))
            st.session_state.hist_vol = calculate_historical_volatility(ticker)
            if st.session_state.live_price:
                st.success(f"Live Price: ${st.session_state.live_price:,.2f}")
            if st.session_state.hist_vol:
                 st.success(f"1-Yr Hist. Vol: {st.session_state.hist_vol:.2%}")
        except Exception:
            st.error(f"Failed to fetch data for {ticker}.")

    S = st.number_input("Current Stock Price (S):", value=st.session_state.get('live_price', 150.0), min_value=0.01, format="%.2f")
    K = st.number_input("Strike Price (K):", value=155.0, min_value=0.01, format="%.2f")
    
    expiry_date = st.date_input("Expiration Date:", value=date.today() + timedelta(days=90))
    T = max((expiry_date - date.today()).days / 365.0, 1e-9) # Avoid T=0
    st.sidebar.caption(f"Time to Expiration (T): {T:.4f} years")

    r = st.slider("Risk-Free Rate (r, %):", 0.0, 10.0, 5.0, 0.1) / 100
    sigma = st.slider("Volatility (σ, %):", 0.1, 100.0, st.session_state.get('hist_vol', 0.20) * 100, 0.5) / 100

# --- Main Content Area ---
results = calculate_greeks(S, K, T, r, sigma)

st.subheader("Calculated Option Prices & Greeks")
col1, col2 = st.columns(2)
with col1:
    st.metric("Call Option Price", f"${results['call_price']:,.3f}", delta=f"{results['call_delta']:.3f} Delta", help="Theoretical price of the call option.")
with col2:
    st.metric("Put Option Price", f"${results['put_price']:,.3f}", delta=f"{results['put_delta']:.3f} Delta", help="Theoretical price of the put option.")

st.markdown("---")
st.subheader("The Greeks: Risk Analysis")
gcol1, gcol2, gcol3 = st.columns(3)
gcol1.metric("Gamma", f"{results['gamma']:.4f}", help="Rate of change of Delta. Indicates how much Delta will change for a $1 move in the stock.")
gcol2.metric("Vega", f"${results['vega']:.4f}", help="Price change for a 1% change in volatility. Shows sensitivity to volatility.")
gcol3.metric("Call/Put Theta", f"${results['call_theta']:.4f} / ${results['put_theta']:.4f}", help="Price decay per day (time decay). Shows how much value the option loses daily.")

# --- Visualizations ---
st.markdown("---")
st.subheader("Payoff & Profit/Loss Diagram at Expiration")

# Create data for the payoff chart
stock_price_range = np.linspace(S * 0.7, S * 1.3, 100)
call_payoff = np.maximum(stock_price_range - K, 0)
put_payoff = np.maximum(K - stock_price_range, 0)
call_profit = call_payoff - results['call_price']
put_profit = put_payoff - results['put_price']

fig = go.Figure()
fig.add_trace(go.Scatter(x=stock_price_range, y=call_profit, mode='lines', name='Call Profit/Loss', line=dict(color='green')))
fig.add_trace(go.Scatter(x=stock_price_range, y=put_profit, mode='lines', name='Put Profit/Loss', line=dict(color='red')))
fig.add_hline(y=0, line_dash="dash", line_color="gray")
fig.add_vline(x=K, line_dash="dash", line_color="blue", annotation_text="Strike Price")
fig.update_layout(
    title="Profit/Loss at Expiration",
    xaxis_title="Stock Price at Expiration ($)",
    yaxis_title="Profit / Loss per Share ($)"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Greeks Sensitivity Analysis")
# Sensitivity analysis for Delta
price_range_sens = np.linspace(S * 0.8, S * 1.2, 50)
delta_sens_data = [calculate_greeks(p, K, T, r, sigma) for p in price_range_sens]

sens_fig = go.Figure()
sens_fig.add_trace(go.Scatter(x=price_range_sens, y=[d['call_delta'] for d in delta_sens_data], mode='lines', name='Call Delta', line=dict(color='green')))
sens_fig.add_trace(go.Scatter(x=price_range_sens, y=[d['put_delta'] for d in delta_sens_data], mode='lines', name='Put Delta', line=dict(color='red')))
sens_fig.add_trace(go.Scatter(x=price_range_sens, y=[d['gamma'] for d in delta_sens_data], mode='lines', name='Gamma', line=dict(color='purple', dash='dot'), yaxis="y2"))

sens_fig.update_layout(
    title="Delta and Gamma vs. Stock Price",
    xaxis_title="Stock Price ($)",
    yaxis_title="Delta",
    yaxis=dict(title="Delta"),
    yaxis2=dict(title="Gamma", overlaying="y", side="right", showgrid=False),
    legend=dict(x=0.01, y=0.99)
)
st.plotly_chart(sens_fig, use_container_width=True)

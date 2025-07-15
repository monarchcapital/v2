# streamlit_app.py
import streamlit as st

st.set_page_config(
    page_title="Monarch: Stock Price Predictor",
    page_icon="👑",
    layout="wide"
)

st.title("Welcome to Monarch: A Comprehensive Stock Analysis Suite 👑")

# --- ADDED IMAGE ---
st.image(
    "https://images.pexels.com/photos/7567569/pexels-photo-7567569.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1",
    caption="Monarch Stock Analysis Suite"
)

st.markdown("""
Welcome to Monarch, your all-in-one platform for financial market analysis and prediction.
Navigate through the pages on the left to explore the full suite of tools:

- **📈 Home:** Perform an in-depth analysis of a single stock, visualize historical data with technical indicators, and forecast future prices using various machine learning models.
- **⚡ Live Predictor:** Get the current live price and a next-day forecast for any stock.
- **💰 Option Pricing:** Calculate theoretical Call and Put option prices using the Black-Scholes model.
- **📋 My Watchlist:** Keep track of multiple stocks, get quick next-day predictions, and leverage ensemble models for robust forecasts.
- **📊 Indian Stock Fundamentals:** Dive deep into the financial health of Indian companies with multiple valuation models like DCF, Graham Value, and PEG Ratio.
- **🇮🇳 NIFTY 50 Analysis:** Compare top-down vs. bottom-up forecasting methodologies for India's benchmark index.
- **🏢 NIFTY 50 Sector Analysis:** Get a sector-wise breakdown and simple trading signals for NIFTY 50 stocks.
- **📈 All NSE Stocks Analysis:** Get a sector-wise breakdown and simple trading signals for a sample of all NSE stocks.
- **🌍 Global Index Analysis:** Analyze and forecast major global indices like the S&P 500, Crude Oil, and Gold.
- **🏆 Model Comparison Backtest:** Rigorously test and compare the historical performance of prediction models using a walk-forward analysis.
""")

st.info("""
**Important Notes:**
- Data is fetched from Yahoo Finance. Ensure your ticker symbols are correct (e.g., `AAPL` for Apple, `RELIANCE.NS` for Reliance Industries).
- All predictions and analyses are for informational and educational purposes only and should not be considered financial advice.
""")

# The actual content of the pages will be loaded from the 'pages' directory automatically by Streamlit.

# streamlit_app.py
import streamlit as st

st.set_page_config(
    page_title="Monarch: Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("Welcome to Monarch: Stock Price Predictor 👑")
st.markdown("""
Navigate through the pages on the left to:
- **📈 Home:** Analyze a single stock in depth, visualize historical data with technical indicators, and forecast future prices using machine learning models.
- **📋 My Watchlist:** Keep track of multiple stocks and get quick next-day predictions for them.
""")

st.info("""
**Important Notes:**
- Data is fetched from Yahoo Finance. Ensure your ticker symbols are correct (e.g., `AAPL` for Apple, `RELIANCE.NS` for Reliance in India).
- Predictions are for informational and educational purposes only and should not be considered financial advice.
""")

# You can add more global information or welcome messages here if needed.
# The actual content of the pages will be loaded from the 'pages' directory automatically by Streamlit.


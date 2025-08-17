# pages/Market_Pulse.py - A daily newsletter-style page for market analysis and news.

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
import plotly.graph_objects as go
from googlesearch import search
import numpy as np
from newspaper import Article, Config

# --- Page Configuration ---
st.set_page_config(page_title="Market Pulse", page_icon="📰", layout="wide")

# --- Helper Functions ---

@st.cache_data(ttl=1800) # Cache data for 30 minutes
def get_market_data(tickers):
    """
    Fetches and processes market data for a list of tickers, robustly handling holidays.
    This logic is adapted from the working Home.py page for maximum reliability.
    """
    try:
        # 1. Find the last two valid trading days using a reliable index
        end_date = date.today()
        start_date = end_date - timedelta(days=15)
        
        # Use a major global index like S&P 500 to find recent trading days
        ref_hist = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
        ref_hist.dropna(inplace=True)
        
        if len(ref_hist) < 2:
            return None # Not enough data to compare

        last_day = ref_hist.index[-2]
        previous_day = ref_hist.index[-3]

        # 2. Download all ticker data for a slightly wider period to ensure coverage
        fetch_start = previous_day.date() - timedelta(days=5)
        fetch_end = last_day.date() + timedelta(days=1)
        all_data = yf.download(tickers, start=fetch_start, end=fetch_end, progress=False, timeout=10)
        
        if all_data.empty:
            return None

        # 3. Forward-fill data to handle non-trading days for specific tickers
        full_date_range = pd.date_range(start=fetch_start, end=last_day.date())
        all_data = all_data.reindex(full_date_range).ffill()

        # 4. Extract the close prices for the last two valid days
        close_prices_last = all_data.loc[pd.to_datetime(last_day)]['Close']
        close_prices_prev = all_data.loc[pd.to_datetime(previous_day)]['Close']

        # 5. Calculate changes and format into a DataFrame
        results_df = pd.DataFrame({
            'price': close_prices_last,
            'prev_price': close_prices_prev
        })
        results_df.dropna(inplace=True)
        
        results_df['change'] = results_df['price'] - results_df['prev_price']
        results_df['pct_change'] = (results_df['change'] / results_df['prev_price']) * 100
        
        results_df.reset_index(inplace=True)
        results_df.rename(columns={'Ticker': 'symbol', 'index': 'symbol'}, inplace=True)
        
        return results_df

    except Exception:
        return None


@st.cache_data(ttl=3600) # Cache news for 60 minutes
def get_financial_news_with_summaries(query, num_results=5):
    """Fetches top financial news headlines and summarizes them."""
    config = Config()
    config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
    config.request_timeout = 10
    
    try:
        search_results = search(query, num_results=num_results, lang="en")
        news_items = []
        for url in search_results:
            try:
                article = Article(url, config=config)
                article.download()
                article.parse()
                article.nlp()
                
                title = article.title if article.title else f"Article from {url.split('//')[1].split('/')[0]}"
                summary = article.summary if article.summary else "Summary could not be generated for this article."

                news_items.append({"title": title, "url": url, "summary": summary})
            except Exception:
                news_items.append({
                    "title": f"Summarization Failed: {url.split('//')[1].split('/')[0]}",
                    "url": url,
                    "summary": "Could not access the content of this article to generate a summary. Please visit the link directly."
                })
        return news_items
    except Exception as e:
        return [{"title": "Error", "url": "", "summary": f"Could not fetch news at the moment: {e}"}]

# --- Main UI ---
st.title(f"📰 Market Pulse: {date.today().strftime('%B %d, %Y')}")
st.markdown("Your daily briefing on the Indian and global financial markets, including top news and performance analysis.")

# --- Action Buttons ---
col1, col2, _ = st.columns([1, 1, 5])
generate_report = col1.button("Generate Report", use_container_width=True)
if col2.button("Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.success("Data cache has been cleared. Click 'Generate Report' for the latest data.")
    st.rerun()

st.markdown("---")

if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False

if generate_report:
    st.session_state.report_generated = True

if st.session_state.report_generated:
    # --- Market Snapshot ---
    st.header("🌐 Global & Indian Market Snapshot")
    
    global_indices = {
        "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "Dow Jones": "^DJI",
        "FTSE 100": "^FTSE", "Nikkei 225": "^N225", "DAX": "^GDAXI"
    }
    indian_indices = {"Nifty 50": "^NSEI", "Sensex": "^BSESN"}
    
    all_tickers = list(global_indices.values()) + list(indian_indices.values())
    indices_df = get_market_data(all_tickers)
    
    if indices_df is not None and not indices_df.empty:
        st.subheader("Indian Indices")
        cols = st.columns(len(indian_indices))
        for i, (name, symbol) in enumerate(indian_indices.items()):
            data = indices_df[indices_df['symbol'] == symbol]
            if not data.empty:
                metric_data = data.iloc[0]
                cols[i].metric(
                    label=name,
                    value=f"{metric_data['price']:.2f}",
                    delta=f"{metric_data['change']:.2f} ({metric_data['pct_change']:.2f}%)"
                )
    
        st.subheader("Global Indices")
        cols = st.columns(len(global_indices))
        for i, (name, symbol) in enumerate(global_indices.items()):
            data = indices_df[indices_df['symbol'] == symbol]
            if not data.empty:
                metric_data = data.iloc[0]
                cols[i].metric(
                    label=name,
                    value=f"{metric_data['price']:.2f}",
                    delta=f"{metric_data['change']:.2f} ({metric_data['pct_change']:.2f}%)"
                )
    else:
        st.warning("Could not fetch live market data for indices. This may be due to market holidays or an API issue.")
    
    st.markdown("---")
    
    # --- Indian Market Deep Dive ---
    st.header("🇮🇳 Indian Market Deep Dive")
    
    sectoral_indices = {
        "NIFTY BANK": "^NSEBANK", "NIFTY IT": "^CNXIT", "NIFTY AUTO": "^CNXAUTO",
        "NIFTY PHARMA": "^CNXPHARMA", "NIFTY FMCG": "^CNXFMCG", "NIFTY METAL": "^CNXMETAL"
    }
    sectoral_df = get_market_data(list(sectoral_indices.values()))
    
    if sectoral_df is not None and not sectoral_df.empty:
        sectoral_df['Sector'] = sectoral_df['symbol'].map({v: k for k, v in sectoral_indices.items()})
        sectoral_df.dropna(subset=['pct_change'], inplace=True)

        if not sectoral_df.empty:
            sectoral_df = sectoral_df.sort_values("pct_change", ascending=False)
            
            fig = go.Figure(go.Bar(
                x=sectoral_df['pct_change'],
                y=sectoral_df['Sector'],
                orientation='h',
                text=sectoral_df['pct_change'].apply(lambda x: f'{x:.2f}%'),
                textposition='auto',
                marker_color=np.where(sectoral_df['pct_change'] > 0, 'green', 'red')
            ))
            fig.update_layout(
                title="Today's Sectoral Performance",
                xaxis_title="Percentage Change (%)",
                yaxis_title="Sector",
                yaxis={'categoryorder':'total ascending'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No valid sectoral data to display after cleaning.")
    else:
        st.warning("Could not fetch data for sectoral analysis. This may be due to market holidays or an API issue.")
    
    st.markdown("---")
    
    # --- Financial News Section ---
    st.header("🗞️ Top Financial News")
    
    news_col1, news_col2 = st.columns(2)
    
    with news_col1:
        st.subheader("Indian Finance News")
        with st.spinner("Fetching and summarizing Indian news..."):
            indian_news = get_financial_news_with_summaries("latest indian stock market business finance news", num_results=5)
            for news_item in indian_news:
                with st.expander(news_item['title']):
                    st.markdown(news_item['summary'])
                    st.markdown(f"[Read full article]({news_item['url']})")
    
    with news_col2:
        st.subheader("Global Finance News")
        with st.spinner("Fetching and summarizing global news..."):
            global_news = get_financial_news_with_summaries("latest global markets financial news", num_results=5)
            for news_item in global_news:
                with st.expander(news_item['title']):
                    st.markdown(news_item['summary'])
                    st.markdown(f"[Read full article]({news_item['url']})")

else:
    st.info("Click 'Generate Report' to load the latest market data and news.")

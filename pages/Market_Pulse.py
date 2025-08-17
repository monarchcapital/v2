# pages/Market_Pulse.py - A daily newsletter-style page for market analysis and news.
# Upgraded to use the Google Gemini API for advanced, synthesized summaries.

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
import plotly.graph_objects as go
import numpy as np
import google.generativeai as genai
from fpdf import FPDF # This will now import from the fpdf2 library
import os
from google.api_core import exceptions as google_exceptions
from requests import exceptions as requests_exceptions

# --- Page Configuration ---
st.set_page_config(page_title="Market Pulse", page_icon="📈", layout="wide")

# --- Hide Streamlit Menu & Footer ---
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- API Key Configuration ---
# Use Streamlit secrets for secure API key management.
# The key should be stored in your .streamlit/secrets.toml file.
# Example: GOOGLE_API_KEY = "your_secret_key_here"
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except (KeyError, FileNotFoundError):
    st.warning("Google AI API Key is not found. Please add it to your Streamlit secrets.", icon="⚠️")
    GOOGLE_API_KEY = None

# --- Helper Functions ---

@st.cache_data(ttl=1800) # Cache data for 30 minutes
def get_market_data(tickers):
    """
    Fetches and processes market data for a list of tickers, robustly handling holidays and network errors.
    """
    try:
        # Establish a recent date range using a reliable global index
        end_date = date.today()
        start_date = end_date - timedelta(days=15)
        
        ref_hist = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
        ref_hist.dropna(inplace=True)
        
        if len(ref_hist) < 2:
            st.error("Not enough historical data to determine a recent date range.")
            return None

        # Set a generous fetch window based on the reference index
        fetch_start = ref_hist.index[-10].date() # Look back ~10 trading days
        fetch_end = end_date + timedelta(days=1)
        all_data = yf.download(tickers, start=fetch_start, end=fetch_end, progress=False, timeout=10)
        
        if all_data.empty:
            st.warning(f"No data returned from yfinance for tickers: {tickers}")
            return None

        # Handle yfinance's multi-level columns by safely selecting 'Close' prices
        if isinstance(all_data.columns, pd.MultiIndex):
            close_prices = all_data.get('Close')
        else:
            close_prices = all_data[['Close']] # Ensure it's a DataFrame

        if close_prices is None or close_prices.empty:
            st.warning(f"Could not retrieve 'Close' price data for tickers: {tickers}")
            return None

        # Clean the data by removing any days where all tickers were non-trading
        valid_prices = close_prices.dropna(how='all')

        if len(valid_prices) < 2:
            st.warning(f"Not enough recent trading data available for tickers: {tickers} to calculate change.")
            return None
            
        # **FIX**: Determine the last two trading days from the ACTUAL data for the requested tickers
        last_available_day = valid_prices.index[-1]
        previous_available_day = valid_prices.index[-2]

        close_prices_last = valid_prices.loc[last_available_day]
        close_prices_prev = valid_prices.loc[previous_available_day]

        results_df = pd.DataFrame({
            'price': close_prices_last,
            'prev_price': close_prices_prev
        })
        results_df.dropna(inplace=True)

        if results_df.empty:
            st.warning(f"Data for {tickers} could not be processed for the required dates. It may be a holiday period.")
            return None
        
        results_df['change'] = results_df['price'] - results_df['prev_price']
        results_df['pct_change'] = (results_df['change'] / results_df['prev_price']) * 100
        
        results_df.reset_index(inplace=True)
        results_df.rename(columns={'Ticker': 'symbol', 'index': 'symbol'}, inplace=True)
        
        return results_df

    except requests_exceptions.RequestException as e:
        st.error(f"Network error fetching market data: {e}. Please check your connection.")
        return None
    except (KeyError, IndexError) as e:
        st.error(f"Data processing error: {e}. The data from the source may be in an unexpected format.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred in get_market_data: {e}")
        return None

@st.cache_data(ttl=3600) # Cache news for 60 minutes
def get_synthesized_market_summary(query, market_data_context=""):
    """
    Uses the Gemini API to generate a synthesized financial summary for a specific query.
    """
    if not GOOGLE_API_KEY:
        return {"title": "API Key Missing", "summary": "Cannot generate news summary without a Google AI API Key."}
        
    try:
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        prompt = f"""
        As an expert financial market analyst, generate a detailed market summary based on the latest news about '{query}'.
        Incorporate the following recent market data for context if relevant: {market_data_context}.

        Your summary should be structured like a professional newsletter article:
        1.  **Main Headline:** A compelling title that captures the essence of the news for '{query}'.
        2.  **Key Takeaways:** Start with 3-4 bullet points summarizing the most critical information.
        3.  **Thematic Analysis:** Identify 2-3 key themes (e.g., Sectoral Performance, Macroeconomic Factors, Corporate Earnings). For each theme, provide a detailed paragraph.
        4.  **Outlook:** Conclude with a brief, data-driven outlook.
        5.  **Formatting:** Use Markdown with bold headings for each section. Start the entire response with a main heading for the topic using '### {query.title()}'.
        """
        response = model.generate_content(prompt)
        return {"title": f"Analysis: {query.title()}", "summary": response.text}

    except google_exceptions.GoogleAPICallError as e:
        return {"title": "Error: Google API Call Failed", "summary": f"Could not generate summary due to an API error: {e}"}
    except Exception as e:
        return {"title": "Error Generating Summary", "summary": f"An unexpected error occurred while generating the news summary: {e}"}

def generate_pdf_from_markdown(markdown_text, title, indices_data=None, sectoral_data=None):
    """
    Generates a PDF from a markdown string and dataframes, returning it as bytes.
    Uses fpdf2 and includes a font for full Unicode support.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- Font Setup for Unicode Support ---
    # To use this, you must have the DejaVuSans.ttf file in the same directory.
    # You can download it from: https://dejavu-fonts.github.io/
    try:
        pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
        pdf.set_font('DejaVu', '', 12)
    except FileNotFoundError:
        # Fallback to Arial if the font file is not found
        pdf.set_font("Arial", '', 12)
        st.warning("DejaVuSans.ttf font not found. PDF may not render all characters correctly. Using fallback Arial font.")

    # Clean text to remove characters not supported by the fallback font
    def clean_text(text):
        # Replace emojis and other special symbols with a placeholder or remove them
        return ''.join(c for c in text if ord(c) < 256)

    # Main Title
    pdf.set_font_size(18)
    pdf.multi_cell(0, 10, clean_text(title), 0, 'C')
    pdf.ln(10)

    # --- Market Snapshot Section ---
    if indices_data:
        pdf.set_font_size(16)
        pdf.cell(0, 10, clean_text("🌍 Global & Indian Market Snapshot"), ln=1, align='L')
        
        # Indian Indices
        pdf.set_font_size(12)
        pdf.cell(0, 8, clean_text("🇮🇳 Indian Indices"), ln=1, align='L')
        for _, row in indices_data['indian'].iterrows():
            pdf.set_font_size(11)
            delta = f"{row['change']:.2f} ({row['pct_change']:.2f}%)"
            pdf.cell(0, 6, clean_text(f"  - {row['name']}: {row['price']:.2f} ({delta})"), ln=1, align='L')
        pdf.ln(3)

        # Global Indices
        pdf.set_font_size(12)
        pdf.cell(0, 8, "Global Indices", ln=1, align='L')
        for _, row in indices_data['global'].iterrows():
            pdf.set_font_size(11)
            delta = f"{row['change']:.2f} ({row['pct_change']:.2f}%)"
            pdf.cell(0, 6, clean_text(f"  - {row['name']}: {row['price']:.2f} ({delta})"), ln=1, align='L')
        pdf.ln(5)

    # --- Sectoral Performance Section ---
    if sectoral_data is not None and not sectoral_data.empty:
        pdf.set_font_size(16)
        pdf.cell(0, 10, clean_text("🇮🇳 Indian Market Deep Dive"), ln=1, align='L')
        pdf.set_font_size(12)
        pdf.cell(0, 8, "Today's Sectoral Performance", ln=1, align='L')
        for _, row in sectoral_data.iterrows():
            pdf.set_font_size(11)
            delta = f"{row['pct_change']:.2f}%"
            pdf.cell(0, 6, clean_text(f"  - {row['Sector']}: {delta}"), ln=1, align='L')
        pdf.ln(5)

    # --- AI Analysis Section ---
    pdf.set_font_size(16)
    pdf.cell(0, 10, clean_text("🤖 Monarch News Analysis"), ln=1, align='L')
    pdf.set_font_size(12) # Reset font for the body
    
    # Simple markdown parsing for PDF
    for line in markdown_text.split('\n'):
        cleaned_line = clean_text(line.strip())
        if not cleaned_line:
            pdf.ln(2)
            continue
        
        if cleaned_line.startswith('### '):
            pdf.set_font_size(14)
            pdf.multi_cell(0, 8, cleaned_line.replace('### ', ''))
            pdf.set_font_size(12)
        elif cleaned_line.startswith('**'):
            pdf.set_font(style='B')
            pdf.multi_cell(0, 8, cleaned_line.replace('**', ''))
            pdf.set_font(style='')
        elif cleaned_line.startswith('* '):
            # **FIX**: Use a standard hyphen for bullet points to ensure compatibility
            pdf.multi_cell(0, 8, f"- {cleaned_line.replace('* ', '')}")
        elif cleaned_line.startswith('---'):
            pdf.ln(5)
            pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
            pdf.ln(5)
        else:
            pdf.multi_cell(0, 8, cleaned_line)
        
        pdf.ln(2)
        
    # **FIX**: Ensure the output is a bytes object for st.download_button
    return bytes(pdf.output())

# --- Main UI ---
st.title(f"📈 Market Pulse: {date.today().strftime('%B %d, %Y')}")
st.markdown("Your daily briefing on the Indian and global financial markets, powered by AI-driven analysis.")

# --- Action Buttons ---
col1, col2, _ = st.columns([1, 1, 5])
generate_report = col1.button("Generate Report", use_container_width=True, type="primary")
if col2.button("Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.success("Data cache has been cleared. Click 'Generate Report' for the latest data.")
    st.rerun()

st.markdown("---")

if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False

if generate_report:
    st.session_state.report_generated = True
    # Clear previous analysis content
    st.session_state.pop('combined_summary', None)
    st.session_state.pop('combined_title', None)


if st.session_state.report_generated:
    # --- Market Snapshot ---
    st.header("🌍 Global & Indian Market Snapshot")
    
    global_indices = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "Dow Jones": "^DJI", "FTSE 100": "^FTSE", "Nikkei 225": "^N225", "DAX": "^GDAXI"}
    indian_indices = {"Nifty 50": "^NSEI", "Sensex": "^BSESN"}
    all_tickers = list(global_indices.values()) + list(indian_indices.values())
    
    with st.spinner("Fetching latest market data..."):
        indices_df = get_market_data(all_tickers)
    
    market_context_for_llm = ""
    if indices_df is not None and not indices_df.empty:
        # Prepare data for PDF and UI
        indian_df = indices_df[indices_df['symbol'].isin(indian_indices.values())].copy()
        indian_df['name'] = indian_df['symbol'].map({v: k for k, v in indian_indices.items()})
        
        global_df = indices_df[indices_df['symbol'].isin(global_indices.values())].copy()
        global_df['name'] = global_df['symbol'].map({v: k for k, v in global_indices.items()})

        st.session_state.indices_data_for_pdf = {'indian': indian_df, 'global': global_df}

        st.subheader("Indian Indices")
        cols = st.columns(len(indian_indices))
        for i, (name, symbol) in enumerate(indian_indices.items()):
            data = indian_df[indian_df['symbol'] == symbol]
            if not data.empty:
                metric_data = data.iloc[0]
                cols[i].metric(label=name, value=f"{metric_data['price']:.2f}", delta=f"{metric_data['change']:.2f} ({metric_data['pct_change']:.2f}%)")
                market_context_for_llm += f"{name}: {metric_data['price']:.2f} ({metric_data['pct_change']:.2f}% change). "
        
        st.subheader("Global Indices")
        cols = st.columns(len(global_indices))
        for i, (name, symbol) in enumerate(global_indices.items()):
            data = global_df[global_df['symbol'] == symbol]
            if not data.empty:
                metric_data = data.iloc[0]
                cols[i].metric(label=name, value=f"{metric_data['price']:.2f}", delta=f"{metric_data['change']:.2f} ({metric_data['pct_change']:.2f}%)")
    else:
        st.warning("Could not fetch live market data for indices.")
    
    st.markdown("---")
    
    # --- Indian Market Deep Dive ---
    st.header("🇮🇳 Indian Market Deep Dive")
    sectoral_indices = {"NIFTY BANK": "^NSEBANK", "NIFTY IT": "^CNXIT", "NIFTY AUTO": "^CNXAUTO", "NIFTY PHARMA": "^CNXPHARMA", "NIFTY FMCG": "^CNXFMCG", "NIFTY METAL": "^CNXMETAL"}
    
    with st.spinner("Analyzing sectoral performance..."):
        sectoral_df = get_market_data(list(sectoral_indices.values()))

    if sectoral_df is not None and not sectoral_df.empty:
        sectoral_df['Sector'] = sectoral_df['symbol'].map({v: k for k, v in sectoral_indices.items()})
        sectoral_df.dropna(subset=['pct_change'], inplace=True)
        if not sectoral_df.empty:
            sectoral_df = sectoral_df.sort_values("pct_change", ascending=False)
            st.session_state.sectoral_data_for_pdf = sectoral_df # Store for PDF
            fig = go.Figure(go.Bar(x=sectoral_df['pct_change'], y=sectoral_df['Sector'], orientation='h', text=sectoral_df['pct_change'].apply(lambda x: f'{x:.2f}%'), textposition='auto', marker_color=np.where(sectoral_df['pct_change'] > 0, 'green', 'red')))
            fig.update_layout(title="Today's Sectoral Performance", xaxis_title="Percentage Change (%)", yaxis_title="Sector", yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not fetch data for sectoral analysis.")
    
    st.markdown("---")
    
    # --- AI-Powered Financial News Section ---
    st.header("🤖 Monarch News Analysis")
    
    topic_options = ["Indian Market Overview", "Global Market Overview", "Indian IT Sector", "US Inflation & Fed Policy"]
    selected_topics = st.multiselect("Choose topics for analysis (select one or more):", topic_options)

    custom_topics_input = st.text_area("Enter custom topics for analysis (one per line):", placeholder="e.g., Impact of crude oil prices on the Indian economy\nFuture of EV stocks in India")

    all_queries = selected_topics
    if custom_topics_input:
        custom_topics = [topic.strip() for topic in custom_topics_input.split('\n') if topic.strip()]
        all_queries.extend(custom_topics)

    if st.button("Analyze Selected Topics", use_container_width=True, disabled=(not GOOGLE_API_KEY)):
        if all_queries:
            all_summaries = []
            with st.spinner("Generating AI-powered analysis... This may take a moment."):
                for query in all_queries:
                    summary_data = get_synthesized_market_summary(query, market_context_for_llm)
                    all_summaries.append(summary_data['summary'])
            
            st.session_state.combined_summary = "\n\n---\n\n".join(all_summaries)
            st.session_state.combined_title = "Consolidated Market Pulse Report"
            st.rerun()

    if 'combined_summary' in st.session_state:
        st.subheader("Monarch Market Newsletter")
        with st.container(border=True):
            st.markdown(st.session_state.combined_summary)
        
        st.markdown("---")
        
        pdf_bytes = generate_pdf_from_markdown(
            st.session_state.combined_summary, 
            st.session_state.combined_title,
            indices_data=st.session_state.get('indices_data_for_pdf'),
            sectoral_data=st.session_state.get('sectoral_data_for_pdf')
        )
        
        st.download_button(
            label="📄 Download as PDF",
            data=pdf_bytes,
            file_name=f"Market_Pulse_Report_{date.today().strftime('%Y_%m_%d')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

else:
    st.info("Click 'Generate Report' to load the latest market data and news.")

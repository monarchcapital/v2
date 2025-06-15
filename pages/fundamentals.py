# pages/Fundamentals.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import numpy_financial as npf
from datetime import datetime

st.set_page_config(page_title="Monarch: Indian Stock Fundamentals", layout="wide")

st.header("📊 Indian Stock Fundamentals")
st.markdown("""
This page provides key fundamental financial data and valuation insights for Indian companies.
You can view their income statements, balance sheets, cash flow statements, and fair value estimates.
""")

# --- Stock Ticker Input ---
ticker_input = st.text_area(
    "Enter Indian Stock Ticker Symbol(s) (e.g., RELIANCE.NS, TCS.NS, HDFCBANK.NS - separate by comma or new line):",
    "RELIANCE.NS, TCS.NS"
)

# Process multiple tickers
tickers = [t.strip().upper() for t in ticker_input.replace('\n', ',').split(',') if t.strip()]

# Valuation Parameters in Sidebar
st.sidebar.header("📈 Valuation Parameters")

# Margin of Safety Sliders
st.sidebar.subheader("Margin of Safety Settings")
mos_dcf = st.sidebar.slider("DCF Margin of Safety (%)", 0, 50, 20, key='dcf_mos')
mos_graham = st.sidebar.slider("Graham Formula Margin of Safety (%)", 0, 50, 25, key='graham_mos')
mos_peg = st.sidebar.slider("PEG Ratio Margin of Safety (%)", 0, 50, 15, key='peg_mos')

# DCF Parameters
st.sidebar.subheader("DCF Model Parameters")
growth_rate = st.sidebar.slider("Growth Rate (%)", 5, 30, 12, key='growth_rate') / 100
terminal_growth = st.sidebar.slider("Terminal Growth (%)", 1, 10, 3, key='terminal_growth') / 100
discount_rate = st.sidebar.slider("Discount Rate (%)", 5, 20, 10, key='discount_rate') / 100

# Graham Formula Parameters
st.sidebar.subheader("Graham Formula Parameters")
graham_multiplier = st.sidebar.slider("Graham Multiplier", 15.0, 25.0, 22.5, step=0.5, key='graham_multiplier')
max_pe = st.sidebar.slider("Max P/E Ratio", 10, 25, 15, key='max_pe')

# PEG Ratio Parameters
st.sidebar.subheader("PEG Ratio Parameters")
peg_ratio = st.sidebar.slider("Target PEG Ratio", 0.5, 2.0, 1.0, step=0.1, key='peg_ratio')

@st.cache_data(ttl=3600) # Cache data for 1 hour
def fetch_fundamental_data(ticker_symbol):
    """Fetches fundamental data using yfinance."""
    try:
        stock = yf.Ticker(ticker_symbol)

        # Basic Info
        info = stock.info

        # Financial Statements
        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        # News from Yahoo Finance
        news = stock.news
        
        return info, income_stmt, balance_sheet, cash_flow, news
    except Exception as e:
        st.warning(f"Could not fetch data for {ticker_symbol}. Please check the ticker symbol and try again. Error: {e}")
        return None, None, None, None, None

def calculate_dcf_fair_value(info, cash_flow, growth_rate, terminal_growth, discount_rate):
    """Calculate fair value using Discounted Cash Flow model"""
    try:
        # Get required financial metrics
        # Try to get Free Cash Flow from cash flow statement
        if 'Free Cash Flow' in cash_flow.index:
            current_fcf = cash_flow.loc['Free Cash Flow', cash_flow.columns[0]]
        else:
            # Calculate FCF if not directly available: Operating Cash Flow - Capital Expenditures
            operating_cash = cash_flow.loc['Total Cash From Operating Activities', cash_flow.columns[0]]
            capital_expenditures = abs(cash_flow.loc['Capital Expenditures', cash_flow.columns[0]])
            current_fcf = operating_cash + capital_expenditures  # CapEx is negative so we add
        
        shares_outstanding = info.get('sharesOutstanding')
        cash = info.get('totalCash')
        debt = info.get('totalDebt')
        current_price = info.get('currentPrice')
        
        if not all([current_fcf, shares_outstanding, cash, debt, current_price]):
            return None, None, None
        
        # Financial assumptions
        years = 5  # Projection period
        
        # Calculate projected cash flows
        cash_flows = [current_fcf * ((1 + growth_rate) ** year) for year in range(1, years+1)]
        
        # Calculate terminal value
        terminal_value = cash_flows[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
        
        # Discount cash flows
        discounted_cash_flows = [cf / ((1 + discount_rate) ** (i+1)) for i, cf in enumerate(cash_flows)]
        discounted_terminal_value = terminal_value / ((1 + discount_rate) ** years)
        
        # Calculate enterprise value
        enterprise_value = sum(discounted_cash_flows) + discounted_terminal_value
        
        # Adjust for cash and debt to get equity value
        equity_value = enterprise_value - debt + cash
        
        # Calculate fair value per share
        fair_value = equity_value / shares_outstanding
        
        return fair_value, equity_value
    except Exception as e:
        return None, None

def calculate_graham_fair_value(info, graham_multiplier, max_pe):
    """Calculate fair value using Graham Formula"""
    try:
        eps = info.get('trailingEps')
        bvps = info.get('bookValue')
        
        if not eps or not bvps:
            return None
            
        # Basic Graham Formula
        raw_value = np.sqrt(graham_multiplier * eps * bvps)
        
        # Apply P/E limit
        pe = info.get('trailingPE')
        if pe and pe > max_pe:
            return min(raw_value, eps * max_pe)
        return raw_value
    except:
        return None

def calculate_peg_fair_value(info, peg_ratio):
    """Calculate fair value using PEG Ratio"""
    try:
        eps = info.get('trailingEps')
        growth = info.get('earningsGrowth')  # This might not be available
        
        # If earnings growth isn't available, use analyst estimate or default
        if not growth:
            # Try to get analyst growth estimate
            growth = info.get('revenueGrowth')
            if not growth:
                growth = 0.10  # Default to 10% if no growth available
        
        # Calculate fair value
        return eps * (1 + growth) * peg_ratio
    except:
        return None

# Initialize summary data
summary_data = []

if tickers:
    # Fetch data and calculate valuations for each ticker
    for ticker in tickers:
        info, income_stmt, balance_sheet, cash_flow, news = fetch_fundamental_data(ticker)
        
        if info:
            current_price = info.get('currentPrice')
            
            # Calculate all fair values
            dcf_value, dcf_equity = calculate_dcf_fair_value(
                info, cash_flow, 
                growth_rate, terminal_growth, discount_rate
            )
            graham_value = calculate_graham_fair_value(info, graham_multiplier, max_pe)
            peg_value = calculate_peg_fair_value(info, peg_ratio)
            
            # Apply margin of safety
            dcf_adjusted = dcf_value * (1 - mos_dcf/100) if dcf_value else None
            graham_adjusted = graham_value * (1 - mos_graham/100) if graham_value else None
            peg_adjusted = peg_value * (1 - mos_peg/100) if peg_value else None
            
            # Calculate average fair value
            fair_values = [v for v in [dcf_adjusted, graham_adjusted, peg_adjusted] if v]
            avg_fair_value = np.mean(fair_values) if fair_values else None
            
            summary_data.append({
                "Ticker": ticker,
                "Current Price": f"₹{current_price:,.2f}" if current_price else "N/A",
                "DCF Value": f"₹{dcf_value:,.2f}" if dcf_value else "N/A",
                f"DCF (MoS {mos_dcf}%)": f"₹{dcf_adjusted:,.2f}" if dcf_adjusted else "N/A",
                "Graham Value": f"₹{graham_value:,.2f}" if graham_value else "N/A",
                f"Graham (MoS {mos_graham}%)": f"₹{graham_adjusted:,.2f}" if graham_adjusted else "N/A",
                "PEG Value": f"₹{peg_value:,.2f}" if peg_value else "N/A",
                f"PEG (MoS {mos_peg}%)": f"₹{peg_adjusted:,.2f}" if peg_adjusted else "N/A",
                "Avg Fair Value": f"₹{avg_fair_value:,.2f}" if avg_fair_value else "N/A"
            })
    
    # Display summary table at the top
    if summary_data:
        st.subheader("📊 Fair Value Summary")
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
    
    tabs = st.tabs([ticker for ticker in tickers])  # Create tabs for each ticker
    
    for idx, tab in enumerate(tabs):
        with tab:
            ticker = tickers[idx]
            st.markdown(f"## Detailed Analysis: {ticker}")
            info, income_stmt, balance_sheet, cash_flow, news = fetch_fundamental_data(ticker)

            if info and info.get('longName'): 
                company_name = info.get('longName', ticker)
                current_price = info.get('currentPrice', 0)
                
                # Calculate all fair values
                dcf_value, dcf_equity = calculate_dcf_fair_value(
                    info, cash_flow, 
                    growth_rate, terminal_growth, discount_rate
                )
                graham_value = calculate_graham_fair_value(info, graham_multiplier, max_pe)
                peg_value = calculate_peg_fair_value(info, peg_ratio)
                
                # Apply margin of safety
                dcf_adjusted = dcf_value * (1 - mos_dcf/100) if dcf_value else None
                graham_adjusted = graham_value * (1 - mos_graham/100) if graham_value else None
                peg_adjusted = peg_value * (1 - mos_peg/100) if peg_value else None
                
                # Calculate average fair value
                fair_values = [v for v in [dcf_adjusted, graham_adjusted, peg_adjusted] if v]
                avg_fair_value = np.mean(fair_values) if fair_values else None
                
                # Display valuation summary
                st.subheader("💎 Valuation Summary")
                
                # Create valuation cards
                col1, col2, col3, col4 = st.columns(4)
                
                # Current Price
                with col1:
                    st.metric("Current Price", 
                              f"₹{current_price:,.2f}" if current_price else "N/A",
                              delta=None)
                
                # Average Fair Value
                with col2:
                    if avg_fair_value:
                        delta = f"{(avg_fair_value - current_price)/current_price*100:.2f}%" if current_price else None
                        st.metric("Average Fair Value", 
                                  f"₹{avg_fair_value:,.2f}", 
                                  delta=delta)
                    else:
                        st.metric("Average Fair Value", "N/A")
                
                # Margin of Safety
                with col3:
                    if avg_fair_value and current_price:
                        mos = ((avg_fair_value - current_price) / avg_fair_value) * 100
                        st.metric("Margin of Safety", 
                                  f"{mos:.2f}%",
                                  delta="Undervalued" if mos > 0 else "Overvalued")
                    else:
                        st.metric("Margin of Safety", "N/A")
                
                # Recommendation
                with col4:
                    if avg_fair_value and current_price:
                        if current_price < avg_fair_value * 0.8:
                            rec = "Strong Buy"
                            color = "green"
                        elif current_price < avg_fair_value:
                            rec = "Buy"
                            color = "lightgreen"
                        elif current_price < avg_fair_value * 1.2:
                            rec = "Hold"
                            color = "orange"
                        else:
                            rec = "Sell"
                            color = "red"
                        st.markdown(f"<h3 style='text-align: center; color: {color};'>{rec}</h3>", 
                                    unsafe_allow_html=True)
                    else:
                        st.metric("Recommendation", "N/A")
                
                # Valuation Models Table
                st.subheader("📐 Fair Value Models")
                valuation_data = {
                    "Model": ["Discounted Cash Flow", "Graham Formula", "PEG Ratio"],
                    "Fair Value": [
                        f"₹{dcf_value:,.2f}" if dcf_value else "N/A", 
                        f"₹{graham_value:,.2f}" if graham_value else "N/A", 
                        f"₹{peg_value:,.2f}" if peg_value else "N/A"
                    ],
                    "Margin of Safety": [f"{mos_dcf}%", f"{mos_graham}%", f"{mos_peg}%"],
                    "Adjusted Value": [
                        f"₹{dcf_adjusted:,.2f}" if dcf_adjusted else "N/A", 
                        f"₹{graham_adjusted:,.2f}" if graham_adjusted else "N/A", 
                        f"₹{peg_adjusted:,.2f}" if peg_adjusted else "N/A"
                    ],
                    "Valuation": [
                        "Undervalued" if dcf_adjusted and current_price < dcf_adjusted else "Overvalued" if dcf_adjusted else "N/A",
                        "Undervalued" if graham_adjusted and current_price < graham_adjusted else "Overvalued" if graham_adjusted else "N/A",
                        "Undervalued" if peg_adjusted and current_price < peg_adjusted else "Overvalued" if peg_adjusted else "N/A"
                    ]
                }
                st.dataframe(pd.DataFrame(valuation_data), use_container_width=True)
                
                # Model Explanations
                with st.expander("Model Explanations"):
                    st.markdown("""
                    **1. Discounted Cash Flow (DCF) Model:**
                    - Projects future cash flows and discounts them to present value
                    - Incorporates growth rate, discount rate, and terminal value
                    - Accounts for company's cash and debt
                    - **Formula:** Fair Value = (Sum of Discounted Cash Flows + Terminal Value + Cash - Debt) / Shares Outstanding
                    
                    **2. Graham Formula:**
                    - Developed by Benjamin Graham (Warren Buffett's mentor)
                    - Conservative valuation method for defensive investors
                    - **Formula:** Fair Value = √(22.5 × EPS × Book Value per Share)
                    - Applies a P/E limit of 15
                    
                    **3. PEG Ratio Valuation:**
                    - Considers both price/earnings ratio and growth rate
                    - **Formula:** Fair Value = EPS × (1 + Growth Rate) × Target PEG Ratio
                    - Useful for growth companies
                    """)
                
                # Detailed Valuation Breakdown
                st.subheader("🔍 Detailed Valuation Breakdown")
                
                # DCF Details
                if dcf_value:
                    st.markdown("#### DCF Valuation Details")
                    st.markdown(f"""
                    - **Base DCF Value:** ₹{dcf_value:,.2f}
                    - **Applied Margin of Safety:** {mos_dcf}%
                    - **Adjusted Fair Value:** ₹{dcf_adjusted:,.2f}
                    - **Current Price vs DCF:** {'**Undervalued** ⬆️' if current_price < dcf_adjusted else '**Overvalued** ⬇️'}
                    """)
                    
                    # Display DCF inputs
                    dcf_inputs = {
                        "Projection Period": "5 years",
                        "Growth Rate": f"{growth_rate*100:.1f}%",
                        "Terminal Growth": f"{terminal_growth*100:.1f}%",
                        "Discount Rate": f"{discount_rate*100:.1f}%",
                        "Shares Outstanding": f"{info.get('sharesOutstanding', 0):,.0f}",
                        "Total Cash": f"₹{info.get('totalCash', 0)/10000000:,.2f} Cr",
                        "Total Debt": f"₹{info.get('totalDebt', 0)/10000000:,.2f} Cr",
                        "Calculated Equity Value": f"₹{dcf_equity/10000000:,.2f} Cr" if dcf_equity else "N/A"
                    }
                    st.write(pd.DataFrame([dcf_inputs]).T.rename(columns={0: 'Value'}))
                    st.markdown("---")
                
                # Graham Formula Details
                if graham_value:
                    st.markdown("#### Graham Formula Details")
                    eps = info.get('trailingEps', 'N/A')
                    bvps = info.get('bookValue', 'N/A')
                    
                    st.markdown(f"""
                    - **EPS:** ₹{eps:,.2f}
                    - **Book Value per Share:** ₹{bvps:,.2f}
                    - **Base Graham Value:** ₹{graham_value:,.2f}
                    - **Applied Margin of Safety:** {mos_graham}%
                    - **Adjusted Fair Value:** ₹{graham_adjusted:,.2f}
                    - **Current Price vs Graham:** {'**Undervalued** ⬆️' if current_price < graham_adjusted else '**Overvalued** ⬇️'}
                    """)
                    st.markdown("---")
                
                # PEG Ratio Details
                if peg_value:
                    st.markdown("#### PEG Ratio Valuation Details")
                    eps = info.get('trailingEps', 'N/A')
                    growth = info.get('earningsGrowth') or info.get('revenueGrowth') or 0.10
                    
                    st.markdown(f"""
                    - **EPS:** ₹{eps:,.2f}
                    - **Growth Rate:** {growth*100:.1f}%
                    - **Target PEG Ratio:** {peg_ratio}
                    - **Base PEG Value:** ₹{peg_value:,.2f}
                    - **Applied Margin of Safety:** {mos_peg}%
                    - **Adjusted Fair Value:** ₹{peg_adjusted:,.2f}
                    - **Current Price vs PEG:** {'**Undervalued** ⬆️' if current_price < peg_adjusted else '**Overvalued** ⬇️'}
                    """)
                    st.markdown("---")
                
                # Continue with the rest of the analysis (Company Overview, Financials, etc.)
                with st.expander(f"View Full Company Details for {company_name}"):
                    # --- Company Overview ---
                    st.markdown("#### Company Overview")
                    overview_data = {
                        "Sector": info.get('sector', 'N/A'),
                        "Industry": info.get('industry', 'N/A'),
                        "Full Time Employees": f"{info.get('fullTimeEmployees', 'N/A'):,}" if isinstance(info.get('fullTimeEmployees'), int) else info.get('fullTimeEmployees', 'N/A'),
                        "Website": info.get('website', 'N/A'),
                        "Business Summary": info.get('longBusinessSummary', 'N/A')
                    }
                    st.write(pd.DataFrame([overview_data]).T.rename(columns={0: 'Value'}))

                    st.markdown("---")

                    # --- Key Ratios ---
                    st.markdown("#### Key Financial Ratios")
                    ratios_data = {
                        "Market Cap": info.get('marketCap', 'N/A'),
                        "Trailing P/E": info.get('trailingPE', 'N/A'),
                        "Forward P/E": info.get('forwardPE', 'N/A'),
                        "PEG Ratio": info.get('pegRatio', 'N/A'),
                        "Dividend Yield": info.get('dividendYield', 'N/A'),
                        "Profit Margins": info.get('profitMargins', 'N/A'),
                        "Return on Equity": info.get('returnOnEquity', 'N/A'),
                        "Debt to Equity": info.get('debtToEquity', 'N/A'),
                        "Current Ratio": info.get('currentRatio', 'N/A'),
                        "Book Value": info.get('bookValue', 'N/A'),
                        "Price to Book": info.get('priceToBook', 'N/A'),
                    }
                    # Format numbers
                    formatted_ratios = {}
                    for key, value in ratios_data.items():
                        if isinstance(value, (int, float)):
                            if "Yield" in key or "Margins" in key or "ROE" in key:
                                formatted_ratios[key] = f"{value:.2%}" 
                            elif "Cap" in key:
                                # Convert to crores
                                value_cr = value / 10000000
                                formatted_ratios[key] = f"₹{value_cr:,.0f} Cr" 
                            else:
                                formatted_ratios[key] = f"{value:,.2f}" 
                        else:
                            formatted_ratios[key] = value
                    st.write(pd.DataFrame([formatted_ratios]).T.rename(columns={0: 'Value'}))
                    
                    st.markdown("---")

                    # --- Analyst Estimates ---
                    st.markdown("#### Analyst Fair Value Estimates")
                    
                    if info.get('targetMeanPrice') and info.get('numberOfAnalystOpinions', 0) > 0:
                        fair_value_data = {
                            "Current Price": info.get('currentPrice', 'N/A'),
                            "Analyst Target Mean Price": info.get('targetMeanPrice', 'N/A'),
                            "Analyst Target Median Price": info.get('targetMedianPrice', 'N/A'),
                            "Recommendation": info.get('recommendationKey', 'N/A').title(),
                            "Number of Analysts": info.get('numberOfAnalystOpinions', 'N/A'),
                        }
                        
                        # Calculate upside/downside
                        current = info.get('currentPrice')
                        target = info.get('targetMeanPrice')
                        if current and target:
                            upside = ((target - current) / current) * 100
                            fair_value_data["Upside Potential"] = f"{upside:+.2f}%"
                        
                        formatted_fair_value = {}
                        for key, value in fair_value_data.items():
                            if isinstance(value, (int, float)):
                                formatted_fair_value[key] = f"₹{value:,.2f}" 
                            else:
                                formatted_fair_value[key] = value
                        
                        st.write(pd.DataFrame([formatted_fair_value]).T.rename(columns={0: 'Value'}))
                    
                    else:
                        st.info(f"Analyst consensus estimates not available for {ticker}")
                    
                    st.markdown("---")

                    # --- Valuation Metrics ---
                    st.markdown("#### Market Metrics")
                    valuation_data = {
                        "52-Week Range": f"₹{info.get('fiftyTwoWeekLow', 'N/A'):,.2f} - ₹{info.get('fiftyTwoWeekHigh', 'N/A'):,.2f}",
                        "Beta (5Y Monthly)": info.get('beta', 'N/A'),
                        "Avg. Volume (3m)": info.get('averageVolume', 'N/A'),
                        "Shares Outstanding": f"{info.get('sharesOutstanding', 'N/A'):,}" if isinstance(info.get('sharesOutstanding'), (int, float)) else 'N/A',
                    }
                    
                    # Format volume in lakhs
                    if isinstance(valuation_data["Avg. Volume (3m)"], (int, float)):
                        vol_lakhs = valuation_data["Avg. Volume (3m)"] / 100000
                        valuation_data["Avg. Volume (3m)"] = f"{vol_lakhs:,.0f} Lakh"
                    
                    st.write(pd.DataFrame([valuation_data]).T.rename(columns={0: 'Value'}))
                    
                    st.markdown("---")

                    # --- Latest News Section ---
                    st.markdown("#### Latest News")
                    if news:
                        for item in news:
                            # Safely get news details
                            title = item.get('title', 'News Article')
                            publisher = item.get('publisher', 'Unknown Source')
                            link = item.get('link', '#')
                            
                            # Handle publication time
                            pub_time = item.get('providerPublishTime')
                            if pub_time:
                                try:
                                    pub_date = datetime.fromtimestamp(pub_time)
                                    pub_date_str = pub_date.strftime('%Y-%m-%d %H:%M')
                                except:
                                    pub_date_str = "Date not available"
                            else:
                                pub_date_str = "Date not available"
                            
                            # Display news card
                            with st.container():
                                col1, col2 = st.columns([1, 3])
                                with col1:
                                    # Display thumbnail if available
                                    thumbnail = item.get('thumbnail')
                                    if thumbnail and thumbnail.get('resolutions'):
                                        try:
                                            st.image(thumbnail['resolutions'][0]['url'], width=150)
                                        except:
                                            pass
                                with col2:
                                    st.subheader(title)
                                    st.caption(f"**{publisher}** | {pub_date_str}")
                                    st.write(f"[Read full article]({link})")
                                st.markdown("---")
                    else:
                        st.info(f"No recent news found for {ticker}.")
                    st.markdown("---")

                    # --- Financial Statements ---
                    # Income Statement
                    if income_stmt is not None and not income_stmt.empty:
                        st.markdown("#### Income Statement (Annual in ₹ Crores)")
                        # Convert to crores
                        income_stmt_cr = income_stmt / 10000000
                        st.dataframe(income_stmt_cr.T.applymap(lambda x: f"₹{x:,.2f}" if isinstance(x, (int, float)) else x))
                        st.markdown("---")

                    # Balance Sheet
                    if balance_sheet is not None and not balance_sheet.empty:
                        st.markdown("#### Balance Sheet (Annual in ₹ Crores)")
                        balance_sheet_cr = balance_sheet / 10000000
                        st.dataframe(balance_sheet_cr.T.applymap(lambda x: f"₹{x:,.2f}" if isinstance(x, (int, float)) else x))
                        st.markdown("---")

                    # Cash Flow Statement
                    if cash_flow is not None and not cash_flow.empty:
                        st.markdown("#### Cash Flow Statement (Annual in ₹ Crores)")
                        cash_flow_cr = cash_flow / 10000000
                        st.dataframe(cash_flow_cr.T.applymap(lambda x: f"₹{x:,.2f}" if isinstance(x, (int, float)) else x))
            else:
                st.error(f"⚠️ No data available for {ticker}. Please verify the ticker symbol.")
else:
    st.info("ℹ️ Please enter one or more stock ticker symbols to view their fundamental data.")

# --- Explanation Section ---
st.markdown("### Understanding the Valuation Models")
with st.expander("Click to learn more about valuation methods"):
    st.markdown("""
    **Valuation Models Explained:**
    
    1. **Discounted Cash Flow (DCF):**
    - Projects future cash flows and discounts them to present value
    - Most appropriate for companies with predictable cash flows
    - Incorporates growth rate, discount rate, and terminal value
    - Accounts for company's cash and debt
    
    2. **Graham Formula:**
    - Developed by Benjamin Graham (Warren Buffett's mentor)
    - Conservative valuation method for defensive investors
    - Formula: Fair Value = √(22.5 × EPS × Book Value per Share)
    - Applies a P/E limit of 15 to prevent overvaluation
    
    3. **PEG Ratio Valuation:**
    - Considers both price/earnings ratio and growth rate
    - Formula: Fair Value = EPS × (1 + Growth Rate) × Target PEG Ratio
    - Useful for growth companies where earnings are expanding
    
    **Margin of Safety (MoS):**
    - A risk management principle where you only buy when the price is significantly below the estimated intrinsic value
    - The percentage discount applied to the calculated fair value
    - Higher MoS provides more protection against valuation errors
    - Adjust using the sliders in the sidebar
    
    **Recommendation Guide:**
    - **Strong Buy:** Current price < 80% of average fair value
    - **Buy:** Current price < 100% of average fair value
    - **Hold:** Current price within 20% of fair value
    - **Sell:** Current price > 120% of fair value
    
    **Important Disclaimer:**  
    This data is for informational purposes only. Consult a SEBI-registered financial advisor before making investment decisions. Past performance doesn't guarantee future results.
    """)

st.markdown("---")
st.caption("© Monarch Stock Analysis | Data Source: Yahoo Finance | Note: All values in Indian Rupees (₹)")
# pages/fundamentals.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Monarch: Investment Analysis", layout="wide")

st.header("📊 Investment Analysis & Portfolio Management")
st.markdown("""
This advanced dashboard provides a holistic view of a company's investment potential, combining multiple valuation models, financial health diagnostics, and peer analysis to support sophisticated investment decisions.
""")

# --- Helper Functions ---
@st.cache_data(ttl=3600)
def fetch_data_bundle(ticker_symbol):
    """Fetches a comprehensive data bundle for a given ticker."""
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        if not info or info.get('quoteType') != 'EQUITY':
            st.warning(f"No equity data found for {ticker_symbol}.")
            return {}
        
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        if financials.empty or balance_sheet.empty or cash_flow.empty:
            st.warning(f"Could not retrieve full financial statements for {ticker_symbol}.")
            return {"info": info}

        return {
            "info": info, 
            "financials": financials, 
            "balance_sheet": balance_sheet,
            "cash_flow": cash_flow, 
            "history": stock.history(period="5y")
        }
    except Exception as e:
        st.error(f"An error occurred while fetching data for {ticker_symbol}: {e}")
        return {}

# --- Calculation Functions ---
def calculate_dcf(info, cash_flow, g, t_g, wacc):
    try:
        fcf = info.get('freeCashflow', cash_flow.loc['Total Cash From Operating Activities'].iloc[0] + cash_flow.loc['Capital Expenditure'].iloc[0])
        shares, cash, debt = info.get('sharesOutstanding'), info.get('totalCash'), info.get('totalDebt')
        if not all([fcf, shares, cash is not None, debt is not None]): return None
        if wacc <= t_g: return None # Avoid division by zero or negative denominator
        
        future_fcf = [fcf * ((1 + g) ** y) for y in range(1, 6)]
        terminal_value = future_fcf[-1] * (1 + t_g) / (wacc - t_g)
        discounted_fcf = [cf / ((1 + wacc) ** (i+1)) for i, cf in enumerate(future_fcf)]
        enterprise_value = sum(discounted_fcf) + (terminal_value / ((1 + wacc) ** 5))
        equity_value = enterprise_value - debt + cash
        return equity_value / shares
    except (KeyError, TypeError, ZeroDivisionError, IndexError): return None

def calculate_graham_number(info):
    try:
        eps, bvps = info.get('trailingEps'), info.get('bookValue')
        if not all([eps, bvps]) or eps <= 0 or bvps <= 0: return None
        return np.sqrt(22.5 * eps * bvps)
    except (TypeError, ValueError): return None

def calculate_ddm(info, wacc, t_g):
    """Calculates the Dividend Discount Model (DDM) value."""
    try:
        dps = info.get('dividendRate')
        if not dps or dps <= 0: return None
        if wacc <= t_g: return None
        return dps * (1 + t_g) / (wacc - t_g)
    except (TypeError, ZeroDivisionError): return None

def calculate_ev_ebitda_value(info):
    """Calculates fair value based on Enterprise Value / EBITDA multiple."""
    try:
        ebitda, shares, debt, cash = info.get('ebitda'), info.get('sharesOutstanding'), info.get('totalDebt'), info.get('totalCash')
        industry = info.get('industry')
        if not all([ebitda, shares, debt is not None, cash is not None]): return None

        industry_multiples = {'Software—Infrastructure': 25.0, 'Semiconductors': 20.0, 'Banks—Regional': 10.0, 'Technology': 22.0, 'Healthcare': 18.0}
        multiple = industry_multiples.get(industry, 15.0)

        enterprise_value = ebitda * multiple
        equity_value = enterprise_value - debt + cash
        return equity_value / shares
    except (TypeError, ZeroDivisionError): return None

def calculate_ps_value(info):
    """Calculates fair value based on Price/Sales multiple."""
    try:
        revenue_per_share = info.get('revenuePerShare')
        industry = info.get('industry')
        if not revenue_per_share: return None

        industry_multiples = {'Software—Infrastructure': 12.0, 'Semiconductors': 8.0, 'Banks—Regional': 3.0, 'Technology': 9.0, 'Healthcare': 5.0}
        multiple = industry_multiples.get(industry, 2.5)

        return revenue_per_share * multiple
    except (TypeError, ZeroDivisionError): return None

def calculate_piotroski_f_score(financials, balance_sheet, cash_flow, info):
    try:
        score = 0
        ni_y0 = financials.loc['Net Income'].iloc[0]
        cfo_y0 = cash_flow.loc['Total Cash From Operating Activities'].iloc[0]
        score += 1 if ni_y0 > 0 else 0
        score += 1 if cfo_y0 > 0 else 0
        assets_y1 = balance_sheet.loc['Total Assets'].iloc[1]
        ni_y1 = financials.loc['Net Income'].iloc[1]
        assets_y2 = balance_sheet.loc['Total Assets'].iloc[2]
        roa_y0 = ni_y0 / assets_y1
        roa_y1 = ni_y1 / assets_y2
        score += 1 if roa_y0 > roa_y1 else 0
        score += 1 if cfo_y0 > ni_y0 else 0
        debt_y0 = balance_sheet.loc['Total Liab'].iloc[0]
        debt_y1 = balance_sheet.loc['Total Liab'].iloc[1]
        score += 1 if (debt_y0 / assets_y1) < (debt_y1 / assets_y2) else 0
        current_assets_y0 = balance_sheet.loc['Total Current Assets'].iloc[0]
        current_liab_y0 = balance_sheet.loc['Total Current Liabilities'].iloc[0]
        current_assets_y1 = balance_sheet.loc['Total Current Assets'].iloc[1]
        current_liab_y1 = balance_sheet.loc['Total Current Liabilities'].iloc[1]
        current_ratio_y0 = current_assets_y0 / current_liab_y0
        current_ratio_y1 = current_assets_y1 / current_liab_y1
        score += 1 if current_ratio_y0 > current_ratio_y1 else 0
        if 'Diluted Average Shares' in financials.index:
            shares_y0 = financials.loc['Diluted Average Shares'].iloc[0]
            shares_y1 = financials.loc['Diluted Average Shares'].iloc[1]
            score += 1 if shares_y0 <= shares_y1 else 0
        else: score += 1
        gm_y0 = financials.loc['Gross Profit'].iloc[0] / financials.loc['Total Revenue'].iloc[0]
        gm_y1 = financials.loc['Gross Profit'].iloc[1] / financials.loc['Total Revenue'].iloc[1]
        score += 1 if gm_y0 > gm_y1 else 0
        revenue_y0 = financials.loc['Total Revenue'].iloc[0]
        revenue_y1 = financials.loc['Total Revenue'].iloc[1]
        asset_turnover_y0 = revenue_y0 / assets_y1
        asset_turnover_y1 = revenue_y1 / assets_y2
        score += 1 if asset_turnover_y0 > asset_turnover_y1 else 0
        return score
    except (KeyError, IndexError, ZeroDivisionError, TypeError): return "N/A"

def generate_investment_thesis(data, valuations, scores, prices):
    """Generates a structured list of bullish and bearish points."""
    bull_points = []
    bear_points = []

    if prices['buy'] and prices['current'] < prices['buy']:
        upside_potential = ((prices['fair'] / prices['current']) - 1) * 100
        bull_points.append(f"**Price is Undervalued:** Current price of ${prices['current']:.2f} is below our Monarch Buy Target of ${prices['buy']:.2f}.")
        bull_points.append(f"**Significant Upside:** Our blended fair value is ${prices['fair']:.2f}, representing a potential upside of **{upside_potential:.2f}%**.")
    elif prices['fair'] and prices['current'] > prices['fair']:
        downside_risk = ((prices['fair'] / prices['current']) - 1) * 100
        bear_points.append(f"**Price is Overvalued:** Current price of ${prices['current']:.2f} is above our Monarch Fair Value of ${prices['fair']:.2f}.")
        bear_points.append(f"**Potential Downside:** The price may be inflated, with a potential downside of **{downside_risk:.2f}%** to reach its intrinsic value.")

    for model, value in valuations.items():
        if value and prices['current']:
            if value > prices['current']:
                bull_points.append(f"The **{model} model** estimates a value of ${value:.2f}, suggesting the stock is undervalued from this perspective.")
            else:
                bear_points.append(f"The **{model} model** estimates a value of ${value:.2f}, suggesting the stock is overvalued from this perspective.")

    if isinstance(scores['piotroski'], int):
        if scores['piotroski'] >= 7:
            bull_points.append(f"**Excellent Financial Health:** A high Piotroski F-Score of **{scores['piotroski']}/9** indicates strong fundamentals and profitability.")
        elif scores['piotroski'] <= 3:
            bear_points.append(f"**Weak Financial Health:** A low Piotroski F-Score of **{scores['piotroski']}/9** signals potential weaknesses in the company's financial standing.")

    beta = data['info'].get('beta')
    if beta:
        if beta < 1.0:
            bull_points.append(f"**Low Volatility:** A Beta of **{beta:.2f}** suggests the stock is less volatile than the overall market, which can be attractive for conservative investors.")
        elif beta > 1.5:
            bear_points.append(f"**High Volatility:** A Beta of **{beta:.2f}** indicates the stock is significantly more volatile than the market, implying higher risk.")

    return bull_points, bear_points

# --- UI Layout ---
ticker_input = st.text_area(
    "Enter Stock Ticker Symbol(s) (e.g., RELIANCE.NS, AAPL, MSFT - separate by comma or new line):",
    "MSFT, AAPL",
    height=100
)
tickers = [t.strip().upper() for t in ticker_input.replace('\n', ',').split(',') if t.strip()]

with st.sidebar:
    st.header("📈 Valuation & Portfolio Config")
    margin_of_safety = st.slider("Required Margin of Safety (%)", 0, 50, 20, key='main_mos')
    
    st.subheader("Valuation Model Weighting")
    weight_dcf = st.slider("DCF Weight (%)", 0, 100, 30, key='w_dcf')
    weight_ddm = st.slider("DDM Weight (%)", 0, 100, 15, key='w_ddm')
    weight_graham = st.slider("Graham Number Weight (%)", 0, 100, 15, key='w_graham')
    weight_ev_ebitda = st.slider("EV/EBITDA Multiple Weight (%)", 0, 100, 10, key='w_ev_ebitda')
    weight_ps = st.slider("P/S Multiple Weight (%)", 0, 100, 10, key='w_ps')
    weight_analyst = st.slider("Analyst Target Weight (%)", 0, 100, 20, key='w_analyst')
    
    total_weight = weight_dcf + weight_ddm + weight_graham + weight_ev_ebitda + weight_ps + weight_analyst
    if total_weight != 100:
        st.warning(f"Weights must sum to 100. Current sum: {total_weight}")

    with st.expander("DCF & DDM Model Parameters"):
        growth_rate = st.slider("FCF/Dividend Growth Rate (5y) (%)", -10, 30, 10, key='g_rate') / 100
        terminal_growth = st.slider("Terminal Growth Rate (%)", 0, 10, 3, key='t_growth') / 100
        discount_rate = st.slider("Discount Rate (WACC) (%)", 5, 20, 9, key='d_rate') / 100

# --- Main Application Logic ---
if not tickers:
    st.info("ℹ️ Please enter one or more stock ticker symbols to begin the analysis.")
else:
    for ticker in tickers:
        st.markdown(f"---")
        st.markdown(f"## Analysis for: **{ticker}**")
        
        with st.spinner(f"Fetching comprehensive data for {ticker}..."):
            data = fetch_data_bundle(ticker)
        
        if not data or "info" not in data:
            st.error(f"Could not retrieve sufficient data for {ticker}."); continue

        info = data["info"]
        st.header(f"{info.get('longName', ticker)}")
        st.markdown(f"**Sector:** {info.get('sector', 'N/A')} | **Industry:** {info.get('industry', 'N/A')}")
        
        current_price = info.get('currentPrice', info.get('previousClose'))
        if not current_price: st.warning("Could not determine current price."); continue

        dcf_val = calculate_dcf(info, data['cash_flow'], growth_rate, terminal_growth, discount_rate) if 'cash_flow' in data else None
        graham_val = calculate_graham_number(info)
        ddm_val = calculate_ddm(info, discount_rate, terminal_growth)
        ev_ebitda_val = calculate_ev_ebitda_value(info)
        ps_val = calculate_ps_value(info)
        analyst_target_val = info.get('targetMeanPrice')
        
        valuations = {
            'DCF': dcf_val, 
            'Graham Number': graham_val,
            'Dividend Discount (DDM)': ddm_val,
            'EV/EBITDA Multiple': ev_ebitda_val,
            'P/S Multiple': ps_val,
            'Analyst Target': analyst_target_val
        }
        weights = {
            'DCF': weight_dcf, 
            'Graham Number': weight_graham,
            'Dividend Discount (DDM)': weight_ddm,
            'EV/EBITDA Multiple': weight_ev_ebitda,
            'P/S Multiple': weight_ps,
            'Analyst Target': weight_analyst
        }
        
        valid_vals = [valuations[k] for k, w in weights.items() if valuations[k] is not None and w > 0]
        valid_weights = [w for k, w in weights.items() if valuations[k] is not None and w > 0]
        
        weighted_avg = np.average(valid_vals, weights=valid_weights) if valid_vals and sum(valid_weights) > 0 else None
        
        # Corrected Logic: Buy Target is Fair Value with MoS, Sell Target is Fair Value
        buy_price = weighted_avg * (1 - margin_of_safety/100) if weighted_avg else None
        sell_price = weighted_avg # Sell target IS the fair value

        prices = {'current': current_price, 'buy': buy_price, 'sell': sell_price, 'fair': weighted_avg}
        piotroski_score = calculate_piotroski_f_score(data['financials'], data['balance_sheet'], data['cash_flow'], info) if all(k in data for k in ["financials", "balance_sheet", "cash_flow"]) else "N/A"
        scores = {'piotroski': piotroski_score}

        summary_tab, thesis_tab, valuation_tab, analyst_tab, health_tab = st.tabs(["Summary", "Investment Thesis", "Valuation Models", "Analyst Consensus", "Financial Health"])

        with summary_tab:
            st.subheader("Monarch Price Targets")
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Current Price", f"${current_price:,.2f}")
            m_col2.metric(f"Monarch Buy Target ({margin_of_safety}% MoS)", f"${buy_price:,.2f}" if buy_price else "N/A", delta_color="inverse")
            m_col3.metric("Monarch Sell Target (Fair Value)", f"${sell_price:,.2f}" if sell_price else "N/A")

            if weighted_avg and current_price:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta", value = current_price,
                    title = {'text': "Price vs. Fair Value Range"},
                    delta = {'reference': weighted_avg, 'relative': True, 'valueformat': '.1%'},
                    gauge = {
                        'axis': {'range': [None, max(current_price, weighted_avg) * 1.2]},
                        'steps' : [
                            {'range': [0, buy_price if buy_price else 0], 'color': "rgba(40, 167, 69, 0.7)"},
                            {'range': [buy_price if buy_price else 0, weighted_avg], 'color': "rgba(255, 193, 7, 0.7)"}],
                        'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.9, 'value': weighted_avg},
                        'bar': {'color': 'rgba(0, 123, 255, 0.8)'}
                    }))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

        with thesis_tab:
            st.subheader("Monarch's Data-Driven Thesis")
            bull, bear = generate_investment_thesis(data, valuations, scores, prices)
            
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.markdown("#### Bullish Case (Reasons to Consider Buying)")
                if bull:
                    for point in bull: st.markdown(f"- ✅ {point}")
                else:
                    st.info("No strong bullish signals detected based on current data and settings.")
            with t_col2:
                st.markdown("#### Bearish Case (Reasons for Caution)")
                if bear:
                    for point in bear: st.markdown(f"- ⚠️ {point}")
                else:
                    st.info("No strong bearish signals detected based on current data and settings.")

        with valuation_tab:
            st.subheader("Intrinsic Value Breakdown")
            val_data = [{"Valuation Model": model, "Estimated Fair Value": f"${value:,.2f}", "Upside/Downside": f"{((value / current_price) - 1) * 100:.2f}%"} for model, value in valuations.items() if value is not None]
            if val_data: st.dataframe(pd.DataFrame(val_data), use_container_width=True, hide_index=True)
            else: st.info("No valuation models could be calculated.")

        with analyst_tab:
            st.subheader("Wall Street Analyst Consensus")
            recommendation = info.get('recommendationKey', 'N/A').replace('_', ' ').title()
            target_mean = info.get('targetMeanPrice')
            target_high = info.get('targetHighPrice')
            target_low = info.get('targetLowPrice')
            num_analysts = info.get('numberOfAnalystOpinions')

            if recommendation != 'N/A':
                st.metric(f"Consensus Recommendation ({num_analysts} Analysts)", recommendation)
            if target_mean and target_low and target_high:
                fig = go.Figure(go.Indicator(
                    mode="number+gauge+delta", value=current_price,
                    delta={'reference': target_mean},
                    title={'text': "Current Price vs. Analyst Mean Target"},
                    gauge={'shape': "bullet", 'axis': {'range': [target_low, target_high]},
                           'threshold': {'line': {'color': "red", 'width': 2}, 'thickness': 0.75, 'value': target_mean},
                           'steps': [{'range': [target_low, target_high], 'color': "lightgray"}]}))
                fig.update_layout(height=150)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f"Analysts have set price targets from **${target_low:,.2f}** to **${target_high:,.2f}**, with a mean of **${target_mean:,.2f}**.")
            else:
                st.info("No analyst price targets available for this stock.")

        with health_tab:
            st.subheader("Financial Health & Quality Score")
            st.metric("Piotroski F-Score (0-9)", f"{piotroski_score}/9" if isinstance(piotroski_score, int) else "N/A")
            if isinstance(piotroski_score, int):
                st.progress(piotroski_score / 9)
                st.caption("A score of 7-9 is strong, while 0-3 is weak. It measures financial strength.")

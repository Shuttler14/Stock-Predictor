import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from predictor import StockPredictor

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MarketMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM STYLING ---
st.markdown("""
<style>
    .metric-container {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE THE BRAIN ---
@st.cache_resource
def load_predictor():
    return StockPredictor()

bot = load_predictor()

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🧠 MarketMind")
    st.caption("Real-Time Stock Prediction System")
    st.markdown("---")
    
    # 1. Input
    ticker = st.text_input("Enter Ticker Symbol", value="AAPL", help="e.g., NVDA, TSLA, GC=F (Gold)").upper()
    
    # 2. Settings
    st.markdown("### Analysis Configuration")
    days = st.slider("Prediction Horizon (Days)", 1, 5, 1)
    
    # Mapping Toggles to Logic Modes
    st.markdown("### Active Modules")
    tech_on = st.checkbox("Technical Analysis", value=True, disabled=True, help="Always active (Base Model)")
    news_on = st.checkbox("News Sentiment", value=True)
    fund_on = st.checkbox("Fundamental Data", value=True)
    
    # Determine Mode based on toggles
    mode = 'technical'
    if news_on and fund_on:
        mode = 'all'
    elif news_on:
        mode = 'news'
    elif fund_on:
        mode = 'fundamental'
        
    st.markdown("---")
    run_btn = st.button("🚀 Run Prediction", type="primary", use_container_width=True)
    
    st.info(f"Current Mode: **{mode.upper()}**")

# --- MAIN DASHBOARD ---
st.title(f"MarketMind Analysis: {ticker}")

if run_btn:
    if not ticker:
        st.warning("Please enter a valid ticker symbol.")
    else:
        with st.spinner(f"Fetching data, reading news, and training model for {ticker}..."):
            try:
                # CALL THE BACKEND
                result = bot.predict_and_explain(ticker, days_ahead=days, mode=mode)
                
                if not result['predicted_price']:
                    st.error(result['explanation'])
                else:
                    # --- TOP LEVEL METRICS ---
                    col1, col2, col3, col4 = st.columns(4)
                    
                    curr_price = result['current_price']
                    pred_price = result['predicted_price']
                    diff = pred_price - curr_price
                    pct_change = (diff / curr_price) * 100
                    
                    with col1:
                        st.metric("Current Price", f"${curr_price:,.2f}")
                    
                    with col2:
                        color = "normal"
                        if diff > 0: color = "inverse"
                        st.metric(f"Predicted ({days} Day)", f"${pred_price:,.2f}", f"{pct_change:+.2f}%", delta_color=color)
                        
                    with col3:
                        sent = result['sentiment_score']
                        icon = "😐"
                        if sent > 0.15: icon = "🚀 Bullish"
                        elif sent < -0.15: icon = "🐻 Bearish"
                        st.metric("News Sentiment", f"{sent:.2f}", icon)
                        
                    with col4:
                        verdict = "BUY / LONG" if diff > 0 else "SELL / SHORT"
                        st.metric("AI Verdict", verdict)

                    # --- EXPLANATION SECTION ---
                    st.markdown("### 📝 AI Logic Explanation")
                    st.success(f"**Reasoning:** {result['explanation']}")

                    # --- CHARTING SECTION ---
                    st.markdown("### 📉 Technical View")
                    
                    # Get data for plotting
                    hist_data = bot.analyzer.get_historical_data(ticker)
                    hist_data = bot.analyzer.add_technical_indicators(hist_data)
                    chart_data = hist_data.tail(100) # Last 100 days
                    
                    fig = go.Figure()
                    
                    # Candlestick
                    fig.add_trace(go.Candlestick(
                        x=chart_data.index,
                        open=chart_data['Open'],
                        high=chart_data['High'],
                        low=chart_data['Low'],
                        close=chart_data['Close'],
                        name="Price"
                    ))
                    
                    # Indicators
                    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['SMA_50'], line=dict(color='orange', width=1), name="SMA 50"))
                    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['SMA_200'], line=dict(color='blue', width=1), name="SMA 200"))
                    
                    fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # --- NEWS FEED ---
                    if result['headlines']:
                        st.markdown("### 📰 Contextual News")
                        for i, head in enumerate(result['headlines'][:5]):
                            st.text(f"{i+1}. {head}")

            except Exception as e:
                st.error(f"System Error: {e}")
                st.caption("Try a different ticker or check your internet connection.")

else:
    # Idle State
    st.info("👈 Enter a ticker (e.g., AAPL, NVDA) and click 'Run Prediction' to start.")
    
    # Optional: Quick Market Glance
    st.markdown("#### Quick Tips:")
    st.markdown("""
    * **Technical Only:** Pure price action analysis.
    * **News Sentiment:** Scrapes Google News for realtime sentiment.
    * **Ensemble (All):** Combines price trends with fundamental valuation and news.
    """)
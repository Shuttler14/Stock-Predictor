import streamlit as st
import plotly.graph_objects as go
from predictor import StockPredictor

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MarketMind AI",
    page_icon="📈",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #262730;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE PREDICTOR ---
@st.cache_resource
def get_predictor():
    return StockPredictor()

bot = get_predictor()

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🧠 MarketMind")
    st.markdown("---")
    
    ticker_input = st.text_input("Stock Ticker", value="AAPL").upper()
    
    st.markdown("### Analysis Mode")
    mode_selection = st.radio(
        "Select Prediction Logic:",
        ('All (Ensemble)', 'Technical Only', 'Fundamental Only', 'News Only')
    )
    
    # Map friendly names to logic keys
    mode_map = {
        'All (Ensemble)': 'all',
        'Technical Only': 'technical',
        'Fundamental Only': 'fundamental',
        'News Only': 'news'
    }
    selected_mode = mode_map[mode_selection]
    
    st.markdown("---")
    predict_btn = st.button("🔮 Analyze & Predict", type="primary")
    
    st.markdown("#### About")
    st.info(
        "MarketMind uses Random Forest Regression "
        "combined with VADER sentiment analysis to "
        "forecast short-term price movements."
    )

# --- MAIN APP LOGIC ---

st.title(f"Real-Time Analysis: {ticker_input}")

if predict_btn:
    with st.spinner(f"Fetching data and training AI model for {ticker_input}..."):
        try:
            # Run the Prediction Engine
            result = bot.predict_and_explain(ticker_input, days_ahead=1, mode=selected_mode)
            
            if not result['predicted_price']:
                st.error(result['explanation'])
            else:
                # --- DISPLAY METRICS ---
                col1, col2, col3, col4 = st.columns(4)
                
                # Calculate change
                current = result['current_price']
                pred = result['predicted_price']
                delta = pred - current
                delta_pct = (delta / current) * 100
                
                with col1:
                    st.metric("Current Price", f"${current:.2f}")
                
                with col2:
                    st.metric("Predicted (Next Close)", f"${pred:.2f}", f"{delta_pct:.2f}%")
                
                with col3:
                    sent_score = result['sentiment_score']
                    emoji = "😐"
                    if sent_score > 0.1: emoji = "🙂 Bullish"
                    elif sent_score < -0.1: emoji = "dV Bearish"
                    st.metric("News Sentiment", f"{sent_score:.2f}", emoji)
                    
                with col4:
                    # Recommendation based on prediction direction
                    action = "BUY / HOLD" if delta > 0 else "SELL / WAIT"
                    color = "green" if delta > 0 else "red"
                    st.markdown(f"**AI Verdict:**")
                    st.markdown(f":{color}[**{action}**]")

                # --- EXPLANATION BLOCK ---
                st.markdown("### 📝 AI Logic Explanation")
                st.success(result['explanation'])

                # --- CHARTS ---
                st.markdown("### 📊 Market Visualization")
                
                # Fetch history for plotting
                hist_df = bot.analyzer.get_historical_data(ticker_input)
                hist_df = bot.analyzer.add_technical_indicators(hist_df)
                
                # Filter to last 6 months for clearer view
                plot_data = hist_df.tail(120)
                
                fig = go.Figure()

                # Candlestick
                fig.add_trace(go.Candlestick(
                    x=plot_data.index,
                    open=plot_data['Open'],
                    high=plot_data['High'],
                    low=plot_data['Low'],
                    close=plot_data['Close'],
                    name='Price'
                ))

                # Add SMAs
                fig.add_trace(go.Scatter(
                    x=plot_data.index, 
                    y=plot_data['SMA_50'], 
                    line=dict(color='orange', width=1.5), 
                    name='SMA 50'
                ))
                
                fig.add_trace(go.Scatter(
                    x=plot_data.index, 
                    y=plot_data['SMA_200'], 
                    line=dict(color='blue', width=1.5), 
                    name='SMA 200'
                ))

                fig.update_layout(
                    height=500,
                    template="plotly_dark",
                    title_text=f"{ticker_input} Price Action + Technicals",
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)

                # --- NEWS SECTION ---
                st.markdown("### 📰 Latest News Headlines")
                if result['headlines']:
                    for headline in result['headlines'][:5]:
                        st.text(f"• {headline}")
                else:
                    st.text("No recent news found.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.warning("Check if the ticker symbol is correct or if your internet connection is active.")

else:
    # Initial State
    st.info("👈 Enter a stock ticker in the sidebar and click 'Analyze' to start.")
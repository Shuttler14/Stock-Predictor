import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from market_engine import MarketAnalyzer

class StockPredictor:
    def __init__(self):
        self.analyzer = MarketAnalyzer()
        
    def predict_and_explain(self, ticker, days_ahead=1, mode='all'):
        """
        Orchestrates data fetching, model training, and explanation generation.
        
        Args:
            ticker (str): Stock symbol (e.g., 'AAPL')
            days_ahead (int): Prediction horizon (default 1 day)
            mode (str): 'technical', 'fundamental', 'news', or 'all'
            
        Returns:
            dict: Prediction results and textual explanation.
        """
        
        # 1. Fetch Data
        print(f"Fetching data for {ticker}...")
        df = self.analyzer.get_historical_data(ticker)
        df = self.analyzer.add_technical_indicators(df)
        
        # Get live context
        sentiment_score, headlines = self.analyzer.get_live_news_sentiment(ticker)
        fundamentals = self.analyzer.get_fundamentals(ticker)
        
        if df.empty or len(df) < 200:
            return {
                'predicted_price': None, 
                'explanation': "Insufficient historical data for analysis (need > 200 days).",
                'sentiment_score': sentiment_score,
                'headlines': headlines
            }

        # 2. Prepare Dataset for Training (Technical Base)
        # We drop the last 'days_ahead' rows for training as they have no target
        feature_cols = ['Close', 'SMA_50', 'SMA_200', 'RSI']
        
        # Drop rows where indicators haven't calculated yet (NaNs)
        model_data = df.dropna().copy()
        
        # Create Target (Future Price)
        model_data['Target'] = model_data['Close'].shift(-days_ahead)
        
        # Remove the final rows which have NaN target after shifting
        train_data = model_data.dropna()
        
        X = train_data[feature_cols]
        y = train_data['Target']
        
        # 3. Train Model (Random Forest)
        # We train on the specific ticker's recent history to capture its volatility profile
        rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X, y)
        
        # 4. Make Base Prediction
        # We use the *very last row* of known data to predict the next step
        last_row = model_data.iloc[[-1]][feature_cols]
        base_prediction = rf.predict(last_row)[0]
        current_price = last_row['Close'].values[0]
        
        # 5. Apply Mode Logic
        final_prediction = base_prediction
        explanation_parts = []
        
        # -- Mode: Technical (Default RF output) --
        if mode == 'technical':
            explanation_parts.append("Prediction based strictly on price action and moving averages.")
            
        # -- Mode: Fundamental (Heuristic Adjustment) --
        # Simulating fundamental impact: "Regression to mean" based on P/E
        pe_ratio = fundamentals.get('forwardPE')
        if mode == 'fundamental' or mode == 'all':
            if pe_ratio:
                # Heuristic: Standard Market P/E is roughly 20-25
                if pe_ratio > 35:
                    adjustment = 0.99 # Dampen prediction slightly (Overvalued)
                    explanation_parts.append(f"Fundamentals: P/E is high ({pe_ratio:.1f}), suggesting overvaluation.")
                    if mode == 'fundamental': final_prediction = current_price * adjustment # Pure fundamental drift
                    else: final_prediction *= adjustment # Apply to technical base
                    
                elif pe_ratio < 15 and pe_ratio > 0:
                    adjustment = 1.01 # Boost prediction slightly (Undervalued)
                    explanation_parts.append(f"Fundamentals: P/E is low ({pe_ratio:.1f}), suggesting value.")
                    if mode == 'fundamental': final_prediction = current_price * adjustment
                    else: final_prediction *= adjustment
                else:
                    explanation_parts.append("Fundamentals: P/E ratio is within normal neutral range.")
            else:
                 explanation_parts.append("Fundamentals: No P/E data available (skipping adjustment).")

        # -- Mode: News (Sentiment Adjustment) --
        # Simulating news impact: Sentiment score (-1 to 1) influences short-term price
        if mode == 'news' or mode == 'all':
            # Impact factor: How much news can move price (e.g., 2%)
            news_impact = 0.02 * sentiment_score 
            
            if abs(sentiment_score) > 0.1:
                if mode == 'news':
                    # If mode is ONLY news, we start from current price and apply drift
                    final_prediction = current_price * (1 + news_impact)
                else:
                    # Apply to technical base
                    final_prediction = final_prediction * (1 + news_impact)
                
                state = "Positive" if sentiment_score > 0 else "Negative"
                explanation_parts.append(f"News: {state} market sentiment detected ({sentiment_score:.2f}).")
            else:
                explanation_parts.append("News: Market sentiment is Neutral.")

        # 6. Generate Logic Explanation (Rule-Based)
        
        # RSI Check
        last_rsi = last_row['RSI'].values[0]
        if last_rsi > 70:
            explanation_parts.append("Technicals: RSI is Overbought (Risk of pullback).")
        elif last_rsi < 30:
            explanation_parts.append("Technicals: RSI is Oversold (Potential bounce).")
        
        # Trend Check
        last_sma200 = last_row['SMA_200'].values[0]
        if current_price > last_sma200:
            explanation_parts.append("Trend: Price is above the 200-day Moving Average (Long-term Bullish).")
        else:
            explanation_parts.append("Trend: Price is below the 200-day Moving Average (Long-term Bearish).")

        # 7. Final Formatting
        explanation_text = " | ".join(explanation_parts)
        
        return {
            'ticker': ticker,
            'current_price': round(current_price, 2),
            'predicted_price': round(final_prediction, 2),
            'explanation': explanation_text,
            'sentiment_score': sentiment_score,
            'headlines': headlines,
            'mode': mode
        }

# --- Quick Test Block ---
if __name__ == "__main__":
    bot = StockPredictor()
    result = bot.predict_and_explain("NVDA", mode='all')
    
    print("\n--- Prediction Report ---")
    print(f"Ticker: {result['ticker']}")
    print(f"Current: ${result['current_price']}")
    print(f"Predicted: ${result['predicted_price']}")
    print(f"Reasoning: {result['explanation']}")
    print(f"Top News: {result['headlines'][0] if result['headlines'] else 'None'}")
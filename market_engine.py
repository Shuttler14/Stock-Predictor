import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob

class MarketAnalyzer:
    """
    A robust engine for fetching stock data, calculating technicals,
    analyzing news sentiment, and retrieving fundamentals.
    """

    def get_historical_data(self, ticker: str, period: str = '2y') -> pd.DataFrame:
        """
        Fetches historical OHLCV data from yfinance.
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df.empty:
                print(f"Error: No data found for ticker symbol '{ticker}'.")
                return pd.DataFrame()
            
            # Ensure index is datetime and sorted
            df.index = pd.to_datetime(df.index)
            return df
        
        except Exception as e:
            print(f"Error fetching historical data for {ticker}: {e}")
            return pd.DataFrame()

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates and adds SMA_50, SMA_200, and RSI_14 to the DataFrame.
        """
        if df.empty:
            return df
            
        try:
            df = df.copy()

            # 1. Simple Moving Averages (SMA)
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()

            # 2. Relative Strength Index (RSI - 14)
            delta = df['Close'].diff()
            
            # Separate gains and losses
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

            # Calculate RS and RSI
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # Fill NaN values (optional: usually better to drop them for ML, but keeping them here for visualization)
            # df.dropna(inplace=True) 
            
            return df

        except Exception as e:
            print(f"Error adding technical indicators: {e}")
            return df

    def get_live_news_sentiment(self, ticker: str):
        """
        Fetches news from Google News RSS and analyzes sentiment using TextBlob.
        Returns: (sentiment_score, top_headlines_list)
        """
        # Using Google News RSS Feed (More stable than scraping raw HTML)
        url = f"https://news.google.com/rss/search?q={ticker}+stock+when:1d&hl=en-US&gl=US&ceid=US:en"
        
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, features="xml")
            items = soup.findAll('item')
            
            headlines = []
            sentiment_scores = []
            
            # Process top 10 headlines
            for item in items[:10]:
                title = item.title.text
                headlines.append(title)
                
                # Analyze sentiment
                blob = TextBlob(title)
                sentiment_scores.append(blob.sentiment.polarity)
            
            if not sentiment_scores:
                return 0.0, ["No recent news found."]
            
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            
            return avg_sentiment, headlines

        except requests.exceptions.RequestException as e:
            print(f"Network error fetching news for {ticker}: {e}")
            return 0.0, ["Error fetching news."]
        except Exception as e:
            print(f"Error parsing news for {ticker}: {e}")
            return 0.0, ["Error analyzing news."]

    def get_fundamentals(self, ticker: str) -> dict:
        """
        Retrieves key fundamental metrics: Forward PE, Market Cap, Dividend Yield.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            fundamentals = {
                'forwardPE': info.get('forwardPE', None),
                'marketCap': info.get('marketCap', None),
                'dividendYield': info.get('dividendYield', None)
            }
            
            return fundamentals
            
        except Exception as e:
            print(f"Error fetching fundamentals for {ticker}: {e}")
            return {'forwardPE': None, 'marketCap': None, 'dividendYield': None}

# --- Quick Test Block ---
if __name__ == "__main__":
    analyzer = MarketAnalyzer()
    ticker = "AAPL"
    
    print(f"--- Testing {ticker} ---")
    
    # Test Data
    df = analyzer.get_historical_data(ticker)
    print(f"Data Shape: {df.shape}")
    
    # Test Indicators
    df = analyzer.add_technical_indicators(df)
    print(f"Tail with Indicators:\n{df[['Close', 'SMA_50', 'RSI']].tail(3)}")
    
    # Test Fundamentals
    fund = analyzer.get_fundamentals(ticker)
    print(f"Fundamentals: {fund}")
    
    # Test Sentiment
    score, headlines = analyzer.get_live_news_sentiment(ticker)
    print(f"Sentiment Score: {score}")
    print(f"Top Headline: {headlines[0] if headlines else 'None'}")
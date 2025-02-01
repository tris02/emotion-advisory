#!/usr/bin/env python3
"""
Detects market events using rule-based triggers and sentiment analysis.

Install with:
    pip install yfinance transformers torch requests pandas
"""

import yfinance as yf
from transformers import pipeline
import pandas as pd
import time
import re
import datetime

# Global Variables
MARKET_DROP_THRESHOLD = -0.05  # 5% drop in major index
VIX_SPIKE_THRESHOLD = 0.10     # 10% spike in VIX

# Instead of loading FinBERT at the top, use a function to load it when needed.
_sentiment_pipeline = None

def get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        try:
            print("Loading FinBERT sentiment analysis model...")
            _sentiment_pipeline = pipeline("text-classification", model="yiyanghkust/finbert-tone")
        except Exception as e:
            print(f"Error loading FinBERT model: {e}")
            _sentiment_pipeline = None
    return _sentiment_pipeline

def check_market_volatility():
    """
    Check S&P 500 and VIX for significant movements.
    """
    try:
        # Fetch the latest S&P 500 and VIX data
        sp500 = yf.Ticker("^GSPC").history(period="1d", interval="5m")
        vix = yf.Ticker("^VIX").history(period="1d", interval="5m")

        # Use .iloc for positional indexing to avoid deprecation warnings
        sp500_change = (sp500['Close'].iloc[-1] / sp500['Close'].iloc[0]) - 1
        vix_change = (vix['Close'].iloc[-1] / vix['Close'].iloc[0]) - 1

        # Print Market Data
        print(f"\n[Market Data - {datetime.datetime.now()}]")
        print(f"S&P 500 Change: {sp500_change:.2%}")
        print(f"VIX Change: {vix_change:.2%}")

        # Check for rule-based triggers
        alerts = []
        if sp500_change <= MARKET_DROP_THRESHOLD:
            alerts.append(f"ALERT: S&P 500 dropped by {sp500_change:.2%}")
        if vix_change >= VIX_SPIKE_THRESHOLD:
            alerts.append(f"ALERT: VIX spiked by {vix_change:.2%}")
        
        return alerts
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return []

def analyze_sentiment(text):
    """
    Analyze sentiment of financial text using FinBERT.
    """
    pipeline_instance = get_sentiment_pipeline()
    if pipeline_instance is None:
        print("FinBERT model not loaded; cannot perform sentiment analysis.")
        return None, None
    result = pipeline_instance(text)
    sentiment = result[0]['label']
    score = result[0]['score']
    return sentiment, score

def keyword_alert(text):
    """
    Detect panic-inducing keywords in financial text.
    """
    keywords = ["market crash", "unexpected rate hike", "economic downturn", "recession"]
    if any(re.search(rf"\b{keyword}\b", text.lower()) for keyword in keywords):
        return f"KEYWORD ALERT: Panic-inducing phrase detected in text: '{text}'"
    return None

def fetch_text_data():
    """
    Simulate incoming text data for analysis (replaceable with real APIs).
    """
    sample_texts = [
        "The stock market is crashing due to unexpected Fed rate hikes.",
        "Good news: Inflation rates are lower than anticipated.",
        "Analysts predict an economic downturn in Q3 due to rising oil prices.",
        "The S&P 500 has reached a 3-month high amidst recovery hopes."
    ]
    return sample_texts

def market_event_detection_system():
    """
    Main function to run market event detection combining rule-based triggers and sentiment analysis.
    """
    print("Starting Market Event Detection System...\n")
    while True:
        try:
            # Step 1: Check Market Volatility (Rule-Based)
            market_alerts = check_market_volatility()
            if market_alerts:
                print("\n[Rule-Based Alerts]")
                for alert in market_alerts:
                    print(alert)
            
            # Step 2: Analyze Sentiment of Text Data
            print("\n[Sentiment Analysis]")
            text_data = fetch_text_data()
            for text in text_data:
                sentiment, score = analyze_sentiment(text)
                keyword_flag = keyword_alert(text)
                print(f"Text: {text}")
                print(f"Sentiment: {sentiment}, Confidence: {score:.4f}")
                if keyword_flag:
                    print(keyword_flag)
            
            # Wait before polling again (e.g., every 1 minute)
            time.sleep(60)
        except KeyboardInterrupt:
            print("\nStopping Market Event Detection System...")
            break
        except Exception as e:
            print(f"Error in system loop: {e}")
            time.sleep(60)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    market_event_detection_system()

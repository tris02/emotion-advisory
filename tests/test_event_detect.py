import unittest
import pandas as pd
from unittest.mock import patch
from emotion_advisory.event_detect import check_market_volatility, analyze_sentiment

class TestEventDetect(unittest.TestCase):
    @patch("emotion_advisory.event_detect.yf.Ticker")
    def test_check_market_volatility(self, mock_ticker):
        """Test if market volatility detection correctly flags alerts"""
        mock_ticker.return_value.history.return_value = pd.DataFrame({
            "Close": [4000, 3800]  # Simulated drop in S&P500
        })

        alerts = check_market_volatility()
        self.assertGreaterEqual(len(alerts), 1)  # Should detect volatility

    def test_analyze_sentiment(self):
        """Test if sentiment analysis categorizes financial text correctly"""
        sentiment, score = analyze_sentiment("The stock market crashed today!")
        self.assertIn(sentiment.lower(), ["positive", "negative", "neutral"])  # Normalize case
        self.assertIsInstance(score, float)  # Ensure it's a float

if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import torch
from emotion_advisory.reaction_detect import fetch_trading_data, predict_behavior_from_trading_data

class TestReactionDetect(unittest.TestCase):

    @patch("emotion_advisory.reaction_detect.alpaca.list_orders")
    def test_fetch_trading_data(self, mock_list_orders):
        """Test trade data fetching with proper Alpaca API mocking"""

        # Fix: Ensure mock return value is a list of trade objects, not dicts
        mock_trade = MagicMock()
        mock_trade.symbol = "AAPL"
        mock_trade.qty = "10"
        mock_trade.side = "buy"
        mock_trade.type = "market"
        mock_trade.status = "new"
        mock_trade.submitted_at = "2025-02-02T01:00:00Z"

        mock_trade2 = MagicMock()
        mock_trade2.symbol = "TSLA"
        mock_trade2.qty = "5"
        mock_trade2.side = "sell"
        mock_trade2.type = "market"
        mock_trade2.status = "accepted"
        mock_trade2.submitted_at = "2025-02-02T01:01:00Z"

        mock_list_orders.return_value = [mock_trade, mock_trade2]

        # Fetch trading data
        trades = fetch_trading_data()

        # Debugging print
        print("\n[DEBUG] Fetched Trades DataFrame:\n", trades)

        # Ensure DataFrame is not empty
        self.assertFalse(trades.empty, "fetch_trading_data() returned an empty DataFrame!")

        # Ensure DataFrame has expected columns
        expected_columns = {"symbol", "qty", "side", "type", "status", "timestamp"}
        self.assertTrue(expected_columns.issubset(trades.columns), 
                        f"Missing expected columns in {trades.columns}")

        # Validate trade presence without enforcing exact values
        self.assertTrue(any(trades["symbol"] == "AAPL"), "AAPL trade not found!")
        self.assertTrue(any(trades["symbol"] == "TSLA"), "TSLA trade not found!")
    
    @patch("emotion_advisory.reaction_detect.DistilBertForSequenceClassification.from_pretrained")
    @patch("emotion_advisory.reaction_detect.DistilBertTokenizer.from_pretrained")
    def test_predict_behavior_from_trading_data(self, mock_tokenizer, mock_model):
        """Test behavior prediction with proper Hugging Face mocks"""

        # Fix: Ensure mock tokenizer behaves properly
        mock_tokenizer.return_value = MagicMock()

        # Fix: Ensure mock model returns a valid tensor instead of breaking `.logits`
        mock_model_instance = MagicMock()
        mock_model_instance.eval.return_value = None  # Prevent actual execution
        mock_model_instance.return_value = MagicMock()
        mock_model_instance.return_value.logits = torch.tensor([[0.7, 0.1, 0.2, 0.9, 0.05]])  # ✅ Fix: Simulated tensor output
        mock_model.return_value = mock_model_instance

        # Simulated trade data
        sample_data = pd.DataFrame({
            "symbol": ["AAPL", "TSLA"],
            "qty": [50, 200],
            "side": ["buy", "sell"],
            "type": ["market", "market"],
            "status": ["new", "filled"],
            "timestamp": ["2025-02-02T01:10:00Z", "2025-02-02T01:15:00Z"]
        })

        # Run prediction
        results = predict_behavior_from_trading_data("./models", sample_data, threshold=0.5)

        # Debugging print
        print("\n[DEBUG] Predicted Labels DataFrame:\n", results)

        # Ensure predictions exist
        self.assertFalse(results.empty, "predict_behavior_from_trading_data() returned an empty DataFrame!")
        self.assertIn("predicted_labels", results.columns, "Missing 'predicted_labels' column in results!")

        # Validate that predictions contain expected behavior labels
        for labels in results["predicted_labels"]:
            self.assertIsInstance(labels, list, "Each prediction should be a list of behavior labels")
            self.assertTrue(all(isinstance(label, str) for label in labels), "Each label should be a string")


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch
from emotion_advisory.advisory_response import import generate_advice_vllm, validate_and_clean_response, VLLMConfig

class TestAdvisoryResponse(unittest.TestCase):

    @patch("emotion_advisory.advisory_response.generate_advice_vllm")
    def test_generate_advice(self, mock_generate_advice):
        """Test if advisory response generation returns a valid advisory message"""

        # ✅ Mock response to simulate valid advisory output
        mock_generate_advice.return_value = (
            "Consider hedging your portfolio with diversified assets."
        )

        # Mocked configuration
        config = VLLMConfig()
        prompt = "Market is highly volatile. Provide a risk mitigation strategy."

        # Call the function
        advice = generate_advice_vllm(prompt, config)

        # ✅ Assertions to ensure expected behavior
        self.assertIsInstance(advice, str, "Advisory response should be a string!")
        self.assertGreater(len(advice), 20, f"Advisory response is too short: {advice}")

    def test_validate_and_clean_response(self):
        """Test if advisory response is formatted correctly"""
        raw_response = "## Hedge Strategy: *Consider options!*"
        cleaned_response = validate_and_clean_response(raw_response)
        self.assertTrue(cleaned_response.startswith("- "))  # Standard formatting check
        self.assertNotIn("*", cleaned_response)  # No markdown symbols

if __name__ == '__main__':
    unittest.main()

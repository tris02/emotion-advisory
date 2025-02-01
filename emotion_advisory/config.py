import os

# API Keys (Masked for Security)
API_KEY = os.getenv("ALPACA_API_KEY", "your_api_key_here")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "your_secret_key_here")

# Base URLs for Different Modules
BASE_URLS = {
    "default": "https://paper-api.alpaca.markets",
    "verify_simulation": "https://paper-api.alpaca.markets/v2/orders",
}

# Function to Fetch Appropriate Base URL
def get_base_url(module_name="default"):
    return BASE_URLS.get(module_name, BASE_URLS["default"])


#!/usr/bin/env python3
"""
Verify trading simulation with Alpaca using direct REST API.
"""

from alpaca_trade_api import REST
import requests
import pandas as pd
from emotion_advisory.config import API_KEY, SECRET_KEY, get_base_url

# Alpaca API Configuration
BASE_URL = get_base_url("verify_simulation")
alpaca = REST(API_KEY, SECRET_KEY, BASE_URL)

# Fetch Recent Orders Using REST API
def fetch_recent_orders():
    try:
        # Request headers with API credentials
        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": API_KEY,
            "APCA-API-SECRET-KEY": SECRET_KEY
        }

        # Make a GET request to the orders endpoint
        response = requests.get(BASE_URL, headers=headers)
        if response.status_code != 200:
            print(f"Error fetching orders: {response.status_code} - {response.text}")
            return pd.DataFrame(columns=["symbol", "qty", "side", "type", "status", "submitted_at"])
        
        # Parse the JSON response
        orders = response.json()
        if not orders:
            print("No orders found.")
            return pd.DataFrame(columns=["symbol", "qty", "side", "type", "status", "submitted_at"])
        
        # Extract relevant order details
        order_data = []
        for order in orders:
            order_data.append({
                "symbol": order["symbol"],
                "qty": int(order["qty"]),
                "side": order["side"],
                "type": order["type"],
                "status": order["status"],
                "submitted_at": order["submitted_at"]
            })
        return pd.DataFrame(order_data)
    except Exception as e:
        print(f"Error fetching orders: {e}")
        return pd.DataFrame(columns=["symbol", "qty", "side", "type", "status", "submitted_at"])

# Verify Simulation Scenarios
def verify_simulation_with_orders():
    print("Fetching order data...")
    order_data = fetch_recent_orders()
    if order_data.empty:
        print("No order data available to verify.")
        return

    # Check Frequent Trades
    print("Checking frequent trades...")
    aapl_orders = order_data[(order_data["symbol"] == "AAPL") & (order_data["side"] == "buy") & (order_data["status"].isin(["new", "accepted", "filled"]))]
    if len(aapl_orders) >= 10:
        print("Frequent trades simulation: PASSED")
    else:
        print(f"Frequent trades simulation: FAILED (Found {len(aapl_orders)} orders)")

    # Check Aggressive Shorting Across Multiple Stocks
    print("Checking aggressive shorting across multiple stocks...")
    shorting_symbols = ["NVDA", "META"]  # Symbols to verify
    shorting_passed = True
    for symbol in shorting_symbols:
        short_orders = order_data[(order_data["symbol"] == symbol) & (order_data["side"] == "sell") & (order_data["status"].isin(["new", "accepted", "filled"]))]
        if len(short_orders) >= 3:  # Expect at least 3 short-sell orders per stock
            print(f"Aggressive shorting simulation for {symbol}: PASSED")
        else:
            print(f"Aggressive shorting simulation for {symbol}: FAILED (Found {len(short_orders)} orders)")
            shorting_passed = False
    if shorting_passed:
        print("Aggressive shorting simulation across multiple stocks: PASSED")
    else:
        print("Aggressive shorting simulation across multiple stocks: FAILED")

    # Check High-Risk Allocation
    print("Checking high-risk allocation...")
    tsla_orders = order_data[(order_data["symbol"] == "TSLA") & (order_data["side"] == "buy") & (order_data["status"].isin(["new", "accepted", "filled"]))]
    if not tsla_orders.empty:
        print("High-risk allocation simulation: PASSED")
    else:
        print("High-risk allocation simulation: FAILED")

if __name__ == "__main__":
    print("Verifying trading simulation with orders...")
    verify_simulation_with_orders()

#!/usr/bin/env python3
"""
Simulate trading actions with Alpaca using direct REST API.
"""

from alpaca_trade_api import REST
import time
from emotion_advisory.config import API_KEY, SECRET_KEY, get_base_url

# Alpaca API Configuration
BASE_URL = get_base_url("default")  # Uses default Alpaca endpoint
alpaca = REST(API_KEY, SECRET_KEY, BASE_URL)

# Simulate Frequent Trades During Market Volatility
def simulate_frequent_trades():
    for _ in range(15):  # Simulate 15 trades
        alpaca.submit_order(
            symbol="AAPL",
            qty=1,
            side="buy",
            type="market",
            time_in_force="gtc"
        )
        time.sleep(1)  # Pause briefly between trades

# Simulating Aggressive Shorting across Multiple Stocks
def simulate_aggressive_shorting_multi():
    symbols = ["NVDA", "META"]  # Stocks to short
    try:
        for symbol in symbols:
            # Fetch the latest quote
            latest_quote = alpaca.get_latest_quote(symbol)
            current_price = latest_quote.bid_price
            if current_price <= 0:
                print(f"Invalid price for {symbol}.")
                continue

            # Simulate aggressive shorting for each stock
            for i in range(3):  # 3 successive short-sell orders per stock
                qty_to_short = (i + 1) * 3  # Increase quantity by 3 each time
                alpaca.submit_order(
                    symbol=symbol,
                    qty=qty_to_short,
                    side="sell",
                    type="market",
                    time_in_force="gtc"
                )
                print(f"Submitted short-sell order for {qty_to_short} shares of {symbol} at ${current_price:.2f}.")
                time.sleep(1)  # Pause briefly between trades
    except Exception as e:
        print(f"Error simulating multi-stock aggressive shorting: {e}")


# Simulate Over-Concentration in High-Risk Assets
def simulate_high_risk_allocation():
   
    # Allocate 100% of the portfolio to a high-risk asset
    buying_power = float(alpaca.get_account().cash)
    high_risk_symbol = "TSLA"
    
    # Fetch the latest quote
    latest_quote = alpaca.get_latest_quote(high_risk_symbol)
    
    # Use ask_price instead of askprice
    if latest_quote.ask_price > 0:  # Ensure a valid ask price
        qty_to_buy = round(int(buying_power // latest_quote.ask_price)/5,0)
        if qty_to_buy > 0:
            alpaca.submit_order(
                symbol=high_risk_symbol,
                qty=qty_to_buy,
                side="buy",
                type="market",
                time_in_force="gtc"
            )
            print(f"Submitted buy order for {qty_to_buy} shares of TSLA.")
        else:
            print("Insufficient buying power to purchase shares.")
    else:
        print(f"Invalid ask price for {high_risk_symbol}.")

if __name__ == "__main__":
    print("Simulating frequent trades...")
    simulate_frequent_trades()

    print("Simulating high-risk allocation...")
    simulate_high_risk_allocation()

    print("Simulating aggressive shorting across multiple stocks...")
    simulate_aggressive_shorting_multi()

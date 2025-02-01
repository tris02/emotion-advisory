# Usage Guide

## Running the Advisory System
To run the **Emotion-Aware Advisory System**, use the following command:
```bash
python main.py
```

## Module-Specific Execution
You can execute individual modules for specific functionalities:

### Market Event Detection
```bash
python -m emotion_advisory.event_detect
```
This will analyze **market volatility**, news sentiment, and generate alerts.

### Trading Behavior Analysis
```bash
python -m emotion_advisory.reaction_detect
```
This module evaluates **user trading behavior** and identifies **emotional triggers**.

### Simulated Trading
```bash
python -m emotion_advisory.simulate_trade
```
This runs **trade simulations** for frequent trading, aggressive shorting, and high-risk allocation.

### Trade Verification
```bash
python -m emotion_advisory.verify_simulation
```
This verifies **trading patterns and behaviors** against recorded market data.

### Generating Financial Advisory
```bash
python -m emotion_advisory.advisory_response
```
This module generates **adaptive AI-driven trading recommendations**.

## Logging and Debugging
- **Logs** are stored in the `logs/` directory.
- You can view logs in real-time with:
```bash
tail -f logs/advisory_system.log
```

## Updating the System
If updates are available, pull the latest changes:
```bash
git pull origin main
pip install --upgrade .
```


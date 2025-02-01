import logging
import pandas as pd
from emotion_advisory.event_detect import check_market_volatility
from emotion_advisory.reaction_detect import fetch_trading_data, predict_behavior_from_trading_data
from emotion_advisory.advisory_response import run_advisory_app
from emotion_advisory.simulate_trade import simulate_frequent_trades, simulate_aggressive_shorting_multi, simulate_high_risk_allocation
from emotion_advisory.verify_simulation import verify_simulation_with_orders

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model Path for Behavior Prediction
DISTILBERT_MODEL_PATH = "./distilbert_trading_model_multi"

def main():
    try:
        logger.info("==============================")
        logger.info("Starting Emotion-Aware Advisory System...")
        logger.info("==============================")
        
        # Step 1: Running Trade Simulations (Prepares Data for Testing)
        logger.info("Running trade simulations...")
        simulate_frequent_trades()
        simulate_high_risk_allocation()
        simulate_aggressive_shorting_multi()
        
        # Step 2: Verifying Simulated Trades
        logger.info("Verifying simulated trades...")
        verify_simulation_with_orders()
        
        # Step 3: Market Event Detection
        logger.info("Analyzing market events...")
        market_alerts = check_market_volatility()
        market_context = "Market Conditions: " + ("; ".join(market_alerts) if market_alerts else "No significant volatility events detected.")
        
        # Step 4: Trading Behavior Detection
        logger.info("Fetching trading data...")
        trades = fetch_trading_data()
        if trades.empty:
            logger.warning("No trading activity detected.")
            print("\n[Advisory Response]\n- No actionable trading activity found in the portfolio.")
            return
        
        logger.info("Predicting trading behaviors...")
        behavior_df = predict_behavior_from_trading_data(DISTILBERT_MODEL_PATH, trades)
        
        # Step 5: Generate Professional Advisory
        logger.info("==============================")
        logger.info("Generating financial advisory...")
        logger.info("==============================")
        run_advisory_app()
        
        logger.info("==============================")
        logger.info("Emotion-Aware Advisory System Execution Completed Successfully.")
        logger.info("==============================")
    
    except Exception as e:
        logger.critical(f"System failure: {e}", exc_info=True)
        print("\n[System Error]\n- An unexpected issue occurred in the advisory pipeline.")

if __name__ == "__main__":
    main()

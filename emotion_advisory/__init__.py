# __init__.py

import logging

# Basic logging setup for all modules
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Mark this as a package
__all__ = ["advisory_response", "event_detect", "reaction_detect", "simulate_trade", "verify_simulation"]

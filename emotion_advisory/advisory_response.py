#!/usr/bin/env python3
"""
Advisory system with recommendations for financial risk management.
"""

import os
import re
import requests
import pandas as pd
import logging
from typing import List, Tuple
from collections import defaultdict
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("financial_advisory.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import financial analysis modules
from emotion_advisory.event_detect import check_market_volatility
from emotion_advisory.reaction_detect import fetch_trading_data, predict_behavior_from_trading_data

DISTILBERT_MODEL_PATH = "./distilbert_trading_model_multi"

@dataclass
class VLLMConfig:
    """Optimized configuration for financial advisory generation"""
    temperature: float = 0.7  # Balance between creativity and focus
    max_tokens: int = 256     # Sufficient for detailed recommendations
    top_p: float = 0.95       # Broad token sampling
    repetition_penalty: float = 1.2  # Prevent redundant content

def get_fallback_advice() -> str:
    """Professional fallback advisory with concrete parameters"""
    return "- Immediately implement position sizing limits (max 2% per trade) and set volatility-adjusted stop-loss orders at 5% below current market price."

def validate_and_clean_response(response: str) -> str:
    """Enforces professional financial advisory standards"""
    # Check for minimum quality criteria
    if not response or len(response) < 25 or not any(c.isalpha() for c in response):
        return get_fallback_advice()
    
    # Professional formatting checks
    response = response.strip()
    if not response.startswith('- '):
        response = '- ' + response
    
    # Remove markdown and ensure financial terminology
    response = re.sub(r'[*#]', '', response)
    required_terms = {'exposure', 'stop-loss', 'hedge', 'position', 'volatility', 'portfolio'}
    if not any(term in response.lower() for term in required_terms):
        return get_fallback_advice()
    
    # Truncate to first complete sentence
    sentence_end = max((response.find(c) for c in '.!?'), default=-1)
    if sentence_end != -1:
        response = response[:sentence_end+1]
    
    return response

def generate_advice_vllm(prompt: str, config: VLLMConfig, stop_marker: str = "\n||END_RECOMMENDATION||") -> str:
    """Enhanced LLM query handler with professional financial guardrails"""
    url = "http://localhost:8000/v1/completions"
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "prompt": prompt,
        "max_new_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "repetition_penalty": config.repetition_penalty,
        "stop": stop_marker
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("choices"):
            generated = data["choices"][0].get("text", "").strip()
            return validate_and_clean_response(generated)
        return get_fallback_advice()
    
    except requests.exceptions.RequestException as e:
        logger.error("Market data API error: %s", str(e))
        return get_fallback_advice()

@dataclass
class TradePattern:
    """Enhanced trading pattern analysis with risk prioritization"""
    symbol: str
    action: str
    total_qty: int
    trade_count: int
    avg_qty: float
    risk_score: float  # Numeric risk score 0-1

def analyze_trade_patterns(behavior_df: pd.DataFrame) -> List[TradePattern]:
    """Quantitative risk analysis engine"""
    patterns = []
    symbol_data = defaultdict(lambda: defaultdict(list))
    
    for _, row in behavior_df.iterrows():
        symbol_data[row['symbol']][row['side']].append(row['qty'])
    
    for symbol, sides in symbol_data.items():
        for action, quantities in sides.items():
            total_qty = sum(quantities)
            trade_count = len(quantities)
            avg_qty = total_qty / trade_count
            
            # Quantitative risk scoring
            quantity_risk = min(avg_qty / 100, 1.0)  # Normalize based on 100 share baseline
            frequency_risk = min(trade_count / 10, 1.0)  # Normalize based on 10 trades baseline
            risk_score = 0.6 * quantity_risk + 0.4 * frequency_risk
            
            patterns.append(TradePattern(
                symbol=symbol,
                action=action,
                total_qty=total_qty,
                trade_count=trade_count,
                avg_qty=avg_qty,
                risk_score=risk_score
            ))
    
    return sorted(patterns, key=lambda x: -x.risk_score)

def generate_behavior_context(behavior_df: pd.DataFrame) -> Tuple[str, List[str]]:
    """Institutional-grade risk analysis reporting"""
    patterns = analyze_trade_patterns(behavior_df)
    risk_factors = []
    pattern_descriptions = []
    
    for pattern in patterns[:3]:  # Top 3 risks only
        desc = (f"{pattern.action.upper()} concentration in {pattern.symbol}: "
                f"{pattern.total_qty} shares across {pattern.trade_count} trades "
                f"(Risk Score: {pattern.risk_score:.2f}/1.0)")
        pattern_descriptions.append(desc)
        
        if pattern.risk_score > 0.7:
            risk_factors.append(
                f"CRITICAL {pattern.action} exposure in {pattern.symbol} "
                f"(Score: {pattern.risk_score:.2f})"
            )

    behavior_indicators = {label for labels in behavior_df["predicted_labels"] for label in labels}
    behavior_indicators.discard("calm")
    
    context = (
        "Portfolio Risk Analysis:\n"
        f"{'\n'.join(pattern_descriptions)}\n\n"
        "Behavioral Red Flags:\n"
        f"- {', '.join(sorted(behavior_indicators))}"
    )
    
    return context, risk_factors

def build_enhanced_prompt(market_context: str, behavior_context: str, risk_factors: List[str]) -> str:
    """Institutional investor-grade prompt engineering"""
    return (
        f"# Professional Financial Advisory Request\n\n"
        f"## Market Context\n{market_context}\n\n"
        f"## Portfolio Analysis\n{behavior_context}\n\n"
        f"## Critical Risk Priorities\n"
        f"{chr(10).join(risk_factors[:3])}\n\n"  # Top 3 risks only
        f"## Advisory Requirements\n"
        "Generate ONE professional risk mitigation strategy for the highest priority risk with:\n"
        "- Specific position sizing (% of portfolio)\n"
        "- Technical indicators (e.g., moving averages, RSI)\n"
        "- Hedge mechanisms (options, futures)\n"
        "- Clear numeric parameters\n\n"
        "## Example Response\n"
        "- Immediately reduce TSLA long exposure to 1.5% of portfolio value, implement 3% trailing stop-loss "
        "based on 20-day volatility (current IV: 45%), and hedge with weekly 650-strike puts covering 75% of exposure.\n\n"
        "## Required Recommendation\n- "
    )

def run_advisory_app() -> None:
    """Institutional-grade advisory workflow"""
    try:
        logger.info("Initiating portfolio risk analysis...")
        
        # Market Conditions Analysis
        market_alerts = check_market_volatility()
        market_context = "Market Conditions: " + ("; ".join(market_alerts) if market_alerts 
                          else "No significant volatility events detected in major indices")
        
        # Trading Behavior Analysis
        trades = fetch_trading_data()
        if trades.empty:
            logger.warning("No trading activity detected")
            print("\n[Advisory Response]\n- No actionable trading activity found in current portfolio.")
            return
            
        behavior_df = predict_behavior_from_trading_data(DISTILBERT_MODEL_PATH, trades)
        behavior_context, risk_factors = generate_behavior_context(behavior_df)
        
        # Generate Professional Advisory
        prompt = build_enhanced_prompt(market_context, behavior_context, risk_factors)
        config = VLLMConfig(temperature=0.7, max_tokens=256)
        
        raw_advice = generate_advice_vllm(prompt, config)
        final_advice = validate_and_clean_response(raw_advice)

        print("\n[Professional Advisory]")
        print(final_advice)
        logger.info("Advisory process completed successfully")
        
    except Exception as e:
        logger.critical("System failure in advisory pipeline: %s", str(e))
        print("\n[Fallback Advisory]\n- Implement immediate 50% position reduction across all high-volatility assets and consult senior risk management.")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    
    try:
        import torch
        torch.multiprocessing.set_sharing_strategy('file_system')
    except ImportError:
        logger.warning("PyTorch multiprocessing config unavailable")
    
    try:
        run_advisory_app()
    except KeyboardInterrupt:
        logger.info("Advisory process interrupted by user")
    except Exception as e:
        logger.critical("Fatal system error: %s", str(e), exc_info=True)

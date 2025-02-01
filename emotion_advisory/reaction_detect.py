#!/usr/bin/env python3
"""
Detects trading behaviors and predicts future actions using DistilBERT.
"""

from alpaca_trade_api import REST
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments
)
import pandas as pd
import torch
import random
from datetime import datetime, timedelta
import os
from emotion_advisory.config import API_KEY, SECRET_KEY, get_base_url

# Alpaca API Configuration
BASE_URL = get_base_url("default")  # Uses default Alpaca endpoint
alpaca = REST(API_KEY, SECRET_KEY, BASE_URL)

# Directory for the multi-label model
MULTI_LABEL_MODEL_DIR = "./distilbert_trading_model_multi"

# 1. Behavior Set for Multi-Label
ALL_BEHAVIORS = [
    "calm",
    "frequent_trading",
    "large_buy",
    "aggressive_shorting",
    "high_risk",
]

# 2. Fetch User Trading Data from Alpaca
def fetch_trading_data():
    try:
        orders = alpaca.list_orders(status="all")
        if not orders:
            print("No orders found in Alpaca account.")
            return pd.DataFrame()
        
        trading_data = []
        for order in orders:
            trading_data.append({
                "symbol": order.symbol,
                "qty": int(order.qty),
                "side": order.side,
                "type": order.type,
                "status": order.status,
                "timestamp": order.submitted_at
            })
        return pd.DataFrame(trading_data)
    except Exception as e:
        print(f"Error fetching trading data: {e}")
        return pd.DataFrame()

# 3. Simulate Training Data
def simulate_training_data(num_samples=200):
    symbols = ["AAPL", "TSLA", "NVDA", "META", "AMZN"]
    sides = ["buy", "sell"]
    simulated_data = []

    for _ in range(num_samples):
        symbol = random.choice(symbols)
        side = random.choice(sides)
        qty = random.randint(1, 50)
        timestamp = datetime.now() - timedelta(minutes=random.randint(0, 1000))
        
        simulated_data.append({
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": "market",
            "status": "filled",
            "timestamp": timestamp.isoformat()
        })

    # Simulate frequent trading behavior for AAPL
    frequent_trading_symbol = "AAPL"
    base_time = datetime.now()

    for i in range(10):
        timestamp = base_time - timedelta(seconds=i * 5)
        simulated_data.append({
            "symbol": frequent_trading_symbol,
            "qty": random.randint(1, 10),
            "side": "buy",
            "type": "market",
            "status": "filled",
            "timestamp": timestamp.isoformat()
        })

    return pd.DataFrame(simulated_data)

# 4. Multi-Label Behavior Logic
def identify_behavior_vector(row):
    """
    Return a 5-element list indicating which behaviors apply:
      Index 0 -> calm
      Index 1 -> frequent_trading
      Index 2 -> large_buy
      Index 3 -> aggressive_shorting
      Index 4 -> high_risk
    By default, we set calm=1 only if no other behaviors are triggered.
    """
    vec = [0, 0, 0, 0, 0]

    # frequent_trading
    if row.get('trade_count', 0) >= 5:
        vec[1] = 1
    
    # large_buy
    if row['qty'] > 10 and row['side'] == "buy":
        vec[2] = 1
    
    # aggressive_shorting
    if row['side'] == "sell" and row['qty'] > 5:
        vec[3] = 1
    
    # high_risk
    if row['symbol'] in ["TSLA", "NVDA", "META"]:
        vec[4] = 1

    # calm -> only if none of the others triggered
    if sum(vec[1:]) == 0:
        vec[0] = 1
    
    return vec

def label_trading_behavior(trading_data):
    if trading_data.empty:
        print("No trading data available for labeling.")
        return pd.DataFrame()

    trading_data['timestamp'] = pd.to_datetime(trading_data['timestamp'])
    trading_data.sort_values(by=['symbol', 'timestamp'], inplace=True)

    # Detect frequent trading using rolling 3-minute window
    trade_counts = (
        trading_data.set_index('timestamp')
        .groupby('symbol')['symbol']
        .rolling('3min')
        .count()
        .reset_index(name='trade_count')
    )

    trading_data = pd.merge(
        trading_data,
        trade_counts,
        on=['symbol', 'timestamp'],
        how='left'
    )

    trading_data['behavior_vector'] = trading_data.apply(identify_behavior_vector, axis=1)
    return trading_data

# 5. Prepare Data for DistilBERT
def prepare_training_data(trading_data):
    if trading_data.empty:
        print("No labeled trading data available for training.")
        return pd.DataFrame()
    
    trading_data['text'] = trading_data.apply(
        lambda row: f"{row['side']} {row['qty']} shares of {row['symbol']} on {row['timestamp']}",
        axis=1
    )
    return trading_data[['text', 'behavior_vector']]

# 6. Fine-Tune DistilBERT for Multi-Label
def train_distilbert_model(train_texts, train_label_vecs):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    train_encodings = tokenizer(list(train_texts), truncation=True, padding="max_length")

    class TradingDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, label_vecs):
            self.encodings = encodings
            self.label_vecs = label_vecs

        def __len__(self):
            return len(self.label_vecs)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.label_vecs[idx], dtype=torch.float)
            return item

    train_dataset = TradingDataset(train_encodings, train_label_vecs)

    try:
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=5,
            problem_type="multi_label_classification"
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

    if model is None:
        raise ValueError(f"Failed to load DistilBERT model. Ensure the model is trained and exists.")

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        evaluation_strategy="no",
        save_strategy="steps",
        logging_dir="./logs",
        logging_steps=50,
        save_steps=50,
        learning_rate=5e-5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    trainer.train()
    model.save_pretrained(MULTI_LABEL_MODEL_DIR, safe_serialization=False)
    return model

# 7. Real-Time Multi-Label Prediction with Calm Fallback
def predict_behavior_from_trading_data(model_path, trading_data, threshold=0.5):
    """
    1. Create 'text' if missing
    2. Tokenize
    3. Forward pass
    4. Sigmoid + threshold for each label
    5. If no labels are predicted, use ["calm"]
    """
    if 'text' not in trading_data.columns:
        if 'timestamp' in trading_data.columns:
            time_col = 'timestamp'
        else:
            time_col = 'submitted_at'
        trading_data['text'] = trading_data.apply(
            lambda row: f"{row['side']} {row['qty']} shares of {row['symbol']} on {row[time_col]}",
            axis=1
        )

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=5,
        problem_type="multi_label_classification"
    )
    model.eval()

    results = []
    for idx, row in trading_data.iterrows():
        text = row['text']
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]
            probs = torch.sigmoid(logits).tolist()

        predicted_labels = []
        for i, p in enumerate(probs):
            if p >= threshold:
                predicted_labels.append(ALL_BEHAVIORS[i])

        # --- Fallback to calm if no label predicted ---
        if len(predicted_labels) == 0:
            predicted_labels = ["calm"]

        results.append(predicted_labels)

    trading_data["predicted_labels"] = results
    return trading_data

# Main Workflow
if __name__ == "__main__":
    # 1) Simulate training data
    simulated_training_data = simulate_training_data()
    
    # 2) Label them with multi-label vectors
    labeled_training_data = label_trading_behavior(simulated_training_data)
    
    # 3) Prepare data for DistilBERT
    training_data = prepare_training_data(labeled_training_data)
    print("\nSimulated Training Data (multi-label):")
    print(training_data.head(10))

    # 4) Train DistilBERT and save to MULTI_LABEL_MODEL_DIR
    train_texts = training_data['text'].values
    train_label_vecs = training_data['behavior_vector'].values
    model = train_distilbert_model(train_texts, train_label_vecs)

    # 5) Fetch real trading data
    real_trading_data = fetch_trading_data()
    if real_trading_data.empty:
        print("No trading data found in Alpaca. Exiting.")
        exit()

    # 6) Optionally label it for debugging
    real_labeled_data = label_trading_behavior(real_trading_data)
    real_labeled_data['text'] = real_labeled_data.apply(
        lambda row: f"{row['side']} {row['qty']} shares of {row['symbol']} on {row['timestamp']}",
        axis=1
    )

    # 7) Predict with fallback to calm if empty
    predictions = predict_behavior_from_trading_data(MULTI_LABEL_MODEL_DIR, real_labeled_data, threshold=0.5)
    print("\nMulti-Label Predictions from Alpaca Data:")
    print(predictions[['text', 'predicted_labels']])

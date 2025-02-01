# Emotion-Aware Advisory System

## Overview
The **Emotion-Aware Advisory System** is a simulated AI-powered trading advisory tool that integrates market event detection, behavior analysis, and real-time trading recommendations. It leverages **Alpaca API** for simulated trading & market insights and **machine learning models** FinBert, DistilBert and DeepSeek to detect trading behaviors, mitigate emotionally driven decisions, and provide rational financial guidance.

## Features
- **Real-time market event detection** (news sentiment, volatility monitoring).
- **AI-powered behavior recognition** (analyzes trading patterns and emotional triggers).
- **Simulated trading scenarios** for strategy testing.
- **Adaptive financial advisory recommendations** using NLP models.

## Installation

### Prerequisites
Ensure you have **Python 3.8+** installed. You also need **pip** and **git**.

### Clone Repository
```bash
git clone https://github.com/tris02/emotion-advisory.git
cd emotion-advisory
```

### Install Package
```bash
pip install .
```

## Usage Guide

### Set Up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows (CMD)
```

### Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Configure API Keys
Edit emotion_advisory/config.py and replace placeholders:
```bash
API_KEY = os.getenv("ALPACA_API_KEY", "your_api_key_here")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "your_secret_key_here")
```

### Running the System

#### 1️. Start vLLM Server
The advisory system requires vLLM for AI-driven recommendations. Start the server:
```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --tensor-parallel-size 1 --max-model-len 1024 --enforce-eager
```
Verify vLLM is running:
```bash
ss -tulnp | grep 8000   # Linux/macOS (alternative to netstat)
```
You should see an output similar to:
```bash
tcp LISTEN 0 128 0.0.0.0:8000
```

#### 2. Running the Advisory System
To **initialize the DistilBert model**:
```bash
python -m emotion_advisory.reaction_detect
```
To run the **Emotion-Aware Advisory System**, use the following command:
```bash
python -m emotion_advisory.main or python main.py
```

## Contributing
- Fork the repository.
- Create a new branch (`feature-xyz`).
- Commit your changes and push to GitHub.
- Open a Pull Request (PR).

## License
This project is licensed under the **MIT License**. See [LICENSE](../LICENSE) for details.

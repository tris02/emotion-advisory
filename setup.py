from setuptools import setup, find_packages

setup(
    name="emotion_advisory",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "yfinance",
        "transformers",
        "torch",
        "requests",
        "pandas",
        "alpaca-trade-api",
        "accelerate"
    ],
)

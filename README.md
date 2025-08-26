# ML-Powered Futures Trading Bot: A Quant Finance Case Study

## Overview
This repository demonstrates a machine learning-based trading strategy for futures markets, focusing on backtesting and educational purposes. Built in Python with PyTorch for LSTM-based predictions, it incorporates technical indicators (MACD, RSI, ATR), volatility filters, and risk management (Kelly criterion). The strategy targets multi-symbol trading (e.g., NQ, ES, CL futures) for diversification.

**Key Features:**
- LSTM neural network for sequence prediction on price features.
- Backtesting on historical data via yfinance.
- Modular design for easy extension (e.g., add new indicators).
- Conservative risk params to emphasize safety.

This is a case study from my portfolio, showcasing quant development skills. For live trading or custom versions, contact me—I've used similar bots to pass prop firm evals and generate consistent returns. **Disclaimer: This is for educational use only. Trading involves risk; past performance isn't indicative of future results. Not financial advice—consult a professional.**

## Architecture
- **Data Prep**: Fetch OHLCV from yfinance; compute features (returns, MACD diff, RSI, ATR norm, volume norm).
- **ML Model**: LSTM processes sequences (20 lags) to predict up/down moves (binary classification).
- **Signals**: Generate buy/sell based on indicators + ML filter (>0.6 confidence for long, <0.4 for short).
- **Risk Management**: ATR for vol skips, Kelly for sizing (capped conservatively).
- **Backtest**: Simulates multi-symbol portfolio growth.

<img width="2850" height="1976" alt="architecture" src="https://github.com/user-attachments/assets/e3859869-38f9-4bb7-94d7-2549394edeb0" />

## Installation
1. Clone the repo: `git clone https://github.com/yourusername/ml-futures-trading-bot-case-study.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run backtest: `python main.py` (adjust params in code).

Requirements (in requirements.txt):

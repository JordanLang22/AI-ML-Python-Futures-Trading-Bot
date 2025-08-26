import pandas as pd
import numpy as np
import yfinance as yf
from models import TradingLSTM, SeqDataset, train_model, load_model, get_ml_prediction
from indicators import calculate_macd, calculate_rsi, calculate_atr, prepare_features, calculate_kelly, get_sentiment
from utils import is_day_trading_hours
import os  # For env vars

# Params (optimized for high returns; conservative defaults for public demo to promote risk awareness)
symbols = ['NQ=F', 'ES=F', 'CL=F']  # Multi-symbol
allocation = 1.0 / len(symbols)
macd_fast, macd_slow, macd_signal = 5, 13, 4
rsi_period = 14
rsi_buy_threshold, rsi_sell_threshold = 40, 60
volume_multiplier = 1.1
initial_capital = 5000
base_position_size = 1  # Micros
risk_per_trade = 0.001  # Redacted: Original used 0.01 for aggressive sizing; lowered here for demo conservatism
stop_loss_pct, take_profit_pct = 0.015, 0.04
atr_period = 14
vol_threshold_multiplier = 2.0  # For filter
sequence_length = 20  # For LSTM
# Redacted: Original had alpha_vantage_key = 'YOUR_ALPHA_VANTAGE_KEY'; replaced with env var for security
alpha_vantage_key = os.getenv('ALPHA_VANTAGE_KEY', None)  # Use environment variables for keys in production

# Backtest (multi-symbol)
def backtest(timeframe='1h', start='2025-01-01', end='2025-08-07'):  # Redacted: Original used longer periods (e.g., 2022-2025); shortened to recent months to avoid API abuse in public forks
    portfolios = {}
    for sym in symbols:
        data = yf.download(sym, start=start, end=end, interval=timeframe)
        model, scaler = train_model(data)
        atr = calculate_atr(data)
        avg_atr = atr.mean()
        close = data['Close']
        volume = data['Volume']
        avg_volume = volume.rolling(20).mean()
        macd, macd_sig = calculate_macd(close)
        rsi = calculate_rsi(close)

        signals = pd.Series(0, index=close.index)
        buy_cond = (macd > macd_sig) & (rsi < rsi_buy_threshold) & (volume > avg_volume * volume_multiplier)
        sell_cond = (macd < macd_sig) & (rsi > rsi_sell_threshold) & (volume > avg_volume * volume_multiplier)
        signals[buy_cond] = 1
        signals[sell_cond] = -1

        # ML filter
        ml_preds = [get_ml_prediction(model, scaler, data.iloc[max(0, i-50):i+1]) for i in range(len(data))]
        ml_preds = pd.Series(ml_preds, index=data.index)
        signals = signals.where((signals == 1) & (ml_preds > 0.6) | (signals == -1) & (ml_preds < 0.4), 0)

        # Vol filter
        signals[atr > avg_atr * vol_threshold_multiplier] = 0

        returns = close.pct_change() * signals.shift(1)
        kelly_f = calculate_kelly(returns.dropna().rolling(100).apply(lambda x: pd.Series(x)))  # Rolling
        adj_returns = returns * kelly_f.shift()  # Approx
        port = (initial_capital * allocation) * (1 + adj_returns).cumprod()
        portfolios[sym] = port

    total_port = sum(portfolios.values())
    print(f'Final Value: {total_port.iloc[-1]:.2f}, Return %: {((total_port.iloc[-1] - initial_capital) / initial_capital * 100):.2f}%')

# Run backtest()
backtest()
# Redacted: Original call to live_trade('KEY', 'SECRET') was here; omitted as live trading is not included in public version.

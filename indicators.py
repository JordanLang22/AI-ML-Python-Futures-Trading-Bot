import pandas as pd
import numpy as np
import requests  # For sentiment (redacted in public version)

def calculate_macd(data, fast=5, slow=13, signal=4):
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)

def calculate_atr(df, period=14):
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.DataFrame(index=df.index)
    tr['HL'] = high - low
    tr['HC'] = abs(high - close.shift())
    tr['LC'] = abs(low - close.shift())
    tr['TR'] = tr.max(axis=1)
    atr = tr['TR'].rolling(window=period).mean()  # Simple avg for init
    return atr

def prepare_features(df, seq_len):
    close = df['Close']
    macd, macd_sig = calculate_macd(close)
    rsi = calculate_rsi(close)
    atr = calculate_atr(df)
    returns = close.pct_change().shift(-1)
    features = pd.DataFrame({
        'return': close.pct_change(),
        'macd_diff': macd - macd_sig,
        'rsi': rsi,
        'atr_norm': atr / close,
        'vol_norm': df['Volume'] / df['Volume'].rolling(20).mean()
    }).dropna()
    labels = (returns > 0).astype(int).iloc[:-1]
    features = features.iloc[:-1]
    return features.values, labels.values

# Redacted: Original function for get_sentiment using Alpha Vantage API was here; removed to avoid exposing paid APIs and rate limits. In public version, use mock or optional implementation.
def get_sentiment(symbol='NDX'):  # Mock for demo
    return 0.5  # Neutral; implement your own sentiment source in private

def calculate_kelly(returns):
    # Redacted: Original Kelly criterion used aggressive capping at 0.5; added comments for risk awareness
    # Note: Kelly can lead to high volatility; use conservatively or with half-Kelly in practice
    wins = returns[returns > 0]
    losses = -returns[returns < 0]
    if len(losses) == 0: return 0.25  # Lowered from 0.5 for demo
    p = len(wins) / len(returns)
    r = wins.mean() / losses.mean() if len(losses) > 0 else 1
    kelly = p - (1 - p) / r
    return min(max(kelly, 0), 0.25)  # Capped lower for safety in public version

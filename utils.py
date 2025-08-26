from datetime import datetime, timezone, timedelta

def is_day_trading_hours():
    now = datetime.now(timezone.utc) - timedelta(hours=4)
    open_time = now.replace(hour=9, minute=30, second=0)
    close_time = now.replace(hour=16, minute=0, second=0)
    return open_time <= now <= close_time and now.weekday() < 5

# Redacted: Original entire live_trade function was here, including broker connections (e.g., ccxt.binance), order execution (e.g., create_market_buy_order), position fetching, and loop for real-time trading. Removed for safety and liability; this public version is for educational backtesting only. Implement live trading privately with your own broker API and consult a financial advisor.
# def live_trade(broker_key, broker_secret):
#     # Placeholder: Integrate with your broker API here for live execution.
#     pass

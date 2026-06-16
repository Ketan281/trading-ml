import pandas as pd
import numpy as np
import os
import json

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Indicators ────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return round(float(100 - (100 / (1 + rs.iloc[-1]))), 2)

def compute_macd(series):
    ema12  = series.ewm(span=12).mean()
    ema26  = series.ewm(span=26).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return "bullish" if macd.iloc[-1] > signal.iloc[-1] else "bearish"

def compute_ema(series, period=20):
    return round(float(series.ewm(span=period).mean().iloc[-1]), 2)

def compute_atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return round(float(tr.rolling(period).mean().iloc[-1]), 2)

def compute_bollinger(series, period=20):
    ma  = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    price = series.iloc[-1]
    if price > upper.iloc[-1]:
        return "above_upper"
    elif price < lower.iloc[-1]:
        return "below_lower"
    else:
        return "inside_bands"

def compute_vwap(df):
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vwap    = (typical * df["Volume"]).cumsum() / df["Volume"].cumsum()
    price   = df["Close"].iloc[-1]
    return "above_vwap" if price > vwap.iloc[-1] else "below_vwap"

def compute_trend(series):
    ema20 = series.ewm(span=20).mean().iloc[-1]
    ema50 = series.ewm(span=50).mean().iloc[-1]
    price = series.iloc[-1]
    if price > ema20 > ema50:
        return "uptrend"
    elif price < ema20 < ema50:
        return "downtrend"
    else:
        return "sideways"

def compute_volatility(atr, price):
    pct = (atr / price) * 100
    if pct > 2:
        return "high"
    elif pct > 1:
        return "medium"
    else:
        return "low"

# ── Compute Indicators From a DataFrame ───────────────
def compute_indicators(df, symbol):
    """Build the indicator dict from an OHLCV DataFrame. Reusable for any
    symbol regardless of where its data lives (data/ or data/historical/)."""
    if df is None or len(df) < 60:
        return None
    close = df["Close"]
    price = round(float(close.iloc[-1]), 2)
    atr   = compute_atr(df)
    return {
        "symbol":       symbol,
        "price":        price,
        "trend":        compute_trend(close),
        "rsi":          compute_rsi(close),
        "macd":         compute_macd(close),
        "ema_20":       compute_ema(close, 20),
        "ema_50":       compute_ema(close, 50),
        "bollinger":    compute_bollinger(close),
        "vwap":         compute_vwap(df),
        "atr":          atr,
        "volatility":   compute_volatility(atr, price)
    }


# ── Analyze One Symbol ────────────────────────────────
def analyze(symbol):
    path = os.path.join(DATA_DIR, f"{symbol}_daily.csv")
    if not os.path.exists(path):
        print(f"  ⚠ No data found for {symbol}")
        return None

    df     = pd.read_csv(path, index_col="Date", parse_dates=True)
    result = compute_indicators(df, symbol)
    if result is None:
        print(f"  ⚠ Not enough data for {symbol}")
        return None

    out_path = os.path.join(OUTPUT_DIR, f"{symbol}_indicators.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  ✅ {symbol} → {out_path}")
    return result

# ── Main ─────────────────────────────────────────────
if __name__ == "__main__":
    symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS"]
    print("=" * 50)
    print("  Trading AI — Indicator Pipeline")
    print("=" * 50)

    for s in symbols:
        r = analyze(s)
        if r:
            print(f"  {json.dumps(r, indent=2)}\n")

    print("✅ All indicators computed and saved to /outputs!")
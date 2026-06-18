"""
Forex data pipeline — live OHLC candles for major currency pairs via yfinance.

Supports multiple timeframes for multi-timeframe analysis.  yfinance maps forex
pairs as e.g. EURUSD=X.  We normalise to "EUR/USD" display names.
"""

import yfinance as yf
import pandas as pd
from functools import lru_cache
from datetime import datetime, timedelta

PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X",
    "USD/CHF": "USDCHF=X",
    "NZD/USD": "NZDUSD=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
}

# pip value per standard lot (100k units) — for P&L calculation
PIP_INFO = {
    "EUR/USD": {"pip": 0.0001, "pip_value": 10.0},
    "GBP/USD": {"pip": 0.0001, "pip_value": 10.0},
    "AUD/USD": {"pip": 0.0001, "pip_value": 10.0},
    "NZD/USD": {"pip": 0.0001, "pip_value": 10.0},
    "USD/CAD": {"pip": 0.0001, "pip_value": 10.0},
    "USD/CHF": {"pip": 0.0001, "pip_value": 10.0},
    "EUR/GBP": {"pip": 0.0001, "pip_value": 10.0},
    "EUR/JPY": {"pip": 0.01,   "pip_value": 6.5},
    "USD/JPY": {"pip": 0.01,   "pip_value": 6.5},
    "GBP/JPY": {"pip": 0.01,   "pip_value": 6.5},
}

# Typical retail spreads in pips
SPREADS = {
    "EUR/USD": 1.0, "GBP/USD": 1.5, "USD/JPY": 1.2, "AUD/USD": 1.5,
    "USD/CAD": 1.8, "USD/CHF": 1.5, "NZD/USD": 2.0, "EUR/GBP": 1.8,
    "EUR/JPY": 2.0, "GBP/JPY": 3.0,
}

TIMEFRAMES = {
    "15m": {"interval": "15m", "period": "5d"},
    "1h":  {"interval": "1h",  "period": "30d"},
    "4h":  {"interval": "1h",  "period": "60d"},   # yfinance max for 1h is 730d; we resample
    "1d":  {"interval": "1d",  "period": "200d"},
}


def _yf_symbol(pair: str) -> str:
    return PAIRS.get(pair, pair.replace("/", "") + "=X")


def fetch_candles(pair: str, interval: str = "15m", period: str = "5d") -> pd.DataFrame:
    sym = _yf_symbol(pair)
    t = yf.Ticker(sym)
    df = t.history(period=period, interval=interval)
    if df.empty:
        return df
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    return df


def fetch_multi_timeframe(pair: str) -> dict:
    """Fetch candles across all analysis timeframes. Returns {tf: DataFrame}."""
    result = {}
    for tf, params in TIMEFRAMES.items():
        df = fetch_candles(pair, params["interval"], params["period"])
        if df.empty:
            continue
        if tf == "4h" and params["interval"] == "1h":
            df = df.resample("4h").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum"
            }).dropna()
        result[tf] = df
    return result


def candles_to_list(df: pd.DataFrame) -> list:
    """Convert DataFrame to list of dicts for the API/frontend."""
    out = []
    for ts, row in df.iterrows():
        out.append({
            "time": ts.strftime("%Y-%m-%d %H:%M"),
            "open": round(float(row["Open"]), 5),
            "high": round(float(row["High"]), 5),
            "low": round(float(row["Low"]), 5),
            "close": round(float(row["Close"]), 5),
            "volume": int(row["Volume"]) if row["Volume"] else 0,
        })
    return out


def current_price(pair: str) -> float | None:
    df = fetch_candles(pair, "1m", "1d")
    if df.empty:
        return None
    return round(float(df["Close"].iloc[-1]), 5)


def list_pairs():
    return list(PAIRS.keys())

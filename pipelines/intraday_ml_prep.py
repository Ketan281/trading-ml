"""
Intraday ML preparation layer — label generation, feature assembly,
dataset creation for future XGBoost / LightGBM / CatBoost training.

Creates labels for:
  - Next 15/30/60 minute direction
  - Volatility expansion
  - Breakout probability
  - Trend continuation

Generates train-ready datasets with walk-forward split metadata.
Does NOT train models — only prepares the infrastructure.
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FEATURE_DIR = os.path.join(ROOT, "data", "features", "intraday")
DATASET_DIR = os.path.join(ROOT, "data", "ml_datasets", "intraday")
os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

INDEX_YF = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}

LABEL_HORIZONS = [15, 30, 60]
MIN_BARS_FOR_LABELS = 100
VOL_EXPAND_THRESHOLD = 1.3
BREAKOUT_ATR_MULT = 1.5


# ── Label Generators ────────────────────────────────

def _forward_return(close, horizon_bars):
    return close.shift(-horizon_bars) / close - 1


def label_direction(close, horizon_bars, threshold=0.001):
    fwd = _forward_return(close, horizon_bars)
    labels = pd.Series(1, index=close.index, dtype=int)
    labels[fwd > threshold] = 2
    labels[fwd < -threshold] = 0
    labels[fwd.isna()] = np.nan
    return labels


def label_volatility_expansion(df, horizon_bars=12, lookback=20):
    atr_now = _rolling_atr(df, lookback)
    future_range = df["High"].rolling(horizon_bars).max().shift(-horizon_bars) - \
                   df["Low"].rolling(horizon_bars).min().shift(-horizon_bars)
    ratio = future_range / atr_now.replace(0, np.nan)
    labels = (ratio > VOL_EXPAND_THRESHOLD).astype(float)
    labels[ratio.isna()] = np.nan
    return labels


def label_breakout(df, horizon_bars=12, lookback=20):
    high_channel = df["High"].rolling(lookback).max()
    low_channel = df["Low"].rolling(lookback).min()
    future_high = df["High"].rolling(horizon_bars).max().shift(-horizon_bars)
    future_low = df["Low"].rolling(horizon_bars).min().shift(-horizon_bars)
    breakout_up = future_high > high_channel
    breakout_down = future_low < low_channel
    labels = pd.Series(0, index=df.index, dtype=int)
    labels[breakout_up] = 1
    labels[breakout_down] = 1
    labels[future_high.isna()] = np.nan
    return labels.astype(float)


def label_trend_continuation(df, horizon_bars=12):
    close = df["Close"]
    e20 = close.ewm(span=20, adjust=False).mean()
    current_trend = (close > e20).astype(int) * 2 - 1
    future_close = close.shift(-horizon_bars)
    future_trend = (future_close > e20.shift(-horizon_bars)).astype(int) * 2 - 1
    continuation = (current_trend == future_trend).astype(float)
    continuation[future_close.isna()] = np.nan
    return continuation


# ── Feature Engineering ─────────────────────────────

def _rolling_atr(df, period=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_intraday_features(df):
    if df is None or len(df) < 50:
        return None
    c = df["Close"]
    h, l, v = df["High"], df["Low"], df.get("Volume", pd.Series(0, index=df.index))

    feats = pd.DataFrame(index=df.index)

    feats["return_1"] = c.pct_change(1)
    feats["return_3"] = c.pct_change(3)
    feats["return_5"] = c.pct_change(5)
    feats["return_10"] = c.pct_change(10)
    feats["return_20"] = c.pct_change(20)

    feats["rsi_14"] = _rsi(c, 14)
    feats["rsi_7"] = _rsi(c, 7)

    ema8 = c.ewm(span=8, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    feats["ema_gap_8"] = (c - ema8) / ema8
    feats["ema_gap_21"] = (c - ema21) / ema21
    feats["ema_gap_50"] = (c - ema50) / ema50
    feats["ema_cross_8_21"] = (ema8 > ema21).astype(int)

    atr = _rolling_atr(df, 14)
    feats["atr_pct"] = atr / c
    feats["range_pct"] = (h - l) / c
    feats["range_vs_atr"] = (h - l) / atr.replace(0, np.nan)

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    feats["bb_width"] = (2 * bb_std) / bb_mid.replace(0, np.nan)
    feats["bb_position"] = (c - (bb_mid - 2 * bb_std)) / (4 * bb_std).replace(0, np.nan)

    if v.sum() > 0:
        feats["vol_ratio_5"] = v / v.rolling(5).mean().replace(0, np.nan)
        feats["vol_ratio_20"] = v / v.rolling(20).mean().replace(0, np.nan)
        tp = (h + l + c) / 3
        pv = (tp * v).cumsum()
        vv = v.cumsum().replace(0, np.nan)
        vwap = pv / vv
        feats["vwap_dist"] = (c - vwap) / vwap.replace(0, np.nan)
    else:
        feats["vol_ratio_5"] = 1.0
        feats["vol_ratio_20"] = 1.0
        feats["vwap_dist"] = 0.0

    feats["dist_high_20"] = (c - h.rolling(20).max()) / c
    feats["dist_low_20"] = (c - l.rolling(20).min()) / c

    rv5 = feats["return_1"].rolling(5).std() * np.sqrt(252)
    rv20 = feats["return_1"].rolling(20).std() * np.sqrt(252)
    feats["rv_5"] = rv5
    feats["rv_20"] = rv20
    feats["vol_ratio_rv"] = rv5 / rv20.replace(0, np.nan)

    feats["streak"] = _streak(c)

    if hasattr(df.index, "hour"):
        feats["hour"] = df.index.hour
        feats["minute"] = df.index.minute
        feats["session_pct"] = ((df.index.hour - 9) * 60 + df.index.minute - 15) / 375
        feats["session_pct"] = feats["session_pct"].clip(0, 1)

    return feats


def _streak(close):
    signs = np.sign(close.diff())
    streak = pd.Series(0, index=close.index, dtype=int)
    for i in range(1, len(signs)):
        if signs.iloc[i] == signs.iloc[i - 1] and signs.iloc[i] != 0:
            streak.iloc[i] = streak.iloc[i - 1] + int(signs.iloc[i])
        elif signs.iloc[i] != 0:
            streak.iloc[i] = int(signs.iloc[i])
    return streak


# ── Dataset Assembly ────────────────────────────────

def build_labeled_dataset(df, symbol="UNKNOWN"):
    feats = build_intraday_features(df)
    if feats is None:
        return None

    for h in LABEL_HORIZONS:
        bars = h  # for 5m bars, h minutes = h/5 bars; adjust if interval != 1m
        feats[f"label_dir_{h}m"] = label_direction(df["Close"], bars)

    feats["label_vol_expand"] = label_volatility_expansion(df)
    feats["label_breakout"] = label_breakout(df)
    feats["label_trend_cont"] = label_trend_continuation(df)

    feats = feats.dropna(subset=[f"label_dir_{LABEL_HORIZONS[0]}m"])

    return feats


def walk_forward_splits(df, n_splits=5, test_ratio=0.2):
    n = len(df)
    if n < 100:
        return []
    min_train = int(n * 0.3)
    splits = []
    for i in range(n_splits):
        train_end = min_train + int((n - min_train) * (i + 1) / (n_splits + 1))
        test_end = min(train_end + int(n * test_ratio), n)
        splits.append({
            "fold": i + 1,
            "train_start": 0,
            "train_end": train_end,
            "test_start": train_end,
            "test_end": test_end,
            "train_rows": train_end,
            "test_rows": test_end - train_end,
        })
    return splits


def save_dataset(df, symbol, interval="5m"):
    if df is None or df.empty:
        return None
    path = os.path.join(DATASET_DIR, f"{symbol}_{interval}_ml_dataset.parquet")
    df.to_parquet(path, index=True)

    feature_cols = [c for c in df.columns if not c.startswith("label_")]
    label_cols = [c for c in df.columns if c.startswith("label_")]
    splits = walk_forward_splits(df)

    meta = {
        "symbol": symbol,
        "interval": interval,
        "created": datetime.now().isoformat(),
        "rows": len(df),
        "n_features": len(feature_cols),
        "features": feature_cols,
        "labels": label_cols,
        "walk_forward_splits": splits,
        "recommended_models": ["xgboost", "lightgbm", "catboost"],
        "notes": "Labels are forward-looking — NEVER leak into features. "
                 "Use walk-forward splits only.",
    }
    meta_path = os.path.join(DATASET_DIR, f"{symbol}_{interval}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {"path": path, "meta_path": meta_path, "rows": len(df),
            "features": len(feature_cols), "labels": len(label_cols)}


# ── Data Readiness Check ────────────────────────────

def check_data_readiness(symbol, interval="5m"):
    try:
        import yfinance as yf
        yf_sym = INDEX_YF.get(symbol, f"{symbol}.NS")
        df = yf.Ticker(yf_sym).history(period="60d", interval=interval)
    except Exception:
        df = None

    if df is None or df.empty:
        return {"symbol": symbol, "ready": False, "reason": "No data available",
                "rows": 0, "sufficient_for_training": False}

    df = df.rename(columns=str.title)
    rows = len(df)
    has_volume = "Volume" in df.columns and df["Volume"].sum() > 0
    sufficient = rows >= 500

    return {
        "symbol": symbol,
        "interval": interval,
        "ready": True,
        "rows": rows,
        "has_volume": has_volume,
        "date_range": [str(df.index[0].date()), str(df.index[-1].date())],
        "sufficient_for_training": sufficient,
        "min_required": 500,
        "recommendation": "Ready for dataset creation" if sufficient
                          else f"Need {500 - rows} more bars before training",
    }


def prepare_symbol(symbol, interval="5m"):
    readiness = check_data_readiness(symbol, interval)
    if not readiness["ready"]:
        return readiness

    try:
        import yfinance as yf
        yf_sym = INDEX_YF.get(symbol, f"{symbol}.NS")
        df = yf.Ticker(yf_sym).history(period="60d", interval=interval)
        df = df.rename(columns=str.title)
    except Exception:
        return {"symbol": symbol, "error": "Failed to fetch data"}

    dataset = build_labeled_dataset(df, symbol)
    if dataset is None or dataset.empty:
        return {"symbol": symbol, "error": "Failed to build dataset"}

    save_info = save_dataset(dataset, symbol, interval)
    return {**readiness, **save_info, "status": "dataset_created"}


if __name__ == "__main__":
    symbols = sys.argv[1:] or ["NIFTY", "BANKNIFTY", "RELIANCE"]
    print("=" * 64)
    print("  INTRADAY ML PREPARATION LAYER")
    print("=" * 64)

    for s in symbols:
        print(f"\n  Checking {s}...")
        readiness = check_data_readiness(s)
        print(f"    Data: {'OK' if readiness['ready'] else 'NO DATA'}")
        if readiness["ready"]:
            print(f"    Rows: {readiness['rows']}")
            print(f"    Range: {readiness.get('date_range', ['?', '?'])}")
            print(f"    Training ready: {readiness['sufficient_for_training']}")

            if readiness["sufficient_for_training"]:
                result = prepare_symbol(s)
                if "error" not in result:
                    print(f"    Dataset: {result.get('rows', 0)} rows, "
                          f"{result.get('features', 0)} features, "
                          f"{result.get('labels', 0)} labels")
                    print(f"    Saved: {result.get('path', '?')}")
                else:
                    print(f"    Error: {result['error']}")
            else:
                print(f"    {readiness['recommendation']}")

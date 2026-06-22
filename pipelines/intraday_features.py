"""
Intraday feature store -- structured features from 5m and 15m bars.

Computes RVOL, opening range, gap stats, VWAP distance, momentum,
range/volume expansion, time-of-day, and volatility compression/expansion
for use by downstream models and dashboards.
"""

import json
import os
import sys
from datetime import datetime, time as dtime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipelines.intraday import fetch_intraday, INDEX_YF

MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)
SESSION_MINUTES = 375  # 9:15 to 15:30
OR_MINUTES = 15
OR_BARS_5M = 3

FEATURE_DIR = os.path.join(ROOT, "data", "features", "intraday")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _today_bars(df):
    """Return bars for the most recent trading day in the dataframe."""
    last_day = df.index[-1].date()
    return df[df.index.map(lambda x: x.date() == last_day)]


def _session_groups(df):
    """Group bars by trading day (date)."""
    return df.groupby(df.index.date)


def _safe_float(v, default=0.0):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return default
    return float(v)


# ---------------------------------------------------------------------------
# 1. Relative Volume (RVOL)
# ---------------------------------------------------------------------------

def compute_rvol(bars_5m, lookback_days=20):
    """Relative volume: current cumulative session vol vs average cumulative
    vol at the same time-of-day over prior sessions."""
    if bars_5m is None or len(bars_5m) < OR_BARS_5M:
        return {"rvol": None, "rvol_raw_ratio": None}

    today = _today_bars(bars_5m)
    if today.empty:
        return {"rvol": None, "rvol_raw_ratio": None}

    current_cum_vol = float(today["Volume"].sum())
    bars_into_session = len(today)

    groups = _session_groups(bars_5m)
    past_cum_vols = []
    today_date = today.index[-1].date()
    for day, grp in groups:
        if day >= today_date:
            continue
        subset = grp.iloc[:bars_into_session]
        if len(subset) >= max(1, bars_into_session - 2):
            past_cum_vols.append(float(subset["Volume"].sum()))

    past_cum_vols = past_cum_vols[-lookback_days:]
    if not past_cum_vols:
        return {"rvol": None, "rvol_raw_ratio": None}

    avg_cum_vol = np.mean(past_cum_vols)
    if avg_cum_vol <= 0:
        return {"rvol": None, "rvol_raw_ratio": None}

    rvol = current_cum_vol / avg_cum_vol
    return {
        "rvol": round(rvol, 3),
        "rvol_raw_ratio": round(rvol, 3),
    }


# ---------------------------------------------------------------------------
# 2. Opening Range Metrics
# ---------------------------------------------------------------------------

def compute_opening_range(bars_5m):
    """Opening range (first 15 min) metrics: high, low, width, distance,
    breakout flag."""
    if bars_5m is None or len(bars_5m) < OR_BARS_5M:
        return {
            "or_high": None, "or_low": None, "or_width": None,
            "or_distance_pct": None, "or_breakout": "none",
        }

    today = _today_bars(bars_5m)
    if len(today) < OR_BARS_5M:
        return {
            "or_high": None, "or_low": None, "or_width": None,
            "or_distance_pct": None, "or_breakout": "none",
        }

    or_bars = today.iloc[:OR_BARS_5M]
    or_high = float(or_bars["High"].max())
    or_low = float(or_bars["Low"].min())
    or_width = or_high - or_low

    px = float(today["Close"].iloc[-1])
    mid = (or_high + or_low) / 2.0
    or_distance_pct = ((px - mid) / mid * 100) if mid > 0 else 0.0

    if px > or_high:
        breakout = "above"
    elif px < or_low:
        breakout = "below"
    else:
        breakout = "none"

    return {
        "or_high": round(or_high, 2),
        "or_low": round(or_low, 2),
        "or_width": round(or_width, 2),
        "or_distance_pct": round(or_distance_pct, 3),
        "or_breakout": breakout,
    }


# ---------------------------------------------------------------------------
# 3. Gap Statistics
# ---------------------------------------------------------------------------

def compute_gap(bars_5m, prev_close=None):
    """Gap percentage, fill percentage, direction."""
    default = {"gap_pct": None, "gap_fill_pct": None, "gap_direction": "none"}
    if bars_5m is None or len(bars_5m) < OR_BARS_5M:
        return default

    today = _today_bars(bars_5m)
    if today.empty:
        return default

    if prev_close is None:
        groups = _session_groups(bars_5m)
        days_sorted = sorted(groups.groups.keys())
        today_date = today.index[-1].date()
        prior_days = [d for d in days_sorted if d < today_date]
        if not prior_days:
            return default
        prev_day = bars_5m[bars_5m.index.map(lambda x: x.date() == prior_days[-1])]
        prev_close = float(prev_day["Close"].iloc[-1])

    if prev_close <= 0:
        return default

    today_open = float(today["Open"].iloc[0])
    gap_pct = (today_open - prev_close) / prev_close * 100

    if abs(gap_pct) < 0.01:
        return {"gap_pct": 0.0, "gap_fill_pct": 0.0, "gap_direction": "flat"}

    gap_direction = "up" if gap_pct > 0 else "down"

    gap_size = today_open - prev_close
    current_px = float(today["Close"].iloc[-1])
    if gap_direction == "up":
        filled = max(0.0, today_open - min(float(today["Low"].min()), current_px))
    else:
        filled = max(0.0, max(float(today["High"].max()), current_px) - today_open)

    gap_fill_pct = min(1.0, filled / abs(gap_size)) if abs(gap_size) > 0 else 0.0

    return {
        "gap_pct": round(gap_pct, 3),
        "gap_fill_pct": round(gap_fill_pct, 3),
        "gap_direction": gap_direction,
    }


# ---------------------------------------------------------------------------
# 4. VWAP Distance and Bands
# ---------------------------------------------------------------------------

def compute_vwap_features(bars_5m):
    """VWAP distance percentage, slope, and standard deviation bands."""
    default = {
        "vwap": None, "vwap_distance_pct": None, "vwap_slope": "flat",
        "vwap_upper_1sd": None, "vwap_lower_1sd": None,
        "vwap_upper_2sd": None, "vwap_lower_2sd": None,
    }
    if bars_5m is None or len(bars_5m) < OR_BARS_5M:
        return default

    today = _today_bars(bars_5m)
    if len(today) < 2:
        return default

    tp = (today["High"] + today["Low"] + today["Close"]) / 3.0
    cum_vol = today["Volume"].cumsum().replace(0, np.nan)
    cum_tp_vol = (tp * today["Volume"]).cumsum()
    vwap_series = cum_tp_vol / cum_vol

    vwap_val = _safe_float(vwap_series.iloc[-1])
    if vwap_val <= 0:
        return default

    px = float(today["Close"].iloc[-1])
    vwap_distance_pct = (px - vwap_val) / vwap_val * 100

    deviation = tp - vwap_series
    cum_sq_dev = (deviation ** 2 * today["Volume"]).cumsum()
    variance = cum_sq_dev / cum_vol
    std = np.sqrt(variance)
    current_std = _safe_float(std.iloc[-1])

    lookback = min(6, len(vwap_series))
    if lookback >= 2:
        slope_val = float(vwap_series.iloc[-1] - vwap_series.iloc[-lookback])
        if slope_val > 0.01:
            vwap_slope = "rising"
        elif slope_val < -0.01:
            vwap_slope = "falling"
        else:
            vwap_slope = "flat"
    else:
        vwap_slope = "flat"

    return {
        "vwap": round(vwap_val, 2),
        "vwap_distance_pct": round(vwap_distance_pct, 3),
        "vwap_slope": vwap_slope,
        "vwap_upper_1sd": round(vwap_val + current_std, 2),
        "vwap_lower_1sd": round(vwap_val - current_std, 2),
        "vwap_upper_2sd": round(vwap_val + 2 * current_std, 2),
        "vwap_lower_2sd": round(vwap_val - 2 * current_std, 2),
    }


# ---------------------------------------------------------------------------
# 5. Intraday Momentum Score
# ---------------------------------------------------------------------------

def compute_intraday_momentum(bars_5m):
    """Composite momentum score 0-100 from price position in day range,
    direction of last 6 bars, and momentum acceleration."""
    if bars_5m is None or len(bars_5m) < 6:
        return {"momentum_score": None, "momentum_direction": "neutral"}

    today = _today_bars(bars_5m)
    if len(today) < 3:
        return {"momentum_score": None, "momentum_direction": "neutral"}

    day_high = float(today["High"].max())
    day_low = float(today["Low"].min())
    px = float(today["Close"].iloc[-1])
    day_range = day_high - day_low

    if day_range <= 0:
        position_score = 50.0
    else:
        position_score = (px - day_low) / day_range * 100

    tail = today.tail(min(6, len(today)))
    closes = tail["Close"].values
    up_bars = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i - 1])
    down_bars = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i - 1])
    total_moves = max(1, up_bars + down_bars)
    direction_score = (up_bars / total_moves) * 100

    if len(closes) >= 4:
        mid = len(closes) // 2
        first_half_chg = closes[mid] - closes[0]
        second_half_chg = closes[-1] - closes[mid]
        if abs(first_half_chg) > 0:
            accel = (second_half_chg - first_half_chg) / abs(first_half_chg)
            accel_score = 50 + min(50, max(-50, accel * 50))
        else:
            accel_score = 50.0
    else:
        accel_score = 50.0

    momentum = 0.4 * position_score + 0.35 * direction_score + 0.25 * accel_score
    momentum = max(0, min(100, momentum))

    if momentum >= 60:
        direction = "bullish"
    elif momentum <= 40:
        direction = "bearish"
    else:
        direction = "neutral"

    return {
        "momentum_score": round(momentum, 1),
        "momentum_direction": direction,
    }


# ---------------------------------------------------------------------------
# 6. Range Expansion Score
# ---------------------------------------------------------------------------

def compute_range_expansion(bars_5m):
    """Current day range vs average day range."""
    if bars_5m is None or len(bars_5m) < OR_BARS_5M:
        return {"range_expansion": None, "range_trend": "unknown"}

    today = _today_bars(bars_5m)
    if today.empty:
        return {"range_expansion": None, "range_trend": "unknown"}

    current_range = float(today["High"].max() - today["Low"].min())

    groups = _session_groups(bars_5m)
    today_date = today.index[-1].date()
    past_ranges = []
    for day, grp in groups:
        if day >= today_date:
            continue
        bars_count = min(len(today), len(grp))
        subset = grp.iloc[:bars_count]
        past_ranges.append(float(subset["High"].max() - subset["Low"].min()))

    if not past_ranges:
        return {"range_expansion": None, "range_trend": "unknown"}

    avg_range = np.mean(past_ranges)
    if avg_range <= 0:
        return {"range_expansion": None, "range_trend": "unknown"}

    ratio = current_range / avg_range
    trend = "expanding" if ratio > 1.1 else "contracting" if ratio < 0.9 else "normal"

    return {
        "range_expansion": round(ratio, 3),
        "range_trend": trend,
    }


# ---------------------------------------------------------------------------
# 7. Volume Expansion Score
# ---------------------------------------------------------------------------

def compute_volume_expansion(bars_5m):
    """Current session volume vs average session volume, plus trend direction."""
    if bars_5m is None or len(bars_5m) < OR_BARS_5M:
        return {"volume_expansion": None, "volume_trend": "unknown"}

    today = _today_bars(bars_5m)
    if today.empty:
        return {"volume_expansion": None, "volume_trend": "unknown"}

    current_vol = float(today["Volume"].sum())

    groups = _session_groups(bars_5m)
    today_date = today.index[-1].date()
    past_vols = []
    for day, grp in groups:
        if day >= today_date:
            continue
        bars_count = min(len(today), len(grp))
        subset = grp.iloc[:bars_count]
        past_vols.append(float(subset["Volume"].sum()))

    if not past_vols:
        return {"volume_expansion": None, "volume_trend": "unknown"}

    avg_vol = np.mean(past_vols)
    if avg_vol <= 0:
        return {"volume_expansion": None, "volume_trend": "unknown"}

    ratio = current_vol / avg_vol

    recent_bars = today.tail(min(6, len(today)))
    if len(recent_bars) >= 3:
        first_half = recent_bars.iloc[: len(recent_bars) // 2]["Volume"].mean()
        second_half = recent_bars.iloc[len(recent_bars) // 2 :]["Volume"].mean()
        if second_half > first_half * 1.1:
            vol_trend = "increasing"
        elif second_half < first_half * 0.9:
            vol_trend = "decreasing"
        else:
            vol_trend = "steady"
    else:
        vol_trend = "unknown"

    return {
        "volume_expansion": round(ratio, 3),
        "volume_trend": vol_trend,
    }


# ---------------------------------------------------------------------------
# 8. Time-of-Day Features
# ---------------------------------------------------------------------------

def compute_time_features(bars_5m):
    """Minutes since open, time bucket, percentage of session elapsed."""
    if bars_5m is None or bars_5m.empty:
        return {
            "minutes_since_open": None, "time_bucket": "unknown",
            "pct_session_elapsed": None,
        }

    last_ts = bars_5m.index[-1]
    t = last_ts.time()

    open_dt = datetime.combine(last_ts.date(), MARKET_OPEN)
    current_dt = datetime.combine(last_ts.date(), t)
    minutes_since_open = max(0, (current_dt - open_dt).total_seconds() / 60)

    pct_elapsed = min(1.0, minutes_since_open / SESSION_MINUTES)

    if minutes_since_open <= 15:
        bucket = "open_15m"
    elif minutes_since_open <= 90:
        bucket = "morning"
    elif minutes_since_open <= 210:
        bucket = "midday"
    elif minutes_since_open <= 345:
        bucket = "afternoon"
    else:
        bucket = "close_30m"

    return {
        "minutes_since_open": round(minutes_since_open, 1),
        "time_bucket": bucket,
        "pct_session_elapsed": round(pct_elapsed, 4),
    }


# ---------------------------------------------------------------------------
# 9. Volatility Compression and 10. Volatility Expansion
# ---------------------------------------------------------------------------

def compute_vol_compression(bars_5m):
    """Realized vol last 30m vs last 2h, Bollinger squeeze detection,
    and ATR-based volatility expansion flag."""
    default = {
        "vol_compression": False,
        "vol_compression_ratio": None,
        "bb_squeeze": False,
        "vol_expansion": False,
        "vol_expansion_ratio": None,
    }
    if bars_5m is None or len(bars_5m) < 24:
        return default

    today = _today_bars(bars_5m)
    if len(today) < 10:
        return default

    closes = today["Close"]
    returns = closes.pct_change().dropna()
    if len(returns) < 10:
        return default

    # Compression: vol of last 6 bars (~30m) vs last 24 bars (~2h)
    last_6 = returns.tail(6)
    last_24 = returns.tail(min(24, len(returns)))
    vol_30m = float(last_6.std()) if len(last_6) >= 3 else None
    vol_2h = float(last_24.std()) if len(last_24) >= 6 else None

    compression = False
    compression_ratio = None
    if vol_30m is not None and vol_2h is not None and vol_2h > 0:
        compression_ratio = vol_30m / vol_2h
        compression = compression_ratio < 0.6

    # Bollinger Band squeeze: BB width < 20th percentile of the session
    bb_period = min(20, len(closes))
    bb_squeeze = False
    if bb_period >= 5:
        ma = closes.rolling(bb_period).mean()
        std = closes.rolling(bb_period).std()
        bb_width = (2 * std / ma).dropna()
        if len(bb_width) >= 5:
            current_width = float(bb_width.iloc[-1])
            pctile_20 = float(bb_width.quantile(0.20))
            bb_squeeze = current_width < pctile_20

    # Expansion: spike in recent 5m ATR vs session average ATR
    highs = today["High"]
    lows = today["Low"]
    prev_close = today["Close"].shift(1)
    tr = pd.concat([
        highs - lows,
        (highs - prev_close).abs(),
        (lows - prev_close).abs(),
    ], axis=1).max(axis=1)

    session_avg_atr = float(tr.mean())
    recent_atr = float(tr.tail(3).mean()) if len(tr) >= 3 else session_avg_atr

    vol_expansion = False
    vol_expansion_ratio = None
    if session_avg_atr > 0:
        vol_expansion_ratio = recent_atr / session_avg_atr
        vol_expansion = vol_expansion_ratio > 1.5

    return {
        "vol_compression": compression,
        "vol_compression_ratio": round(compression_ratio, 3) if compression_ratio is not None else None,
        "bb_squeeze": bb_squeeze,
        "vol_expansion": vol_expansion,
        "vol_expansion_ratio": round(vol_expansion_ratio, 3) if vol_expansion_ratio is not None else None,
    }


# ---------------------------------------------------------------------------
# Main composite function
# ---------------------------------------------------------------------------

def compute_intraday_features(symbol, bars_5m=None, bars_15m=None):
    """Compute all intraday features for a symbol.
    Fetches bars if not provided. Returns a flat dict of feature values."""
    if bars_5m is None:
        bars_5m = fetch_intraday(symbol, "5m", period="60d")
    if bars_15m is None:
        bars_15m = fetch_intraday(symbol, "15m", period="60d")

    if bars_5m is None or bars_5m.empty:
        return {"symbol": symbol, "error": "no 5m data available"}

    ts = bars_5m.index[-1]
    features = {
        "symbol": symbol,
        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    features.update(compute_rvol(bars_5m))
    features.update(compute_opening_range(bars_5m))
    features.update(compute_gap(bars_5m))
    features.update(compute_vwap_features(bars_5m))
    features.update(compute_intraday_momentum(bars_5m))
    features.update(compute_range_expansion(bars_5m))
    features.update(compute_volume_expansion(bars_5m))
    features.update(compute_time_features(bars_5m))
    features.update(compute_vol_compression(bars_5m))

    return features


# ---------------------------------------------------------------------------
# Feature store: compute and persist for multiple symbols
# ---------------------------------------------------------------------------

def build_feature_store(symbols, save=True):
    """Compute and store features for multiple symbols.
    Saves to data/features/intraday/{symbol}_intraday_features.json"""
    os.makedirs(FEATURE_DIR, exist_ok=True)
    results = {}
    for sym in symbols:
        print(f"  computing features for {sym} ...")
        feat = compute_intraday_features(sym)
        results[sym] = feat
        if save:
            path = os.path.join(FEATURE_DIR, f"{sym}_intraday_features.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(feat, f, indent=2, default=str)
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pprint

    target = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    print(f"Computing intraday features for {target} ...")
    features = compute_intraday_features(target)
    pprint.pprint(features, width=100)

    save_path = os.path.join(FEATURE_DIR, f"{target}_intraday_features.json")
    os.makedirs(FEATURE_DIR, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2, default=str)
    print(f"\nSaved to {save_path}")

"""
Multi-timeframe analysis engine.

Analyses 1m / 5m / 15m / 1h / daily bars simultaneously and produces:
  - Per-TF trend, momentum, and structure reads
  - Timeframe Agreement Score (how many TFs agree on direction)
  - Trend Alignment Score (weighted score across TFs)
  - Counter-trend detection (lower TF diverges from higher TF trend)
  - Trend continuation probability (alignment + momentum + volume)

All outputs are rule-based. No ML.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yfinance as yf

INDEX_YF = {
    "NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK",
    "FINNIFTY": "^CNXFIN", "SENSEX": "^BSESN",
}

TIMEFRAMES = [
    {"label": "1m",  "interval": "1m",  "period": "5d",  "weight": 0.05},
    {"label": "5m",  "interval": "5m",  "period": "5d",  "weight": 0.15},
    {"label": "15m", "interval": "15m", "period": "5d",  "weight": 0.25},
    {"label": "1h",  "interval": "60m", "period": "1mo", "weight": 0.25},
    {"label": "1d",  "interval": "1d",  "period": "6mo", "weight": 0.30},
]


def _fetch(symbol, interval, period):
    yf_sym = INDEX_YF.get(symbol, f"{symbol}.NS")
    try:
        df = yf.Ticker(yf_sym).history(period=period, interval=interval)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df = df.rename(columns=str.title)
    return df


def _ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(df, period=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _macd(close, fast=12, slow=26, sig=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_f - ema_s
    signal_line = macd_line.ewm(span=sig, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def _trend_direction(df):
    if df is None or len(df) < 50:
        return "neutral", 0.0
    c = df["Close"]
    e8 = _ema(c, 8).iloc[-1]
    e21 = _ema(c, 21).iloc[-1]
    e50 = _ema(c, 50).iloc[-1] if len(c) >= 50 else e21
    price = c.iloc[-1]
    rsi = _rsi(c).iloc[-1]
    _, _, hist = _macd(c)
    macd_h = hist.iloc[-1] if len(hist) > 0 else 0

    score = 0
    if price > e8:  score += 1
    if price > e21: score += 1
    if price > e50: score += 1
    if e8 > e21:    score += 1
    if e21 > e50:   score += 1
    if rsi > 50:    score += 1
    if macd_h > 0:  score += 1

    if price < e8:  score -= 1
    if price < e21: score -= 1
    if price < e50: score -= 1
    if e8 < e21:    score -= 1
    if e21 < e50:   score -= 1
    if rsi < 50:    score -= 1
    if macd_h < 0:  score -= 1

    norm = score / 7.0
    if norm >= 0.5:
        return "bullish", norm
    elif norm <= -0.5:
        return "bearish", norm
    return "neutral", norm


def _tf_analysis(df, label):
    if df is None or len(df) < 20:
        return None
    c = df["Close"]
    direction, strength = _trend_direction(df)
    rsi_val = float(_rsi(c).iloc[-1]) if len(c) >= 15 else 50.0
    atr_val = float(_atr(df).iloc[-1]) if len(df) >= 15 else 0.0
    atr_pct = atr_val / c.iloc[-1] * 100 if c.iloc[-1] > 0 else 0.0

    mom_5 = (c.iloc[-1] / c.iloc[-6] - 1) * 100 if len(c) > 6 else 0.0
    vol_ratio = 1.0
    if "Volume" in df.columns and len(df) >= 20:
        avg_vol = df["Volume"].iloc[-20:].mean()
        if avg_vol > 0:
            vol_ratio = df["Volume"].iloc[-1] / avg_vol

    return {
        "timeframe": label,
        "direction": direction,
        "strength": round(strength, 3),
        "rsi": round(rsi_val, 1),
        "momentum_5bar": round(mom_5, 2),
        "atr_pct": round(atr_pct, 2),
        "volume_ratio": round(vol_ratio, 2),
        "price": round(float(c.iloc[-1]), 2),
    }


def _agreement_score(tf_results):
    if not tf_results:
        return 0.0, "neutral"
    bullish = sum(1 for r in tf_results if r["direction"] == "bullish")
    bearish = sum(1 for r in tf_results if r["direction"] == "bearish")
    total = len(tf_results)
    if bullish > bearish:
        return round(bullish / total * 100, 1), "bullish"
    elif bearish > bullish:
        return round(bearish / total * 100, 1), "bearish"
    return round(50.0, 1), "neutral"


def _alignment_score(tf_results, tf_configs):
    if not tf_results:
        return 50.0
    score = 0.0
    total_weight = 0.0
    for r, cfg in zip(tf_results, tf_configs):
        w = cfg["weight"]
        total_weight += w
        if r["direction"] == "bullish":
            score += w * (50 + r["strength"] * 50)
        elif r["direction"] == "bearish":
            score += w * (50 + r["strength"] * 50)
        else:
            score += w * 50
    return round(score / total_weight, 1) if total_weight > 0 else 50.0


def _counter_trend(tf_results):
    if len(tf_results) < 3:
        return False, ""
    higher_tf = tf_results[-1]["direction"]
    lower_tfs = [r["direction"] for r in tf_results[:2]]
    if higher_tf == "neutral":
        return False, ""
    opposite = "bearish" if higher_tf == "bullish" else "bullish"
    diverging = sum(1 for d in lower_tfs if d == opposite)
    if diverging >= 1:
        return True, f"Lower TFs ({','.join(lower_tfs)}) vs daily ({higher_tf})"
    return False, ""


def _continuation_probability(tf_results, agreement_pct, consensus_dir):
    if not tf_results or consensus_dir == "neutral":
        return 50.0
    score = 0.0
    score += agreement_pct * 0.40

    aligned_strength = np.mean([
        abs(r["strength"]) for r in tf_results
        if r["direction"] == consensus_dir
    ]) if any(r["direction"] == consensus_dir for r in tf_results) else 0
    score += aligned_strength * 100 * 0.30

    avg_vol = np.mean([r["volume_ratio"] for r in tf_results])
    vol_boost = min(avg_vol / 1.5, 1.0) * 100
    score += vol_boost * 0.15

    rsi_vals = [r["rsi"] for r in tf_results if r["direction"] == consensus_dir]
    if rsi_vals:
        avg_rsi = np.mean(rsi_vals)
        if consensus_dir == "bullish":
            rsi_score = min(avg_rsi / 70, 1.0) * 100
        else:
            rsi_score = min((100 - avg_rsi) / 70, 1.0) * 100
        score += rsi_score * 0.15

    return round(max(0, min(100, score)), 1)


def multi_timeframe_read(symbol):
    tf_results = []
    tf_configs_used = []

    for cfg in TIMEFRAMES:
        df = _fetch(symbol, cfg["interval"], cfg["period"])
        r = _tf_analysis(df, cfg["label"])
        if r:
            tf_results.append(r)
            tf_configs_used.append(cfg)

    if not tf_results:
        return {"symbol": symbol, "error": "No data for any timeframe"}

    agreement_pct, consensus_dir = _agreement_score(tf_results)
    alignment = _alignment_score(tf_results, tf_configs_used)
    is_counter, counter_detail = _counter_trend(tf_results)
    continuation = _continuation_probability(
        tf_results, agreement_pct, consensus_dir
    )

    return {
        "symbol": symbol,
        "timeframes": tf_results,
        "n_timeframes": len(tf_results),
        "agreement_score": agreement_pct,
        "consensus_direction": consensus_dir,
        "alignment_score": alignment,
        "counter_trend_detected": is_counter,
        "counter_trend_detail": counter_detail,
        "continuation_probability": continuation,
        "consolidated_score": round(
            0.35 * agreement_pct +
            0.35 * alignment +
            0.30 * continuation, 1
        ),
    }


def multi_timeframe_batch(symbols):
    results = {}
    for s in symbols:
        results[s] = multi_timeframe_read(s)
    return results


if __name__ == "__main__":
    symbols = sys.argv[1:] or ["RELIANCE", "NIFTY", "HDFCBANK"]
    print("=" * 64)
    print("  MULTI-TIMEFRAME ANALYSIS ENGINE")
    print("=" * 64)
    for s in symbols:
        r = multi_timeframe_read(s)
        if "error" in r:
            print(f"\n  {s}: {r['error']}")
            continue
        print(f"\n  {s}")
        print(f"  {'TF':<6} {'DIR':<10} {'STR':>6} {'RSI':>6} {'MOM%':>7} {'RVOL':>6}")
        for tf in r["timeframes"]:
            print(f"  {tf['timeframe']:<6} {tf['direction']:<10} "
                  f"{tf['strength']:>6.3f} {tf['rsi']:>6.1f} "
                  f"{tf['momentum_5bar']:>6.2f}% {tf['volume_ratio']:>6.2f}")
        print(f"  ---")
        print(f"  Agreement    : {r['agreement_score']}% ({r['consensus_direction']})")
        print(f"  Alignment    : {r['alignment_score']}")
        print(f"  Counter-trend: {'YES - ' + r['counter_trend_detail'] if r['counter_trend_detected'] else 'No'}")
        print(f"  Continuation : {r['continuation_probability']}%")
        print(f"  CONSOLIDATED : {r['consolidated_score']}")

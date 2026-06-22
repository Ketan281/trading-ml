"""
Market Regime Intelligence V2 — day-type detection + strategy adaptation.

Extends the existing 4-class macro regime (bull/sideways/bear/volatile from
models/regime_detector and pipelines/market_regime) with intraday day-type
classification. Every strategy adapts its parameters to the detected regime.

Day Types (10):
  trend_day, range_day, breakout_day, mean_reversion_day,
  vol_expansion, vol_contraction, panic_selling,
  short_covering_rally, risk_on, risk_off

Signal flow:
  detect_day_type() → classify_regime_v2() → adapt_signal()
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Strategy Adaptation Rules ──────────────────────────
STRATEGY_ADAPTATION = {
    "trend_day": {
        "bias": "follow",
        "stops": "trail_wide",
        "targets": "extended",
        "sizing_multiplier": 1.0,
        "playbook": "Follow the trend. Trail stops wide (2x ATR). Let winners run. "
                    "Add on pullbacks to VWAP. Avoid fading.",
    },
    "range_day": {
        "bias": "fade",
        "stops": "tight",
        "targets": "to_range_bound",
        "sizing_multiplier": 0.6,
        "playbook": "Fade extremes of the range. Tight stops beyond range bounds. "
                    "Take profit at the opposite range boundary. Small size.",
    },
    "breakout_day": {
        "bias": "follow",
        "stops": "below_breakout",
        "targets": "measured_move",
        "sizing_multiplier": 0.8,
        "playbook": "Enter on confirmed breakout with volume. Stop below breakout "
                    "level. Target measured move (range height projected). Watch for "
                    "false breakouts.",
    },
    "mean_reversion_day": {
        "bias": "fade",
        "stops": "beyond_extreme",
        "targets": "to_mean",
        "sizing_multiplier": 0.5,
        "playbook": "Fade the extreme move back towards VWAP/mean. Tight stops "
                    "beyond the extreme. Target the mean. Small position — "
                    "catching knives is dangerous.",
    },
    "vol_expansion": {
        "bias": "reduce_exposure",
        "stops": "wide_atr",
        "targets": "quick_scalp",
        "sizing_multiplier": 0.4,
        "playbook": "Volatility expanding — reduce size, widen stops. Only take "
                    "A+ setups. Quick profits. Avoid holding overnight.",
    },
    "vol_contraction": {
        "bias": "prepare_breakout",
        "stops": "tight",
        "targets": "extended",
        "sizing_multiplier": 0.7,
        "playbook": "Volatility compressing — breakout likely ahead. Position small, "
                    "tight stops. If breakout triggers, add size.",
    },
    "panic_selling": {
        "bias": "no_new_longs",
        "stops": "wide",
        "targets": "none",
        "sizing_multiplier": 0.0,
        "playbook": "PANIC — do NOT buy the dip. No new longs. Protect existing "
                    "positions. Wait for stabilization (2-3 consecutive green bars "
                    "with declining volume).",
    },
    "short_covering_rally": {
        "bias": "no_new_longs",
        "stops": "tight",
        "targets": "quick",
        "sizing_multiplier": 0.3,
        "playbook": "Short squeeze / relief rally — NOT a trend reversal. Do not "
                    "chase. If already long, take partial profits. New longs only "
                    "with A+ conviction.",
    },
    "risk_on": {
        "bias": "follow",
        "stops": "trail_moderate",
        "targets": "extended",
        "sizing_multiplier": 1.0,
        "playbook": "Broad risk-on: beta and cyclicals leading. Full position sizing. "
                    "Trail stops. Rotate into high-beta names.",
    },
    "risk_off": {
        "bias": "defensive",
        "stops": "tight",
        "targets": "conservative",
        "sizing_multiplier": 0.5,
        "playbook": "Risk-off: defensives leading, beta lagging. Reduce gross "
                    "exposure. Favour IT, Pharma, FMCG. Avoid metals, realty, PSU banks.",
    },
}

# ── Scoring Helpers ────────────────────────────────────

def _safe_atr(df, period=14):
    if df is None or len(df) < period + 1:
        return None
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]),
                               np.abs(l[1:] - c[:-1])))
    if len(tr) < period:
        return None
    return float(np.mean(tr[-period:]))


def _adx_proxy(df, period=14):
    """Simplified directional movement proxy (0-100). High = strong trend."""
    if df is None or len(df) < period + 2:
        return 50.0
    closes = df["Close"].values[-(period + 1):]
    ups = np.maximum(np.diff(closes), 0)
    downs = np.maximum(-np.diff(closes), 0)
    sum_up = np.sum(ups)
    sum_down = np.sum(downs)
    total = sum_up + sum_down
    if total == 0:
        return 50.0
    di_diff = abs(sum_up - sum_down) / total
    return float(min(di_diff * 100, 100))


def _intraday_range_ratio(df):
    """Ratio of today's range to recent average range."""
    if df is None or len(df) < 21:
        return 1.0
    ranges = (df["High"] - df["Low"]).values
    today_range = ranges[-1]
    avg_range = np.mean(ranges[-21:-1])
    if avg_range == 0:
        return 1.0
    return float(today_range / avg_range)


def _volume_ratio(df):
    """Today's volume vs 20-day average."""
    if df is None or "Volume" not in df.columns or len(df) < 21:
        return 1.0
    vols = df["Volume"].values
    today_vol = vols[-1]
    avg_vol = np.mean(vols[-21:-1])
    if avg_vol == 0:
        return 1.0
    return float(today_vol / avg_vol)


def _bollinger_bandwidth(df, period=20):
    if df is None or len(df) < period:
        return 50.0
    closes = df["Close"].values[-period:]
    std = np.std(closes)
    mean = np.mean(closes)
    if mean == 0:
        return 50.0
    return float((std / mean) * 100)


def _close_vs_range(df):
    """Where in today's range did we close? 0=low, 1=high."""
    if df is None or len(df) < 1:
        return 0.5
    h, l, c = float(df["High"].iloc[-1]), float(df["Low"].iloc[-1]), float(df["Close"].iloc[-1])
    rng = h - l
    if rng == 0:
        return 0.5
    return float((c - l) / rng)


def _daily_return(df):
    if df is None or len(df) < 2:
        return 0.0
    return float((df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100)


# ── Day-Type Scorers ──────────────────────────────────

def _score_trend_day(df, breadth=None):
    adx = _adx_proxy(df, 14)
    rng = _intraday_range_ratio(df)
    cvr = _close_vs_range(df)
    vol = _volume_ratio(df)
    score = 0.0
    if adx > 40:
        score += 30
    elif adx > 25:
        score += 15
    if rng > 1.5:
        score += 25
    elif rng > 1.2:
        score += 15
    if cvr > 0.8 or cvr < 0.2:
        score += 20
    if vol > 1.3:
        score += 15
    if breadth and breadth.get("composite_score", 50) > 65:
        score += 10
    return min(score, 100)


def _score_range_day(df, breadth=None):
    adx = _adx_proxy(df, 14)
    rng = _intraday_range_ratio(df)
    cvr = _close_vs_range(df)
    score = 0.0
    if adx < 20:
        score += 30
    elif adx < 30:
        score += 15
    if 0.7 < rng < 1.1:
        score += 25
    if 0.3 < cvr < 0.7:
        score += 20
    vol = _volume_ratio(df)
    if vol < 0.9:
        score += 15
    bw = _bollinger_bandwidth(df)
    if bw < 2.0:
        score += 10
    return min(score, 100)


def _score_breakout_day(df, breadth=None):
    rng = _intraday_range_ratio(df)
    vol = _volume_ratio(df)
    bw = _bollinger_bandwidth(df, 20)
    score = 0.0
    if rng > 1.8:
        score += 30
    elif rng > 1.4:
        score += 20
    if vol > 1.5:
        score += 25
    elif vol > 1.2:
        score += 15
    if bw < 1.5:
        score += 20
    cvr = _close_vs_range(df)
    if cvr > 0.85 or cvr < 0.15:
        score += 15
    if breadth and abs(breadth.get("composite_score", 50) - 50) > 20:
        score += 10
    return min(score, 100)


def _score_mean_reversion_day(df, breadth=None):
    ret = _daily_return(df)
    prev_ret = 0.0
    if df is not None and len(df) >= 3:
        prev_ret = float((df["Close"].iloc[-2] / df["Close"].iloc[-3] - 1) * 100)
    score = 0.0
    if abs(prev_ret) > 2.0 and np.sign(ret) != np.sign(prev_ret):
        score += 35
    elif abs(prev_ret) > 1.0 and np.sign(ret) != np.sign(prev_ret):
        score += 20
    cvr = _close_vs_range(df)
    if 0.35 < cvr < 0.65:
        score += 20
    adx = _adx_proxy(df)
    if adx < 25:
        score += 15
    vol = _volume_ratio(df)
    if vol < 1.0:
        score += 10
    if breadth and 40 < breadth.get("composite_score", 50) < 60:
        score += 10
    return min(score, 100)


def _score_vol_expansion(df, breadth=None):
    if df is None or len(df) < 22:
        return 0.0
    closes = df["Close"].values
    recent_vol = np.std(closes[-5:]) / np.mean(closes[-5:]) * 100
    hist_vol = np.std(closes[-22:-5]) / np.mean(closes[-22:-5]) * 100
    score = 0.0
    if hist_vol > 0:
        ratio = recent_vol / hist_vol
        if ratio > 2.0:
            score += 40
        elif ratio > 1.5:
            score += 25
        elif ratio > 1.2:
            score += 15
    rng = _intraday_range_ratio(df)
    if rng > 1.5:
        score += 25
    vol = _volume_ratio(df)
    if vol > 1.3:
        score += 15
    bw = _bollinger_bandwidth(df)
    if bw > 3.0:
        score += 10
    return min(score, 100)


def _score_vol_contraction(df, breadth=None):
    if df is None or len(df) < 22:
        return 0.0
    closes = df["Close"].values
    recent_vol = np.std(closes[-5:]) / np.mean(closes[-5:]) * 100
    hist_vol = np.std(closes[-22:-5]) / np.mean(closes[-22:-5]) * 100
    score = 0.0
    if hist_vol > 0:
        ratio = recent_vol / hist_vol
        if ratio < 0.5:
            score += 40
        elif ratio < 0.7:
            score += 25
    rng = _intraday_range_ratio(df)
    if rng < 0.6:
        score += 25
    bw = _bollinger_bandwidth(df)
    if bw < 1.0:
        score += 20
    vol = _volume_ratio(df)
    if vol < 0.7:
        score += 15
    return min(score, 100)


def _score_panic_selling(df, breadth=None):
    ret = _daily_return(df)
    vol = _volume_ratio(df)
    cvr = _close_vs_range(df)
    score = 0.0
    if ret < -3.0:
        score += 35
    elif ret < -2.0:
        score += 25
    elif ret < -1.5:
        score += 15
    if vol > 2.0:
        score += 25
    elif vol > 1.5:
        score += 15
    if cvr < 0.15:
        score += 20
    if breadth and breadth.get("composite_score", 50) < 25:
        score += 15
    rng = _intraday_range_ratio(df)
    if rng > 2.0:
        score += 10
    return min(score, 100)


def _score_short_covering(df, breadth=None):
    ret = _daily_return(df)
    prev_ret = 0.0
    if df is not None and len(df) >= 3:
        prev_ret = float((df["Close"].iloc[-2] / df["Close"].iloc[-3] - 1) * 100)
    score = 0.0
    if prev_ret < -1.5 and ret > 1.5:
        score += 35
    elif prev_ret < -1.0 and ret > 1.0:
        score += 20
    vol = _volume_ratio(df)
    if vol > 1.5:
        score += 20
    cvr = _close_vs_range(df)
    if cvr > 0.85:
        score += 20
    if breadth and breadth.get("composite_score", 50) < 40:
        score += 15
    return min(score, 100)


def _score_risk_on(df, breadth=None):
    score = 0.0
    ret = _daily_return(df)
    if ret > 0.5:
        score += 20
    if breadth:
        bs = breadth.get("composite_score", 50)
        if bs > 70:
            score += 35
        elif bs > 60:
            score += 20
        regime = breadth.get("regime", "neutral")
        if regime in ("strong_bullish", "bullish"):
            score += 20
    vol = _volume_ratio(df)
    if vol > 1.1:
        score += 10
    cvr = _close_vs_range(df)
    if cvr > 0.6:
        score += 10
    return min(score, 100)


def _score_risk_off(df, breadth=None):
    score = 0.0
    ret = _daily_return(df)
    if ret < -0.3:
        score += 15
    if breadth:
        bs = breadth.get("composite_score", 50)
        if bs < 30:
            score += 35
        elif bs < 40:
            score += 20
        regime = breadth.get("regime", "neutral")
        if regime in ("strong_bearish", "bearish"):
            score += 20
    vol = _volume_ratio(df)
    if vol > 1.2:
        score += 10
    cvr = _close_vs_range(df)
    if cvr < 0.4:
        score += 10
    return min(score, 100)


_SCORERS = {
    "trend_day":            _score_trend_day,
    "range_day":            _score_range_day,
    "breakout_day":         _score_breakout_day,
    "mean_reversion_day":   _score_mean_reversion_day,
    "vol_expansion":        _score_vol_expansion,
    "vol_contraction":      _score_vol_contraction,
    "panic_selling":        _score_panic_selling,
    "short_covering_rally": _score_short_covering,
    "risk_on":              _score_risk_on,
    "risk_off":             _score_risk_off,
}


# ── Public API ─────────────────────────────────────────

def detect_day_type(index_data=None):
    """Classify today's day type from index OHLCV data.

    Args:
        index_data: DataFrame with OHLCV columns, or None to auto-load NIFTY.

    Returns:
        dict with day_type, confidence, all_scores, strategy_adaptation, playbook.
    """
    df = index_data
    if df is None:
        try:
            from pipelines.market_regime import _load_index
            df = _load_index("NIFTY")
        except Exception:
            pass

    breadth = None
    try:
        from pipelines.breadth import breadth_read
        breadth = breadth_read()
    except Exception:
        pass

    scores = {}
    for name, scorer in _SCORERS.items():
        try:
            scores[name] = scorer(df, breadth)
        except Exception:
            scores[name] = 0.0

    if not scores or max(scores.values()) == 0:
        return {
            "day_type": "range_day",
            "confidence": 0.3,
            "all_scores": scores,
            "strategy_adaptation": STRATEGY_ADAPTATION["range_day"],
            "playbook": STRATEGY_ADAPTATION["range_day"]["playbook"],
            "breadth_regime": breadth.get("regime") if breadth else None,
        }

    best = max(scores, key=scores.get)
    best_score = scores[best]
    second = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0
    confidence = min((best_score - second) / 100 + best_score / 100, 1.0)

    return {
        "day_type": best,
        "confidence": round(confidence, 3),
        "all_scores": {k: round(v, 1) for k, v in scores.items()},
        "strategy_adaptation": STRATEGY_ADAPTATION[best],
        "playbook": STRATEGY_ADAPTATION[best]["playbook"],
        "breadth_regime": breadth.get("regime") if breadth else None,
    }


def classify_regime_v2(index_data=None):
    """Full regime classification: macro regime + day type + strategy rules.

    Returns dict with macro_regime, day_type, regime_confidence,
    strategy_rules, allowed/forbidden strategies, sizing/stop/target styles.
    """
    macro_regime = "unknown"
    try:
        from pipelines.market_regime import _load_index, regime_at
        df = _load_index("NIFTY")
        if df is not None:
            result = regime_at(df)
            macro_regime = result.get("label", "unknown") if isinstance(result, dict) else "unknown"
    except Exception:
        pass

    day = detect_day_type(index_data)
    day_type = day["day_type"]
    adapt = day["strategy_adaptation"]

    forbidden = []
    allowed = ["equity_long", "equity_short", "buy_call", "buy_put",
               "bull_call_spread", "bear_put_spread", "iron_condor",
               "long_straddle", "long_strangle"]

    if day_type == "panic_selling":
        forbidden = ["equity_long", "buy_call", "bull_call_spread"]
        allowed = [s for s in allowed if s not in forbidden]
    elif day_type == "short_covering_rally":
        forbidden = ["equity_long", "buy_call"]
        allowed = [s for s in allowed if s not in forbidden]
    elif day_type == "vol_expansion":
        allowed.append("long_straddle")
        allowed.append("long_strangle")
    elif day_type == "vol_contraction":
        forbidden = ["long_straddle", "long_strangle"]
        allowed = [s for s in allowed if s not in forbidden]

    if macro_regime in ("bear", "volatile"):
        for s in ["equity_long", "buy_call", "bull_call_spread"]:
            if s not in forbidden:
                forbidden.append(s)
        allowed = [s for s in allowed if s not in forbidden]

    return {
        "macro_regime": macro_regime,
        "day_type": day_type,
        "regime_confidence": day["confidence"],
        "day_scores": day["all_scores"],
        "strategy_rules": adapt,
        "allowed_strategies": list(set(allowed)),
        "forbidden_strategies": list(set(forbidden)),
        "sizing_multiplier": adapt["sizing_multiplier"],
        "stop_style": adapt["stops"],
        "target_style": adapt["targets"],
        "playbook": adapt["playbook"],
    }


def adapt_signal(signal, regime=None):
    """Modify a trade signal based on the current regime.

    Args:
        signal: dict with at least final_action, fused_confidence.
        regime: output of classify_regime_v2(), or None to compute.

    Returns:
        Modified signal dict (original is NOT mutated).
    """
    sig = dict(signal)
    if regime is None:
        regime = classify_regime_v2()

    day_type = regime.get("day_type", "range_day")
    adapt = regime.get("strategy_rules", STRATEGY_ADAPTATION["range_day"])
    action = sig.get("final_action", "hold")
    confidence = sig.get("fused_confidence", 0.5)

    sig["regime_v2"] = {
        "day_type": day_type,
        "macro_regime": regime.get("macro_regime", "unknown"),
        "sizing_multiplier": adapt["sizing_multiplier"],
        "stop_style": adapt["stops"],
        "target_style": adapt["targets"],
    }

    if day_type == "panic_selling" and action == "buy":
        sig["final_action"] = "no_trade"
        sig.setdefault("warnings", []).append(
            "BLOCKED: Panic selling detected — no new longs")
        sig["fused_confidence"] = 0.0

    elif day_type == "short_covering_rally" and action == "buy":
        sig.setdefault("warnings", []).append(
            "Short covering rally — likely unsustainable, extreme caution")
        sig["fused_confidence"] = confidence * 0.4

    elif day_type == "range_day":
        sig["fused_confidence"] = confidence * 0.8
        sig.setdefault("notes", []).append("Range day — tightened targets")

    elif day_type == "trend_day" and action in ("buy", "sell"):
        sig["fused_confidence"] = min(confidence * 1.1, 1.0)
        sig.setdefault("notes", []).append("Trend day — widened stops, extended targets")

    elif day_type == "vol_expansion":
        sig["fused_confidence"] = confidence * 0.6
        sig.setdefault("warnings", []).append(
            "Volatility expanding — reduced confidence and size")

    elif day_type == "risk_off" and action == "buy":
        sig["fused_confidence"] = confidence * 0.7
        sig.setdefault("warnings", []).append(
            "Risk-off environment — favour defensives only")

    sig["fused_confidence"] = round(sig.get("fused_confidence", 0.5), 4)
    return sig


# ── CLI ────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  MARKET REGIME INTELLIGENCE V2")
    print("=" * 60)

    day = detect_day_type()
    print(f"\n  Day Type   : {day['day_type']}")
    print(f"  Confidence : {day['confidence']:.1%}")
    print(f"  Breadth    : {day.get('breadth_regime', 'N/A')}")
    print(f"\n  Scores:")
    for dt, sc in sorted(day["all_scores"].items(), key=lambda x: -x[1]):
        bar = "#" * int(sc / 5)
        print(f"    {dt:<25} {sc:5.1f}  {bar}")
    print(f"\n  Playbook: {day['playbook']}")

    regime = classify_regime_v2()
    print(f"\n  Macro Regime : {regime['macro_regime']}")
    print(f"  Allowed      : {', '.join(regime['allowed_strategies'])}")
    print(f"  Forbidden    : {', '.join(regime['forbidden_strategies']) or 'none'}")
    print(f"  Sizing Mult  : {regime['sizing_multiplier']}")

    test_signal = {"symbol": "RELIANCE", "final_action": "buy",
                   "fused_confidence": 0.75}
    adapted = adapt_signal(test_signal, regime)
    print(f"\n  Signal Test:")
    print(f"    Before: {test_signal['final_action']} @ {test_signal['fused_confidence']}")
    print(f"    After : {adapted['final_action']} @ {adapted['fused_confidence']}")

    out_path = os.path.join(OUTPUT_DIR, "regime_v2.json")
    json.dump(regime, open(out_path, "w"), indent=2, default=str)
    print(f"\n  Saved → {out_path}")

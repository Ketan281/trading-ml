"""
Trade management intelligence — dynamic stops, partial profit booking,
trailing logic, time-based and volatility-based exits, exit confidence,
and trade quality scoring.

Wraps and extends the existing stops.py engine with full lifecycle
management. No ML — all rule-based.
"""

import os
import sys
from datetime import datetime, time as dtime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipelines.stops import dynamic_stops, _atr, _trend

# ── Config ──────────────────────────────────────────
ATR_PERIOD = 14
CAPITAL = 1_000_000
RISK_PCT = 0.01

PARTIAL_TARGETS = [
    {"label": "T1", "r_multiple": 1.0, "exit_pct": 0.33},
    {"label": "T2", "r_multiple": 2.0, "exit_pct": 0.33},
    {"label": "T3", "r_multiple": 3.0, "exit_pct": 0.34},
]

TIME_EXIT_WARN = dtime(14, 30)
TIME_EXIT_HARD = dtime(15, 10)

MAX_HOLD_BARS_INTRADAY = 60
MAX_HOLD_DAYS_SWING = 20

VOL_EXPAND_THRESHOLD = 1.5
VOL_CONTRACT_THRESHOLD = 0.6


# ── Dynamic Stop Methods ────────────────────────────

def compute_dynamic_stop(df, entry, direction="long"):
    base = dynamic_stops(df, entry)
    if not base:
        return None
    if direction == "short":
        stop = 2 * entry - base["stop"]
        risk_pct = (stop - entry) / entry * 100
        base["stop"] = round(stop, 2)
        base["risk_pct"] = round(abs(risk_pct), 2)
        base["direction"] = "short"
    else:
        base["direction"] = "long"
    return base


def compute_breakeven_stop(entry, current_price, initial_stop, direction="long"):
    if direction == "long":
        r_distance = entry - initial_stop
        if r_distance <= 0:
            return initial_stop
        pnl = current_price - entry
        if pnl >= r_distance:
            return round(entry + r_distance * 0.1, 2)
        return initial_stop
    else:
        r_distance = initial_stop - entry
        if r_distance <= 0:
            return initial_stop
        pnl = entry - current_price
        if pnl >= r_distance:
            return round(entry - r_distance * 0.1, 2)
        return initial_stop


def compute_trailing_stop(df, entry, current_price, direction="long",
                          method="chandelier", lookback=22, multiplier=3.0):
    if df is None or len(df) < lookback + ATR_PERIOD:
        return None
    atr = float(_atr(df, ATR_PERIOD).iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        return None

    if method == "chandelier":
        if direction == "long":
            hh = float(df["High"].iloc[-lookback:].max())
            trail = hh - multiplier * atr
        else:
            ll = float(df["Low"].iloc[-lookback:].min())
            trail = ll + multiplier * atr
    elif method == "atr":
        if direction == "long":
            trail = current_price - multiplier * atr
        else:
            trail = current_price + multiplier * atr
    elif method == "ema":
        e21 = df["Close"].ewm(span=21, adjust=False).mean().iloc[-1]
        if direction == "long":
            trail = float(e21) - 0.5 * atr
        else:
            trail = float(e21) + 0.5 * atr
    else:
        trail = current_price - multiplier * atr if direction == "long" else current_price + multiplier * atr

    return round(trail, 2)


# ── Partial Profit Booking ──────────────────────────

def compute_profit_targets(entry, stop, direction="long"):
    r_distance = abs(entry - stop)
    targets = []
    for t in PARTIAL_TARGETS:
        if direction == "long":
            price = entry + t["r_multiple"] * r_distance
        else:
            price = entry - t["r_multiple"] * r_distance
        targets.append({
            "label": t["label"],
            "r_multiple": t["r_multiple"],
            "price": round(price, 2),
            "exit_pct": t["exit_pct"],
        })
    return targets


def evaluate_partial_exits(entry, stop, current_price, direction="long"):
    targets = compute_profit_targets(entry, stop, direction)
    executed = []
    pending = []
    for t in targets:
        if direction == "long":
            hit = current_price >= t["price"]
        else:
            hit = current_price <= t["price"]
        t["hit"] = hit
        if hit:
            executed.append(t)
        else:
            pending.append(t)
    return {"executed": executed, "pending": pending, "targets": targets}


# ── Time-Based Exit Logic ───────────────────────────

def time_exit_check(entry_time, current_time=None, trade_type="intraday"):
    now = current_time or datetime.now().time()
    if trade_type == "intraday":
        if now >= TIME_EXIT_HARD:
            return {"action": "exit_now", "reason": "Hard time cutoff reached",
                    "urgency": 1.0}
        if now >= TIME_EXIT_WARN:
            minutes_left = (TIME_EXIT_HARD.hour * 60 + TIME_EXIT_HARD.minute) - \
                           (now.hour * 60 + now.minute)
            return {"action": "prepare_exit", "reason": f"{minutes_left}m to hard cutoff",
                    "urgency": round(1 - minutes_left / 40, 2)}
        return {"action": "hold", "reason": "Within trading hours", "urgency": 0.0}
    return {"action": "hold", "reason": "Swing trade — no time exit", "urgency": 0.0}


# ── Volatility-Based Exit Logic ─────────────────────

def volatility_exit_check(df, entry, current_price, direction="long"):
    if df is None or len(df) < 30:
        return {"action": "hold", "reason": "Insufficient data"}
    atr = float(_atr(df, ATR_PERIOD).iloc[-1])
    atr_prev = float(_atr(df, ATR_PERIOD).iloc[-6])
    if atr_prev <= 0:
        return {"action": "hold", "reason": "Cannot compute vol ratio"}

    vol_ratio = atr / atr_prev
    pnl = (current_price - entry) if direction == "long" else (entry - current_price)
    pnl_r = pnl / atr if atr > 0 else 0

    if vol_ratio >= VOL_EXPAND_THRESHOLD and pnl_r > 1.0:
        return {
            "action": "tighten_stop",
            "reason": f"Vol expanding {vol_ratio:.2f}x with profit — protect gains",
            "vol_ratio": round(vol_ratio, 2),
        }
    if vol_ratio >= VOL_EXPAND_THRESHOLD and pnl_r < -0.5:
        return {
            "action": "exit",
            "reason": f"Vol expanding {vol_ratio:.2f}x against position — cut loss",
            "vol_ratio": round(vol_ratio, 2),
        }
    if vol_ratio <= VOL_CONTRACT_THRESHOLD:
        return {
            "action": "hold_tight",
            "reason": f"Vol contracting {vol_ratio:.2f}x — expect expansion, stay positioned",
            "vol_ratio": round(vol_ratio, 2),
        }
    return {"action": "hold", "reason": "Normal volatility", "vol_ratio": round(vol_ratio, 2)}


# ── Exit Confidence Score ───────────────────────────

def exit_confidence_score(partial_info, time_info, vol_info, trailing_stop,
                          current_price, entry, direction="long"):
    score = 0.0
    pnl = (current_price - entry) if direction == "long" else (entry - current_price)
    in_profit = pnl > 0

    n_targets_hit = len(partial_info.get("executed", []))
    score += n_targets_hit * 15

    if time_info.get("action") == "exit_now":
        score += 30
    elif time_info.get("action") == "prepare_exit":
        score += 15

    if vol_info.get("action") == "exit":
        score += 25
    elif vol_info.get("action") == "tighten_stop":
        score += 10

    if trailing_stop and direction == "long" and current_price < trailing_stop:
        score += 30
    elif trailing_stop and direction == "short" and current_price > trailing_stop:
        score += 30

    if not in_profit and n_targets_hit == 0:
        score = max(score, 10)

    return round(min(100, score), 1)


# ── Trade Quality Score ─────────────────────────────

def trade_quality_score(entry, stop, current_price, direction="long",
                        rsi=50, volume_ratio=1.0, alignment_score=50):
    r_distance = abs(entry - stop)
    if r_distance == 0:
        return 0.0
    pnl = (current_price - entry) if direction == "long" else (entry - current_price)
    r_multiple = pnl / r_distance

    score = 0.0
    if r_multiple >= 3:   score += 35
    elif r_multiple >= 2: score += 28
    elif r_multiple >= 1: score += 20
    elif r_multiple >= 0: score += 10
    else:                 score += max(0, 10 + r_multiple * 10)

    if 40 <= rsi <= 60:
        score += 5
    elif (direction == "long" and 50 < rsi < 70) or \
         (direction == "short" and 30 < rsi < 50):
        score += 15
    else:
        score += 8

    if volume_ratio >= 1.5: score += 15
    elif volume_ratio >= 1.0: score += 10
    else: score += 5

    score += alignment_score * 0.35

    return round(min(100, max(0, score)), 1)


# ── Full Trade Management Read ──────────────────────

def trade_management_read(df, entry, stop, current_price,
                          direction="long", trade_type="intraday",
                          rsi=50, volume_ratio=1.0, alignment_score=50):
    result = {"entry": entry, "stop": stop, "current_price": current_price,
              "direction": direction, "trade_type": trade_type}

    dynamic = compute_dynamic_stop(df, entry, direction)
    result["dynamic_stop"] = dynamic

    be_stop = compute_breakeven_stop(entry, current_price, stop, direction)
    result["breakeven_stop"] = be_stop

    trail = compute_trailing_stop(df, entry, current_price, direction)
    result["trailing_stop"] = trail

    partials = evaluate_partial_exits(entry, stop, current_price, direction)
    result["partial_exits"] = partials

    time_check = time_exit_check(None, None, trade_type)
    result["time_exit"] = time_check

    vol_check = volatility_exit_check(df, entry, current_price, direction)
    result["volatility_exit"] = vol_check

    active_stop = stop
    candidates = [s for s in [stop, be_stop, trail] if s is not None]
    if direction == "long":
        active_stop = max(candidates) if candidates else stop
    else:
        active_stop = min(candidates) if candidates else stop
    result["active_stop"] = round(active_stop, 2)

    pnl = (current_price - entry) if direction == "long" else (entry - current_price)
    pnl_pct = pnl / entry * 100
    result["unrealized_pnl"] = round(pnl, 2)
    result["unrealized_pnl_pct"] = round(pnl_pct, 2)
    result["r_multiple"] = round(pnl / abs(entry - stop), 2) if abs(entry - stop) > 0 else 0

    exit_conf = exit_confidence_score(partials, time_check, vol_check,
                                      trail, current_price, entry, direction)
    result["exit_confidence"] = exit_conf

    quality = trade_quality_score(entry, stop, current_price, direction,
                                  rsi, volume_ratio, alignment_score)
    result["trade_quality"] = quality

    if exit_conf >= 70:
        result["recommendation"] = "EXIT"
    elif exit_conf >= 40:
        result["recommendation"] = "TIGHTEN_STOP"
    elif time_check["action"] == "prepare_exit":
        result["recommendation"] = "PREPARE_EXIT"
    else:
        result["recommendation"] = "HOLD"

    return result


if __name__ == "__main__":
    from models.cross_sectional import load_prices
    symbol = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    prices = load_prices(universe={symbol})
    df = prices.get(symbol)
    if df is None:
        print(f"No data for {symbol}")
        sys.exit(1)

    entry = float(df["Close"].iloc[-5])
    stop_info = dynamic_stops(df, entry)
    stop = stop_info["stop"] if stop_info else entry * 0.95
    current = float(df["Close"].iloc[-1])

    print("=" * 64)
    print("  TRADE MANAGEMENT ENGINE")
    print("=" * 64)
    r = trade_management_read(df, entry, stop, current)
    print(f"\n  {symbol}")
    print(f"  Entry: {r['entry']}  Stop: {r['stop']}  Current: {r['current_price']}")
    print(f"  PnL: {r['unrealized_pnl']} ({r['unrealized_pnl_pct']:.2f}%)  R: {r['r_multiple']}")
    print(f"  Active Stop: {r['active_stop']}")
    print(f"  Trailing Stop: {r['trailing_stop']}")
    print(f"  Breakeven Stop: {r['breakeven_stop']}")
    print(f"\n  Partial Targets:")
    for t in r["partial_exits"]["targets"]:
        status = "HIT" if t["hit"] else "pending"
        print(f"    {t['label']}: {t['price']} ({t['r_multiple']}R, {t['exit_pct']*100:.0f}%) [{status}]")
    print(f"\n  Time Exit: {r['time_exit']['action']} — {r['time_exit']['reason']}")
    print(f"  Vol Exit:  {r['volatility_exit']['action']} — {r['volatility_exit']['reason']}")
    print(f"\n  Exit Confidence: {r['exit_confidence']}/100")
    print(f"  Trade Quality:   {r['trade_quality']}/100")
    print(f"  Recommendation:  {r['recommendation']}")

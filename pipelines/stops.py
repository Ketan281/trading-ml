"""
Dynamic stop-loss selection engine.

A fixed % stop ignores how each stock actually moves. This computes four
professional stop methods and picks the one that fits the stock's current
behaviour:

  • ATR stop        — entry − k×ATR : volatility-scaled, all-purpose
  • Swing-low stop  — below the recent swing low : respects market structure
  • Chandelier stop — highest-high − k×ATR : a TRAILING stop for trends
  • Structure stop  — below the recent consolidation support

Selection logic: strong trend → chandelier (let it run, trail); normal up →
ATR; choppy/range → swing-low or structure (tighter, respects S/R). Returns
every method plus the recommended one with its risk %, so position sizing can
turn the stop distance into a share count.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ATR_PERIOD     = 14
ATR_MULT       = 2.5
CHANDELIER_LB  = 22
CHANDELIER_MULT = 3.0
SWING_LB       = 10
STRUCT_LB      = 20
BUFFER         = 0.004      # 0.4% beyond the level to avoid wick stop-outs
MAX_STOP_PCT   = 0.12       # never risk more than 12% on one name


def _atr(df, period=ATR_PERIOD):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _trend(df):
    c = df["Close"]
    e20, e50 = c.ewm(span=20).mean(), c.ewm(span=50).mean()
    price = c.iloc[-1]
    if price > e20.iloc[-1] > e50.iloc[-1] and e20.iloc[-1] > e20.iloc[-6]:
        return "strong_up"
    if price > e50.iloc[-1]:
        return "up"
    return "weak_or_range"


def dynamic_stops(df, entry=None):
    """All four stops + a recommended pick for a LONG. Returns None if data
    is too short."""
    if df is None or len(df) < max(CHANDELIER_LB, STRUCT_LB, 50):
        return None
    c = df["Close"]
    entry = float(entry if entry is not None else c.iloc[-1])
    atr = float(_atr(df).iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        return None

    methods = {}
    methods["atr"] = entry - ATR_MULT * atr
    methods["swing_low"] = float(df["Low"].iloc[-SWING_LB:].min()) * (1 - BUFFER)
    methods["chandelier"] = float(df["High"].iloc[-CHANDELIER_LB:].max()) - CHANDELIER_MULT * atr
    methods["structure"] = float(df["Low"].iloc[-STRUCT_LB:].min()) * (1 - BUFFER)

    # Keep only valid stops (below entry, within max risk).
    valid = {}
    for k, stop in methods.items():
        if stop <= 0 or stop >= entry:
            continue
        risk = (entry - stop) / entry
        if risk <= MAX_STOP_PCT:
            valid[k] = stop

    trend = _trend(df)
    # Preference order by regime of the stock.
    if trend == "strong_up":
        order = ["chandelier", "atr", "swing_low", "structure"]
    elif trend == "up":
        order = ["atr", "swing_low", "chandelier", "structure"]
    else:
        order = ["swing_low", "structure", "atr", "chandelier"]

    chosen = next((m for m in order if m in valid), None)
    if chosen is None:
        # everything too wide → cap at MAX_STOP_PCT ATR-style
        chosen = "capped"
        valid["capped"] = entry * (1 - MAX_STOP_PCT)

    stop = valid[chosen]
    risk_pct = (entry - stop) / entry
    return {
        "entry": round(entry, 2), "trend": trend,
        "recommended_method": chosen,
        "stop": round(stop, 2), "risk_pct": round(risk_pct * 100, 2),
        "atr": round(atr, 2), "atr_pct": round(atr / entry * 100, 2),
        "all_methods": {k: round(v, 2) for k, v in {**methods}.items()},
        "rationale": _why(chosen, trend),
    }


def _why(method, trend):
    return {
        "chandelier": f"{trend} trend → trailing chandelier lets the winner run",
        "atr": f"{trend} trend → volatility-scaled ATR stop",
        "swing_low": "range/weak trend → stop below the recent swing low (structure)",
        "structure": "range/weak trend → stop below recent consolidation support",
        "capped": "all structural stops too wide → capped at max risk",
    }.get(method, "")


if __name__ == "__main__":
    from models.cross_sectional import load_prices
    syms = sys.argv[1:] or ["RELIANCE", "TCS", "BHEL"]
    prices = load_prices(universe=set(syms))
    print("=" * 64)
    print("  DYNAMIC STOP-LOSS ENGINE")
    print("=" * 64)
    for s in syms:
        df = prices.get(s)
        if df is None:
            print(f"  {s}: no data"); continue
        r = dynamic_stops(df)
        if not r:
            print(f"  {s}: insufficient data"); continue
        print(f"\n  {s}  entry {r['entry']} ({r['trend']}, ATR {r['atr_pct']}%)")
        print(f"     → {r['recommended_method'].upper()} stop {r['stop']} "
              f"(risk {r['risk_pct']}%)")
        print(f"     {r['rationale']}")
        print(f"     all: {r['all_methods']}")

"""
Intraday regime classifier — NIFTY / BANKNIFTY (5m + 15m).

The daily 4-class regime governs the swing book; index-options intraday needs
its OWN, faster read because the playbook flips with the session. This
classifies the live intraday tape into:

  • TRENDING_UP / TRENDING_DOWN — 5m & 15m aligned, price riding VWAP/EMA
  • RANGE_BOUND                  — coiling around VWAP, low realised range
  • VOLATILE_CHOPPY              — big two-way range, no follow-through

…and maps each to an options stance (buy premium / sell premium / directional).
Rule-based for now (honest: the ML version waits on the intraday history the
collector is recording — 60 days is too little to train).
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipelines.intraday import fetch_intraday, _vwap, _atr, _ema, _opening_range, trend_15m


def classify(symbol):
    df5 = fetch_intraday(symbol, "5m", period="5d")
    df15 = fetch_intraday(symbol, "15m", period="10d")
    if df5 is None or len(df5) < 30:
        return {"symbol": symbol, "regime": "unknown", "note": "no intraday data"}

    df5 = df5.copy()
    df5["vwap"] = _vwap(df5); df5["atr"] = _atr(df5)
    df5["ema9"] = _ema(df5["Close"], 9); df5["ema20"] = _ema(df5["Close"], 20)

    # today's session slice
    last_day = df5.index[-1].date()
    day = df5[df5.index.map(lambda x: x.date() == last_day)]
    if len(day) < 6:
        day = df5.tail(40)

    px = float(day["Close"].iloc[-1]); vwap = float(day["vwap"].iloc[-1])
    # Yahoo indices have no intraday volume → VWAP is NaN; fall back to the
    # session typical-price mean as the anchor.
    if not np.isfinite(vwap):
        vwap = float(((day["High"] + day["Low"] + day["Close"]) / 3).mean())
    t15 = trend_15m(df15)
    e9, e20 = float(day["ema9"].iloc[-1]), float(day["ema20"].iloc[-1])

    # realised intraday vol (5m returns, annualised-ish) and range vs ATR
    r = day["Close"].pct_change()
    rv = float(r.std()) if len(r) > 3 else 0.0
    day_range = (day["High"].max() - day["Low"].min())
    atr = float(day["atr"].iloc[-1]) if np.isfinite(day["atr"].iloc[-1]) else day_range / 10
    range_in_atr = day_range / atr if atr else 0
    above_vwap = px > vwap
    rv_pct = float((r.abs().tail(8).mean()) / (r.abs().mean() + 1e-9))  # recent vs day

    # ── classify ─────────────────────────────────────────
    trending_up = above_vwap and e9 > e20 and t15 == "up"
    trending_dn = (not above_vwap) and e9 < e20 and t15 == "down"
    if trending_up:
        regime, stance = "trending_up", "directional longs / buy CE on dips to VWAP"
    elif trending_dn:
        regime, stance = "trending_down", "directional shorts / buy PE on pops to VWAP"
    elif range_in_atr > 9 or rv_pct > 1.6:
        regime, stance = "volatile_choppy", "buy premium (straddle) / avoid selling naked"
    else:
        regime, stance = "range_bound", "sell premium (iron condor) around VWAP"

    return {
        "symbol": symbol, "regime": regime, "stance": stance,
        "as_of": df5.index[-1].strftime("%Y-%m-%d %H:%M"),
        "price": round(px, 1), "vwap": round(vwap, 1),
        "above_vwap": bool(above_vwap), "trend_15m": t15,
        "range_in_atr": round(range_in_atr, 1),
        "ema9_vs_20": "bull" if e9 > e20 else "bear",
    }


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["NIFTY", "BANKNIFTY"]):
        r = classify(s)
        print("=" * 60)
        print(f"  INTRADAY REGIME — {s}  ({r.get('as_of','?')})")
        print("=" * 60)
        print(f"  Regime    : {r['regime'].upper()}")
        if "stance" in r:
            print(f"  Stance    : {r['stance']}")
            print(f"  Price/VWAP: {r['price']} / {r['vwap']} "
                  f"({'above' if r['above_vwap'] else 'below'}) | 15m {r['trend_15m']} | "
                  f"EMA9v20 {r['ema9_vs_20']} | range {r['range_in_atr']}×ATR")
        print()

"""
Market-breadth model — is the WHOLE market participating, or just a few names?

A rising index on narrow breadth (a handful of heavyweights) is fragile; a
rising index on broad breadth is durable. This reads the internals of the full
~470-stock universe:

  • Advance/Decline    — how many stocks rose vs fell today (+ AD ratio)
  • DMA participation  — % of stocks above their 50-DMA and 200-DMA
  • New highs / lows    — 52-week new highs vs new lows (leadership vs distress)
  • Breadth trend       — is participation expanding or contracting?

Combined into a 0-100 breadth score + a plain-English signal. Computed from
price history alone (leak-free) and returns a time series, so it doubles as a
feature for the regime classifier and the ensemble.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.cross_sectional import load_prices


# ── Build an aligned close matrix (date × symbol) ─────
def _close_matrix(prices):
    closes = {s: df["Close"] for s, df in prices.items() if len(df) > 220}
    mat = pd.DataFrame(closes).sort_index()
    return mat.dropna(how="all")


def breadth_series(prices=None, days=250):
    """Time series of the core breadth internals (last `days` sessions)."""
    if prices is None:
        prices = load_prices()
    mat = _close_matrix(prices)
    ma50, ma200 = mat.rolling(50).mean(), mat.rolling(200).mean()
    hi52, lo52 = mat.rolling(252).max(), mat.rolling(252).min()
    up = mat.diff() > 0

    out = pd.DataFrame(index=mat.index)
    out["pct_above_50dma"]  = (mat > ma50).sum(axis=1) / mat.notna().sum(axis=1) * 100
    out["pct_above_200dma"] = (mat > ma200).sum(axis=1) / mat.notna().sum(axis=1) * 100
    out["advances"] = up.sum(axis=1)
    out["declines"] = (~up & mat.notna()).sum(axis=1)
    out["ad_ratio"] = out["advances"] / out["declines"].replace(0, np.nan)
    out["new_highs"] = (mat >= hi52).sum(axis=1)
    out["new_lows"]  = (mat <= lo52).sum(axis=1)
    out["nh_nl_diff"] = out["new_highs"] - out["new_lows"]
    return out.tail(days)


# ── Composite read for the latest session ─────────────
def breadth_read(prices=None):
    s = breadth_series(prices, days=60)
    if s.empty:
        return {"score": None, "signal": "no data"}
    last = s.iloc[-1]
    a50, a200 = last["pct_above_50dma"], last["pct_above_200dma"]
    a50_5d = s["pct_above_50dma"].iloc[-5] if len(s) >= 5 else a50
    trend = a50 - a50_5d                                   # expanding vs contracting

    # 0-100 score: participation (both DMAs), leadership (NH-NL), trend.
    score = (0.35 * a50 + 0.35 * a200 +
             0.20 * np.clip(50 + last["nh_nl_diff"], 0, 100) +
             0.10 * np.clip(50 + trend * 5, 0, 100))
    score = float(round(np.clip(score, 0, 100), 1))

    if score >= 65 and trend >= 0:
        signal = "broad & strengthening — healthy participation"
    elif score >= 55:
        signal = "constructive — majority participating"
    elif score >= 40:
        signal = "mixed / narrowing — selective market"
    elif score >= 25:
        signal = "weak — few stocks holding up (fragile)"
    else:
        signal = "very weak — broad distribution"

    return {
        "score": score, "signal": signal,
        "pct_above_50dma": round(a50, 1), "pct_above_200dma": round(a200, 1),
        "advances": int(last["advances"]), "declines": int(last["declines"]),
        "ad_ratio": round(float(last["ad_ratio"]), 2) if last["ad_ratio"] == last["ad_ratio"] else None,
        "new_highs": int(last["new_highs"]), "new_lows": int(last["new_lows"]),
        "participation_trend_5d": round(float(trend), 1),
        "as_of": str(s.index[-1].date()),
    }


if __name__ == "__main__":
    r = breadth_read()
    print("=" * 60)
    print(f"  MARKET BREADTH  ({r['as_of']})")
    print("=" * 60)
    print(f"  Breadth score   : {r['score']}/100  →  {r['signal']}")
    print(f"  % above 50-DMA  : {r['pct_above_50dma']}%   "
          f"(5d trend {r['participation_trend_5d']:+})")
    print(f"  % above 200-DMA : {r['pct_above_200dma']}%")
    print(f"  Advances/Declines: {r['advances']}/{r['declines']}  "
          f"(AD ratio {r['ad_ratio']})")
    print(f"  New highs/lows  : {r['new_highs']} / {r['new_lows']}")

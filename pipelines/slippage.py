"""
Slippage estimator — what the fill REALLY costs, before you trade.

Backtests assume you trade at the close; reality charges you the spread plus
market impact, and impact grows with how much of the day's volume you demand.
This estimates the round-trip drag so position sizing and expected-return maths
stay honest.

Model (square-root market impact — the industry-standard Almgren form):

    slippage_bps ≈ half_spread_bps + k · daily_vol_bps · √(order_value / ADV)

  • half_spread  : tighter for liquid names (scaled by turnover)
  • impact term  : grows with participation rate (order vs average daily value)
    and the stock's volatility — exactly how real impact behaves

HONEST SCOPE: this is a PARAMETRIC estimator, not a model trained on your fills
(we have no execution data). It is calibrated to sensible NSE defaults and is
deliberately conservative; treat it as a planning number, refine k once you
have real fills.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.cross_sectional import load_prices

IMPACT_K   = 0.4        # impact coefficient (conservative; tune with real fills)
ADV_DAYS   = 30
MIN_HALF_SPREAD_BPS = 1.0
MAX_HALF_SPREAD_BPS = 60.0


def _adv_and_vol(df):
    r = df.tail(ADV_DAYS)
    adv = float((r["Close"] * r["Volume"]).median())                 # ₹/day
    vol = float(df["Close"].pct_change().tail(60).std())             # daily
    return adv, vol


def _half_spread_bps(adv_cr):
    """Wider spread for thinner names. ~1bp for very liquid, capped at 60bp."""
    if adv_cr <= 0:
        return MAX_HALF_SPREAD_BPS
    bps = 8.0 / np.sqrt(adv_cr)            # 100cr ADV → ~0.8bp, 1cr → 8bp
    return float(np.clip(bps, MIN_HALF_SPREAD_BPS, MAX_HALF_SPREAD_BPS))


def estimate(symbol, order_value, df=None):
    """Estimate one-way slippage in bps + ₹ for an order of `order_value` ₹."""
    if df is None:
        df = load_prices(universe={symbol}).get(symbol)
    if df is None or len(df) < ADV_DAYS:
        return None
    adv, vol = _adv_and_vol(df)
    adv_cr = adv / 1e7
    half_spread = _half_spread_bps(adv_cr)
    participation = order_value / adv if adv else 1.0
    impact_bps = IMPACT_K * (vol * 1e4) * np.sqrt(min(participation, 1.0))
    one_way_bps = half_spread + impact_bps
    return {
        "symbol": symbol, "order_value": round(order_value),
        "adv_cr": round(adv_cr, 1), "daily_vol_pct": round(vol * 100, 2),
        "participation_pct": round(participation * 100, 2),
        "half_spread_bps": round(half_spread, 1),
        "impact_bps": round(float(impact_bps), 1),
        "one_way_bps": round(float(one_way_bps), 1),
        "round_trip_bps": round(float(one_way_bps) * 2, 1),
        "one_way_cost": round(order_value * one_way_bps / 1e4),
        "round_trip_cost": round(order_value * one_way_bps / 1e4 * 2),
        "liquidity": ("excellent" if one_way_bps < 6 else "ok" if one_way_bps < 20
                      else "expensive — size down / use limits"),
    }


def slippage_for_book(book_path=None):
    book_path = book_path or os.path.join(ROOT, "outputs", "portfolio_book.json")
    if not os.path.exists(book_path):
        print("  ⚠ No portfolio_book.json — run pipelines/portfolio_book.py first.")
        return None
    import json
    book = json.load(open(book_path))
    syms = [h["symbol"] for h in book["holdings"]]
    prices = load_prices(universe=set(syms))
    print("=" * 72)
    print("  SLIPPAGE ESTIMATE — portfolio book (round-trip)")
    print("=" * 72)
    print(f"  {'SYMBOL':<12}{'ORDER ₹':>11}{'ADVcr':>7}{'PART%':>7}"
          f"{'SPRD':>6}{'IMPACT':>7}{'RT bps':>7}{'RT ₹':>9}  LIQUIDITY")
    total = 0
    for h in book["holdings"]:
        e = estimate(h["symbol"], h["capital"], prices.get(h["symbol"]))
        if not e:
            continue
        total += e["round_trip_cost"]
        print(f"  {e['symbol']:<12}{e['order_value']:>11,}{e['adv_cr']:>7}"
              f"{e['participation_pct']:>7}{e['half_spread_bps']:>6}"
              f"{e['impact_bps']:>7}{e['round_trip_bps']:>7}"
              f"{e['round_trip_cost']:>9,}  {e['liquidity']}")
    deployed = sum(h["capital"] for h in book["holdings"])
    print("  " + "-" * 70)
    print(f"  Total round-trip slippage ₹{total:,}  "
          f"({total/deployed*1e4:.0f} bps of ₹{deployed:,} deployed)")
    return total


if __name__ == "__main__":
    if len(sys.argv) > 2:
        e = estimate(sys.argv[1], float(sys.argv[2]))
        print(e)
    else:
        slippage_for_book()

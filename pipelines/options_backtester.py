"""
Option strategy backtester (BS-synthetic) — would systematic structures work?

We have NO historical option prices, so we cannot backtest real fills. What we
CAN do, honestly, is SYNTHETIC: price each leg with Black-Scholes at entry using
trailing realised vol (plus a variance-risk-premium markup, because implied vol
historically trades above subsequent realised — that premium is exactly what
sellers harvest), hold to weekly expiry, and settle at intrinsic value.

This answers "did systematically selling weekly premium / buying it pay off on
NIFTY/BANKNIFTY over 10 years?" — directionally, net of costs.

KEY ASSUMPTIONS (results are sensitive to these — stated, not hidden):
  • entry IV = trailing RV20 × IV_RV_RATIO  (the variance risk premium)
  • settle at expiry intrinsic (European, hold to expiry, no early management)
  • flat per-leg cost; no real bid-ask/skew/slippage
This is a SANITY CHECK on structure edge, NOT proof of a live-tradeable system.
Real validation arrives with the option-chain data the collector is recording.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.index_options_model import load_index

HORIZON      = 5            # weekly expiry (trading days)
IV_RV_RATIO  = 1.15         # implied ≈ 15% above realised (variance risk premium)
RISK_FREE    = 0.065
# Cost is a fraction of the PREMIUM transacted (brokerage+STT+slippage), NOT of
# spot — charging spot-based cost wrongly crushes multi-leg structures.
COST_PREMIUM_FRAC = 0.03
TRADING_DAYS = 252


def _bs(S, K, T, sigma, kind):
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if kind == "C" else (K - S))
    d1 = (np.log(S / K) + (RISK_FREE + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "C":
        return S * norm.cdf(d1) - K * np.exp(-RISK_FREE * T) * norm.cdf(d2)
    return K * np.exp(-RISK_FREE * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _legs(kind, spot, sigma, T):
    """Return legs [(opt_kind, strike, side)] for a structure. side +1 long / -1 short.
    OTM strikes placed at ~1σ expected move."""
    em = spot * sigma * np.sqrt(T)             # 1σ move to expiry
    atm = round(spot / 50) * 50
    o1 = round((spot + em) / 50) * 50; o1p = round((spot - em) / 50) * 50
    o2 = round((spot + 2 * em) / 50) * 50; o2p = round((spot - 2 * em) / 50) * 50
    return {
        "short_straddle":  [("C", atm, -1), ("P", atm, -1)],
        "short_strangle":  [("C", o1, -1), ("P", o1p, -1)],
        "iron_condor":     [("C", o1, -1), ("C", o2, +1),
                            ("P", o1p, -1), ("P", o2p, +1)],
        "long_straddle":   [("C", atm, +1), ("P", atm, +1)],
        "bull_put_credit": [("P", o1p, -1), ("P", o2p, +1)],
    }[kind]


def backtest(symbol="NIFTY", kind="iron_condor", iv_ratio=IV_RV_RATIO):
    df = load_index(symbol)
    if df is None or len(df) < 300:
        return None
    c = df["Close"]
    rv20 = c.pct_change().rolling(20).std() * np.sqrt(TRADING_DAYS)
    dates = df.index
    rb = range(60, len(df) - HORIZON, HORIZON)     # weekly, non-overlapping

    pnls = []
    for i in rb:
        spot = float(c.iloc[i]); sig_rv = float(rv20.iloc[i])
        if not np.isfinite(sig_rv) or sig_rv <= 0:
            continue
        iv = sig_rv * iv_ratio
        T = HORIZON / TRADING_DAYS
        spot_exp = float(c.iloc[i + HORIZON])
        legs = _legs(kind, spot, iv, T)
        pnl = 0.0
        for ok, K, side in legs:
            entry = _bs(spot, K, T, iv, ok)
            intrinsic = max(0.0, (spot_exp - K) if ok == "C" else (K - spot_exp))
            pnl += side * (intrinsic - entry)          # long: gain intrinsic−cost
            pnl -= COST_PREMIUM_FRAC * entry            # cost ∝ premium transacted
        pnls.append(pnl / spot)                         # normalise to % of spot
    if not pnls:
        return None

    p = np.array(pnls)
    wins = p[p > 0]
    ann = TRADING_DAYS / HORIZON
    sharpe = (p.mean() * ann) / (p.std() * np.sqrt(ann)) if p.std() > 0 else float("nan")
    return {
        "symbol": symbol, "structure": kind, "iv_ratio": iv_ratio,
        "cycles": len(p), "years": round(len(p) / ann, 1),
        "avg_pnl_pct": round(float(p.mean()) * 100, 3),
        "total_pnl_pct": round(float(p.sum()) * 100, 1),
        "win_rate_pct": round(len(wins) / len(p) * 100, 1),
        "avg_win_pct": round(float(wins.mean()) * 100, 3) if len(wins) else 0,
        "avg_loss_pct": round(float(p[p <= 0].mean()) * 100, 3) if (p <= 0).any() else 0,
        "worst_pct": round(float(p.min()) * 100, 2),
        "sharpe": round(float(sharpe), 2),
    }


def run(symbol="NIFTY"):
    print("=" * 76)
    print(f"  OPTION STRATEGY BACKTESTER (BS-SYNTHETIC) — {symbol}")
    print(f"  ⚠ Synthetic pricing (BS + RV×{IV_RV_RATIO} IV proxy); sanity check, "
          f"not live proof")
    print("=" * 76)
    print(f"  {'STRUCTURE':<18}{'CYCLES':>7}{'WIN%':>7}{'AVG%':>8}"
          f"{'AVGWIN':>8}{'AVGLOSS':>9}{'WORST':>8}{'SHARPE':>8}")
    rows = []
    for k in ("short_straddle", "short_strangle", "iron_condor",
              "bull_put_credit", "long_straddle"):
        r = backtest(symbol, k)
        if not r:
            continue
        rows.append(r)
        print(f"  {k:<18}{r['cycles']:>7}{r['win_rate_pct']:>7}"
              f"{r['avg_pnl_pct']:>8}{r['avg_win_pct']:>8}{r['avg_loss_pct']:>9}"
              f"{r['worst_pct']:>8}{r['sharpe']:>8}")
    print(f"\n  ({rows[0]['years'] if rows else 0} yrs weekly cycles. Premium-selling")
    print("   win-rate is high but watch AVGLOSS/WORST — the tail is the risk.)")
    return rows


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["NIFTY"]):
        run(s)

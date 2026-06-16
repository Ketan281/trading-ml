"""
Monte Carlo risk engine — what could this book actually do over the next month?

A single backtest number hides the distribution of outcomes. This simulates the
constructed book forward thousands of times by BOOTSTRAPPING real historical
daily joint returns (which preserves cross-correlation AND fat tails — far more
honest than a Gaussian), then reads the risk off the distribution:

  • VaR 95 / 99      — the loss you breach only 5% / 1% of the time
  • CVaR (ES)        — the AVERAGE loss in those tail cases (what it costs when
                       it goes wrong)
  • Expected / worst max drawdown across the simulated paths
  • P(loss), P(>X% gain), terminal-return distribution

Bootstrapping joint return vectors keeps the book's diversification real: if two
holdings crashed together historically, they crash together in the sim.
"""

import os
import sys
import json

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.cross_sectional import load_prices

HORIZON   = 21          # ~1 month forward
N_SIMS    = 10_000
LOOKBACK  = 252         # history to bootstrap from
CAPITAL   = 1_000_000


def _returns(symbols, days=LOOKBACK):
    prices = load_prices(universe=set(symbols))
    cols = {s: prices[s]["Close"].pct_change().tail(days)
            for s in symbols if s in prices}
    return pd.DataFrame(cols).dropna(how="any")


def simulate(weights, horizon=HORIZON, n_sims=N_SIMS, seed=42):
    """weights: dict {symbol: weight_fraction} (sum ≤ 1; remainder = cash).
    Bootstraps historical joint daily returns. Returns the risk distribution."""
    rng = np.random.default_rng(seed)
    syms = list(weights)
    rets = _returns(syms)
    if rets.empty or len(rets) < 60:
        return None
    syms = [s for s in syms if s in rets.columns]
    w = np.array([weights[s] for s in syms])
    R = rets[syms].to_numpy()                     # (days × names)
    n_days = len(R)

    terminal, maxdds = np.empty(n_sims), np.empty(n_sims)
    for i in range(n_sims):
        idx = rng.integers(0, n_days, horizon)    # iid bootstrap of day-vectors
        path_daily = R[idx] @ w                   # portfolio daily returns
        equity = np.cumprod(1 + path_daily)
        terminal[i] = equity[-1] - 1
        peak = np.maximum.accumulate(equity)
        maxdds[i] = (equity / peak - 1).min()

    pct = lambda a, p: float(np.percentile(a, p))
    var95, var99 = -pct(terminal, 5), -pct(terminal, 1)
    cvar95 = -float(terminal[terminal <= pct(terminal, 5)].mean())
    return {
        "names": len(syms), "horizon_days": horizon, "sims": n_sims,
        "gross_exposure_pct": round(float(w.sum()) * 100, 1),
        "exp_return_pct": round(float(terminal.mean()) * 100, 2),
        "median_return_pct": round(pct(terminal, 50) * 100, 2),
        "vol_pct": round(float(terminal.std()) * 100, 2),
        "var95_pct": round(var95 * 100, 2), "var99_pct": round(var99 * 100, 2),
        "cvar95_pct": round(cvar95 * 100, 2),
        "exp_maxdd_pct": round(float(maxdds.mean()) * 100, 2),
        "worst_maxdd_pct": round(float(maxdds.min()) * 100, 2),
        "p_loss_pct": round(float((terminal < 0).mean()) * 100, 1),
        "p_gain_5pct": round(float((terminal > 0.05).mean()) * 100, 1),
        "best_case_pct": round(pct(terminal, 99) * 100, 2),
        "worst_case_pct": round(pct(terminal, 1) * 100, 2),
    }


def from_book(book_path=None, capital=CAPITAL):
    book_path = book_path or os.path.join(ROOT, "outputs", "portfolio_book.json")
    if not os.path.exists(book_path):
        print("  ⚠ No portfolio_book.json — run pipelines/portfolio_book.py first.")
        return None
    book = json.load(open(book_path))
    weights = {h["symbol"]: h["weight_pct"] / 100 for h in book["holdings"]}
    res = simulate(weights)
    if res is None:
        print("  ⚠ Not enough return history to simulate."); return None

    print("=" * 66)
    print(f"  MONTE CARLO RISK — {res['names']} names, {res['horizon_days']}d, "
          f"{res['sims']:,} sims (bootstrap)")
    print("=" * 66)
    print(f"  Gross exposure   : {res['gross_exposure_pct']}%")
    print(f"  Expected return  : {res['exp_return_pct']:+}%  "
          f"(median {res['median_return_pct']:+}%, vol {res['vol_pct']}%)")
    print(f"  VaR 95 / 99      : -{res['var95_pct']}% / -{res['var99_pct']}%  "
          f"(₹{round(res['var95_pct']/100*capital):,} / "
          f"₹{round(res['var99_pct']/100*capital):,})")
    print(f"  CVaR 95 (ES)     : -{res['cvar95_pct']}%  "
          f"(avg loss in the worst 5% of months)")
    print(f"  Expected max DD  : {res['exp_maxdd_pct']}%  "
          f"(worst path {res['worst_maxdd_pct']}%)")
    print(f"  P(loss)          : {res['p_loss_pct']}%  | "
          f"P(>5% gain): {res['p_gain_5pct']}%")
    print(f"  1st–99th pctile  : {res['worst_case_pct']}%  …  {res['best_case_pct']}%")
    return res


if __name__ == "__main__":
    from_book()

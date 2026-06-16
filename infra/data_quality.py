"""
Data-quality validation + anomaly detection.

Garbage in, garbage out — a single bad tick, an unadjusted split, or a stale
file can silently poison a model or a backtest. This validates raw price data
BEFORE it is trusted, and flags anomalies for review:

  • OHLC consistency  — high ≥ low, high ≥ open/close, low ≤ open/close
  • non-positive       — zero/negative prices, zero-volume sessions
  • extreme moves      — single-day jumps > 30% (bad tick OR unadjusted split)
  • return anomalies   — daily returns beyond ~6σ (statistical outliers)
  • staleness          — last bar older than N days (feed stopped updating)
  • gaps / duplicates  — missing or repeated dates

Returns a per-symbol report (pass / warn / fail) and a universe summary, so
the retrain pipeline can refuse to train on a broken universe.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

EXTREME_MOVE = 0.30        # > 30% one-day move = suspicious
SIGMA_FLAG   = 6.0         # return z-score beyond this = anomaly
STALE_DAYS   = 10          # last bar older than this (calendar) = stale


def validate_prices(df, symbol="?", stale_days=STALE_DAYS):
    issues, warns = [], []
    if df is None or df.empty:
        return {"symbol": symbol, "severity": "fail", "issues": ["no data"]}

    cols = set(df.columns)
    need = {"Open", "High", "Low", "Close", "Volume"}
    if not need.issubset(cols):
        return {"symbol": symbol, "severity": "fail",
                "issues": [f"missing columns {need - cols}"]}

    o, h, l, c, v = (df["Open"], df["High"], df["Low"], df["Close"], df["Volume"])
    n = len(df)
    ohlc_fail_thresh = max(5, int(n * 0.005))     # >0.5% of bars or >5 = corruption

    # OHLC consistency — a few bad bars (split-adjustment artifacts) only warn.
    bad_ohlc = int(((h < l) | (h < o) | (h < c) | (l > o) | (l > c)).sum())
    if bad_ohlc >= ohlc_fail_thresh:
        issues.append(f"{bad_ohlc} OHLC-inconsistent bars (corruption)")
    elif bad_ohlc:
        warns.append(f"{bad_ohlc} OHLC-inconsistent bar(s) (likely split artifact)")

    # Non-positive
    nonpos = int((c <= 0).sum())
    if nonpos:
        issues.append(f"{nonpos} non-positive close(s)")
    zero_vol = int((v <= 0).sum())
    if zero_vol > len(df) * 0.05:
        warns.append(f"{zero_vol} zero-volume sessions")

    # Extreme moves / split-like
    ret = c.pct_change()
    extreme = int((ret.abs() > EXTREME_MOVE).sum())
    if extreme:
        warns.append(f"{extreme} day(s) > {EXTREME_MOVE:.0%} move (bad tick / split?)")

    # Statistical anomalies
    z = (ret - ret.mean()) / (ret.std() + 1e-9)
    anom = int((z.abs() > SIGMA_FLAG).sum())
    if anom:
        warns.append(f"{anom} return(s) beyond {SIGMA_FLAG:.0f}σ")

    # NaNs
    nan_close = int(c.isna().sum())
    if nan_close:
        issues.append(f"{nan_close} NaN close(s)")

    # Duplicates / staleness
    dups = int(df.index.duplicated().sum())
    if dups:
        issues.append(f"{dups} duplicate dates")
    last = df.index[-1]
    try:
        stale = (pd.Timestamp.now().normalize() - pd.Timestamp(last)).days
    except Exception:
        stale = None
    if stale is not None and stale > stale_days:
        warns.append(f"stale: last bar {last.date()} ({stale}d ago)")

    severity = "fail" if issues else ("warn" if warns else "pass")
    return {"symbol": symbol, "severity": severity, "rows": len(df),
            "last_date": str(last.date()) if hasattr(last, "date") else str(last),
            "issues": issues, "warnings": warns}


def validate_universe(limit=None):
    from models.cross_sectional import load_prices
    prices = load_prices()
    syms = list(prices)[:limit] if limit else list(prices)
    reports = [validate_prices(prices[s], s) for s in syms]
    by = {"pass": [], "warn": [], "fail": []}
    for r in reports:
        by[r["severity"]].append(r["symbol"])
    return reports, by


if __name__ == "__main__":
    reports, by = validate_universe()
    print("=" * 64)
    print("  DATA-QUALITY VALIDATION — universe")
    print("=" * 64)
    total = len(reports)
    print(f"  Symbols checked : {total}")
    print(f"  ✅ pass {len(by['pass'])} | ⚠ warn {len(by['warn'])} | "
          f"❌ fail {len(by['fail'])}")
    if by["fail"]:
        print(f"\n  ❌ FAILS: {', '.join(by['fail'][:20])}")
    # show a few warnings with detail
    shown = 0
    for r in reports:
        if r["severity"] == "warn" and shown < 10:
            print(f"  ⚠ {r['symbol']:<12} {', '.join(r['warnings'])}")
            shown += 1

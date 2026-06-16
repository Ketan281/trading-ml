"""
Daily intraday-bar collector — builds the history deep learning will need.

THE PROBLEM IT SOLVES
---------------------
yfinance only serves a rolling ~60 days of 5m bars (and ~7 days of 1m). That
window is far too short to train/validate a deep intraday model — which is
exactly why intraday is rule-based today. This collector runs every day and
APPENDS the latest bars to a permanent per-symbol store, de-duplicating by
timestamp. Run it daily and the archive grows past the 60-day wall: after a
few months you have enough genuine intraday history to make a DL intraday
model a legitimate, testable option (not a 60-day overfit).

STORAGE
-------
data/intraday/<interval>/<SYMBOL>.csv   (Datetime index, IST, OHLCV)
Idempotent: re-running on the same day only adds bars not already stored, so
it is safe to schedule, retry, or run twice.

USAGE
-----
    python training/collect_intraday.py              # liquid universe, 5m+15m
    python training/collect_intraday.py --all        # full price universe
    python training/collect_intraday.py SBIN RELIANCE # explicit symbols
"""

import os
import sys
import time
import glob

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipelines.intraday import fetch_intraday, _liquid_universe

OUT_BASE = os.path.join(ROOT, "data", "intraday")

# interval -> yfinance fetch window (a few days of overlap guarantees no gaps
# even if the collector misses a day or two).
INTERVALS = {"5m": "5d", "15m": "1mo"}

MAX_LIQUID   = 150        # default universe size (liquid names = tight spreads)
SLEEP_SEC    = 0.3        # be polite to the data provider
SAVE_COLS    = ["Open", "High", "Low", "Close", "Volume"]


def _store_path(interval, symbol):
    d = os.path.join(OUT_BASE, interval)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{symbol}.csv")


def _append(interval, symbol, new_df):
    """Merge new bars into the symbol's store, de-duped by timestamp.
    Returns (added_rows, total_rows, last_ts)."""
    if new_df is None or new_df.empty:
        return 0, _existing_count(interval, symbol), None
    new_df = new_df[[c for c in SAVE_COLS if c in new_df.columns]].copy()
    new_df.index.name = "Datetime"

    path = _store_path(interval, symbol)
    if os.path.exists(path):
        old = pd.read_csv(path, index_col="Datetime", parse_dates=True)
        try:
            old.index = old.index.tz_convert("Asia/Kolkata")
        except (TypeError, AttributeError):
            try:
                old.index = old.index.tz_localize("Asia/Kolkata")
            except Exception:
                pass
        before = len(old)
        merged = pd.concat([old, new_df])
    else:
        before = 0
        merged = new_df

    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.to_csv(path)
    added = len(merged) - before
    last_ts = merged.index[-1]
    return added, len(merged), last_ts


def _existing_count(interval, symbol):
    path = _store_path(interval, symbol)
    if not os.path.exists(path):
        return 0
    try:
        return sum(1 for _ in open(path)) - 1
    except Exception:
        return 0


def _resolve_universe(args):
    if args == ["--all"]:
        from models.cross_sectional import load_prices
        return sorted(load_prices().keys())
    if args:
        return args
    return _liquid_universe(max_symbols=MAX_LIQUID)


def collect(symbols=None):
    symbols = symbols or _liquid_universe(max_symbols=MAX_LIQUID)
    print("=" * 66)
    print("  DAILY INTRADAY-BAR COLLECTOR")
    print(f"  {pd.Timestamp.now(tz='Asia/Kolkata').strftime('%Y-%m-%d %H:%M IST')}")
    print(f"  Symbols: {len(symbols)} | intervals: {', '.join(INTERVALS)}")
    print("=" * 66)

    totals = {iv: {"added": 0, "rows": 0, "ok": 0} for iv in INTERVALS}
    for i, sym in enumerate(symbols, 1):
        line = [f"  [{i:>3}/{len(symbols)}] {sym:<12}"]
        for iv, period in INTERVALS.items():
            df = fetch_intraday(sym, iv, period=period)
            added, rows, last_ts = _append(iv, sym, df)
            if rows:
                totals[iv]["ok"] += 1
            totals[iv]["added"] += added
            totals[iv]["rows"]  += rows
            tag = f"{iv}:+{added}({rows})" if rows else f"{iv}:--"
            line.append(tag)
            time.sleep(SLEEP_SEC)
        if i % 20 == 0 or i == len(symbols):
            print("  ".join(line))

    print("\n  " + "─" * 56)
    for iv in INTERVALS:
        t = totals[iv]
        print(f"  {iv:<4} | symbols stored {t['ok']:>3} | "
              f"new bars +{t['added']:<6} | total bars {t['rows']:,}")
    print(f"\n  ✅ Archive → {OUT_BASE}\\<interval>\\<SYMBOL>.csv")
    print("  Run daily (after ~15:35 IST) to grow history past the 60-day wall.")
    return totals


if __name__ == "__main__":
    syms = _resolve_universe(sys.argv[1:])
    collect(syms)

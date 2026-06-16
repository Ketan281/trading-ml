"""
Fetch the full Nifty 500 universe (~500 stocks) of daily OHLCV history.

Why: the cross-sectional ranking model gets stronger with more names to
rank against each other. This expands the universe from ~50 to ~500 and
writes each symbol to data/historical/{SYMBOL}.csv — the same place
models/cross_sectional.py reads from.

It also caches the constituent list and a symbol→industry map (the latter
is the seed for future sector-relative / fundamental features).
"""

import os
import io
import csv
import sys
import json
import time
import urllib.request

import pandas as pd
import yfinance as yf

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST_DIR = os.path.join(ROOT, "data", "historical")
os.makedirs(HIST_DIR, exist_ok=True)

LIST_URL   = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
LIST_CACHE = os.path.join(HIST_DIR, "nifty500_list.csv")
IND_CACHE  = os.path.join(HIST_DIR, "industries.json")

START_DATE = "2014-01-01"
BATCH_SIZE = 50          # tickers per yfinance call
MIN_ROWS   = 260         # ~1 year minimum to be useful


# ── Get the Nifty 500 constituent list ────────────────
def get_constituents():
    """Return list of (symbol, industry). Tries NSE live, falls back to
    a cached copy so the script still works offline."""
    try:
        req = urllib.request.Request(
            LIST_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                     "Accept": "text/csv,*/*"},
        )
        data = urllib.request.urlopen(req, timeout=30).read().decode(
            "utf-8", "ignore")
        with open(LIST_CACHE, "w", encoding="utf-8") as f:
            f.write(data)
        print(f"  ✅ Fetched live Nifty 500 list")
    except Exception as e:
        print(f"  ⚠ Live list failed ({e}); trying cache...")
        if not os.path.exists(LIST_CACHE):
            print("  ❌ No cached list available. Aborting.")
            return []
        with open(LIST_CACHE, encoding="utf-8") as f:
            data = f.read()

    rows = list(csv.DictReader(io.StringIO(data)))
    out  = [(r["Symbol"].strip(), r.get("Industry", "").strip())
            for r in rows if r.get("Symbol")]

    # Persist the industry map for later sector features.
    with open(IND_CACHE, "w", encoding="utf-8") as f:
        json.dump({s: ind for s, ind in out}, f, indent=2)

    return out


# ── Save one ticker's frame ───────────────────────────
def _save_frame(symbol, df):
    if df is None or df.empty:
        return False
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df = df.dropna()
    if len(df) < MIN_ROWS:
        return False
    df.to_csv(os.path.join(HIST_DIR, f"{symbol}.csv"))
    return True


# ── Fetch everything in batches ───────────────────────
def fetch_all(limit=None):
    constituents = get_constituents()
    if not constituents:
        return

    symbols = [s for s, _ in constituents]
    if limit:
        symbols = symbols[:limit]

    print(f"  Universe size: {len(symbols)} symbols")
    print(f"  Period: {START_DATE} → today\n")

    ok, failed = [], []

    for i in range(0, len(symbols), BATCH_SIZE):
        batch   = symbols[i:i + BATCH_SIZE]
        tickers = [f"{s}.NS" for s in batch]
        print(f"  [{i + 1:>3}-{i + len(batch):>3} / {len(symbols)}] "
              f"downloading...")

        try:
            data = yf.download(
                tickers, start=START_DATE, interval="1d",
                auto_adjust=True, group_by="ticker",
                threads=True, progress=False,
            )
        except Exception as e:
            print(f"     ⚠ batch error: {e}")
            failed.extend(batch)
            continue

        for sym in batch:
            tk = f"{sym}.NS"
            try:
                # Single vs multi-ticker frames have different shapes.
                if isinstance(data.columns, pd.MultiIndex):
                    if tk not in data.columns.get_level_values(0):
                        failed.append(sym); continue
                    sub = data[tk]
                else:
                    sub = data
                if _save_frame(sym, sub):
                    ok.append(sym)
                else:
                    failed.append(sym)
            except Exception:
                failed.append(sym)

        time.sleep(1.0)   # be polite to the data provider

    print(f"\n  {'=' * 50}")
    print(f"  ✅ Saved   : {len(ok)}/{len(symbols)}")
    print(f"  ❌ Failed  : {len(failed)}")
    if failed:
        print(f"  Failed sample: {failed[:15]}")
    print(f"  Data dir   : {HIST_DIR}")
    return ok, failed


if __name__ == "__main__":
    # Optional: python training/fetch_universe.py 100  (fetch first 100)
    lim = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    fetch_all(limit=lim)

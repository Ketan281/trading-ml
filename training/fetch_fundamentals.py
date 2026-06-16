"""
Fetch fundamental data (~23 params) for the whole universe.

Why: pure price-momentum ranking surfaces junky, low-quality names. Layering
a fundamental QUALITY score on top lets the screener (a) drop balance-sheet
junk and (b) tilt the final shortlist toward profitable, reasonably-valued,
financially-healthy companies — the way a real long-term trader filters.

IMPORTANT — point-in-time caveat
--------------------------------
yfinance's .info gives only the CURRENT snapshot of each company's
fundamentals (today's PE, today's ROE, ...). It does NOT give clean
historical point-in-time financials. Therefore these features are used ONLY
at screening time (models/fundamentals.py) as a quality filter / tilt — they
are deliberately NOT fed into the historical cross-sectional backtest, because
doing so would leak today's knowledge into past dates and fake the IC.

Output: data/historical/fundamentals.json  → {symbol: {field: value, ...}}
The fetch is RESUMABLE: already-fetched symbols are skipped on re-run.
"""

import os
import sys
import json
import time
import glob

import yfinance as yf

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST_DIR = os.path.join(ROOT, "data", "historical")
OUT_PATH = os.path.join(HIST_DIR, "fundamentals.json")

# Indices are not companies — never fetch fundamentals for them.
EXCLUDE = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"}

# The ~23 fields we keep. Names match yfinance .info keys exactly.
FIELDS = [
    # Valuation
    "trailingPE", "forwardPE", "priceToBook", "pegRatio",
    "enterpriseToEbitda", "priceToSalesTrailing12Months",
    # Profitability
    "returnOnEquity", "returnOnAssets", "profitMargins",
    "operatingMargins", "grossMargins",
    # Growth
    "earningsGrowth", "revenueGrowth", "earningsQuarterlyGrowth",
    # Financial health
    "debtToEquity", "currentRatio", "quickRatio",
    # Shareholder / quality / size
    "dividendYield", "payoutRatio", "freeCashflow",
    "marketCap", "beta", "totalRevenue",
]


def _universe_symbols():
    """Every symbol we have price history for (so the score aligns 1:1 with
    the ranker's universe)."""
    syms = []
    for path in glob.glob(os.path.join(HIST_DIR, "*.csv")):
        name = os.path.basename(path).replace(".csv", "")
        if name in EXCLUDE or name.lower() in ("manifest",):
            continue
        syms.append(name)
    return sorted(syms)


def _load_existing():
    if os.path.exists(OUT_PATH):
        try:
            with open(OUT_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _extract(info):
    """Pull just our FIELDS out of a yfinance .info dict, keeping only
    numeric values (drop None / non-numeric)."""
    row = {}
    for k in FIELDS:
        v = info.get(k)
        if isinstance(v, (int, float)) and v == v:   # numeric & not NaN
            row[k] = float(v)
    return row


def fetch_all(limit=None, save_every=20):
    symbols = _universe_symbols()
    if limit:
        symbols = symbols[:limit]

    data = _load_existing()
    todo = [s for s in symbols if s not in data]

    print("=" * 60)
    print("  Fundamental fetch — yfinance .info snapshot")
    print("=" * 60)
    print(f"  Universe       : {len(symbols)}")
    print(f"  Already cached : {len(symbols) - len(todo)}")
    print(f"  To fetch       : {len(todo)}\n")

    ok = fail = 0
    for i, sym in enumerate(todo, 1):
        try:
            info = yf.Ticker(f"{sym}.NS").info
            row  = _extract(info)
            if row:
                data[sym] = row
                ok += 1
            else:
                fail += 1
        except Exception:
            fail += 1

        if i % save_every == 0 or i == len(todo):
            with open(OUT_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"  [{i:>3}/{len(todo)}] {sym:<14} "
                  f"ok={ok} fail={fail}  (saved)")
        time.sleep(0.3)   # be polite to the data provider

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\n  {'=' * 50}")
    print(f"  ✅ Fundamentals saved: {len(data)} symbols")
    print(f"  📄 {OUT_PATH}")
    return data


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    fetch_all(limit=lim)

"""
Point-in-time (PIT) fundamentals fetch — roadmap #3.

WHY this exists (and how it differs from fetch_fundamentals.py)
--------------------------------------------------------------
fetch_fundamentals.py pulls yfinance's `.info` — a CURRENT snapshot. Using a
snapshot on historical dates leaks the future, so those features are barred
from the backtest and used only as a live screening tilt.

This fetch instead pulls the QUARTERLY STATEMENTS (income statement, balance
sheet, cash-flow). Every column is a real reporting PERIOD-END date. From
those we can build features that are legitimately point-in-time: a value is
only "known" to the market AFTER the results are filed. We model that with a
reporting lag (see models/fundamentals_pit.py), so no future leaks in.

HONEST LIMITATION
-----------------
yfinance returns only ~4-5 recent quarters per symbol. That is plenty to
(a) compute correct lagged YoY growth / margins for live ranking and
(b) START a genuine PIT archive that deepens every time this is run — but it
is NOT deep enough to re-validate the full 10-year IC backtest. That depth
needs a paid PIT fundamentals feed. We build the pipeline correctly now so it
is ready the moment richer history is available.

Output: data/historical/fundamentals_pit.json
    { symbol: { "YYYY-MM-DD" (period_end): { line_item: value, ... }, ... } }
Resumable: already-fetched symbols are skipped on re-run.
"""

import os
import sys
import json
import time
import glob

import yfinance as yf

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST_DIR = os.path.join(ROOT, "data", "historical")
OUT_PATH = os.path.join(HIST_DIR, "fundamentals_pit.json")

EXCLUDE = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"}

# Line items we keep, mapped to the yfinance statement row label(s) that
# carry them. We try each candidate label in order (yfinance renames rows
# across versions) and take the first that exists.
INCOME_ITEMS = {
    "revenue":          ["Total Revenue", "Operating Revenue"],
    "operating_income": ["Operating Income", "Total Operating Income As Reported"],
    "net_income":       ["Net Income", "Net Income Common Stockholders",
                         "Net Income From Continuing Operation Net Minority Interest"],
    "gross_profit":     ["Gross Profit"],
}
BALANCE_ITEMS = {
    "equity":            ["Stockholders Equity", "Total Stockholder Equity",
                          "Common Stock Equity"],
    "total_debt":        ["Total Debt"],
    "current_assets":    ["Current Assets", "Total Current Assets"],
    "current_liab":      ["Current Liabilities", "Total Current Liabilities"],
    "total_assets":      ["Total Assets"],
}
CASHFLOW_ITEMS = {
    "free_cash_flow":    ["Free Cash Flow"],
    "operating_cf":      ["Operating Cash Flow", "Total Cash From Operating Activities"],
}


def _universe_symbols():
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


def _pick(df, candidates):
    """Return the row Series for the first candidate label present in df."""
    if df is None or df.empty:
        return None
    for label in candidates:
        if label in df.index:
            return df.loc[label]
    return None


def _harvest(df, item_map):
    """Turn a statement DataFrame (rows=line items, cols=period-ends) into
    {period_end_str: {item: value}} for the line items we care about."""
    out = {}
    if df is None or df.empty:
        return out
    for item, candidates in item_map.items():
        row = _pick(df, candidates)
        if row is None:
            continue
        for col, val in row.items():
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            if v != v:                      # NaN
                continue
            key = col.date().isoformat() if hasattr(col, "date") else str(col)
            out.setdefault(key, {})[item] = v
    return out


def _fetch_symbol(sym):
    """All three quarterly statements, merged by period-end date."""
    t = yf.Ticker(f"{sym}.NS")
    merged = {}
    for getter, items in (
        (lambda: t.quarterly_financials,     INCOME_ITEMS),
        (lambda: t.quarterly_balance_sheet,  BALANCE_ITEMS),
        (lambda: t.quarterly_cashflow,       CASHFLOW_ITEMS),
    ):
        try:
            harvested = _harvest(getter(), items)
        except Exception:
            harvested = {}
        for period, vals in harvested.items():
            merged.setdefault(period, {}).update(vals)
    # Keep only periods that have at least revenue or net income (a real row).
    return {p: v for p, v in merged.items()
            if "revenue" in v or "net_income" in v}


def fetch_all(limit=None, save_every=15):
    symbols = _universe_symbols()
    if limit:
        symbols = symbols[:limit]

    data = _load_existing()
    todo = [s for s in symbols if s not in data]

    print("=" * 62)
    print("  Point-in-time fundamentals fetch — quarterly statements")
    print("=" * 62)
    print(f"  Universe       : {len(symbols)}")
    print(f"  Already cached : {len(symbols) - len(todo)}")
    print(f"  To fetch       : {len(todo)}\n")

    ok = fail = 0
    for i, sym in enumerate(todo, 1):
        try:
            stmts = _fetch_symbol(sym)
            if stmts:
                data[sym] = stmts
                ok += 1
            else:
                data[sym] = {}          # cache the miss so we don't retry forever
                fail += 1
        except Exception:
            data[sym] = {}
            fail += 1

        if i % save_every == 0 or i == len(todo):
            with open(OUT_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            nq = len(data.get(sym, {}))
            print(f"  [{i:>3}/{len(todo)}] {sym:<14} "
                  f"ok={ok} fail={fail} (last {nq}q)  (saved)")
        time.sleep(0.4)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\n  {'=' * 50}")
    print(f"  ✅ PIT fundamentals saved: {len(data)} symbols")
    print(f"  📄 {OUT_PATH}")
    return data


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    fetch_all(limit=lim)

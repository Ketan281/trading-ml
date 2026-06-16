"""
Portfolio optimizer — turn a ranked candidate list into a DIVERSIFIED book.

The ranker tells you the best names; it does NOT stop you buying ten correlated
financials at once. This builds the actual book under real constraints:

  • SECTOR caps        — no more than X% of the book in one sector
  • CORRELATION caps   — skip a name too correlated with what's already in
  • POSITION cap       — max weight per name
  • GROSS exposure     — scaled by regime + drawdown (via risk_policy)
  • WEIGHTING          — inverse-volatility (calmer names get more)

It is regime-aware: in a bear/volatile market it holds fewer names at lower
gross; in a bull market it runs fuller. Output = the book with weights, sector
exposure, and average pairwise correlation (the diversification it achieved).
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.cross_sectional import load_prices, rank_today
from models.sector_strength import attach_sector_rs
from models.regime_classifier import classify
from pipelines.breadth import breadth_read
from pipelines.risk_policy import effective_limits

CORR_CAP   = 0.75      # don't add a name correlated > this with a holding
CORR_DAYS  = 120
CAPITAL    = 1_000_000


def _returns_matrix(symbols, prices, days=CORR_DAYS):
    cols = {}
    for s in symbols:
        df = prices.get(s)
        if df is not None and len(df) > days + 5:
            cols[s] = df["Close"].pct_change().tail(days)
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).dropna(how="all")


def optimize(candidates=None, capital=CAPITAL, regime=None, current_dd=0.0):
    """candidates: df with ['symbol','score'] (+ optional 'sector'). If None,
    pulls the live ranked list. Returns the constructed book + diagnostics."""
    if candidates is None:
        candidates = rank_today(top_n=50)
        if candidates is None:
            return None
    cand = candidates.copy()
    if "sector" not in cand.columns:
        cand = attach_sector_rs(cand)
    cand = cand.dropna(subset=["symbol"]).reset_index(drop=True)

    # Market context → limits.
    reg = regime or classify("NIFTY")["regime"]
    breadth = breadth_read()
    lim = effective_limits(reg, current_dd, breadth.get("score"))
    max_pos = lim["max_positions"]; max_sector = lim["max_sector_pct"]
    gross = lim["gross_effective"]

    prices = load_prices(universe=set(cand["symbol"]))
    rets = _returns_matrix(list(cand["symbol"]), prices)
    corr = rets.corr() if not rets.empty else pd.DataFrame()
    vol = rets.std() * np.sqrt(252) if not rets.empty else pd.Series(dtype=float)

    # ── Greedy diversified selection (rank order) ────────
    selected, sector_count = [], {}
    skipped = {"sector_cap": [], "correlation": []}
    cand = cand.sort_values("score", ascending=False) if "score" in cand else cand
    for r in cand.itertuples():
        if len(selected) >= max_pos:
            break
        sec = getattr(r, "sector", None)
        # Sector cap (by count proxy against max positions).
        sec_w = (sector_count.get(sec, 0) + 1) / max_pos
        if sec and sec_w > max_sector + 1e-9:
            skipped["sector_cap"].append(r.symbol); continue
        # Correlation cap vs already-selected.
        if selected and not corr.empty and r.symbol in corr.columns:
            held = [s for s in selected if s in corr.columns]
            if held:
                mx = corr.loc[r.symbol, held].abs().max()
                if mx > CORR_CAP:
                    skipped["correlation"].append(r.symbol); continue
        selected.append(r.symbol)
        if sec:
            sector_count[sec] = sector_count.get(sec, 0) + 1

    if not selected:
        return {"regime": reg, "gross": gross, "book": [], "note": "no names passed constraints"}

    # ── Inverse-vol weights, capped, scaled to gross ─────
    inv = pd.Series({s: 1 / (vol.get(s, np.nan)) for s in selected})
    inv = inv.replace([np.inf, -np.inf], np.nan).fillna(inv.mean())
    w = inv / inv.sum()
    cap_w = min(0.20, 1.0 / max(1, len(selected)) * 2)     # per-name cap
    w = w.clip(upper=cap_w); w = w / w.sum() * gross         # scale to gross

    secmap = cand.set_index("symbol").get("sector", pd.Series(dtype=object))
    book = []
    for s in selected:
        book.append({"symbol": s, "weight_pct": round(float(w[s]) * 100, 2),
                     "capital": round(float(w[s]) * capital),
                     "sector": secmap.get(s) if hasattr(secmap, "get") else None,
                     "ann_vol_pct": round(float(vol.get(s, np.nan)) * 100, 1)
                                    if s in vol else None})

    # Diagnostics
    sectors = {}
    for b in book:
        sectors[b["sector"]] = round(sectors.get(b["sector"], 0) + b["weight_pct"], 2)
    held = [b["symbol"] for b in book if b["symbol"] in corr.columns]
    avg_corr = (round(float(corr.loc[held, held].values[np.triu_indices(len(held), 1)].mean()), 3)
                if len(held) > 1 else None)

    return {
        "regime": reg, "breadth_score": breadth.get("score"),
        "gross_target_pct": round(gross * 100, 1),
        "gross_book_pct": round(sum(b["weight_pct"] for b in book), 1),
        "positions": len(book), "max_positions": max_pos,
        "max_sector_pct": max_sector, "book": book,
        "sector_exposure": dict(sorted(sectors.items(), key=lambda x: -x[1])),
        "avg_pairwise_corr": avg_corr,
        "skipped": {k: v[:8] for k, v in skipped.items()},
    }


def _print(res):
    if not res or not res.get("book"):
        print("  No book constructed:", res.get("note") if res else "none"); return
    print("=" * 70)
    print(f"  PORTFOLIO OPTIMIZER  | regime {res['regime'].upper()} | "
          f"breadth {res['breadth_score']} | gross target {res['gross_target_pct']}%")
    print("=" * 70)
    print(f"  {'SYMBOL':<13}{'WT%':>7}{'CAPITAL':>12}{'VOL%':>7}  SECTOR")
    for b in res["book"]:
        print(f"  {b['symbol']:<13}{b['weight_pct']:>7}{b['capital']:>12,}"
              f"{(b['ann_vol_pct'] or 0):>7}  {b['sector']}")
    print("  " + "-" * 64)
    print(f"  Positions {res['positions']}/{res['max_positions']} | "
          f"book gross {res['gross_book_pct']}% | "
          f"avg pairwise corr {res['avg_pairwise_corr']}")
    print(f"  Sector exposure: {res['sector_exposure']}")
    if res["skipped"]["correlation"]:
        print(f"  Skipped (correlation): {res['skipped']['correlation']}")
    if res["skipped"]["sector_cap"]:
        print(f"  Skipped (sector cap) : {res['skipped']['sector_cap']}")


if __name__ == "__main__":
    _print(optimize())

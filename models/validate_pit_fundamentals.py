"""
Do point-in-time fundamentals ADD predictive edge? — roadmap #3 (validation).

This is the honest test. We:
  1. Build the same price-factor panel the ranker uses.
  2. As-of join the leak-free PIT fundamental features (available_date <= date).
  3. Cross-sectionally z-score the fundamentals per date (same treatment as
     price features), so they are comparable across stocks on each day.
  4. Restrict to the window where fundamentals actually exist, and compare
     walk-forward Rank IC of:
         price-only            vs
         price + fundamentals
     on IDENTICAL train/test splits.

If fundamentals add IC over price alone on this window → they earn a place in
the ranker. If not (or if the window is too short to tell) we say so plainly
rather than shipping an unproven feature.

NOTE on depth: yfinance gives only ~4-5 quarters, and YoY growth needs a year
of history, so the usable window is short (recent months). Treat the verdict
here as PROVISIONAL until a deeper PIT feed backfills history.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.cross_sectional import (
    load_prices, build_panel, _make_model, _rank_ic,
    FEATURES_Z, MIN_NAMES, HORIZON,
)
from models.fundamentals_pit import (
    pit_feature_panel, asof_join, PIT_FEATURES,
)

PIT_Z = [c + "_z" for c in PIT_FEATURES]


def _zscore_per_date(panel, cols):
    """Cross-sectional z-score of each col within each date."""
    for c in cols:
        panel[c + "_z"] = panel.groupby("date")[c].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9))
    return panel


def run(n_splits=4):
    print("=" * 64)
    print("  PIT FUNDAMENTALS — does the layer add walk-forward IC?")
    print("=" * 64)

    pit = pit_feature_panel()
    if pit.empty:
        print("  ⚠ No PIT fundamentals cached. Run:")
        print("    python training/fetch_fundamentals_pit.py")
        return None
    print(f"  PIT feature rows : {len(pit)} across "
          f"{pit['symbol'].nunique()} symbols")
    print(f"  PIT date span    : {pit['available_date'].min().date()} → "
          f"{pit['available_date'].max().date()}")

    prices = load_prices()
    panel  = build_panel(prices)

    # As-of join fundamentals (leak-free) and z-score them per date.
    panel = asof_join(panel, pit)
    panel = _zscore_per_date(panel, PIT_FEATURES)

    # Keep only rows that actually have fundamentals (else the comparison is
    # vacuous — fundamentals can't help where they don't exist).
    have = panel[PIT_FEATURES].notna().any(axis=1)
    sub  = panel[have].copy()
    cov  = len(sub) / len(panel)
    print(f"  Panel rows       : {len(panel):,}  | with fundamentals: "
          f"{len(sub):,} ({cov*100:.1f}%)")
    if len(sub) < 1500:
        print("\n  ⚠ Too few rows with fundamentals to validate reliably.")
        print("    This is the expected yfinance depth limit, not a bug.")
        print("    Pipeline is built and leak-free; revisit with deeper data.")
        _coverage(sub)
        return {"status": "insufficient_window", "rows_with_fund": int(len(sub)),
                "coverage": round(cov, 3)}

    # Fill missing fundamental z (sparse names) with 0 = neutral, so the
    # combined model can still use price features on those rows.
    for c in PIT_Z:
        sub[c] = sub[c].fillna(0.0)

    dates = np.sort(sub["date"].unique())
    fold  = len(dates) // (n_splits + 1)
    embargo = HORIZON

    price_ics, both_ics = [], []
    for k in range(1, n_splits + 1):
        tr_end = fold * k
        te_start = tr_end + embargo
        te_end = min(fold * (k + 1), len(dates))
        if te_start >= te_end:
            continue
        tr_d = set(dates[:tr_end]); te_d = set(dates[te_start:te_end])
        tr = sub[sub["date"].isin(tr_d)]
        te = sub[sub["date"].isin(te_d)].copy()
        if len(tr) < 500 or len(te) < 100:
            continue

        m1 = _make_model(); m1.fit(tr[FEATURES_Z], tr["label"])
        te["pred"] = m1.predict_proba(te[FEATURES_Z])[:, 1]
        ic_p = _rank_ic(te)

        m2 = _make_model(); m2.fit(tr[FEATURES_Z + PIT_Z], tr["label"])
        te["pred"] = m2.predict_proba(te[FEATURES_Z + PIT_Z])[:, 1]
        ic_b = _rank_ic(te)

        price_ics.append(ic_p); both_ics.append(ic_b)
        print(f"  Fold {k}: price-only IC {ic_p:+.4f} | "
              f"+fundamentals IC {ic_b:+.4f} | Δ {ic_b - ic_p:+.4f}")

    if not price_ics:
        print("\n  ⚠ No valid folds (window too short).")
        return {"status": "insufficient_window"}

    mp, mb = float(np.nanmean(price_ics)), float(np.nanmean(both_ics))
    delta = mb - mp
    print("\n  " + "─" * 56)
    print(f"  Mean Rank IC  price-only      : {mp:+.4f}")
    print(f"  Mean Rank IC  + fundamentals  : {mb:+.4f}")
    print(f"  Δ from fundamentals           : {delta:+.4f}")
    verdict = ("FUNDAMENTALS ADD IC — worth including"
               if delta > 0.005 else
               "NEUTRAL/NEGATIVE on this window — keep price-only ranker")
    print(f"  Verdict                       : {verdict}")
    print("  (PROVISIONAL — short window; re-run as PIT history deepens.)")
    return {"status": "ok", "ic_price": round(mp, 4),
            "ic_both": round(mb, 4), "delta": round(delta, 4),
            "verdict": verdict, "coverage": round(cov, 3)}


def _coverage(sub):
    if sub.empty:
        return
    cov = sub[PIT_FEATURES].notna().mean().sort_values(ascending=False)
    print("\n  Coverage per feature (of rows that have ANY fundamental):")
    for f, c in cov.items():
        print(f"     {f:<26} {c*100:5.1f}%")


if __name__ == "__main__":
    run()

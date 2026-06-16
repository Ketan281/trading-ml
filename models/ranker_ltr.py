"""
Learning-to-rank ranker (LambdaMART) — roadmap #4 (part A).

The production ranker is an XGBClassifier predicting P(beat the median). That
optimises a POINTWISE loss (is each stock above/below median?) even though
what we actually consume is the ORDERING (top-N book). Learning-to-rank
optimises the ordering directly:

  • XGBRanker with objective rank:ndcg = LambdaMART. It is trained on groups
    (one group per date) and graded relevance (per-date forward-return
    quintile 0..4), so it is rewarded for putting the eventual winners at the
    TOP of each day's list — exactly the book we trade.

This module is VALIDATION-FIRST and non-destructive: it runs the LTR model
and the classifier on IDENTICAL walk-forward splits and compares Rank IC and
long/short spread. The LTR model is saved as a SEPARATE artefact; it only
replaces the production ranker if it wins, and that swap is a deliberate
later step, never silent.
"""

import os
import sys
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.cross_sectional import (
    load_prices, build_panel, _make_model, _rank_ic, _long_short,
    FEATURES_Z, MIN_NAMES, HORIZON, MODELS_DIR,
)

N_RELEVANCE_BINS = 5      # graded relevance: per-date forward-return quintiles


def _make_ranker():
    return xgb.XGBRanker(
        objective        = "rank:ndcg",
        n_estimators     = 300,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        random_state     = 42,
        verbosity        = 0,
    )


def _graded_relevance(df):
    """Per-date forward-return quintile (0=worst .. 4=best). LambdaMART's
    relevance label — higher = should rank higher."""
    return df.groupby("date")["fwd"].transform(
        lambda x: pd.qcut(x.rank(method="first"), N_RELEVANCE_BINS,
                          labels=False, duplicates="drop")
    ).fillna(0).astype(int)


def _fit_ranker(model, tr):
    """XGBRanker needs rows grouped by query (date) and a qid array. Sort by
    date so groups are contiguous, then pass qid."""
    tr = tr.sort_values("date")
    qid = tr["date"].astype("category").cat.codes.to_numpy()
    model.fit(tr[FEATURES_Z], tr["rel"], qid=qid)
    return model


def run(n_splits=5):
    print("=" * 64)
    print("  LEARNING-TO-RANK (LambdaMART) vs classifier — walk-forward")
    print("=" * 64)

    prices = load_prices()
    panel  = build_panel(prices)
    panel["rel"] = _graded_relevance(panel)
    dates = np.sort(panel["date"].unique())
    print(f"  Panel rows: {len(panel):,} | dates: {len(dates)}\n")

    fold = len(dates) // (n_splits + 1)
    embargo = HORIZON
    rows = []

    for k in range(1, n_splits + 1):
        tr_end = fold * k
        te_start = tr_end + embargo
        te_end = min(fold * (k + 1), len(dates))
        if te_start >= te_end:
            continue
        tr_d = set(dates[:tr_end]); te_d = set(dates[te_start:te_end])
        tr = panel[panel["date"].isin(tr_d)]
        te = panel[panel["date"].isin(te_d)].copy()
        if len(tr) < 500 or len(te) < 100:
            continue

        # Baseline classifier
        clf = _make_model(); clf.fit(tr[FEATURES_Z], tr["label"])
        te["pred"] = clf.predict_proba(te[FEATURES_Z])[:, 1]
        ic_c, ls_c = _rank_ic(te), _long_short(te)

        # LambdaMART ranker
        rk = _fit_ranker(_make_ranker(), tr)
        te["pred"] = rk.predict(te[FEATURES_Z])
        ic_r, ls_r = _rank_ic(te), _long_short(te)

        rows.append((ic_c, ls_c, ic_r, ls_r))
        print(f"  Fold {k}:  classifier IC {ic_c:+.4f} LS {ls_c:+.4%}  |  "
              f"LTR IC {ic_r:+.4f} LS {ls_r:+.4%}")

    if not rows:
        print("  ⚠ No valid folds."); return None
    arr = np.array(rows)
    mic_c, mls_c, mic_r, mls_r = arr.mean(axis=0)

    print("\n  " + "─" * 56)
    print(f"  Mean IC   classifier {mic_c:+.4f}  |  LTR {mic_r:+.4f}  "
          f"(Δ {mic_r - mic_c:+.4f})")
    print(f"  Mean L/S  classifier {mls_c:+.4%}  |  LTR {mls_r:+.4%}  "
          f"(Δ {mls_r - mls_c:+.4%})")
    winner = "LTR" if (mic_r > mic_c and mls_r >= mls_c) else "classifier"
    verdict = ("LTR WINS — worth promoting to production ranker"
               if winner == "LTR" else
               "Classifier still best — keep it; LTR saved for reference")
    print(f"  Verdict: {verdict}")

    # Fit final LTR on all data and save as a SEPARATE artefact (no silent swap).
    final = _fit_ranker(_make_ranker(), panel)
    path = os.path.join(MODELS_DIR, "cross_sectional_ltr.pkl")
    with open(path, "wb") as f:
        pickle.dump(final, f)
    meta = {
        "type": "lambdamart_ranker", "objective": "rank:ndcg",
        "features": FEATURES_Z, "horizon": HORIZON,
        "mean_ic_ltr": round(float(mic_r), 4),
        "mean_ic_classifier": round(float(mic_c), 4),
        "mean_ls_ltr": round(float(mls_r), 4),
        "winner": winner, "verdict": verdict,
        "trained": datetime.now().isoformat(),
    }
    with open(os.path.join(MODELS_DIR, "cross_sectional_ltr_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  ✅ LTR model saved (reference) → {path}")
    return meta


if __name__ == "__main__":
    run()

"""
Ensemble meta-model — blend every signal into ONE selection score.

Combines the project's signals cross-sectionally into a single 0-100 score per
stock, with a transparent contribution breakdown:

  • Momentum rank   (the cross-sectional ranker — the ONLY proven-edge signal)
  • Fundamental quality (winsorised quality z-score)
  • Sector RS        (is the stock's sector leading?)
  • Intra-sector RS  (is it a leader WITHIN its sector?)

Why a TRANSPARENT weighted blend, not a trained stacker
-------------------------------------------------------
A black-box meta-learner needs leak-free historical labels for ALL components;
our fundamentals are a current snapshot and the multi-signal history is short,
so a trained stacker would overfit. We already saw learning-to-rank and deep
learning LOSE to the plain ranker. So the ensemble is an explicit weighted
z-score blend — every contribution is visible and auditable, and the ranker
(the proven edge) carries the most weight. Regime/breadth are attached as
CONTEXT (they govern how much to deploy, via risk_policy), not silent tweaks.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.cross_sectional import rank_today
from models.fundamentals import quality_scores
from models.sector_strength import sector_strength
from models.regime_classifier import classify
from pipelines.breadth import breadth_read

# Weights reflect PROVEN edge: the ranker dominates; the rest are tilts.
WEIGHTS = {"momentum": 0.45, "quality": 0.25, "sector_rs": 0.20, "intra_rs": 0.10}

# Dynamic factor weighting by regime (#9): momentum leads in bull/trend,
# quality + sector leadership lead in bear/volatile (defensiveness pays). Every
# row still sums to 1.0 and the ranker keeps meaningful weight everywhere.
REGIME_WEIGHTS = {
    "bull":     {"momentum": 0.55, "quality": 0.15, "sector_rs": 0.20, "intra_rs": 0.10},
    "sideways": {"momentum": 0.40, "quality": 0.25, "sector_rs": 0.20, "intra_rs": 0.15},
    "bear":     {"momentum": 0.30, "quality": 0.40, "sector_rs": 0.20, "intra_rs": 0.10},
    "volatile": {"momentum": 0.30, "quality": 0.35, "sector_rs": 0.25, "intra_rs": 0.10},
    "unknown":  dict(WEIGHTS),
}


def weights_for_regime(regime):
    return dict(REGIME_WEIGHTS.get(regime, WEIGHTS))


def _z(s):
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(), s.std()
    return (s - mu) / (sd + 1e-9)


def ensemble_score(pool=500, regime=None):
    """Blend signals into one score. If `regime` is given (or auto-detected),
    factor weights adapt to it (#9 dynamic factor weighting)."""
    if regime is None:
        try:
            regime = classify("NIFTY")["regime"]
        except Exception:
            regime = "unknown"
    weights = weights_for_regime(regime)

    ranked = rank_today(top_n=pool)
    if ranked is None or ranked.empty:
        return None
    df = ranked[["symbol", "score"]].copy()

    # Fundamentals
    q = quality_scores(symbols=set(df["symbol"]))
    if q is not None:
        df = df.merge(q[["quality_z", "quality_score"]],
                      left_on="symbol", right_index=True, how="left")
    else:
        df["quality_z"] = 0.0; df["quality_score"] = 50.0

    # Sector RS
    sector, stocks = sector_strength()
    if stocks is not None:
        srs = stocks[["symbol", "sector", "intra_sector_rs"]]
        df = df.merge(srs, on="symbol", how="left")
        sec_score = sector.set_index("sector")["rs_score"]
        df["sector_rs_score"] = df["sector"].map(sec_score)
    else:
        df["sector"] = None; df["intra_sector_rs"] = 0.0; df["sector_rs_score"] = 50.0

    df["quality_z"] = df["quality_z"].fillna(0.0)
    df["intra_sector_rs"] = df["intra_sector_rs"].fillna(0.0)
    df["sector_rs_score"] = df["sector_rs_score"].fillna(50.0)

    # Cross-sectional z-scores of each component.
    comp = pd.DataFrame({"symbol": df["symbol"]})
    comp["momentum"]  = _z(df["score"])
    comp["quality"]   = _z(df["quality_z"])
    comp["sector_rs"] = _z(df["sector_rs_score"])
    comp["intra_rs"]  = _z(df["intra_sector_rs"])

    # Weighted contributions (regime-adaptive weights; kept for explainability).
    for k, w in weights.items():
        comp[k + "_contrib"] = comp[k] * w
    comp["blend"] = comp[[k + "_contrib" for k in weights]].sum(axis=1)
    comp["ensemble_score"] = (comp["blend"].rank(pct=True) * 100).round(1)

    out = df.merge(comp, on="symbol")
    out.attrs["regime"] = regime
    out.attrs["weights"] = weights
    out = out.sort_values("ensemble_score", ascending=False).reset_index(drop=True)
    out["ensemble_rank"] = out.index + 1
    return out


def top_picks(n=15, pool=500):
    out = ensemble_score(pool)
    if out is None:
        return None
    ctx = {"regime": classify("NIFTY"), "breadth": breadth_read()}
    return out.head(n), ctx


def contribution_breakdown(row):
    """Which signals drove this stock's ensemble score (for explainability)."""
    parts = {k: round(float(row[k + "_contrib"]), 3) for k in WEIGHTS}
    ordered = sorted(parts.items(), key=lambda x: -abs(x[1]))
    return ordered


if __name__ == "__main__":
    res = top_picks(15)
    if res is None:
        print("  ⚠ Train the ranker first (python models/cross_sectional.py).")
    else:
        picks, ctx = res
        r = ctx["regime"]; b = ctx["breadth"]
        print("=" * 78)
        print(f"  ENSEMBLE META-MODEL — top picks   "
              f"(regime {r['regime'].upper()}, breadth {b['score']})")
        print("=" * 78)
        print(f"  {'#':<3}{'SYMBOL':<13}{'ENS':>5}{'MOM':>6}{'QUAL':>6}"
              f"{'SECT':>6}{'INTRA':>6}  SECTOR")
        for x in picks.itertuples():
            print(f"  {x.ensemble_rank:<3}{x.symbol:<13}{x.ensemble_score:>5}"
                  f"{x.momentum_contrib:>6.2f}{x.quality_contrib:>6.2f}"
                  f"{x.sector_rs_contrib:>6.2f}{x.intra_rs_contrib:>6.2f}  "
                  f"{x.sector or '-'}")
        print("\n  (ENS = 0-100 blended percentile; columns = each signal's "
              "weighted contribution)")

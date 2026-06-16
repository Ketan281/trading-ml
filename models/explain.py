"""
Explainable-AI layer — WHY was each name picked?

Every trade should come with a reason a human can check. This produces two
complementary explanations per stock:

  1. RANKER SHAP — exact per-feature SHAP values from the XGBoost ranker
     (via XGBoost's built-in TreeSHAP, `pred_contribs=True` — no extra
     dependency). Shows which factors (momentum, RSI, distance-from-high…)
     pushed the rank UP or DOWN.
  2. ENSEMBLE breakdown — how much each signal (momentum / quality / sector RS /
     intra-sector RS) contributed to the blended score.

Together they answer "the model likes this stock because its 6-month momentum
and sector strength are strong, despite mediocre quality" — in plain English,
auditable, and impossible to hallucinate (the numbers come from the models).
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.cross_sectional import (
    load_prices, _symbol_factors, FEATURES, FEATURES_Z, MODELS_DIR, MIN_NAMES,
)
from models.ensemble import ensemble_score, WEIGHTS

FEATURE_LABEL = {
    "mom_21_z": "1-month momentum", "mom_63_z": "3-month momentum",
    "mom_126_z": "6-month momentum", "mom_252_z": "12-month momentum",
    "rev_5_z": "short-term reversal", "vol_21_z": "volatility",
    "rsi_14_z": "RSI", "dist_high_z": "distance from 1yr high",
    "ma_ratio_z": "price vs 200d MA",
}


def _ranker_shap(symbols=None):
    """Per-feature SHAP contributions from the saved ranker. SHAP is computed
    over the FULL universe (the model scores cross-sectionally, so the z-scores
    must be too); the requested `symbols` are sliced from the result."""
    path = os.path.join(MODELS_DIR, "cross_sectional_xgb.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        model = pickle.load(f)

    prices = load_prices()                   # full universe, not just queries
    rows = []
    for sym, df in prices.items():
        f = _symbol_factors(df).dropna()
        if f.empty:
            continue
        last = f.iloc[-1]
        rows.append({"symbol": sym, **{c: float(last[c]) for c in FEATURES}})
    cs = pd.DataFrame(rows)
    if len(cs) < MIN_NAMES:
        return None
    for c in FEATURES:                       # cross-sectional z like training
        cs[c + "_z"] = (cs[c] - cs[c].mean()) / (cs[c].std() + 1e-9)

    booster = model.get_booster()
    import xgboost as xgb
    dm = xgb.DMatrix(cs[FEATURES_Z])
    contribs = booster.predict(dm, pred_contribs=True)   # rows × (n_feat + bias)
    shap = pd.DataFrame(contribs[:, :-1], columns=FEATURES_Z)
    shap["symbol"] = cs["symbol"].values
    return shap.set_index("symbol")


def explain(symbols, top_k=3):
    """Return a per-symbol explanation dict combining ranker SHAP + ensemble."""
    shap = _ranker_shap(symbols)
    ens = ensemble_score(pool=500)
    ens = ens.set_index("symbol") if ens is not None else None

    out = {}
    for sym in symbols:
        info = {"symbol": sym, "drivers": [], "ensemble": []}
        if shap is not None and sym in shap.index:
            s = shap.loc[sym].sort_values(key=abs, ascending=False)
            for feat, val in s.head(top_k).items():
                info["drivers"].append({
                    "factor": FEATURE_LABEL.get(feat, feat),
                    "effect": "↑" if val > 0 else "↓",
                    "shap": round(float(val), 4)})
        if ens is not None and sym in ens.index:
            row = ens.loc[sym]
            info["ensemble_score"] = float(row["ensemble_score"])
            parts = [(k, round(float(row[k + "_contrib"]), 3)) for k in WEIGHTS]
            info["ensemble"] = sorted(parts, key=lambda x: -abs(x[1]))
            info["sector"] = row.get("sector")
        out[sym] = info
    return out


def narrative(info):
    """One-paragraph plain-English reason from the structured explanation."""
    if not info.get("drivers") and not info.get("ensemble"):
        return f"{info['symbol']}: no explanation available."
    ups = [d["factor"] for d in info["drivers"] if d["effect"] == "↑"]
    downs = [d["factor"] for d in info["drivers"] if d["effect"] == "↓"]
    bits = [f"{info['symbol']} ranks well on " + ", ".join(ups) if ups else
            f"{info['symbol']}"]
    if downs:
        bits.append("held back by " + ", ".join(downs))
    if info.get("ensemble"):
        lead = info["ensemble"][0]
        names = {"momentum": "price momentum", "quality": "fundamental quality",
                 "sector_rs": "sector strength", "intra_rs": "intra-sector leadership"}
        bits.append(f"the blended score is driven most by {names.get(lead[0], lead[0])}")
    return "; ".join(bits) + "."


if __name__ == "__main__":
    syms = sys.argv[1:]
    if not syms:                                   # default: explain top ensemble picks
        ens = ensemble_score(pool=500)
        syms = list(ens.head(5)["symbol"]) if ens is not None else []
    exp = explain(syms)
    print("=" * 74)
    print("  EXPLAINABLE AI — why each name was picked")
    print("=" * 74)
    for sym in syms:
        info = exp.get(sym, {})
        sc = info.get("ensemble_score")
        print(f"\n  ── {sym}  (ensemble {sc if sc is not None else '?'}) ──")
        for d in info.get("drivers", []):
            print(f"     {d['effect']} {d['factor']:<24} SHAP {d['shap']:+}")
        if info.get("ensemble"):
            print("     ensemble contributions: " +
                  ", ".join(f"{k} {v:+.2f}" for k, v in info["ensemble"]))
        print(f"     → {narrative(info)}")

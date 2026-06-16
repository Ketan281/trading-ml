"""
Probability calibration for the ranker's confidence — roadmap #4 (part B).

The classifier outputs P(beat the median), and the screener surfaces that as a
"confidence". But a raw XGBoost probability is usually MIS-calibrated: when it
says 0.70 the real hit-rate might be 0.58. A confidence number you can't trust
is worse than none. This fixes that.

Method (leak-free)
------------------
• Collect OUT-OF-FOLD predictions walk-forward (each test fold scored by a
  model trained only on earlier dates) — so the calibration is measured on
  data the model never saw, the same discipline as the IC backtest.
• Fit an ISOTONIC regression mapping raw prob → empirical P(label=1).
• Report reliability BEFORE vs AFTER with two honest numbers:
      Brier score (lower = better) and
      ECE = expected calibration error (avg |confidence − accuracy| over bins).
• Save the fitted calibrator so the screener can map raw score → trustworthy
  probability at inference time.
"""

import os
import sys
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.cross_sectional import (
    load_prices, build_panel, _make_model,
    FEATURES_Z, HORIZON, MODELS_DIR,
)


def _brier(p, y):
    return float(np.mean((p - y) ** 2))


def _ece(p, y, bins=10):
    """Expected calibration error: bin by predicted prob, average the gap
    between mean confidence and actual accuracy in each bin."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        conf = p[m].mean()
        acc  = y[m].mean()
        ece += (m.sum() / len(p)) * abs(conf - acc)
    return float(ece)


def run(n_splits=5):
    print("=" * 60)
    print("  PROBABILITY CALIBRATION — confidence you can trust")
    print("=" * 60)

    prices = load_prices()
    panel  = build_panel(prices)
    dates  = np.sort(panel["date"].unique())
    fold   = len(dates) // (n_splits + 1)
    embargo = HORIZON

    oof_p, oof_y = [], []
    for k in range(1, n_splits + 1):
        tr_end = fold * k
        te_start = tr_end + embargo
        te_end = min(fold * (k + 1), len(dates))
        if te_start >= te_end:
            continue
        tr_d = set(dates[:tr_end]); te_d = set(dates[te_start:te_end])
        tr = panel[panel["date"].isin(tr_d)]
        te = panel[panel["date"].isin(te_d)]
        if len(tr) < 500 or len(te) < 100:
            continue
        m = _make_model(); m.fit(tr[FEATURES_Z], tr["label"])
        oof_p.append(m.predict_proba(te[FEATURES_Z])[:, 1])
        oof_y.append(te["label"].to_numpy())

    if not oof_p:
        print("  ⚠ Not enough data for calibration."); return None
    p = np.concatenate(oof_p); y = np.concatenate(oof_y).astype(float)
    print(f"  Out-of-fold predictions: {len(p):,}")

    # Fit isotonic on the OOF pairs (the leak-free calibration set).
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p, y)
    p_cal = iso.predict(p)

    b0, b1 = _brier(p, y), _brier(p_cal, y)
    e0, e1 = _ece(p, y),   _ece(p_cal, y)
    print("\n  " + "─" * 50)
    print(f"  {'metric':<18}{'raw':>12}{'calibrated':>14}")
    print(f"  {'Brier (↓)':<18}{b0:>12.4f}{b1:>14.4f}")
    print(f"  {'ECE   (↓)':<18}{e0:>12.4f}{e1:>14.4f}")
    improved = (e1 <= e0 and b1 <= b0)
    print(f"  Calibration {'IMPROVED' if improved else 'no better'} "
          f"(ECE {e0:.3f}→{e1:.3f})")

    # Reliability table (a few bins) so the improvement is visible.
    print("\n  Reliability (raw):  conf → actual hit-rate")
    edges = np.linspace(0, 1, 6)
    idx = np.clip(np.digitize(p, edges) - 1, 0, 4)
    for b in range(5):
        m = idx == b
        if m.any():
            print(f"     [{edges[b]:.1f}-{edges[b+1]:.1f}]  "
                  f"conf {p[m].mean():.3f}  actual {y[m].mean():.3f}  "
                  f"(n={int(m.sum())})")

    # Refit a final calibrator on ALL oof pairs and save.
    path = os.path.join(MODELS_DIR, "confidence_calibrator.pkl")
    with open(path, "wb") as f:
        pickle.dump(iso, f)
    meta = {"type": "isotonic_confidence_calibrator",
            "brier_raw": round(b0, 4), "brier_cal": round(b1, 4),
            "ece_raw": round(e0, 4), "ece_cal": round(e1, 4),
            "improved": bool(improved), "n_oof": int(len(p)),
            "trained": datetime.now().isoformat()}
    with open(os.path.join(MODELS_DIR, "confidence_calibrator_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  ✅ Calibrator saved → {path}")
    return meta


# Inference helper the screener can import.
def calibrate_scores(scores):
    """Map raw ranker probabilities → calibrated probabilities. Falls back to
    the raw scores untouched if no calibrator has been fitted yet."""
    path = os.path.join(MODELS_DIR, "confidence_calibrator.pkl")
    if not os.path.exists(path):
        return np.asarray(scores, dtype=float)
    with open(path, "rb") as f:
        iso = pickle.load(f)
    return iso.predict(np.asarray(scores, dtype=float))


if __name__ == "__main__":
    run()

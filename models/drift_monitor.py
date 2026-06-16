"""
Model drift & edge-decay monitor — roadmap #5 (part B).

A model with a measured backtest IC is not trustworthy forever. Two things
silently erode it:

  1. EDGE DECAY  — the alpha gets arbitraged away; recent out-of-sample IC
     drifts toward zero even though the historical backtest looked great.
  2. FEATURE DRIFT — the live feature distribution moves away from what the
     model trained on (regime change, universe change). Predictions then come
     from a part of feature space the model never learned.

This monitor measures both with honest, leak-free numbers and raises a flag
BEFORE you find out the hard way in live P&L:

  • Recent OOS IC : train on all but the last `live_periods` rebalance dates,
    measure Rank IC on those held-out recent dates, compare to the backtest
    mean IC. A large shortfall = decay.
  • Feature PSI   : Population Stability Index of each feature, training era
    vs the live window. PSI > 0.25 = significant shift (industry rule of
    thumb), 0.10–0.25 = moderate.

Output: a health report (OK / WARN / ALERT) saved to outputs/monitoring/.
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.cross_sectional import (
    load_prices, build_panel, _make_model, _rank_ic,
    FEATURES, FEATURES_Z, HORIZON, MODELS_DIR,
)

OUT_DIR = os.path.join(ROOT, "outputs", "monitoring")
os.makedirs(OUT_DIR, exist_ok=True)

LIVE_PERIODS   = 6        # most-recent rebalances treated as "live" (OOS)
PSI_WARN       = 0.10
PSI_ALERT      = 0.25
IC_DECAY_WARN  = 0.5      # recent IC < 50% of backtest IC → warn
IC_DECAY_ALERT = 0.0      # recent IC <= 0 → alert (edge gone)


def _psi(expected, actual, bins=10):
    """Population Stability Index between two samples of one feature."""
    expected = np.asarray(expected, dtype=float)
    actual   = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual   = actual[~np.isnan(actual)]
    if len(expected) < 50 or len(actual) < 20:
        return float("nan")
    # Bin edges from the expected (training) distribution quantiles.
    qs = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(expected, qs))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    e_hist = np.histogram(expected, bins=edges)[0] / len(expected)
    a_hist = np.histogram(actual,   bins=edges)[0] / len(actual)
    eps = 1e-6
    e_hist = np.clip(e_hist, eps, None)
    a_hist = np.clip(a_hist, eps, None)
    return float(np.sum((a_hist - e_hist) * np.log(a_hist / e_hist)))


def _backtest_ic():
    meta = os.path.join(MODELS_DIR, "cross_sectional_meta.json")
    if os.path.exists(meta):
        try:
            with open(meta) as f:
                return float(json.load(f).get("mean_ic", float("nan")))
        except Exception:
            pass
    return float("nan")


def run(live_periods=LIVE_PERIODS):
    print("=" * 60)
    print("  MODEL DRIFT & EDGE-DECAY MONITOR")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    prices = load_prices()
    panel  = build_panel(prices)
    dates  = np.sort(panel["date"].unique())
    rb     = list(dates[::HORIZON])
    if len(rb) < live_periods + 4:
        print("  ⚠ Not enough history to monitor."); return None

    live_dates  = set(rb[-live_periods:])
    cutoff      = rb[-live_periods]
    train       = panel[panel["date"] < cutoff]
    live        = panel[panel["date"].isin(
                    {d for d in panel["date"].unique() if d >= cutoff})].copy()

    # ── Edge decay: train on history, score the held-out live window ──
    bt_ic = _backtest_ic()
    model = _make_model()
    model.fit(train[FEATURES_Z], train["label"])
    live["pred"] = model.predict_proba(live[FEATURES_Z])[:, 1]
    live_ic = _rank_ic(live)

    ratio = (live_ic / bt_ic) if (bt_ic and bt_ic == bt_ic and bt_ic > 0) else float("nan")
    if live_ic <= IC_DECAY_ALERT:
        ic_status = "ALERT"
    elif ratio == ratio and ratio < IC_DECAY_WARN:
        ic_status = "WARN"
    else:
        ic_status = "OK"

    print(f"\n  EDGE DECAY")
    print(f"    Backtest mean IC      : {bt_ic:+.4f}")
    print(f"    Recent OOS IC ({live_periods} rebal): {live_ic:+.4f}")
    print(f"    Retention             : "
          f"{ratio*100:.0f}% of backtest" if ratio == ratio else
          "    Retention             : n/a")
    print(f"    Status                : {ic_status}")

    # ── Feature drift: PSI training era vs live window ──
    print(f"\n  FEATURE DRIFT (PSI: <{PSI_WARN} ok, "
          f"{PSI_WARN}-{PSI_ALERT} moderate, >{PSI_ALERT} significant)")
    psis = {}
    worst = "OK"
    for f in FEATURES:
        p = _psi(train[f], live[f])
        psis[f] = round(p, 4) if p == p else None
        if p != p:
            tag = "n/a"
        elif p > PSI_ALERT:
            tag, worst = "ALERT", "ALERT"
        elif p > PSI_WARN:
            tag = "WARN"; worst = "WARN" if worst != "ALERT" else worst
        else:
            tag = "ok"
        bar = "█" * min(20, int((p if p == p else 0) / 0.02))
        print(f"    {f:<12} {('%.3f'%p) if p==p else ' n/a ':>7} {tag:<6} {bar}")

    overall = "ALERT" if "ALERT" in (ic_status, worst) else \
              "WARN" if "WARN" in (ic_status, worst) else "OK"
    print("\n  " + "─" * 50)
    print(f"  OVERALL HEALTH: {overall}")
    if overall == "OK":
        print("  → Model is behaving as backtested. No action needed.")
    elif overall == "WARN":
        print("  → Soft signal of decay/shift. Watch closely; consider retrain.")
    else:
        print("  → Edge gone or features shifted hard. RETRAIN / investigate "
              "before trusting live signals.")

    report = {
        "timestamp": datetime.now().isoformat(),
        "backtest_ic": round(bt_ic, 4) if bt_ic == bt_ic else None,
        "live_ic": round(float(live_ic), 4) if live_ic == live_ic else None,
        "ic_retention": round(float(ratio), 3) if ratio == ratio else None,
        "ic_status": ic_status,
        "feature_psi": psis,
        "feature_status": worst,
        "overall": overall,
        "live_periods": live_periods,
    }
    path = os.path.join(OUT_DIR, "drift_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  ✅ Saved → {path}")
    return report


if __name__ == "__main__":
    run()

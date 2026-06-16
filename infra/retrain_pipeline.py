"""
Automated walk-forward retrain pipeline — the self-maintaining loop.

Chains the backbone into one safe, repeatable retrain:

  1. DATA-QUALITY GATE  — validate the universe; abort if too much is broken
  2. RETRAIN            — walk-forward train the cross-sectional ranker
  3. REGISTER           — version the model with its metrics + data hash
  4. PROMOTION GATE     — promote to production ONLY if it does not degrade the
                          current production IC (never ship a worse model)
  5. DRIFT CHECK        — run the drift monitor and record the verdict

This is what you schedule (e.g., weekly). It is conservative by design: a fresh
model that underperforms the incumbent is registered but NOT promoted, so live
behaviour never silently gets worse — the same honesty bar as the rest of the
system.
"""

import os
import sys
import json
import pickle
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from infra.data_quality import validate_universe
from infra.feature_store import universe_hash
from infra import model_registry as registry
from models.cross_sectional import load_prices, train, MODELS_DIR

MODEL_NAME   = "cross_sectional_ranker"
MAX_FAIL_PCT = 0.10        # abort retrain if >10% of universe fails QC
IC_TOLERANCE = 0.002       # allow tiny noise; promote if new ≥ prod − tolerance


def run(force_promote=False):
    print("=" * 70)
    print("  AUTOMATED WALK-FORWARD RETRAIN PIPELINE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── 1. Data-quality gate ─────────────────────────────
    reports, by = validate_universe()
    total = len(reports); fails = len(by["fail"])
    fail_pct = fails / total if total else 1.0
    print(f"  [1] Data quality: {len(by['pass'])} pass / {len(by['warn'])} warn "
          f"/ {fails} fail  ({fail_pct:.1%})")
    if fail_pct > MAX_FAIL_PCT:
        print(f"  ⛔ ABORT — {fail_pct:.1%} of universe failed QC "
              f"(> {MAX_FAIL_PCT:.0%}). Fix data before retraining.")
        return {"status": "aborted_data_quality", "fail_pct": fail_pct}

    # ── 2. Retrain (walk-forward) ────────────────────────
    print("  [2] Retraining ranker (walk-forward)…")
    data_hash = universe_hash(load_prices())
    result = train()                      # trains, validates, saves pkl + meta
    if not result:
        print("  ⛔ Training failed."); return {"status": "train_failed"}

    # ── 3. Register the new version ──────────────────────
    with open(os.path.join(MODELS_DIR, "cross_sectional_xgb.pkl"), "rb") as f:
        model = pickle.load(f)
    entry = registry.register(
        MODEL_NAME, model,
        metrics={"mean_ic": round(result["mean_ic"], 4),
                 "mean_ls": round(result["mean_ls"], 4)},
        params={"type": "xgboost_classifier"}, data_hash=data_hash,
        tags=[result.get("verdict", "")])
    new_ic = entry["metrics"]["mean_ic"]

    # ── 4. Promotion gate ────────────────────────────────
    prod = registry._find(MODEL_NAME, "production")
    prod_ic = prod["metrics"].get("mean_ic") if prod else None
    if prod_ic is None or force_promote or new_ic >= prod_ic - IC_TOLERANCE:
        registry.promote(MODEL_NAME, entry["version"])
        promoted = True
        verdict = (f"promoted v{entry['version']} (IC {new_ic:+.4f}"
                   + (f" vs prod {prod_ic:+.4f})" if prod_ic is not None else ", first model)"))
    else:
        promoted = False
        verdict = (f"NOT promoted — new IC {new_ic:+.4f} < production "
                   f"{prod_ic:+.4f} − tol; keeping incumbent")
    print(f"  [4] Promotion: {verdict}")

    # ── 5. Drift check ───────────────────────────────────
    drift = None
    try:
        from models.drift_monitor import run as drift_run
        print("  [5] Drift check…")
        drift = drift_run()
    except Exception as e:
        print(f"  [5] Drift check skipped: {e}")

    report = {"status": "ok", "timestamp": datetime.now().isoformat(),
              "data_quality": {"pass": len(by["pass"]), "warn": len(by["warn"]),
                               "fail": fails}, "new_version": entry["version"],
              "new_ic": new_ic, "production_ic_before": prod_ic,
              "promoted": promoted, "verdict": verdict,
              "drift_overall": drift.get("overall") if drift else None,
              "data_hash": data_hash}
    out = os.path.join(ROOT, "outputs", "retrain_report.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  ✅ Retrain complete → {out}")
    print(f"     {verdict}"
          + (f" | drift: {drift['overall']}" if drift else ""))
    return report


if __name__ == "__main__":
    run(force_promote="--force" in sys.argv)

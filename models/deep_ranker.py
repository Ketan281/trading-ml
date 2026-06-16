"""
Deep-learning cross-sectional ranker — honest head-to-head vs XGBoost.

The user asked for deep learning behind the calls. This builds a neural net on
the SAME 10-year daily factor panel, validates it on the SAME walk-forward
folds with the SAME metrics (Rank IC, long/short spread) as the production
XGBoost ranker, and reports — without spin — which one actually wins.

Why the honest framing matters
------------------------------
On tabular cross-sectional factor data, gradient-boosted trees usually beat
neural nets, and our own LambdaMART test already underperformed XGBoost on IC.
So this is a fair experiment, not a foregone conclusion dressed as progress:
the DL model is PROMOTED only if it beats the tree out-of-sample. If it loses,
we keep XGBoost and say so.

Model: a small MLP (9 z-scored factors → 64 → 32 → 1) with batch-norm and
dropout, trained as a probability classifier on the beat-median label with
early stopping on a time-ordered validation slice (so it can't overfit the
whole training window).
"""

import os
import sys
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.cross_sectional import (
    load_prices, build_panel, _make_model, _rank_ic, _long_short,
    FEATURES_Z, HORIZON, MIN_NAMES, MODELS_DIR,
)

# torch is imported lazily so the module imports even before install.
EPOCHS      = 30
BATCH       = 8192
LR          = 1e-3
PATIENCE    = 4
HIDDEN      = (64, 32)
DROPOUT     = 0.2
SEED        = 42


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def _build_mlp(n_features):
    torch, nn = _torch()
    torch.manual_seed(SEED)
    layers, prev = [], n_features
    for h in HIDDEN:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                   nn.ReLU(), nn.Dropout(DROPOUT)]
        prev = h
    layers += [nn.Linear(prev, 1)]
    return nn.Sequential(*layers)


def _train_mlp(Xtr, ytr, Xva, yva):
    """Train the MLP with early stopping; return the best-state model."""
    torch, nn = _torch()
    dev = "cpu"
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    yva_t = torch.tensor(yva, dtype=torch.float32).view(-1, 1)

    model = _build_mlp(Xtr.shape[1]).to(dev)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()

    n = len(Xtr_t)
    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            out = model(Xtr_t[idx])
            loss = lossf(out, ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = lossf(model(Xva_t), yva_t).item()
        if vloss < best_val - 1e-5:
            best_val, best_state, bad = vloss, \
                {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _predict_mlp(model, X):
    torch, _ = _torch()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.sigmoid(logits).numpy().ravel()


def run(n_splits=5):
    print("=" * 66)
    print("  DEEP-LEARNING RANKER vs XGBOOST — honest walk-forward")
    print("=" * 66)
    try:
        _torch()
    except Exception:
        print("  ⚠ PyTorch not installed. Run:")
        print("    python -m pip install torch --index-url "
              "https://download.pytorch.org/whl/cpu")
        return None

    prices = load_prices()
    panel  = build_panel(prices)
    dates  = np.sort(panel["date"].unique())
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
        tr_dates = dates[:tr_end]
        te_d = set(dates[te_start:te_end])
        tr = panel[panel["date"].isin(set(tr_dates))]
        te = panel[panel["date"].isin(te_d)].copy()
        if len(tr) < 500 or len(te) < 100:
            continue

        # Time-ordered validation slice = last 15% of train dates (no leak).
        cut = tr_dates[int(len(tr_dates) * 0.85)]
        tr_in = tr[tr["date"] < cut]
        tr_va = tr[tr["date"] >= cut]
        if len(tr_va) < 100:
            tr_in, tr_va = tr, tr

        # XGBoost baseline
        xgb = _make_model(); xgb.fit(tr[FEATURES_Z], tr["label"])
        te["pred"] = xgb.predict_proba(te[FEATURES_Z])[:, 1]
        ic_x, ls_x = _rank_ic(te), _long_short(te)

        # Deep MLP
        mlp = _train_mlp(tr_in[FEATURES_Z].to_numpy(), tr_in["label"].to_numpy(),
                         tr_va[FEATURES_Z].to_numpy(), tr_va["label"].to_numpy())
        te["pred"] = _predict_mlp(mlp, te[FEATURES_Z].to_numpy())
        ic_d, ls_d = _rank_ic(te), _long_short(te)

        rows.append((ic_x, ls_x, ic_d, ls_d))
        print(f"  Fold {k}:  XGB IC {ic_x:+.4f} LS {ls_x:+.4%}  |  "
              f"DL IC {ic_d:+.4f} LS {ls_d:+.4%}")

    if not rows:
        print("  ⚠ No valid folds."); return None
    arr = np.array(rows)
    mic_x, mls_x, mic_d, mls_d = arr.mean(axis=0)

    print("\n  " + "─" * 58)
    print(f"  Mean IC   XGBoost {mic_x:+.4f}  |  Deep {mic_d:+.4f}  "
          f"(Δ {mic_d - mic_x:+.4f})")
    print(f"  Mean L/S  XGBoost {mls_x:+.4%}  |  Deep {mls_d:+.4%}  "
          f"(Δ {mls_d - mls_x:+.4%})")
    dl_wins = (mic_d > mic_x and mls_d >= mls_x * 0.9)
    verdict = ("DEEP LEARNING WINS — worth promoting" if dl_wins else
               "XGBoost still best — keeping it (DL did not beat the tree)")
    print(f"  Verdict: {verdict}")
    print("  (Honest call: promote DL only on a clear out-of-sample win.)")

    # Save the DL artefact + meta regardless, for reference / future use.
    full = panel
    cut  = dates[int(len(dates) * 0.9)]
    fin  = _train_mlp(full[full["date"] < cut][FEATURES_Z].to_numpy(),
                      full[full["date"] < cut]["label"].to_numpy(),
                      full[full["date"] >= cut][FEATURES_Z].to_numpy(),
                      full[full["date"] >= cut]["label"].to_numpy())
    torch, _ = _torch()
    path = os.path.join(MODELS_DIR, "deep_ranker.pt")
    torch.save(fin.state_dict(), path)
    meta = {"type": "mlp_cross_sectional_ranker", "features": FEATURES_Z,
            "hidden": list(HIDDEN), "dropout": DROPOUT, "horizon": HORIZON,
            "mean_ic_dl": round(float(mic_d), 4),
            "mean_ic_xgb": round(float(mic_x), 4),
            "mean_ls_dl": round(float(mls_d), 4),
            "mean_ls_xgb": round(float(mls_x), 4),
            "dl_wins": bool(dl_wins), "verdict": verdict,
            "trained": datetime.now().isoformat()}
    with open(os.path.join(MODELS_DIR, "deep_ranker_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  ✅ DL model saved (reference) → {path}")
    return meta


if __name__ == "__main__":
    run()

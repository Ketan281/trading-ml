"""
ML Trainer — train once, predict forever.

Walk-forward training with 4-year train / 1-year test blocks:
  Fold 1: Train 2006-2010, Test 2011
  Fold 2: Train 2012-2016, Test 2017
  Fold 3: Train 2018-2024, Test 2025
  Final:  Train ALL data -> production model

Loads data per-fold to avoid MemoryError on large feature stores.

Run locally: python -m engines.ml_trainer
The server only loads the saved model for inference.
"""

import gc
import os
import sys
import json
import time
import pickle
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

log = logging.getLogger("ml_trainer")

MODEL_DIR = os.path.join(ROOT, "models", "ml_v2")
os.makedirs(MODEL_DIR, exist_ok=True)

CROSS_SECTIONAL = {"mom_rank", "vol_rank", "rs_score"}

def _get_raw_features():
    from engines.feature_store import FEATURE_COLS
    return [f for f in FEATURE_COLS if f not in CROSS_SECTIONAL]

RAW_FEATURES = _get_raw_features()
Z_FEATURES = [f + "_z" for f in RAW_FEATURES]

HORIZONS = {
    "5d": 5,
    "10d": 10,
    "21d": 21,
}
MIN_STOCKS_PER_DATE = 20

# Walk-forward folds: (train_start, train_end, test_start, test_end)
WALK_FORWARD_FOLDS = [
    ("2006-01-01", "2010-12-31", "2011-01-01", "2011-12-31"),
    ("2012-01-01", "2016-12-31", "2017-01-01", "2017-12-31"),
    ("2018-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
]

# Train on 5d horizon — picks the best trade for THIS WEEK, not next month
DEFAULT_HORIZON = "5d"


def _load_panel_range(min_date, max_date):
    """Load a date range from feature store. Memory-safe."""
    from engines.feature_store import load_training_panel
    return load_training_panel(min_date=min_date, max_date=max_date, recent_years=None)


def _add_forward_returns(panel, horizon_key="21d"):
    """Add forward return columns using price data."""
    panel = panel.sort_values(["symbol", "date"]).copy()

    for label, horizon in HORIZONS.items():
        fwd_col = f"fwd_{label}"
        panel[fwd_col] = np.nan
        for sym, grp in panel.groupby("symbol"):
            idx = grp.index
            prices = grp["price"].values
            fwd = np.full(len(prices), np.nan)
            for i in range(len(prices) - horizon):
                if prices[i] > 0:
                    fwd[i] = prices[i + horizon] / prices[i] - 1
            panel.loc[idx, fwd_col] = fwd

    return panel


def _zscore_per_date(panel):
    """Cross-sectional z-score normalization per date."""
    for feat in RAW_FEATURES:
        zfeat = feat + "_z"
        if feat in panel.columns:
            panel[zfeat] = panel.groupby("date")[feat].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-9)
            )
        else:
            panel[zfeat] = 0.0
    return panel


def _add_labels(panel, horizon_key="21d"):
    """Label: 1 = top half outperformer, 0 = bottom half."""
    fwd_col = f"fwd_{horizon_key}"
    panel["label"] = panel.groupby("date")[fwd_col].transform(
        lambda x: (x.rank(pct=True) > 0.5).astype(int)
    )
    panel["rel_fwd"] = panel.groupby("date")[fwd_col].transform(
        lambda x: x - x.mean()
    )
    return panel


def _clean_features(panel):
    """Replace inf/nan in z-features."""
    for col in Z_FEATURES:
        if col in panel.columns:
            panel[col] = panel[col].replace([np.inf, -np.inf], 0).fillna(0)
    return panel


def _prepare_panel(panel, horizon_key="21d"):
    """Full prep: forward returns -> z-score -> labels -> clean."""
    panel = _add_forward_returns(panel, horizon_key)
    fwd_col = f"fwd_{horizon_key}"
    panel = panel.dropna(subset=[fwd_col])
    date_counts = panel.groupby("date")["symbol"].transform("count")
    panel = panel[date_counts >= MIN_STOCKS_PER_DATE].copy()
    panel = _zscore_per_date(panel)
    panel = _add_labels(panel, horizon_key)
    panel = _clean_features(panel)
    return panel


def _rank_ic(df):
    """Spearman rank correlation of prediction vs actual relative return."""
    ics = []
    for _, grp in df.groupby("date"):
        if len(grp) < MIN_STOCKS_PER_DATE:
            continue
        pr = grp["pred"].rank().values
        ar = grp["rel_fwd"].rank().values
        if pr.std() == 0 or ar.std() == 0:
            continue
        ics.append(np.corrcoef(pr, ar)[0, 1])
    return float(np.nanmean(ics)) if ics else float("nan")


def _long_short(df, q=0.2):
    """Top quintile minus bottom quintile actual return."""
    spreads = []
    for _, grp in df.groupby("date"):
        if len(grp) < MIN_STOCKS_PER_DATE:
            continue
        n = max(1, int(len(grp) * q))
        s = grp.sort_values("pred", ascending=False)
        top = s.head(n)["rel_fwd"].mean()
        bot = s.tail(n)["rel_fwd"].mean()
        spreads.append(top - bot)
    return float(np.nanmean(spreads)) if spreads else float("nan")


def _precision_at_k(df, k=10):
    """What fraction of our top-k picks actually outperformed?"""
    precs = []
    for _, grp in df.groupby("date"):
        if len(grp) < MIN_STOCKS_PER_DATE:
            continue
        s = grp.sort_values("pred", ascending=False).head(k)
        precs.append(s["label"].mean())
    return float(np.nanmean(precs)) if precs else float("nan")


def _profit_simulation(df, top_n=10):
    """Simulate buying top_n stocks equally weighted each rebalance."""
    daily_rets = []
    for date, grp in df.groupby("date"):
        if len(grp) < MIN_STOCKS_PER_DATE:
            continue
        picks = grp.sort_values("pred", ascending=False).head(top_n)
        avg_fwd = picks["rel_fwd"].mean()
        daily_rets.append(avg_fwd)
    if not daily_rets:
        return {}
    rets = pd.Series(daily_rets)
    log_rets = np.log1p(rets.clip(-0.99, None))
    cum_log = log_rets.cumsum()
    cum = np.exp(cum_log)
    annual_periods = 252 / 21
    n_years = len(rets) / annual_periods
    cagr = float((cum.iloc[-1]) ** (1 / max(n_years, 1)) - 1) if len(cum) > 0 else 0
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(annual_periods))
    max_dd = float((cum / cum.cummax() - 1).min())
    return {
        "cagr": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "win_rate": round(float((rets > 0).mean() * 100), 1),
        "n_periods": len(daily_rets),
        "n_years": round(n_years, 1),
    }


def _make_model():
    """Create XGBoost model with tuned hyperparams for 78-feature set."""
    return xgb.XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=0.3,
        reg_lambda=2.0,
        min_child_weight=15,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
        n_jobs=1,
    )


def train(horizon_key=DEFAULT_HORIZON):
    """Full training with walk-forward validation on 4yr/1yr blocks.

    Folds:
      Train 2006-2010 -> Test 2011
      Train 2012-2016 -> Test 2017
      Train 2018-2024 -> Test 2025

    Then trains final model on ALL available data for production.
    """
    print("=" * 60)
    print(f"  ML TRAINER v4 -- {len(RAW_FEATURES)} Features, Walk-Forward 4yr/1yr")
    print("=" * 60)
    t0 = time.time()

    ic_scores, ls_scores, prec_scores = [], [], []
    all_test_preds = []
    total_train_rows = 0
    total_test_rows = 0
    available_features = None

    # ── Walk-forward validation: load each fold separately ──
    print(f"\n[1/3] Walk-forward validation ({len(WALK_FORWARD_FOLDS)} folds)...\n")

    for fold_i, (tr_start, tr_end, te_start, te_end) in enumerate(WALK_FORWARD_FOLDS, 1):
        print(f"  --- Fold {fold_i}: Train {tr_start[:4]}-{tr_end[:4]} | "
              f"Test {te_start[:4]}-{te_end[:4]} ---")

        # Load train data
        print(f"    Loading train data ({tr_start[:4]}-{tr_end[:4]})...")
        train_panel = _load_panel_range(tr_start, tr_end)
        if train_panel.empty:
            print(f"    [SKIP] No train data for {tr_start}-{tr_end}")
            continue

        train_panel = _prepare_panel(train_panel, horizon_key)
        n_train = len(train_panel)
        n_train_sym = train_panel["symbol"].nunique()
        n_train_dates = train_panel["date"].nunique()
        print(f"    Train: {n_train:,} rows, {n_train_sym} stocks, {n_train_dates} dates")

        if n_train < 1000:
            print(f"    [SKIP] Too few training rows")
            del train_panel; gc.collect()
            continue

        available_features = [f for f in Z_FEATURES if f in train_panel.columns]

        # Train model for this fold
        model = _make_model()
        model.fit(train_panel[available_features], train_panel["label"])

        # Free train memory before loading test
        del train_panel; gc.collect()

        # Load test data
        print(f"    Loading test data ({te_start[:4]}-{te_end[:4]})...")
        test_panel = _load_panel_range(te_start, te_end)
        if test_panel.empty:
            print(f"    [SKIP] No test data for {te_start}-{te_end}")
            del model; gc.collect()
            continue

        test_panel = _prepare_panel(test_panel, horizon_key)
        n_test = len(test_panel)
        n_test_dates = test_panel["date"].nunique()
        print(f"    Test:  {n_test:,} rows, {test_panel['symbol'].nunique()} stocks, "
              f"{n_test_dates} dates")

        if n_test < 200:
            print(f"    [SKIP] Too few test rows")
            del test_panel, model; gc.collect()
            continue

        # Predict
        test_panel["pred"] = model.predict_proba(test_panel[available_features])[:, 1]

        ic = _rank_ic(test_panel)
        ls = _long_short(test_panel)
        prec = _precision_at_k(test_panel, k=10)
        sim = _profit_simulation(test_panel, top_n=10)

        ic_scores.append(ic)
        ls_scores.append(ls)
        prec_scores.append(prec)
        all_test_preds.append(test_panel[["date", "symbol", "pred", "rel_fwd", "label"]].copy())
        total_train_rows += n_train
        total_test_rows += n_test

        print(f"    Rank IC      = {ic:+.4f}")
        print(f"    L/S spread   = {ls:+.4%}")
        print(f"    Precision@10 = {prec:.1%}")
        if sim:
            print(f"    CAGR={sim.get('cagr',0):.1f}%  Sharpe={sim.get('sharpe',0):.3f}  "
                  f"MaxDD={sim.get('max_drawdown',0):.1f}%  WinRate={sim.get('win_rate',0):.0f}%")
        print()

        del test_panel, model; gc.collect()

    # ── Summary ──
    mean_ic = float(np.nanmean(ic_scores)) if ic_scores else 0
    mean_ls = float(np.nanmean(ls_scores)) if ls_scores else 0
    mean_prec = float(np.nanmean(prec_scores)) if prec_scores else 0

    sim_results = {}
    if all_test_preds:
        all_test = pd.concat(all_test_preds, ignore_index=True)
        sim_results = _profit_simulation(all_test, top_n=10)
        del all_test; gc.collect()

    print(f"  {'=' * 50}")
    print(f"  VALIDATION SUMMARY ({len(ic_scores)} folds)")
    print(f"  {'=' * 50}")
    print(f"  Mean Rank IC        : {mean_ic:+.4f}")
    print(f"  Mean L/S spread     : {mean_ls:+.4%}  (per {horizon_key})")
    print(f"  Mean Precision@10   : {mean_prec:.1%}")
    if sim_results:
        print(f"  Combined CAGR       : {sim_results.get('cagr', 0):.1f}%")
        print(f"  Combined Sharpe     : {sim_results.get('sharpe', 0):.3f}")
        print(f"  Max Drawdown        : {sim_results.get('max_drawdown', 0):.1f}%")
        print(f"  Win Rate            : {sim_results.get('win_rate', 0):.1f}%")
    print(f"  Total train rows    : {total_train_rows:,}")
    print(f"  Total test rows     : {total_test_rows:,}")

    edge = "STRONG" if mean_ic > 0.05 else "EDGE" if mean_ic > 0.02 else "WEAK"
    print(f"  Verdict             : {edge}")

    # ── Train final production model on most recent block ──
    # Use 2018-2025 for final model (recent market regime most relevant)
    print(f"\n[2/3] Training FINAL model on 2018-2025...")
    final_panel = _load_panel_range("2018-01-01", "2025-12-31")
    if final_panel.empty:
        print("  ERROR: No data for 2018-2025")
        return {}

    final_panel = _prepare_panel(final_panel, horizon_key)
    n_final = len(final_panel)
    n_final_sym = final_panel["symbol"].nunique()
    print(f"  Final training: {n_final:,} rows, {n_final_sym} stocks, "
          f"{final_panel['date'].nunique()} dates")

    available_features = [f for f in Z_FEATURES if f in final_panel.columns]

    final_model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=0.3,
        reg_lambda=2.0,
        min_child_weight=15,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    final_model.fit(final_panel[available_features], final_panel["label"])

    # Feature importance
    imp = pd.Series(final_model.feature_importances_, index=available_features)
    imp = imp.sort_values(ascending=False)
    print("\n  Top 15 features:")
    for feat, val in imp.head(15).items():
        bar = "#" * int(val * 60)
        print(f"    {feat:<30} {bar} {val:.3f}")

    # Z-score stats for inference
    zscore_stats = {}
    latest_date = final_panel["date"].max()
    latest_data = final_panel[final_panel["date"] == latest_date]
    for feat in RAW_FEATURES:
        if feat in latest_data.columns:
            zscore_stats[feat] = {
                "mean": float(latest_data[feat].mean()),
                "std": float(latest_data[feat].std()),
            }

    # ── Save ──
    print(f"\n[3/3] Saving model...")
    model_id = f"v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_path = os.path.join(MODEL_DIR, f"{model_id}.pkl")
    meta_path = os.path.join(MODEL_DIR, f"{model_id}_meta.json")
    latest_path = os.path.join(MODEL_DIR, "latest.pkl")
    latest_meta_path = os.path.join(MODEL_DIR, "latest_meta.json")

    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    meta = {
        "model_id": model_id,
        "type": "cross_sectional_ranker_v3",
        "n_features": len(available_features),
        "feature_categories": {
            "technical": len([f for f in available_features if not any(
                f.endswith(x) for x in ["_pe_trailing_z", "_roe_z", "_roa_z",
                "_profit_margin_z", "_earnings_growth_z"])]),
            "fundamental": 20,
            "pattern": 3,
        },
        "horizon": horizon_key,
        "horizon_days": HORIZONS[horizon_key],
        "features": available_features,
        "raw_features": RAW_FEATURES,
        "n_symbols": n_final_sym,
        "n_dates": int(final_panel["date"].nunique()),
        "n_samples": n_final,
        "walk_forward": {
            "folds": [
                {"train": f"{s[:4]}-{e[:4]}", "test": f"{ts[:4]}-{te[:4]}"}
                for s, e, ts, te in WALK_FORWARD_FOLDS
            ],
            "fold_ics": [round(x, 4) for x in ic_scores],
            "fold_ls": [round(x, 4) for x in ls_scores],
            "fold_prec": [round(x, 4) for x in prec_scores],
        },
        "validation": {
            "mean_rank_ic": round(mean_ic, 4),
            "mean_ls_spread": round(mean_ls, 4),
            "mean_precision_at_10": round(mean_prec, 4),
            "n_folds": len(ic_scores),
        },
        "backtest": sim_results,
        "feature_importance": {k: round(v, 4) for k, v in imp.head(25).items()},
        "zscore_stats": zscore_stats,
        "edge": edge,
        "trained_at": datetime.now().isoformat(),
        "training_time_sec": round(time.time() - t0, 1),
        "final_train_period": "2018-2025",
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    import shutil
    shutil.copy2(model_path, latest_path)
    shutil.copy2(meta_path, latest_meta_path)

    del final_panel, final_model; gc.collect()

    print(f"\n  [OK] Model saved -> {model_path}")
    print(f"       Also -> {latest_path}")
    print(f"       {len(available_features)} features, {n_final:,} training rows")
    print(f"       Training time: {time.time() - t0:.0f}s")

    return meta


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    result = train()
    if result:
        print(f"\nDone. Edge: {result.get('edge')}, "
              f"Rank IC: {result.get('validation', {}).get('mean_rank_ic')}")

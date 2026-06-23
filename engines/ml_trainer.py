"""
ML Trainer — train once, predict forever.

Builds a cross-sectional ranking model on ALL historical data from the
feature store. The model learns which stocks will outperform the universe
over the next N days, using 35 features across momentum, volatility,
volume flow, trend, and mean-reversion.

Training pipeline:
    1. Load panel from feature store (symbol × date × features)
    2. Compute forward returns as targets
    3. Cross-sectional z-score normalization per date
    4. Walk-forward validation (no future leak)
    5. Train final model on all data
    6. Save to model registry

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

# Features used for training (must match feature_store.FEATURE_COLS minus cross-sectional)
RAW_FEATURES = [
    "ret_1d", "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "rsi_14", "bb_pctb", "dist_ma20", "dist_ma50", "dist_ma200",
    "above_20ema", "above_50ema", "above_200ma", "ma_slope_20", "macd_hist",
    "trend_slope_10",
    "vol_10d", "vol_21d", "atr_ratio", "intraday_range",
    "rel_volume", "vol_trend", "mfi_14", "ad_momentum", "up_down_vol_ratio",
    "vol_price_confirm",
    "sr_proximity", "fib_level", "range_pos_20",
    "sharpe_60d", "stretch_atr",
]
Z_FEATURES = [f + "_z" for f in RAW_FEATURES]

HORIZONS = {
    "5d": 5,
    "10d": 10,
    "21d": 21,
}
MIN_STOCKS_PER_DATE = 20


def _load_panel():
    """Load training panel from feature store."""
    from engines.feature_store import load_training_panel
    panel = load_training_panel()
    if panel.empty:
        raise ValueError("Feature store is empty — run: python -m engines.feature_store")
    return panel


def _add_forward_returns(panel, prices_by_sym):
    """Add forward return columns using price data from feature store."""
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
    """Cross-sectional z-score normalization: each feature normalized across
    all stocks on the same date. This removes market-wide moves and makes
    features comparable across different market conditions."""
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
    """Label: relative rank within each date's cross-section.
    1 = top half (outperformer), 0 = bottom half."""
    fwd_col = f"fwd_{horizon_key}"
    panel["label"] = panel.groupby("date")[fwd_col].transform(
        lambda x: (x.rank(pct=True) > 0.5).astype(int)
    )
    panel["rel_fwd"] = panel.groupby("date")[fwd_col].transform(
        lambda x: x - x.mean()
    )
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


def _profit_simulation(df, top_n=10, capital=100000):
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
    cum = (1 + rets).cumprod()
    total_ret = float(cum.iloc[-1] - 1) if len(cum) > 0 else 0
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252 / 21))
    max_dd = float((cum / cum.cummax() - 1).min())
    return {
        "total_return": round(total_ret * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "win_rate": round(float((rets > 0).mean() * 100), 1),
        "n_periods": len(daily_rets),
    }


def train(horizon_key="21d", n_splits=5):
    """Full training pipeline. Returns metrics dict and saves model."""
    print("=" * 60)
    print("  ML TRAINER — Cross-Sectional Ranking Model v2")
    print("=" * 60)

    t0 = time.time()

    # 1. Load panel
    print("\n[1/6] Loading feature store...")
    panel = _load_panel()
    n_symbols = panel["symbol"].nunique()
    n_dates = panel["date"].nunique()
    print(f"  Loaded: {len(panel):,} rows, {n_symbols} stocks, {n_dates} dates")

    # 2. Forward returns
    print(f"\n[2/6] Computing forward returns ({horizon_key})...")
    panel = _add_forward_returns(panel, None)
    fwd_col = f"fwd_{horizon_key}"
    panel = panel.dropna(subset=[fwd_col])

    # Filter dates with enough stocks
    date_counts = panel.groupby("date")["symbol"].transform("count")
    panel = panel[date_counts >= MIN_STOCKS_PER_DATE].copy()
    print(f"  After filters: {len(panel):,} rows, {panel['date'].nunique()} dates")

    # 3. Z-score normalization
    print("\n[3/6] Cross-sectional z-score normalization...")
    panel = _zscore_per_date(panel)
    panel = _add_labels(panel, horizon_key)

    # Replace infinities and fill NaN
    for col in Z_FEATURES:
        if col in panel.columns:
            panel[col] = panel[col].replace([np.inf, -np.inf], 0).fillna(0)

    # 4. Walk-forward validation
    print(f"\n[4/6] Walk-forward validation ({n_splits} folds)...")
    dates = np.sort(panel["date"].unique())
    fold_size = len(dates) // (n_splits + 1)
    embargo = HORIZONS[horizon_key]

    ic_scores, ls_scores, prec_scores = [], [], []
    all_test_preds = []

    for k in range(1, n_splits + 1):
        train_end = fold_size * k
        test_start = train_end + embargo
        test_end = min(fold_size * (k + 1), len(dates))
        if test_start >= test_end:
            continue

        train_dates = set(dates[:train_end])
        test_dates = set(dates[test_start:test_end])

        tr = panel[panel["date"].isin(train_dates)]
        te = panel[panel["date"].isin(test_dates)].copy()
        if len(tr) < 1000 or len(te) < 200:
            continue

        available_features = [f for f in Z_FEATURES if f in panel.columns]

        model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=10,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
            n_jobs=1,
        )
        model.fit(tr[available_features], tr["label"])
        te["pred"] = model.predict_proba(te[available_features])[:, 1]

        ic = _rank_ic(te)
        ls = _long_short(te)
        prec = _precision_at_k(te, k=10)
        ic_scores.append(ic)
        ls_scores.append(ls)
        prec_scores.append(prec)
        all_test_preds.append(te)

        print(f"  Fold {k}: Rank IC = {ic:+.4f} | "
              f"L/S spread = {ls:+.4%} | "
              f"Precision@10 = {prec:.1%}")

    mean_ic = float(np.nanmean(ic_scores))
    mean_ls = float(np.nanmean(ls_scores))
    mean_prec = float(np.nanmean(prec_scores))

    # Profit simulation on all test data
    sim_results = {}
    if all_test_preds:
        all_test = pd.concat(all_test_preds, ignore_index=True)
        sim_results = _profit_simulation(all_test, top_n=10)

    print(f"\n  {'=' * 50}")
    print(f"  Mean Rank IC        : {mean_ic:+.4f}")
    print(f"  Mean L/S spread     : {mean_ls:+.4%}  (per {horizon_key})")
    print(f"  Mean Precision@10   : {mean_prec:.1%}")
    if sim_results:
        print(f"  Backtest Return     : {sim_results.get('total_return', 0):.1f}%")
        print(f"  Backtest Sharpe     : {sim_results.get('sharpe', 0):.3f}")
        print(f"  Max Drawdown        : {sim_results.get('max_drawdown', 0):.1f}%")
        print(f"  Win Rate            : {sim_results.get('win_rate', 0):.1f}%")

    edge = "STRONG" if mean_ic > 0.05 else "EDGE" if mean_ic > 0.02 else "WEAK"
    print(f"  Verdict             : {edge}")

    # 5. Train final model on ALL data
    print(f"\n[5/6] Training final model on all {len(panel):,} rows...")
    available_features = [f for f in Z_FEATURES if f in panel.columns]

    final_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=10,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    final_model.fit(panel[available_features], panel["label"])

    # Feature importance
    imp = pd.Series(final_model.feature_importances_, index=available_features)
    imp = imp.sort_values(ascending=False)
    print("\n  Top features:")
    for feat, val in imp.head(10).items():
        bar = "█" * int(val * 60)
        print(f"    {feat:<24} {bar} {val:.3f}")

    # 6. Save model + metadata
    print("\n[6/6] Saving model...")
    model_id = f"v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_path = os.path.join(MODEL_DIR, f"{model_id}.pkl")
    meta_path = os.path.join(MODEL_DIR, f"{model_id}_meta.json")
    latest_path = os.path.join(MODEL_DIR, "latest.pkl")
    latest_meta_path = os.path.join(MODEL_DIR, "latest_meta.json")

    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    # Also compute and save the z-score statistics for inference
    zscore_stats = {}
    for feat in RAW_FEATURES:
        if feat in panel.columns:
            # Use the most recent date's cross-sectional stats as reference
            latest_date = panel["date"].max()
            latest_data = panel[panel["date"] == latest_date]
            zscore_stats[feat] = {
                "mean": float(latest_data[feat].mean()),
                "std": float(latest_data[feat].std()),
            }

    meta = {
        "model_id": model_id,
        "type": "cross_sectional_ranker_v2",
        "horizon": horizon_key,
        "horizon_days": HORIZONS[horizon_key],
        "features": available_features,
        "raw_features": RAW_FEATURES,
        "n_symbols": n_symbols,
        "n_dates": int(panel["date"].nunique()),
        "n_samples": len(panel),
        "validation": {
            "mean_rank_ic": round(mean_ic, 4),
            "mean_ls_spread": round(mean_ls, 4),
            "mean_precision_at_10": round(mean_prec, 4),
            "n_folds": n_splits,
        },
        "backtest": sim_results,
        "feature_importance": {k: round(v, 4) for k, v in imp.head(20).items()},
        "zscore_stats": zscore_stats,
        "edge": edge,
        "trained_at": datetime.now().isoformat(),
        "training_time_sec": round(time.time() - t0, 1),
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Copy as "latest"
    import shutil
    shutil.copy2(model_path, latest_path)
    shutil.copy2(meta_path, latest_meta_path)

    print(f"\n  ✅ Model saved → {model_path}")
    print(f"     Also → {latest_path}")
    print(f"     Training time: {time.time() - t0:.0f}s")

    return meta


def train_all_horizons():
    """Train models for 5d, 10d, and 21d horizons."""
    results = {}
    for h in HORIZONS:
        print(f"\n{'#' * 60}")
        print(f"  Training {h} horizon model")
        print(f"{'#' * 60}")
        results[h] = train(horizon_key=h)
        gc.collect()
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    if "--all-horizons" in sys.argv:
        train_all_horizons()
    else:
        horizon = "21d"
        for arg in sys.argv[1:]:
            if arg in HORIZONS:
                horizon = arg
        train(horizon_key=horizon)

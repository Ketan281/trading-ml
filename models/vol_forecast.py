"""
Volatility forecasting model -- NIFTY / BANKNIFTY (next-week realised vol).

Forecasts the MAGNITUDE of annualised realised volatility over next HORIZON
days. Uses real intermarket features (crude, DXY, VIX, India VIX, gold,
USD/INR, FII/DII) alongside vol-specific features. Era-based walk-forward
validation across 4 market regimes. Compared against EWMA baseline.
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

from models.index_options_model import load_index, _load_intermarket, ERA_WINDOWS

HORIZON = 5
TRADING_DAYS = 252
EWMA_LAMBDA = 0.94
MODELS_DIR = os.path.dirname(__file__)
INTEL_DIR = os.path.join(ROOT, "data", "market_intel")

BASE_FEATURES = ["rv_5", "rv_10", "rv_20", "rv_60", "vov", "atr_pct",
                  "parkinson", "ret_abs_5", "dow", "vol_ratio",
                  "bb_width", "range_pct", "skew_20"]


def _build(df):
    c, h, l = df["Close"], df["High"], df["Low"]
    r = c.pct_change()
    f = pd.DataFrame(index=df.index)
    for n in (5, 10, 20, 60):
        f[f"rv_{n}"] = r.rolling(n).std() * np.sqrt(TRADING_DAYS)
    f["vov"] = f["rv_10"].rolling(20).std()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    f["atr_pct"] = tr.rolling(14).mean() / c
    f["parkinson"] = np.sqrt((np.log(h / l) ** 2).rolling(10).mean()
                             / (4 * np.log(2))) * np.sqrt(TRADING_DAYS)
    f["ret_abs_5"] = r.abs().rolling(5).mean()
    f["dow"] = df.index.dayofweek
    f["vol_ratio"] = f["rv_5"] / (f["rv_20"] + 1e-9)
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_width"] = (4 * sd20) / ma20
    f["range_pct"] = (h - l) / c
    f["skew_20"] = r.rolling(20).skew()

    # Target: realised vol over the NEXT horizon days (annualised).
    f["target"] = r.shift(-HORIZON).rolling(HORIZON).std() * np.sqrt(TRADING_DAYS)
    # EWMA baseline
    var = r.ewm(alpha=1 - EWMA_LAMBDA).var()
    f["ewma_vol"] = np.sqrt(var) * np.sqrt(TRADING_DAYS)

    # Join real intermarket features
    im = _load_intermarket()
    if im is not None and len(im) > 0:
        for col in im.columns:
            if col not in f.columns:
                aligned = im[col].reindex(df.index, method="ffill")
                if aligned.notna().sum() > len(df) * 0.3:
                    f[col] = aligned

    return f


def _get_vol_features(data):
    cols = [c for c in BASE_FEATURES if c in data.columns]
    im = _load_intermarket()
    if im is not None:
        for c in im.columns:
            if c in data.columns:
                cols.append(c)
    return cols


def evaluate(symbol="NIFTY"):
    df = load_index(symbol)
    if df is None or len(df) < 600:
        return None
    data = _build(df)
    feat_cols = _get_vol_features(data)
    data = data.dropna(subset=feat_cols + ["target", "ewma_vol"])
    dates = data.index
    n = len(data)

    ml_err, ew_err, ml_corr = [], [], []
    era_results = []

    # Era-based walk-forward
    for era in ERA_WINDOWS:
        train_mask = (dates.year >= era["train_start"]) & (dates.year <= era["train_end"])
        test_mask  = dates.year == era["test_year"]
        tr_idx, te_idx = np.array(train_mask), np.array(test_mask)

        tr, te = data[tr_idx], data[te_idx]
        if len(tr) < 100 or len(te) < 20:
            print(f"     {era['name']}: skipped (train={len(tr)}, test={len(te)})")
            continue

        m = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.04,
                             subsample=0.8, colsample_bytree=0.8, random_state=42,
                             verbosity=0)
        m.fit(tr[feat_cols], tr["target"])
        pred = m.predict(te[feat_cols])

        ml_mae = float(np.mean(np.abs(pred - te["target"])))
        ew_mae = float(np.mean(np.abs(te["ewma_vol"] - te["target"])))
        corr = float(np.corrcoef(pred, te["target"])[0, 1]) if te["target"].std() > 0 else 0

        ml_err.append(ml_mae)
        ew_err.append(ew_mae)
        ml_corr.append(corr)

        era_results.append({
            "era": era["name"],
            "train": f"{era['train_start']}-{era['train_end']}",
            "test": era["test_year"],
            "ml_mae": round(ml_mae, 4), "ewma_mae": round(ew_mae, 4),
            "corr": round(corr, 3),
            "ml_beats_ewma": ml_mae < ew_mae,
            "train_rows": len(tr), "test_rows": len(te),
        })
        tag = "[OK] ML wins" if ml_mae < ew_mae else "[--] EWMA wins"
        print(f"     {era['name']}: Train {era['train_start']}-{era['train_end']} "
              f"({len(tr)}) -> Test {era['test_year']} ({len(te)}) "
              f"| ML MAE {ml_mae:.4f} vs EWMA {ew_mae:.4f} {tag} | corr {corr:.3f}")

    if not ml_err:
        return None

    return {
        "symbol": symbol,
        "n_features": len(feat_cols),
        "ml_mae": round(float(np.mean(ml_err)), 4),
        "ewma_mae": round(float(np.mean(ew_err)), 4),
        "ml_corr": round(float(np.nanmean(ml_corr)), 3),
        "ml_beats_ewma": bool(np.mean(ml_err) < np.mean(ew_err)),
        "era_results": era_results,
    }


def forecast(symbol="NIFTY"):
    """Train on all history, return the live next-horizon vol forecast."""
    df = load_index(symbol)
    data = _build(df)
    feat_cols = _get_vol_features(data)
    train = data.dropna(subset=feat_cols + ["target"])
    print(f"     Training vol forecast on {len(train)} rows x {len(feat_cols)} features")

    m = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.04,
                         subsample=0.8, colsample_bytree=0.8, random_state=42,
                         verbosity=0)
    m.fit(train[feat_cols], train["target"])

    # Save model + features
    with open(os.path.join(MODELS_DIR, f"opt_{symbol}_vol_forecast.pkl"), "wb") as f:
        pickle.dump(m, f)
    with open(os.path.join(MODELS_DIR, f"opt_{symbol}_vol_forecast_features.json"), "w") as f:
        json.dump(feat_cols, f, indent=2)

    # Feature importance
    imp = dict(zip(feat_cols, m.feature_importances_))
    top5 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"     Top 5 vol-forecast features:")
    for name, score in top5:
        bar = "#" * max(1, int(score * 50))
        print(f"       {name:20s} {bar} {score:.3f}")

    latest = data.dropna(subset=feat_cols).iloc[[-1]]
    ml = float(m.predict(latest[feat_cols])[0])
    ewma = float(latest["ewma_vol"].iloc[0])
    cur = float(latest["rv_20"].iloc[0])
    return {"symbol": symbol, "as_of": str(df.index[-1].date()),
            "n_features": len(feat_cols),
            "current_rv20": round(cur * 100, 1),
            "forecast_vol_ml": round(ml * 100, 1),
            "forecast_vol_ewma": round(ewma * 100, 1),
            "direction": "expansion" if ml > cur else "contraction"}


if __name__ == "__main__":
    for sym in (sys.argv[1:] or ["NIFTY", "BANKNIFTY"]):
        print("=" * 60)
        print(f"  VOLATILITY FORECAST -- {sym}")
        print(f"  Real intermarket + era-based walk-forward")
        print("=" * 60)
        ev = evaluate(sym); fc = forecast(sym)
        if ev:
            tag = "[OK] ML beats EWMA" if ev["ml_beats_ewma"] else "[--] EWMA wins (vol is persistent)"
            print(f"  Walk-forward MAE : ML {ev['ml_mae']} vs EWMA {ev['ewma_mae']}  {tag}")
            print(f"  Forecast corr    : {ev['ml_corr']}")
            print(f"  Features used    : {ev['n_features']}")
        print(f"  Current RV(20)   : {fc['current_rv20']}%")
        print(f"  Next-{HORIZON}d forecast: ML {fc['forecast_vol_ml']}% | "
              f"EWMA {fc['forecast_vol_ewma']}%  -> {fc['direction'].upper()}")
        print()

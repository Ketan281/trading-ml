"""
Volatility forecasting model — NIFTY / BANKNIFTY (next-week realised vol).

The index_options_model predicts vol EXPANSION vs CONTRACTION (a direction).
This forecasts the MAGNITUDE: the annualised realised volatility over the next
HORIZON days. That number drives expected-range, straddle pricing, and how much
premium is "fair" — so a calibrated vol forecast is directly tradable.

Method (honest):
  • Features: current realised vol over several windows, vol-of-vol, ATR%,
    Parkinson (high-low) vol, return magnitude, day-of-week.
  • Model: XGBoost regressor, walk-forward validated, compared against the
    standard EWMA (RiskMetrics λ=0.94) baseline — vol is highly persistent, so
    EWMA is a strong baseline and we only claim value if we BEAT it.
  • Reports MAE and correlation vs realised, ML vs EWMA, and the live forecast.
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.index_options_model import load_index

HORIZON = 5
TRADING_DAYS = 252
EWMA_LAMBDA = 0.94

FEATURES = ["rv_5", "rv_10", "rv_20", "rv_60", "vov", "atr_pct",
            "parkinson", "ret_abs_5", "dow"]


def _build(df):
    c, h, l = df["Close"], df["High"], df["Low"]
    r = c.pct_change()
    f = pd.DataFrame(index=df.index)
    for n in (5, 10, 20, 60):
        f[f"rv_{n}"] = r.rolling(n).std() * np.sqrt(TRADING_DAYS)
    f["vov"] = f["rv_10"].rolling(20).std()                 # vol of vol
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    f["atr_pct"] = tr.rolling(14).mean() / c
    # Parkinson high-low range volatility
    f["parkinson"] = np.sqrt((np.log(h / l) ** 2).rolling(10).mean()
                             / (4 * np.log(2))) * np.sqrt(TRADING_DAYS)
    f["ret_abs_5"] = r.abs().rolling(5).mean()
    f["dow"] = df.index.dayofweek
    # Target: realised vol over the NEXT horizon days (annualised).
    f["target"] = r.shift(-HORIZON).rolling(HORIZON).std() * np.sqrt(TRADING_DAYS)
    # EWMA baseline (RiskMetrics) — current EWMA vol as the naive forecast.
    var = r.ewm(alpha=1 - EWMA_LAMBDA).var()
    f["ewma_vol"] = np.sqrt(var) * np.sqrt(TRADING_DAYS)
    return f


def evaluate(symbol="NIFTY", n_splits=6):
    df = load_index(symbol)
    if df is None or len(df) < 600:
        return None
    data = _build(df).dropna(subset=FEATURES + ["target", "ewma_vol"])
    n = len(data); fold = n // (n_splits + 1)
    ml_err, ew_err, ml_corr = [], [], []
    for k in range(1, n_splits + 1):
        tr_end = fold * k; te_s = tr_end + HORIZON; te_e = min(fold * (k + 1), n)
        if te_s >= te_e:
            continue
        tr, te = data.iloc[:tr_end], data.iloc[te_s:te_e]
        if len(tr) < 200 or len(te) < 40:
            continue
        m = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.04,
                             subsample=0.8, colsample_bytree=0.8, random_state=42,
                             verbosity=0)
        m.fit(tr[FEATURES], tr["target"])
        pred = m.predict(te[FEATURES])
        ml_err.append(np.mean(np.abs(pred - te["target"])))
        ew_err.append(np.mean(np.abs(te["ewma_vol"] - te["target"])))
        if te["target"].std() > 0:
            ml_corr.append(np.corrcoef(pred, te["target"])[0, 1])
    return {"symbol": symbol, "ml_mae": round(float(np.mean(ml_err)), 4),
            "ewma_mae": round(float(np.mean(ew_err)), 4),
            "ml_corr": round(float(np.nanmean(ml_corr)), 3),
            "ml_beats_ewma": bool(np.mean(ml_err) < np.mean(ew_err))}


def forecast(symbol="NIFTY"):
    """Train on all history, return the live next-horizon vol forecast (ML +
    EWMA baseline)."""
    df = load_index(symbol)
    data = _build(df)
    train = data.dropna(subset=FEATURES + ["target"])
    m = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.04,
                         subsample=0.8, colsample_bytree=0.8, random_state=42,
                         verbosity=0)
    m.fit(train[FEATURES], train["target"])
    latest = data.dropna(subset=FEATURES).iloc[[-1]]
    ml = float(m.predict(latest[FEATURES])[0])
    ewma = float(latest["ewma_vol"].iloc[0])
    cur = float(latest["rv_20"].iloc[0])
    return {"symbol": symbol, "as_of": str(df.index[-1].date()),
            "current_rv20": round(cur * 100, 1),
            "forecast_vol_ml": round(ml * 100, 1),
            "forecast_vol_ewma": round(ewma * 100, 1),
            "direction": "expansion" if ml > cur else "contraction"}


if __name__ == "__main__":
    for sym in (sys.argv[1:] or ["NIFTY", "BANKNIFTY"]):
        ev = evaluate(sym); fc = forecast(sym)
        print("=" * 60)
        print(f"  VOLATILITY FORECAST — {sym}")
        print("=" * 60)
        if ev:
            tag = "✓ beats EWMA" if ev["ml_beats_ewma"] else "✗ EWMA wins (vol is persistent)"
            print(f"  Walk-forward MAE : ML {ev['ml_mae']} vs EWMA {ev['ewma_mae']}  {tag}")
            print(f"  Forecast corr    : {ev['ml_corr']}")
        print(f"  Current RV(20)   : {fc['current_rv20']}%")
        print(f"  Next-{HORIZON}d forecast: ML {fc['forecast_vol_ml']}% | "
              f"EWMA {fc['forecast_vol_ewma']}%  → {fc['direction'].upper()}")

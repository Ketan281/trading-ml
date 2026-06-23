"""
Index Options ML Trainer — predict NIFTY/BANKNIFTY intraday direction.

Time-series model (not cross-sectional like the stock models).
Trained on 18 years of daily data with market breadth from 473 stocks.

Features:
  - Index technicals (RSI, MACD, BB, Supertrend, gap, range patterns)
  - Market breadth (% stocks up, advance/decline, % above EMAs)
  - Intermarket (VIX, India VIX, FII/DII, crude, gold, DXY, S&P500)
  - Option chain signals (PCR, IV, max pain — where available)
  - Day-of-week, expiry proximity

Label: binary — bullish day (close > open) or bearish day.

Walk-forward:
  Fold 1: 2007-2012 → Test 2013
  Fold 2: 2013-2018 → Test 2019
  Fold 3: 2019-2024 → Test 2025
  Final:  2019-2025 → production

Run: python -m engines.index_options_trainer
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

log = logging.getLogger("index_options_trainer")

MODEL_DIR = os.path.join(ROOT, "models", "intraday")
HIST_DIR = os.path.join(ROOT, "data", "historical")
OC_DIR = os.path.join(ROOT, "data", "option_chain", "agg")
os.makedirs(MODEL_DIR, exist_ok=True)

INDICES = ["NIFTY", "BANKNIFTY"]

WALK_FORWARD_FOLDS = [
    ("2007-01-01", "2012-12-31", "2013-01-01", "2013-12-31"),
    ("2013-01-01", "2018-12-31", "2019-01-01", "2019-12-31"),
    ("2019-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
]

FEATURE_COLS = [
    # Index technicals (previous day, no lookahead)
    "prev_intra_ret", "prev_range", "prev_close_loc", "prev_gap",
    "avg_intra_ret_3d", "avg_intra_ret_5d",
    "avg_range_5d", "avg_range_10d",
    "intra_streak",
    "gap_fill_rate_10d",

    # Overnight / gap (known at open)
    "overnight_gap", "overnight_vs_avg",

    # Momentum (shifted 1d)
    "rsi_14", "rsi_slope_5d",
    "macd_hist", "macd_cross",
    "stoch_k", "stoch_d",
    "bb_pctb", "bb_width",
    "supertrend_signal",

    # Trend (shifted 1d)
    "dist_ema20", "dist_ema50",
    "ema20_slope", "ema50_slope",
    "ema_20_50_cross",
    "adx_14", "trend_slope_10",
    "ret_5d", "ret_10d", "ret_21d",

    # Volatility (shifted 1d)
    "vol_10d", "vol_21d",
    "atr_ratio", "atr_expansion",
    "range_expansion",

    # Market breadth (from all stocks — the secret sauce for index prediction)
    "breadth_adv_pct",         # % of stocks with positive intraday return
    "breadth_up_vol_pct",      # % of volume in advancing stocks
    "breadth_above_ema20_pct", # % of stocks above their 20 EMA
    "breadth_above_ema50_pct", # % above 50 EMA
    "breadth_above_200ma_pct", # % above 200 MA
    "breadth_new_high_pct",    # % at 52-week high
    "breadth_rsi_above50_pct", # % with RSI > 50
    "breadth_adv_pct_5d_avg",  # 5-day avg breadth

    # Intermarket (global cues)
    "vix_level", "vix_change_1d", "vix_regime",
    "ivix_level",
    "crude_ret_5d", "gold_ret_5d",
    "dxy_ret_5d", "usdinr_ret_5d",
    "sp500_ret_5d", "us10y_level",
    "fii_net_5d",

    # Option chain signals (only recent, filled with 0 for historical)
    "pcr_oi", "pcr_vol",
    "atm_iv", "iv_change_1d",
    "max_pain_dist_pct",
    "oi_buildup_signal",  # net OI change direction

    # Calendar
    "day_of_week",
    "days_to_expiry",
    "is_expiry_week",
]


def _load_index(symbol):
    path = os.path.join(HIST_DIR, f"{symbol}.csv")
    df = pd.read_csv(path, parse_dates=["Date"],
                     usecols=["Date", "Open", "High", "Low", "Close", "Volume"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def _compute_breadth_features():
    """Compute daily market breadth from all stock CSVs.
    This is what makes index prediction work — breadth leads indices."""
    stock_csvs = [f for f in os.listdir(HIST_DIR) if f.endswith(".csv")
                  and f[:-4] not in {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"}]

    all_daily = []
    for csv_file in stock_csvs:
        try:
            s = pd.read_csv(os.path.join(HIST_DIR, csv_file), parse_dates=["Date"],
                            usecols=["Date", "Open", "High", "Low", "Close", "Volume"])
            s = s.sort_values("Date").reset_index(drop=True)
            sym = csv_file[:-4]
            s["symbol"] = sym
            s["intra_ret"] = (s["Close"] - s["Open"]) / s["Open"].replace(0, np.nan)
            s["ema20"] = s["Close"].ewm(span=20, adjust=False).mean()
            s["ema50"] = s["Close"].ewm(span=50, adjust=False).mean()
            s["ma200"] = s["Close"].rolling(200).mean()
            s["high_52w"] = s["High"].rolling(252).max()
            delta = s["Close"].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            s["rsi_14"] = 100 - 100 / (1 + rs)
            all_daily.append(s[["Date", "symbol", "intra_ret", "Close", "ema20", "ema50",
                                "ma200", "high_52w", "rsi_14", "Volume"]].copy())
        except Exception:
            continue

    if not all_daily:
        return pd.DataFrame()

    panel = pd.concat(all_daily, ignore_index=True)

    breadth = panel.groupby("Date").apply(lambda g: pd.Series({
        "breadth_adv_pct": (g["intra_ret"] > 0).mean(),
        "breadth_up_vol_pct": g.loc[g["intra_ret"] > 0, "Volume"].sum() / max(g["Volume"].sum(), 1),
        "breadth_above_ema20_pct": (g["Close"] > g["ema20"]).mean(),
        "breadth_above_ema50_pct": (g["Close"] > g["ema50"]).mean(),
        "breadth_above_200ma_pct": (g["Close"] > g["ma200"]).mean(),
        "breadth_new_high_pct": (g["Close"] >= g["high_52w"] * 0.98).mean(),
        "breadth_rsi_above50_pct": (g["rsi_14"] > 50).mean(),
    }), include_groups=False).reset_index()

    breadth["breadth_adv_pct_5d_avg"] = breadth["breadth_adv_pct"].rolling(5).mean()
    return breadth


def _load_option_chain_daily(symbol):
    """Load daily option chain summary (last snapshot per day)."""
    path = os.path.join(OC_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        oc = pd.read_csv(path, parse_dates=["timestamp"])
        oc["Date"] = oc["timestamp"].dt.date
        daily = oc.groupby("Date").last().reset_index()
        daily["Date"] = pd.to_datetime(daily["Date"])
        daily["iv_change_1d"] = daily["atm_iv"].diff()
        daily["oi_buildup_signal"] = np.sign(daily["pe_chg_oi"] - daily["ce_chg_oi"])
        return daily[["Date", "pcr_oi", "pcr_vol", "atm_iv", "iv_change_1d",
                       "max_pain_dist_pct", "oi_buildup_signal"]].copy()
    except Exception:
        return pd.DataFrame()


def _load_intermarket():
    from engines.feature_store import _load_intermarket, _load_fii_net
    im = _load_intermarket()
    fii = _load_fii_net()
    return im, fii


def _compute_index_features(df, symbol, breadth_df, oc_df, im_df, fii_series):
    """Compute all features for one index."""
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    n = len(df)
    f = pd.DataFrame(index=df.index)
    f["Date"] = df["Date"]
    f["symbol"] = symbol

    # Intraday return & range
    intra_ret = np.where(o > 0, (c - o) / o, 0)
    intra_range = np.where(o > 0, (h - l) / o, 0)
    gap = np.zeros(n)
    gap[1:] = np.where(c[:-1] > 0, (o[1:] - c[:-1]) / c[:-1], 0)
    close_loc = np.where(h - l > 0, (c - l) / (h - l), 0.5)

    ir_s = pd.Series(intra_ret)
    rng_s = pd.Series(intra_range)
    gap_s = pd.Series(gap)

    f["prev_intra_ret"] = ir_s.shift(1).values
    f["prev_range"] = rng_s.shift(1).values
    f["prev_close_loc"] = pd.Series(close_loc).shift(1).values
    f["prev_gap"] = gap_s.shift(1).values
    f["avg_intra_ret_3d"] = ir_s.rolling(3).mean().shift(1).values
    f["avg_intra_ret_5d"] = ir_s.rolling(5).mean().shift(1).values
    f["avg_range_5d"] = rng_s.rolling(5).mean().shift(1).values
    f["avg_range_10d"] = rng_s.rolling(10).mean().shift(1).values

    signs = np.sign(intra_ret)
    streak = np.zeros(n)
    for i in range(1, n):
        if signs[i] == signs[i-1] and signs[i] != 0:
            streak[i] = streak[i-1] + signs[i]
        elif signs[i] != 0:
            streak[i] = signs[i]
    f["intra_streak"] = pd.Series(streak).shift(1).values

    gap_filled = np.zeros(n)
    for i in range(1, n):
        if gap[i] > 0 and l[i] <= c[i-1]:
            gap_filled[i] = 1
        elif gap[i] < 0 and h[i] >= c[i-1]:
            gap_filled[i] = 1
    f["gap_fill_rate_10d"] = pd.Series(gap_filled).rolling(10).mean().shift(1).values

    f["overnight_gap"] = gap
    avg_gap = gap_s.rolling(20).mean().shift(1).values
    std_gap = gap_s.rolling(20).std().shift(1).values
    f["overnight_vs_avg"] = np.where(std_gap > 1e-9, (gap - avg_gap) / std_gap, 0)

    # Technicals (all shifted 1 day)
    cs = pd.Series(c)
    hs = pd.Series(h)
    ls = pd.Series(l)

    delta = cs.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    f["rsi_14"] = rsi.shift(1).values
    f["rsi_slope_5d"] = (rsi - rsi.shift(5)).shift(1).values

    ema12 = cs.ewm(span=12).mean()
    ema26 = cs.ewm(span=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9).mean()
    f["macd_hist"] = (macd - macd_signal).shift(1).values
    f["macd_cross"] = (np.sign(macd - macd_signal).diff().abs() > 0).astype(float).shift(1).values

    low14 = ls.rolling(14).min()
    high14 = hs.rolling(14).max()
    stoch_k = 100 * (cs - low14) / (high14 - low14).replace(0, np.nan)
    f["stoch_k"] = stoch_k.shift(1).values
    f["stoch_d"] = stoch_k.rolling(3).mean().shift(1).values

    bb_mid = cs.rolling(20).mean()
    bb_std = cs.rolling(20).std()
    f["bb_pctb"] = ((cs - (bb_mid - 2*bb_std)) / (4*bb_std).replace(0, np.nan)).shift(1).values
    f["bb_width"] = (2*bb_std / bb_mid.replace(0, np.nan)).shift(1).values

    ema20 = cs.ewm(span=20).mean()
    ema50 = cs.ewm(span=50).mean()
    f["dist_ema20"] = ((cs - ema20) / ema20).shift(1).values
    f["dist_ema50"] = ((cs - ema50) / ema50).shift(1).values
    f["ema20_slope"] = (ema20.pct_change(5)).shift(1).values
    f["ema50_slope"] = (ema50.pct_change(5)).shift(1).values
    f["ema_20_50_cross"] = (ema20 > ema50).astype(float).shift(1).values

    # Supertrend (simplified)
    tr = pd.concat([hs-ls, (hs-cs.shift()).abs(), (ls-cs.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(10).mean()
    upper = (hs+ls)/2 + 2*atr
    lower = (hs+ls)/2 - 2*atr
    st_signal = (cs > lower).astype(float)
    f["supertrend_signal"] = st_signal.shift(1).values

    f["adx_14"] = 50.0  # simplified; ADX computation is complex
    f["trend_slope_10"] = cs.pct_change(10).shift(1).values
    f["ret_5d"] = cs.pct_change(5).shift(1).values
    f["ret_10d"] = cs.pct_change(10).shift(1).values
    f["ret_21d"] = cs.pct_change(21).shift(1).values

    vol_10 = cs.pct_change().rolling(10).std() * np.sqrt(252)
    vol_21 = cs.pct_change().rolling(21).std() * np.sqrt(252)
    f["vol_10d"] = vol_10.shift(1).values
    f["vol_21d"] = vol_21.shift(1).values
    f["atr_ratio"] = (atr / cs).shift(1).values
    f["atr_expansion"] = (atr / atr.rolling(10).mean().replace(0, np.nan)).shift(1).values
    avg_range_10 = rng_s.rolling(10).mean().shift(1)
    f["range_expansion"] = np.where(avg_range_10 > 1e-9, rng_s.shift(1) / avg_range_10, 1.0)

    # Market breadth (merge by date)
    if breadth_df is not None and not breadth_df.empty:
        f_dates = pd.to_datetime(f["Date"])
        b = breadth_df.copy()
        b["Date"] = pd.to_datetime(b["Date"])
        for col in [c for c in breadth_df.columns if c.startswith("breadth_")]:
            mapping = dict(zip(b["Date"], b[col]))
            f[col] = f_dates.map(mapping).shift(1).values  # use yesterday's breadth
    else:
        for col in [c for c in FEATURE_COLS if c.startswith("breadth_")]:
            f[col] = 0.5

    # Intermarket
    if im_df is not None and not im_df.empty:
        f_dates = pd.to_datetime(f["Date"])
        im = im_df.copy()
        im.index = pd.to_datetime(im.index)
        for col, src in [("vix_level", "vix_level"), ("vix_regime", "vix_regime"),
                          ("ivix_level", "ivix_level"),
                          ("crude_ret_5d", "crude_ret_5d"), ("gold_ret_5d", "gold_ret_5d"),
                          ("dxy_ret_5d", "dxy_ret_5d"), ("usdinr_ret_5d", "usdinr_ret_5d"),
                          ("sp500_ret_5d", "sp500_ret_5d"), ("us10y_level", "us10y_level")]:
            if src in im.columns:
                mapping = dict(zip(im.index, im[src]))
                f[col] = f_dates.map(mapping).shift(1).fillna(0).values
            else:
                f[col] = 0.0
        if "vix_level" in im.columns:
            vix_mapped = f_dates.map(dict(zip(im.index, im["vix_level"])))
            f["vix_change_1d"] = vix_mapped.diff().shift(1).fillna(0).values
        else:
            f["vix_change_1d"] = 0.0
    else:
        for col in ["vix_level", "vix_change_1d", "vix_regime", "ivix_level",
                     "crude_ret_5d", "gold_ret_5d", "dxy_ret_5d", "usdinr_ret_5d",
                     "sp500_ret_5d", "us10y_level"]:
            f[col] = 0.0

    if fii_series is not None and len(fii_series) > 0:
        f_dates = pd.to_datetime(f["Date"])
        fii_idx = pd.to_datetime(fii_series.index)
        mapping = dict(zip(fii_idx, fii_series.values))
        f["fii_net_5d"] = f_dates.map(mapping).shift(1).fillna(0).values
    else:
        f["fii_net_5d"] = 0.0

    # Option chain (only recent, 0 for historical)
    if oc_df is not None and not oc_df.empty:
        oc = oc_df.copy()
        oc["Date"] = pd.to_datetime(oc["Date"])
        f_dates = pd.to_datetime(f["Date"])
        for col in ["pcr_oi", "pcr_vol", "atm_iv", "iv_change_1d",
                     "max_pain_dist_pct", "oi_buildup_signal"]:
            if col in oc.columns:
                mapping = dict(zip(oc["Date"], oc[col]))
                f[col] = f_dates.map(mapping).shift(1).fillna(0).values
            else:
                f[col] = 0.0
    else:
        for col in ["pcr_oi", "pcr_vol", "atm_iv", "iv_change_1d",
                     "max_pain_dist_pct", "oi_buildup_signal"]:
            f[col] = 0.0

    # Calendar
    f["day_of_week"] = df["Date"].dt.dayofweek.values
    # Days to weekly expiry (Thu for NIFTY, Wed for BANKNIFTY)
    expiry_day = 3 if symbol == "NIFTY" else 2  # Thu=3, Wed=2
    dow = df["Date"].dt.dayofweek.values
    f["days_to_expiry"] = np.array([(expiry_day - d) % 7 for d in dow])
    f["is_expiry_week"] = (f["days_to_expiry"] <= 2).astype(float)

    # Labels
    f["intraday_return"] = intra_ret
    f["intraday_range"] = intra_range
    f["label"] = (intra_ret > 0).astype(int)  # 1 = bullish day
    f["price"] = c
    f["open_price"] = o

    return f


def _build_panel(min_date, max_date, breadth_df, im_df, fii_series):
    frames = []
    for sym in INDICES:
        df = _load_index(sym)
        df = df[(df["Date"] >= min_date) & (df["Date"] <= max_date)].copy()
        if len(df) < 30:
            continue
        oc_df = _load_option_chain_daily(sym)
        feats = _compute_index_features(df, sym, breadth_df, oc_df, im_df, fii_series)
        frames.append(feats)

    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=["label", "prev_intra_ret"])
    return panel


def _clean(panel):
    for col in FEATURE_COLS:
        if col in panel.columns:
            panel[col] = panel[col].replace([np.inf, -np.inf], 0).fillna(0)
    return panel


# ── Evaluation ──

def _evaluate(test, model, features):
    test = test.copy()
    test["pred"] = model.predict_proba(test[features])[:, 1]

    correct = ((test["pred"] > 0.5) == test["label"]).mean()
    n_up = (test["pred"] > 0.5).sum()
    n_down = (test["pred"] <= 0.5).sum()

    # Simulate: go long on bullish prediction, short on bearish
    test["signal_ret"] = np.where(test["pred"] > 0.5,
                                   test["intraday_return"],
                                   -test["intraday_return"])
    rets = test["signal_ret"]
    cum = (1 + rets).cumprod()
    n_years = len(rets) / 252
    cagr = float(cum.iloc[-1] ** (1/max(n_years, 0.01)) - 1) if len(cum) > 0 else 0
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))
    max_dd = float((cum / cum.cummax() - 1).min())
    win_rate = float((rets > 0).mean() * 100)
    avg_win = float(rets[rets > 0].mean() * 100) if (rets > 0).any() else 0
    avg_loss = float(rets[rets < 0].mean() * 100) if (rets < 0).any() else 0

    return {
        "accuracy": round(correct * 100, 1),
        "n_bullish": int(n_up), "n_bearish": int(n_down),
        "cagr": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "win_rate": round(win_rate, 1),
        "avg_win_pct": round(avg_win, 3),
        "avg_loss_pct": round(avg_loss, 3),
        "avg_daily_ret": round(float(rets.mean() * 100), 3),
        "n_days": len(test),
    }


def train():
    print("=" * 64)
    print(f"  INDEX OPTIONS ML TRAINER — NIFTY & BANKNIFTY")
    print(f"  {len(FEATURE_COLS)} features | Walk-Forward 6yr/1yr")
    print("=" * 64)
    t0 = time.time()

    print("\n  Computing market breadth from 473 stocks...")
    breadth_df = _compute_breadth_features()
    print(f"  Breadth: {len(breadth_df)} dates")

    im_df, fii_series = _load_intermarket()

    all_metrics = []
    all_test = []

    print(f"\n[1/3] Walk-forward validation ({len(WALK_FORWARD_FOLDS)} folds)...\n")

    for fold_i, (tr_start, tr_end, te_start, te_end) in enumerate(WALK_FORWARD_FOLDS, 1):
        print(f"  --- Fold {fold_i}: Train {tr_start[:4]}-{tr_end[:4]} | "
              f"Test {te_start[:4]}-{te_end[:4]} ---")

        train_panel = _build_panel(tr_start, tr_end, breadth_df, im_df, fii_series)
        train_panel = _clean(train_panel)
        available = [f for f in FEATURE_COLS if f in train_panel.columns]
        print(f"    Train: {len(train_panel)} rows ({train_panel['symbol'].nunique()} indices, "
              f"{train_panel['Date'].nunique()} dates)")

        if len(train_panel) < 100:
            print("    [SKIP] Too few rows")
            continue

        model = xgb.XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.6,
            reg_alpha=0.5, reg_lambda=3.0, min_child_weight=10,
            eval_metric="logloss", random_state=42, verbosity=0, n_jobs=1,
        )
        model.fit(train_panel[available], train_panel["label"])
        del train_panel; gc.collect()

        test_panel = _build_panel(te_start, te_end, breadth_df, im_df, fii_series)
        test_panel = _clean(test_panel)
        print(f"    Test:  {len(test_panel)} rows")

        if len(test_panel) < 50:
            del model; gc.collect()
            continue

        metrics = _evaluate(test_panel, model, available)
        all_metrics.append(metrics)
        all_test.append(test_panel.copy())

        print(f"    Accuracy     = {metrics['accuracy']:.1f}%")
        print(f"    CAGR={metrics['cagr']:.1f}%  Sharpe={metrics['sharpe']:.3f}  "
              f"MaxDD={metrics['max_drawdown']:.1f}%  WinRate={metrics['win_rate']:.0f}%  "
              f"AvgDaily={metrics['avg_daily_ret']:.3f}%")
        print()
        del test_panel, model; gc.collect()

    # Summary
    if all_metrics:
        avg_acc = np.mean([m["accuracy"] for m in all_metrics])
        avg_sharpe = np.mean([m["sharpe"] for m in all_metrics])
        avg_wr = np.mean([m["win_rate"] for m in all_metrics])

        print(f"  {'=' * 54}")
        print(f"  VALIDATION SUMMARY ({len(all_metrics)} folds)")
        print(f"  {'=' * 54}")
        print(f"  Mean Accuracy       : {avg_acc:.1f}%")
        print(f"  Mean Sharpe         : {avg_sharpe:.3f}")
        print(f"  Mean Win Rate       : {avg_wr:.1f}%")

        edge = "STRONG" if avg_acc > 55 else "EDGE" if avg_acc > 52 else "WEAK"
        print(f"  Verdict             : {edge}")
    else:
        edge = "WEAK"

    # Final model
    print(f"\n[2/3] Training FINAL model on 2019-2025...")
    final_panel = _build_panel("2019-01-01", "2025-12-31", breadth_df, im_df, fii_series)
    final_panel = _clean(final_panel)
    available = [f for f in FEATURE_COLS if f in final_panel.columns]
    print(f"  Final: {len(final_panel)} rows, {len(available)} features")

    final_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.6,
        reg_alpha=0.5, reg_lambda=3.0, min_child_weight=10,
        eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1,
    )
    final_model.fit(final_panel[available], final_panel["label"])

    imp = pd.Series(final_model.feature_importances_, index=available)
    imp = imp.sort_values(ascending=False)
    print("\n  Top 15 features:")
    for feat, val in imp.head(15).items():
        bar = "#" * int(val * 60)
        print(f"    {feat:<30} {bar} {val:.3f}")

    # Save
    print(f"\n[3/3] Saving model...")
    model_id = f"index_options_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_path = os.path.join(MODEL_DIR, f"{model_id}.pkl")
    latest_path = os.path.join(MODEL_DIR, "latest_index_options.pkl")
    meta_path = os.path.join(MODEL_DIR, "latest_index_options_meta.json")

    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    meta = {
        "model_id": model_id,
        "type": "index_options_direction",
        "indices": INDICES,
        "n_features": len(available),
        "features": available,
        "walk_forward": {
            "folds": [{"train": f"{s[:4]}-{e[:4]}", "test": f"{ts[:4]}-{te[:4]}"}
                      for s, e, ts, te in WALK_FORWARD_FOLDS],
            "fold_metrics": all_metrics,
        },
        "validation": {
            "mean_accuracy": round(avg_acc, 1) if all_metrics else 0,
            "mean_sharpe": round(avg_sharpe, 3) if all_metrics else 0,
            "mean_win_rate": round(avg_wr, 1) if all_metrics else 0,
        },
        "feature_importance": {k: round(v, 4) for k, v in imp.head(25).items()},
        "edge": edge,
        "trained_at": datetime.now().isoformat(),
        "training_time_sec": round(time.time() - t0, 1),
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    import shutil
    shutil.copy2(model_path, latest_path)

    del final_panel, final_model, breadth_df; gc.collect()

    print(f"\n  [OK] Model saved -> {model_path}")
    print(f"       {len(available)} features")
    print(f"       Training time: {time.time() - t0:.0f}s")

    return meta


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    result = train()
    if result:
        print(f"\nDone. Edge: {result.get('edge')}, "
              f"Accuracy: {result.get('validation', {}).get('mean_accuracy')}%")

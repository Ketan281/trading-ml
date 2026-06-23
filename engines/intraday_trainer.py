"""
Intraday ML Trainer — predict best intraday trade today.

Hybrid approach:
  - 10 years of daily OHLCV → intraday-proxy labels & features
  - 2 months of real 15m candle data → bonus session features
  - Cross-sectional ranking: which stock moves most intraday?

Labels (from daily data):
  - intraday_return = (close - open) / open  (same-day move)
  - intraday_range  = (high - low) / open    (volatility opportunity)
  - gap_return      = (open - prev_close) / prev_close

Walk-forward validation (same as swing model):
  Fold 1: Train 2006-2010, Test 2011
  Fold 2: Train 2012-2016, Test 2017
  Fold 3: Train 2018-2024, Test 2025
  Final:  Train 2018-2025 → production model

Run: python -m engines.intraday_trainer
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

log = logging.getLogger("intraday_trainer")

MODEL_DIR = os.path.join(ROOT, "models", "intraday")
os.makedirs(MODEL_DIR, exist_ok=True)

HIST_DIR = os.path.join(ROOT, "data", "historical")
INTRADAY_DIR = os.path.join(ROOT, "data", "intraday", "15m")
EXCLUDE = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"}

# ── Features derived from daily OHLCV for intraday prediction ──

DAILY_FEATURES = [
    # Prior-day intraday patterns (the model learns "days like yesterday → today")
    "prev_intraday_ret",       # yesterday's (close-open)/open
    "prev_intraday_range",     # yesterday's (high-low)/open
    "prev_gap",                # yesterday's gap open
    "prev_close_loc",          # where close was within day range [0,1]

    # Multi-day intraday patterns
    "avg_intraday_ret_3d",     # avg intraday return last 3 days
    "avg_intraday_ret_5d",     # avg intraday return last 5 days
    "avg_intraday_range_5d",   # avg day range last 5 days
    "avg_intraday_range_10d",  # avg day range last 10 days
    "intraday_ret_streak",     # consecutive same-direction intraday days

    # Gap patterns
    "avg_gap_5d",              # avg gap last 5 days
    "gap_fill_rate_10d",       # % of gaps that filled in last 10 days

    # Overnight positioning
    "overnight_ret",           # today's open vs yesterday's close (gap)
    "overnight_vs_avg",        # today's gap vs average gap

    # Volatility regime for intraday
    "range_expansion",         # today's expected range vs 10d avg
    "range_contraction_days",  # how many days range has been shrinking

    # Volume patterns
    "prev_vol_ratio",          # yesterday's volume vs 20d avg
    "vol_trend_5d",            # volume trending up or down

    # Time-of-week
    "day_of_week",             # 0=Mon, 4=Fri (Monday effect, Friday caution)

    # Price structure for intraday
    "dist_open_ema20",         # gap from EMA20 at open
    "dist_open_prev_high",     # open vs previous day's high
    "dist_open_prev_low",      # open vs previous day's low
    "open_above_prev_close",   # binary: gap up or gap down
]

# Reuse existing features from the swing model feature store
REUSE_FROM_SWING = [
    "ret_1d", "ret_5d", "ret_10d", "ret_21d",
    "rsi_14", "rsi_slope",
    "macd_hist", "macd_signal_cross",
    "bb_pctb", "bb_width",
    "supertrend_signal", "supertrend_dist",
    "adx_14", "trend_slope_10",
    "vol_10d", "vol_21d", "atr_ratio", "intraday_range",
    "rel_volume", "vol_trend", "mfi_14",
    "sr_proximity", "dist_52w_high", "dist_52w_low",
    "dist_ema20", "dist_ema50", "above_200ma",
    "ema_20_50_cross",
    "stochastic_k", "stochastic_d",
    "sharpe_60d",
    "pat_net_score", "pat_bullish_count", "pat_bearish_count",
    "vix_level", "vix_regime", "ivix_level",
    "crude_ret_5d", "gold_ret_5d", "fii_net_5d",
    "pe_trailing", "pb_ratio", "roe", "market_cap_log",
    "value_momentum", "quality_breakout",
]

# 15m session features (only available for recent data)
SESSION_FEATURES = [
    "session_open_range_pct",   # first 30min high-low range / open
    "session_open_direction",   # first 30min return direction
    "session_vwap_dist_open",   # VWAP distance from open at 30min
    "session_vol_ratio_open",   # first 30min volume vs avg first 30min
    "session_prev_last_hour",   # last hour return of previous day
]

ALL_RAW_FEATURES = DAILY_FEATURES + REUSE_FROM_SWING + SESSION_FEATURES
CROSS_SECTIONAL = {"mom_rank", "vol_rank", "rs_score"}
Z_FEATURES = [f + "_z" for f in ALL_RAW_FEATURES]

WALK_FORWARD_FOLDS = [
    ("2006-01-01", "2010-12-31", "2011-01-01", "2011-12-31"),
    ("2012-01-01", "2016-12-31", "2017-01-01", "2017-12-31"),
    ("2018-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
]

MIN_STOCKS_PER_DATE = 20
MIN_HISTORY = 30


# ── Data Loading ──

def _list_symbols():
    csvs = [f[:-4] for f in os.listdir(HIST_DIR) if f.endswith(".csv")]
    return sorted(set(csvs) - EXCLUDE)


def _load_daily(symbol):
    path = os.path.join(HIST_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, parse_dates=["Date"],
                         usecols=["Date", "Open", "High", "Low", "Close", "Volume"])
        df = df.sort_values("Date").reset_index(drop=True)
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        if len(df) < MIN_HISTORY:
            return None
        return df
    except Exception:
        return None


def _load_15m_session_features(symbol):
    """Load real 15m data and compute session-level features per date."""
    path = os.path.join(INTRADAY_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, parse_dates=["Datetime"])
        df["date"] = df["Datetime"].dt.date
        df["time"] = df["Datetime"].dt.time

        session_feats = []
        for date, grp in df.groupby("date"):
            grp = grp.sort_values("Datetime")
            if len(grp) < 4:
                continue
            opn = grp.iloc[0]["Open"]
            first_30 = grp.head(2)  # first 2 × 15m = 30min
            last_hour = grp.tail(4)  # last 4 × 15m = 60min

            feat = {"date": pd.Timestamp(date)}
            feat["session_open_range_pct"] = (first_30["High"].max() - first_30["Low"].min()) / opn if opn > 0 else 0
            feat["session_open_direction"] = 1.0 if first_30.iloc[-1]["Close"] > opn else -1.0
            tp = (grp["High"] + grp["Low"] + grp["Close"]) / 3
            vwap_30 = (tp.head(2) * grp["Volume"].head(2)).sum() / max(grp["Volume"].head(2).sum(), 1)
            feat["session_vwap_dist_open"] = (vwap_30 - opn) / opn if opn > 0 else 0
            avg_vol = grp["Volume"].head(2).mean()
            feat["session_vol_ratio_open"] = avg_vol / max(grp["Volume"].mean(), 1)
            feat["session_prev_last_hour"] = (last_hour.iloc[-1]["Close"] - last_hour.iloc[0]["Open"]) / last_hour.iloc[0]["Open"] if last_hour.iloc[0]["Open"] > 0 else 0
            session_feats.append(feat)

        if not session_feats:
            return None
        return pd.DataFrame(session_feats)
    except Exception:
        return None


# ── Feature Engineering (from daily OHLCV) ──

def compute_intraday_features(df, symbol=None, session_df=None):
    """Compute intraday-prediction features from daily OHLCV.
    These predict: will TODAY have a big move, and in which direction?"""
    o, h, l, c, v = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values, df["Volume"].values
    n = len(df)

    f = pd.DataFrame(index=df.index)
    f["date"] = df["Date"]
    f["symbol"] = symbol
    f["price"] = c
    f["open"] = o
    f["high"] = h
    f["low"] = l

    # -- Intraday return and range (these become LABELS, shifted by 1 for features) --
    intra_ret = np.where(o > 0, (c - o) / o, 0)    # today's intraday return
    intra_range = np.where(o > 0, (h - l) / o, 0)   # today's range
    gap = np.zeros(n)
    gap[1:] = np.where(c[:-1] > 0, (o[1:] - c[:-1]) / c[:-1], 0)
    close_loc = np.where(h - l > 0, (c - l) / (h - l), 0.5)

    # Previous day features (shift by 1 to avoid lookahead)
    f["prev_intraday_ret"] = pd.Series(intra_ret).shift(1).values
    f["prev_intraday_range"] = pd.Series(intra_range).shift(1).values
    f["prev_gap"] = pd.Series(gap).shift(1).values
    f["prev_close_loc"] = pd.Series(close_loc).shift(1).values

    # Multi-day averages
    ir_s = pd.Series(intra_ret)
    rng_s = pd.Series(intra_range)
    gap_s = pd.Series(gap)

    f["avg_intraday_ret_3d"] = ir_s.rolling(3).mean().shift(1).values
    f["avg_intraday_ret_5d"] = ir_s.rolling(5).mean().shift(1).values
    f["avg_intraday_range_5d"] = rng_s.rolling(5).mean().shift(1).values
    f["avg_intraday_range_10d"] = rng_s.rolling(10).mean().shift(1).values

    # Intraday return streak
    signs = np.sign(intra_ret)
    streak = np.zeros(n)
    for i in range(1, n):
        if signs[i] == signs[i - 1] and signs[i] != 0:
            streak[i] = streak[i - 1] + signs[i]
        elif signs[i] != 0:
            streak[i] = signs[i]
    f["intraday_ret_streak"] = pd.Series(streak).shift(1).values

    # Gap patterns
    f["avg_gap_5d"] = gap_s.rolling(5).mean().shift(1).values
    # Gap fill rate: did price return to prev close during the day?
    gap_filled = np.zeros(n)
    for i in range(1, n):
        if gap[i] > 0 and l[i] <= c[i - 1]:  # gap up filled
            gap_filled[i] = 1
        elif gap[i] < 0 and h[i] >= c[i - 1]:  # gap down filled
            gap_filled[i] = 1
    f["gap_fill_rate_10d"] = pd.Series(gap_filled).rolling(10).mean().shift(1).values

    # Overnight / gap
    f["overnight_ret"] = gap
    avg_gap = gap_s.rolling(20).mean().shift(1).values
    std_gap = gap_s.rolling(20).std().shift(1).values
    f["overnight_vs_avg"] = np.where(std_gap > 1e-9, (gap - avg_gap) / std_gap, 0)

    # Range expansion/contraction
    avg_range_10 = rng_s.rolling(10).mean().shift(1).values
    f["range_expansion"] = np.where(avg_range_10 > 1e-9, rng_s.shift(1).values / avg_range_10, 1.0)
    # Days of range contraction
    range_contract = np.zeros(n)
    for i in range(1, n):
        if i > 0 and intra_range[i - 1] < (avg_range_10[i] if not np.isnan(avg_range_10[i]) else intra_range[i - 1]):
            range_contract[i] = range_contract[i - 1] + 1
    f["range_contraction_days"] = range_contract

    # Volume patterns
    vol_s = pd.Series(v, dtype=float)
    vol_20ma = vol_s.rolling(20).mean()
    f["prev_vol_ratio"] = (vol_s / vol_20ma.replace(0, np.nan)).shift(1).values
    f["vol_trend_5d"] = (vol_s.rolling(5).mean() / vol_20ma.replace(0, np.nan)).shift(1).values

    # Day of week
    f["day_of_week"] = df["Date"].dt.dayofweek.values

    # Price structure at open (use yesterday's EMA to avoid lookahead)
    ema20_prev = pd.Series(c).ewm(span=20, adjust=False).mean().shift(1).values
    f["dist_open_ema20"] = np.where(ema20_prev > 0, (o - ema20_prev) / ema20_prev, 0)
    f["dist_open_prev_high"] = np.zeros(n)
    f["dist_open_prev_low"] = np.zeros(n)
    f["open_above_prev_close"] = np.zeros(n)
    for i in range(1, n):
        if h[i - 1] > 0:
            f.iloc[i, f.columns.get_loc("dist_open_prev_high")] = (o[i] - h[i - 1]) / h[i - 1]
        if l[i - 1] > 0:
            f.iloc[i, f.columns.get_loc("dist_open_prev_low")] = (o[i] - l[i - 1]) / l[i - 1]
        f.iloc[i, f.columns.get_loc("open_above_prev_close")] = 1.0 if o[i] > c[i - 1] else 0.0

    # -- Reuse swing model features from feature store --
    # CRITICAL: shift by 1 day to avoid lookahead bias.
    # Swing features use today's close (which IS the intraday return).
    # We must use YESTERDAY's features to predict TODAY's intraday move.
    from engines.feature_store import compute_stock_features
    df_indexed = df.set_index("Date") if "Date" in df.columns and not isinstance(df.index, pd.DatetimeIndex) else df
    swing_feats = compute_stock_features(df_indexed, symbol)
    if swing_feats is not None:
        for col in REUSE_FROM_SWING:
            if col in swing_feats.columns:
                f[col] = swing_feats[col].shift(1).values
            else:
                f[col] = 0.0
    else:
        for col in REUSE_FROM_SWING:
            f[col] = 0.0

    # -- Session features from 15m data --
    if session_df is not None and not session_df.empty:
        session_df = session_df.copy()
        session_df["date"] = pd.to_datetime(session_df["date"])
        f["date"] = pd.to_datetime(f["date"])
        for col in SESSION_FEATURES:
            if col in session_df.columns:
                mapping = dict(zip(session_df["date"], session_df[col]))
                # Shift by 1: use previous session features to predict today
                f[col] = f["date"].map(mapping).shift(1).fillna(0)
            else:
                f[col] = 0.0
    else:
        for col in SESSION_FEATURES:
            f[col] = 0.0

    # -- Intraday labels (what we're predicting) --
    f["intraday_return"] = intra_ret     # (close - open) / open
    f["intraday_range_pct"] = intra_range  # (high - low) / open
    f["gap_return"] = gap

    return f


# ── Panel Building ──

def _build_panel(min_date, max_date):
    """Build cross-sectional panel from daily CSVs for date range."""
    symbols = _list_symbols()
    print(f"    Building panel: {len(symbols)} symbols, {min_date} to {max_date}")

    all_frames = []
    ok = 0
    for sym in symbols:
        df = _load_daily(sym)
        if df is None:
            continue
        df = df[(df["Date"] >= min_date) & (df["Date"] <= max_date)]
        if len(df) < MIN_HISTORY:
            continue

        session_df = _load_15m_session_features(sym)
        feats = compute_intraday_features(df, sym, session_df)
        if feats is not None and len(feats) > MIN_HISTORY:
            all_frames.append(feats)
            ok += 1

    if not all_frames:
        return pd.DataFrame()

    panel = pd.concat(all_frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    print(f"    Loaded: {ok} stocks, {len(panel):,} rows")
    return panel


def _add_labels(panel):
    """Cross-sectional label: top half by intraday return magnitude AND direction."""
    # Primary label: will this stock outperform others intraday?
    panel["label_direction"] = panel.groupby("date")["intraday_return"].transform(
        lambda x: (x.rank(pct=True) > 0.5).astype(int)
    )
    # Secondary: will this stock have a large range (good for options)?
    panel["label_range"] = panel.groupby("date")["intraday_range_pct"].transform(
        lambda x: (x.rank(pct=True) > 0.5).astype(int)
    )
    # Relative intraday return for IC calculation
    panel["rel_intraday"] = panel.groupby("date")["intraday_return"].transform(
        lambda x: x - x.mean()
    )
    return panel


def _zscore_per_date(panel):
    available = [f for f in ALL_RAW_FEATURES if f in panel.columns]
    for feat in available:
        zfeat = feat + "_z"
        panel[zfeat] = panel.groupby("date")[feat].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)
        )
    return panel


def _clean(panel):
    available_z = [f + "_z" for f in ALL_RAW_FEATURES if f in panel.columns]
    for col in available_z:
        if col in panel.columns:
            panel[col] = panel[col].replace([np.inf, -np.inf], 0).fillna(0)
    return panel


def _prepare_panel(panel):
    # Filter dates with enough stocks
    date_counts = panel.groupby("date")["symbol"].transform("count")
    panel = panel[date_counts >= MIN_STOCKS_PER_DATE].copy()
    # Drop rows without valid intraday return
    panel = panel.dropna(subset=["intraday_return"])
    panel = _add_labels(panel)
    panel = _zscore_per_date(panel)
    panel = _clean(panel)
    return panel


# ── Evaluation Metrics ──

def _rank_ic(df):
    ics = []
    for _, grp in df.groupby("date"):
        if len(grp) < MIN_STOCKS_PER_DATE:
            continue
        pr = grp["pred"].rank().values
        ar = grp["rel_intraday"].rank().values
        if pr.std() == 0 or ar.std() == 0:
            continue
        ics.append(np.corrcoef(pr, ar)[0, 1])
    return float(np.nanmean(ics)) if ics else float("nan")


def _long_short(df, q=0.2):
    spreads = []
    for _, grp in df.groupby("date"):
        if len(grp) < MIN_STOCKS_PER_DATE:
            continue
        n = max(1, int(len(grp) * q))
        s = grp.sort_values("pred", ascending=False)
        spreads.append(s.head(n)["rel_intraday"].mean() - s.tail(n)["rel_intraday"].mean())
    return float(np.nanmean(spreads)) if spreads else float("nan")


def _precision_at_k(df, k=10):
    precs = []
    for _, grp in df.groupby("date"):
        if len(grp) < MIN_STOCKS_PER_DATE:
            continue
        s = grp.sort_values("pred", ascending=False).head(k)
        precs.append(s["label_direction"].mean())
    return float(np.nanmean(precs)) if precs else float("nan")


def _profit_sim(df, top_n=10):
    daily_rets = []
    for _, grp in df.groupby("date"):
        if len(grp) < MIN_STOCKS_PER_DATE:
            continue
        picks = grp.sort_values("pred", ascending=False).head(top_n)
        daily_rets.append(picks["intraday_return"].mean())
    if not daily_rets:
        return {}
    rets = pd.Series(daily_rets)
    cum = (1 + rets).cumprod()
    n_years = len(rets) / 252
    cagr = float(cum.iloc[-1] ** (1 / max(n_years, 0.01)) - 1) if len(cum) > 0 else 0
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))
    max_dd = float((cum / cum.cummax() - 1).min())
    return {
        "cagr": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "win_rate": round(float((rets > 0).mean() * 100), 1),
        "avg_daily_ret": round(float(rets.mean() * 100), 3),
        "n_days": len(daily_rets),
    }


# ── Model ──

def _make_model():
    return xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=0.3,
        reg_lambda=2.0,
        min_child_weight=20,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
        n_jobs=1,
    )


# ── Training ──

def train(label_target="direction"):
    """Train intraday model with walk-forward validation.

    label_target: 'direction' (predict intraday return direction)
                  'range' (predict which stocks will have large intraday range — for options)
    """
    label_col = f"label_{label_target}"

    print("=" * 64)
    print(f"  INTRADAY ML TRAINER — {len(ALL_RAW_FEATURES)} Features")
    print(f"  Target: {label_target} | Walk-Forward 4yr/1yr")
    print("=" * 64)
    t0 = time.time()

    ic_scores, ls_scores, prec_scores = [], [], []
    all_test_preds = []
    total_train, total_test = 0, 0
    available_features = None

    print(f"\n[1/3] Walk-forward validation ({len(WALK_FORWARD_FOLDS)} folds)...\n")

    for fold_i, (tr_start, tr_end, te_start, te_end) in enumerate(WALK_FORWARD_FOLDS, 1):
        print(f"  --- Fold {fold_i}: Train {tr_start[:4]}-{tr_end[:4]} | Test {te_start[:4]}-{te_end[:4]} ---")

        print(f"    Loading train data ({tr_start[:4]}-{tr_end[:4]})...")
        train_panel = _build_panel(tr_start, tr_end)
        if train_panel.empty:
            print(f"    [SKIP] No data")
            continue

        train_panel = _prepare_panel(train_panel)
        n_train = len(train_panel)
        print(f"    Train: {n_train:,} rows, {train_panel['symbol'].nunique()} stocks, "
              f"{train_panel['date'].nunique()} dates")

        if n_train < 1000:
            print(f"    [SKIP] Too few rows")
            del train_panel; gc.collect()
            continue

        available_features = [f + "_z" for f in ALL_RAW_FEATURES
                              if f + "_z" in train_panel.columns]

        model = _make_model()
        model.fit(train_panel[available_features], train_panel[label_col])
        del train_panel; gc.collect()

        print(f"    Loading test data ({te_start[:4]}-{te_end[:4]})...")
        test_panel = _build_panel(te_start, te_end)
        if test_panel.empty:
            print(f"    [SKIP] No test data")
            del model; gc.collect()
            continue

        test_panel = _prepare_panel(test_panel)
        n_test = len(test_panel)
        print(f"    Test: {n_test:,} rows, {test_panel['symbol'].nunique()} stocks, "
              f"{test_panel['date'].nunique()} dates")

        if n_test < 200:
            del test_panel, model; gc.collect()
            continue

        test_panel["pred"] = model.predict_proba(test_panel[available_features])[:, 1]

        ic = _rank_ic(test_panel)
        ls = _long_short(test_panel)
        prec = _precision_at_k(test_panel)
        sim = _profit_sim(test_panel)

        ic_scores.append(ic)
        ls_scores.append(ls)
        prec_scores.append(prec)
        all_test_preds.append(test_panel[["date", "symbol", "pred", "rel_intraday",
                                          "label_direction", "intraday_return"]].copy())
        total_train += n_train
        total_test += n_test

        print(f"    Rank IC      = {ic:+.4f}")
        print(f"    L/S spread   = {ls:+.4%}")
        print(f"    Precision@10 = {prec:.1%}")
        if sim:
            print(f"    CAGR={sim['cagr']:.1f}%  Sharpe={sim['sharpe']:.3f}  "
                  f"MaxDD={sim['max_drawdown']:.1f}%  WinRate={sim['win_rate']:.0f}%  "
                  f"AvgDaily={sim['avg_daily_ret']:.3f}%")
        print()

        del test_panel, model; gc.collect()

    # ── Summary ──
    mean_ic = float(np.nanmean(ic_scores)) if ic_scores else 0
    mean_ls = float(np.nanmean(ls_scores)) if ls_scores else 0
    mean_prec = float(np.nanmean(prec_scores)) if prec_scores else 0

    sim_all = {}
    if all_test_preds:
        all_test = pd.concat(all_test_preds, ignore_index=True)
        sim_all = _profit_sim(all_test)
        del all_test; gc.collect()

    print(f"  {'=' * 54}")
    print(f"  VALIDATION SUMMARY ({len(ic_scores)} folds) — target={label_target}")
    print(f"  {'=' * 54}")
    print(f"  Mean Rank IC        : {mean_ic:+.4f}")
    print(f"  Mean L/S spread     : {mean_ls:+.4%}")
    print(f"  Mean Precision@10   : {mean_prec:.1%}")
    if sim_all:
        print(f"  Combined CAGR       : {sim_all.get('cagr', 0):.1f}%")
        print(f"  Combined Sharpe     : {sim_all.get('sharpe', 0):.3f}")
        print(f"  Max Drawdown        : {sim_all.get('max_drawdown', 0):.1f}%")
        print(f"  Win Rate            : {sim_all.get('win_rate', 0):.1f}%")
        print(f"  Avg Daily Return    : {sim_all.get('avg_daily_ret', 0):.3f}%")
    print(f"  Total train rows    : {total_train:,}")
    print(f"  Total test rows     : {total_test:,}")

    edge = "STRONG" if mean_ic > 0.05 else "EDGE" if mean_ic > 0.02 else "WEAK"
    print(f"  Verdict             : {edge}")

    # ── Final model ──
    print(f"\n[2/3] Training FINAL model on 2018-2025...")
    final_panel = _build_panel("2018-01-01", "2025-12-31")
    if final_panel.empty:
        print("  ERROR: No data")
        return {}
    final_panel = _prepare_panel(final_panel)
    n_final = len(final_panel)
    available_features = [f + "_z" for f in ALL_RAW_FEATURES
                          if f + "_z" in final_panel.columns]
    print(f"  Final: {n_final:,} rows, {final_panel['symbol'].nunique()} stocks, "
          f"{final_panel['date'].nunique()} dates, {len(available_features)} features")

    final_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=0.3,
        reg_lambda=2.0,
        min_child_weight=20,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    final_model.fit(final_panel[available_features], final_panel[label_col])

    imp = pd.Series(final_model.feature_importances_, index=available_features)
    imp = imp.sort_values(ascending=False)
    print("\n  Top 15 features:")
    for feat, val in imp.head(15).items():
        bar = "#" * int(val * 60)
        print(f"    {feat:<35} {bar} {val:.3f}")

    # ── Save ──
    print(f"\n[3/3] Saving model...")
    model_id = f"intraday_{label_target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_path = os.path.join(MODEL_DIR, f"{model_id}.pkl")
    latest_path = os.path.join(MODEL_DIR, f"latest_{label_target}.pkl")
    meta_path = os.path.join(MODEL_DIR, f"latest_{label_target}_meta.json")

    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    meta = {
        "model_id": model_id,
        "type": f"intraday_{label_target}_ranker",
        "label_target": label_target,
        "n_features": len(available_features),
        "features": available_features,
        "raw_features": [f for f in ALL_RAW_FEATURES if f + "_z" in available_features],
        "horizon": "intraday (open-to-close)",
        "n_symbols": final_panel["symbol"].nunique(),
        "n_dates": int(final_panel["date"].nunique()),
        "n_samples": n_final,
        "walk_forward": {
            "folds": [{"train": f"{s[:4]}-{e[:4]}", "test": f"{ts[:4]}-{te[:4]}"}
                      for s, e, ts, te in WALK_FORWARD_FOLDS],
            "fold_ics": [round(x, 4) for x in ic_scores],
        },
        "validation": {
            "mean_rank_ic": round(mean_ic, 4),
            "mean_ls_spread": round(mean_ls, 4),
            "mean_precision_at_10": round(mean_prec, 4),
        },
        "backtest": sim_all,
        "feature_importance": {k: round(v, 4) for k, v in imp.head(25).items()},
        "edge": edge,
        "trained_at": datetime.now().isoformat(),
        "training_time_sec": round(time.time() - t0, 1),
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    import shutil
    shutil.copy2(model_path, latest_path)

    del final_panel, final_model; gc.collect()

    print(f"\n  [OK] Model saved -> {model_path}")
    print(f"       Also -> {latest_path}")
    print(f"       {len(available_features)} features, {n_final:,} rows")
    print(f"       Training time: {time.time() - t0:.0f}s")

    return meta


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    target = "direction"
    if len(sys.argv) > 1 and sys.argv[1] == "range":
        target = "range"

    result = train(label_target=target)
    if result:
        print(f"\nDone. Edge: {result.get('edge')}, "
              f"Rank IC: {result.get('validation', {}).get('mean_rank_ic')}")

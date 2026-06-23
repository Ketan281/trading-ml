"""
Feature Store — pre-compute and persist features for all stocks.

Training reads from this store (fast, no CSV re-parsing).
Inference reads the latest row per stock (instant).
Daily update appends one new row per stock.

Storage: SQLite WAL mode — single file, zero config.
"""

import gc
import os
import sys
import json
import time
import sqlite3
import logging
import numpy as np
import pandas as pd
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

log = logging.getLogger("feature_store")
DB_PATH = os.path.join(ROOT, "data", "feature_store.db")
HIST_DIR = os.path.join(ROOT, "data", "historical")
DATA_DIR = os.path.join(ROOT, "data")
EXCLUDE = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"}

FEATURE_COLS = [
    # === TECHNICAL — MOMENTUM & TIMING (best-trade-today signals) ===
    "ret_1d", "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "roc_10", "roc_21", "momentum_score",
    "ret_5d_accel",  # momentum acceleration (today's 5d vs 5 days ago's 5d)

    # === MOVING AVERAGES (EMA 20/50/200 + crossovers) ===
    "dist_ema20", "dist_ema50", "dist_ema200",
    "above_20ema", "above_50ema", "above_200ma",
    "ema20_slope", "ema50_slope",
    "ema_20_50_cross", "ema_50_200_cross",

    # === OSCILLATORS ===
    "rsi_14", "rsi_slope",
    "macd_hist", "macd_signal_cross",
    "bb_pctb", "bb_width",
    "stochastic_k", "stochastic_d",

    # === SUPERTREND & VWAP ===
    "supertrend_signal", "supertrend_dist",
    "vwap_dist",

    # === TREND STRENGTH ===
    "adx_14", "trend_slope_10", "trend_slope_20",

    # === VOLATILITY ===
    "vol_10d", "vol_21d", "atr_ratio", "intraday_range",
    "vol_of_vol", "atr_expansion",

    # === VOLUME & INSTITUTIONAL FLOW ===
    "rel_volume", "vol_trend", "mfi_14", "ad_momentum",
    "up_down_vol_ratio", "vol_price_confirm", "obv_slope",

    # === S/R & STRUCTURE ===
    "sr_proximity", "fib_level", "range_pos_20",
    "dist_52w_high", "dist_52w_low",

    # === QUALITY / RISK ===
    "sharpe_60d", "stretch_atr", "calmar_ratio",

    # === 30 CANDLESTICK PATTERNS (vectorized, computed every bar) ===
    "pat_hammer", "pat_inv_hammer", "pat_shooting_star", "pat_hanging_man",
    "pat_doji", "pat_dragonfly_doji", "pat_gravestone_doji",
    "pat_marubozu", "pat_spinning_top",
    "pat_bull_engulf", "pat_bear_engulf",
    "pat_bull_harami", "pat_bear_harami",
    "pat_piercing", "pat_dark_cloud",
    "pat_morning_star", "pat_evening_star",
    "pat_three_white", "pat_three_black",
    "pat_tweezer_top", "pat_tweezer_bottom",
    "pat_bull_belt", "pat_bear_belt",
    "pat_bull_kicker", "pat_bear_kicker",
    "pat_net_score",  # net pattern score [-1, +1]
    "pat_bullish_count", "pat_bearish_count",

    # === FUNDAMENTALS (20 — static per stock) ===
    "pe_trailing", "pe_forward", "pb_ratio", "peg_ratio",
    "ev_ebitda", "ps_ratio",
    "roe", "roa", "profit_margin", "operating_margin", "gross_margin",
    "earnings_growth", "revenue_growth", "quarterly_earnings_growth",
    "debt_to_equity", "current_ratio", "quick_ratio",
    "dividend_yield", "free_cashflow_yield", "market_cap_log",

    # === INTERACTION: FUNDAMENTAL x TIMING (the "buy best trade today" signals) ===
    "value_momentum",      # cheap (low PE) + rising price
    "quality_breakout",    # high ROE + breaking above 200MA
    "growth_momentum",     # earnings growth + 5d positive return
    "cheap_reversal",      # low PB + RSI oversold + near support
    "quality_trend",       # high margin + supertrend UP + volume surge
    "value_vol_squeeze",   # cheap + low volatility (about to move)

    # === INTERMARKET (global macro — same for all stocks on a given date) ===
    "vix_level", "vix_regime", "ivix_level",
    "crude_ret_5d", "gold_ret_5d", "dxy_ret_5d", "usdinr_ret_5d",
    "sp500_ret_5d", "us10y_level",
    "fii_net_5d",  # FII net buy/sell 5-day sum (crores)

    # === CROSS-SECTIONAL (computed at panel level) ===
    "mom_rank", "vol_rank", "rs_score",
]

FUND_PATH = os.path.join(ROOT, "data", "historical", "fundamentals.json")
INTERMARKET_PATH = os.path.join(ROOT, "data", "market_intel", "intermarket_features.csv")
FII_PATH = os.path.join(ROOT, "data", "market_intel", "FII_DII_history.csv")
_fundamentals_cache = None
_intermarket_cache = None
_fii_cache = None


def _load_intermarket():
    global _intermarket_cache
    if _intermarket_cache is not None:
        return _intermarket_cache
    if os.path.exists(INTERMARKET_PATH):
        try:
            df = pd.read_csv(INTERMARKET_PATH, index_col="Date", parse_dates=True)
            df = df.sort_index()
            _intermarket_cache = df
        except Exception:
            _intermarket_cache = pd.DataFrame()
    else:
        _intermarket_cache = pd.DataFrame()
    return _intermarket_cache


def _load_fii_net():
    global _fii_cache
    if _fii_cache is not None:
        return _fii_cache
    if os.path.exists(FII_PATH):
        try:
            raw = pd.read_csv(FII_PATH)
            raw["date"] = pd.to_datetime(raw["date"], format="mixed", dayfirst=True)
            fii = raw[raw["category"].str.contains("FII", case=False, na=False)].copy()
            fii = fii.set_index("date").sort_index()
            fii_daily = fii["net_value"].groupby(fii.index).sum()
            _fii_cache = fii_daily.rolling(5, min_periods=1).sum()
        except Exception:
            _fii_cache = pd.Series(dtype=float)
    else:
        _fii_cache = pd.Series(dtype=float)
    return _fii_cache


def _load_fundamentals():
    global _fundamentals_cache
    if _fundamentals_cache is not None:
        return _fundamentals_cache
    if os.path.exists(FUND_PATH):
        try:
            with open(FUND_PATH) as f:
                _fundamentals_cache = json.load(f)
        except Exception:
            _fundamentals_cache = {}
    else:
        _fundamentals_cache = {}
    return _fundamentals_cache


def _db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db():
    conn = _db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS features (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            features TEXT NOT NULL,
            price REAL,
            computed_at TEXT,
            PRIMARY KEY (symbol, date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS store_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feat_date ON features(date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feat_sym ON features(symbol)")
    conn.commit()
    conn.close()


def _get_data_dir():
    import glob
    if os.path.isdir(HIST_DIR) and glob.glob(os.path.join(HIST_DIR, "*.csv")):
        return HIST_DIR
    return DATA_DIR


def _vectorized_patterns(o, h, l, c):
    """Compute 25+ candlestick pattern signals as vectorized arrays.
    Each pattern returns +1 (bullish), -1 (bearish), or 0 per bar.
    This runs on EVERY bar, so the model can learn pattern → outcome."""
    n = len(o)
    body = np.abs(c - o)
    rng = np.maximum(h - l, 1e-9)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    bull = (c > o).astype(float)
    bear = (c < o).astype(float)
    body_pct = body / rng

    # Prior trend (5-bar slope)
    prior_slope = np.zeros(n)
    for i in range(5, n):
        prior_slope[i] = (c[i-1] - c[i-5]) / (abs(c[i-5]) + 1e-9)

    up_trend = prior_slope > 0.01
    dn_trend = prior_slope < -0.01

    pat = {}

    # --- SINGLE CANDLE ---
    # Hammer: small body at top, long lower wick, after downtrend
    pat["pat_hammer"] = np.where(
        (lower_wick >= 2 * body) & (upper_wick <= body * 0.5) & dn_trend, 1.0, 0.0)
    # Inverted Hammer: small body at bottom, long upper wick, after downtrend
    pat["pat_inv_hammer"] = np.where(
        (upper_wick >= 2 * body) & (lower_wick <= body * 0.5) & dn_trend, 1.0, 0.0)
    # Shooting Star: long upper wick at top of uptrend
    pat["pat_shooting_star"] = np.where(
        (upper_wick >= 2 * body) & (lower_wick <= body * 0.5) & up_trend, -1.0, 0.0)
    # Hanging Man: hammer shape but in uptrend (bearish)
    pat["pat_hanging_man"] = np.where(
        (lower_wick >= 2 * body) & (upper_wick <= body * 0.5) & up_trend, -1.0, 0.0)
    # Doji: tiny body
    pat["pat_doji"] = np.where(body_pct <= 0.10, np.where(dn_trend, 1.0, np.where(up_trend, -1.0, 0.0)), 0.0)
    # Dragonfly Doji
    pat["pat_dragonfly_doji"] = np.where(
        (body_pct <= 0.10) & (lower_wick >= 0.6 * rng) & dn_trend, 1.0, 0.0)
    # Gravestone Doji
    pat["pat_gravestone_doji"] = np.where(
        (body_pct <= 0.10) & (upper_wick >= 0.6 * rng) & up_trend, -1.0, 0.0)
    # Marubozu: big body, no wicks
    pat["pat_marubozu"] = np.where(body_pct >= 0.90, np.where(bull, 1.0, -1.0), 0.0)
    # Spinning Top: small body, wicks both sides
    pat["pat_spinning_top"] = np.where(
        (body_pct <= 0.30) & (upper_wick >= 0.25 * rng) & (lower_wick >= 0.25 * rng), 0.0, 0.0)
    # Belt Hold
    pat["pat_bull_belt"] = np.where((bull > 0) & (lower_wick <= body * 0.05) & (body_pct >= 0.6) & dn_trend, 1.0, 0.0)
    pat["pat_bear_belt"] = np.where((bear > 0) & (upper_wick <= body * 0.05) & (body_pct >= 0.6) & up_trend, -1.0, 0.0)

    # --- DOUBLE CANDLE ---
    o1 = np.roll(o, 1); c1 = np.roll(c, 1)
    h1 = np.roll(h, 1); l1 = np.roll(l, 1)
    body1 = np.abs(c1 - o1)
    bull1 = c1 > o1; bear1 = c1 < o1

    # Bullish Engulfing: prev red, today green engulfs
    pat["pat_bull_engulf"] = np.where(
        bear1 & (bull > 0) & (o <= c1) & (c >= o1) & (body > body1) & dn_trend, 1.0, 0.0)
    pat["pat_bear_engulf"] = np.where(
        bull1 & (bear > 0) & (o >= c1) & (c <= o1) & (body > body1) & up_trend, -1.0, 0.0)
    # Harami
    pat["pat_bull_harami"] = np.where(
        bear1 & (bull > 0) & (o >= c1) & (c <= o1) & (body < body1 * 0.5) & dn_trend, 1.0, 0.0)
    pat["pat_bear_harami"] = np.where(
        bull1 & (bear > 0) & (o <= c1) & (c >= o1) & (body < body1 * 0.5) & up_trend, -1.0, 0.0)
    # Piercing Line: prev red, today opens below prev low, closes above midpoint
    prev_mid = (o1 + c1) / 2
    pat["pat_piercing"] = np.where(
        bear1 & (bull > 0) & (o < l1) & (c > prev_mid) & dn_trend, 1.0, 0.0)
    # Dark Cloud Cover
    pat["pat_dark_cloud"] = np.where(
        bull1 & (bear > 0) & (o > h1) & (c < prev_mid) & up_trend, -1.0, 0.0)
    # Tweezer
    pat["pat_tweezer_bottom"] = np.where(
        (np.abs(l - l1) <= 0.002 * l) & bear1 & (bull > 0) & dn_trend, 1.0, 0.0)
    pat["pat_tweezer_top"] = np.where(
        (np.abs(h - h1) <= 0.002 * h) & bull1 & (bear > 0) & up_trend, -1.0, 0.0)
    # Kicker
    pat["pat_bull_kicker"] = np.where(
        bear1 & (bull > 0) & (o > o1) & (body_pct >= 0.6) & (body / rng >= 0.5), 1.0, 0.0)
    pat["pat_bear_kicker"] = np.where(
        bull1 & (bear > 0) & (o < o1) & (body_pct >= 0.6) & (body / rng >= 0.5), -1.0, 0.0)

    # --- TRIPLE CANDLE ---
    o2 = np.roll(o, 2); c2 = np.roll(c, 2)
    bull2 = c2 > o2; bear2 = c2 < o2
    body2 = np.abs(c2 - o2)

    # Morning Star: red → small → green (bullish reversal)
    pat["pat_morning_star"] = np.where(
        bear2 & (body1 < body2 * 0.3) & (bull > 0) & (c > (o2 + c2) / 2) & dn_trend, 1.0, 0.0)
    pat["pat_evening_star"] = np.where(
        bull2 & (body1 < body2 * 0.3) & (bear > 0) & (c < (o2 + c2) / 2) & up_trend, -1.0, 0.0)
    # Three White Soldiers / Three Black Crows
    pat["pat_three_white"] = np.where(
        bull2 & bull1 & (bull > 0) & (c > c1) & (c1 > c2) & (body_pct >= 0.5), 1.0, 0.0)
    pat["pat_three_black"] = np.where(
        bear2 & bear1 & (bear > 0) & (c < c1) & (c1 < c2) & (body_pct >= 0.5), -1.0, 0.0)

    # Aggregate
    all_vals = np.stack(list(pat.values()), axis=0)
    pat["pat_net_score"] = np.clip(all_vals.sum(axis=0), -1, 1)
    pat["pat_bullish_count"] = (all_vals > 0).sum(axis=0).astype(float)
    pat["pat_bearish_count"] = (all_vals < 0).sum(axis=0).astype(float)

    # Zero out first 5 bars (need prior trend)
    for k in pat:
        pat[k][:5] = 0.0

    return pat


def compute_stock_features(df, symbol=None):
    """Compute 90+ features: technical + patterns + fundamentals + interactions.
    Patterns are computed vectorized on EVERY bar so the model learns timing."""
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    opn = df["Open"].values
    volume = df["Volume"].values
    n = len(df)

    cs = pd.Series(close, index=df.index)
    hs = pd.Series(high, index=df.index)
    ls = pd.Series(low, index=df.index)
    vs = pd.Series(volume, index=df.index)
    daily_ret = cs.pct_change()

    f = pd.DataFrame(index=df.index)

    # ═══════════════════════════════════════════════════════
    # MOMENTUM & TIMING
    # ═══════════════════════════════════════════════════════
    for d in [1, 5, 10, 21, 63, 126, 252]:
        f[f"ret_{d}d"] = cs.pct_change(d) if n > d else 0.0
    f["roc_10"] = cs.pct_change(10) * 100
    f["roc_21"] = cs.pct_change(21) * 100
    f["momentum_score"] = (
        0.1 * cs.pct_change(5).fillna(0) +
        0.2 * cs.pct_change(21).fillna(0) +
        0.3 * cs.pct_change(63).fillna(0) +
        0.4 * cs.pct_change(252).fillna(0)
    )
    # Momentum acceleration: is momentum INCREASING?
    ret5 = cs.pct_change(5)
    f["ret_5d_accel"] = ret5 - ret5.shift(5)

    # ═══════════════════════════════════════════════════════
    # MOVING AVERAGES: EMA 20, 50, 200 + crossovers
    # ═══════════════════════════════════════════════════════
    ema20 = cs.ewm(span=20).mean()
    ema50 = cs.ewm(span=50).mean()
    ema200 = cs.rolling(200).mean() if n > 200 else cs.ewm(span=200).mean()
    f["dist_ema20"] = cs / ema20 - 1
    f["dist_ema50"] = cs / ema50 - 1
    f["dist_ema200"] = cs / ema200 - 1
    f["above_20ema"] = (cs > ema20).astype(float)
    f["above_50ema"] = (cs > ema50).astype(float)
    f["above_200ma"] = (cs > ema200).astype(float) if n > 200 else 0.5
    f["ema20_slope"] = ema20.pct_change(5)
    f["ema50_slope"] = ema50.pct_change(5)
    ema20_above_50 = (ema20 > ema50).astype(int)
    f["ema_20_50_cross"] = ema20_above_50.diff().fillna(0)
    ema50_above_200 = (ema50 > ema200).astype(int)
    f["ema_50_200_cross"] = ema50_above_200.diff().fillna(0)

    # ═══════════════════════════════════════════════════════
    # OSCILLATORS
    # ═══════════════════════════════════════════════════════
    delta = cs.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss_s = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss_s.replace(0, np.nan))
    f["rsi_14"] = rsi
    f["rsi_slope"] = rsi.diff(5) / 5

    ema12 = cs.ewm(span=12).mean()
    ema26 = cs.ewm(span=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9).mean()
    f["macd_hist"] = macd - macd_signal
    macd_above = (macd > macd_signal).astype(int)
    f["macd_signal_cross"] = macd_above.diff().fillna(0)

    ma20 = cs.rolling(20).mean()
    std20 = cs.rolling(20).std()
    f["bb_pctb"] = (cs - ma20) / (2 * std20 + 1e-9)
    f["bb_width"] = (4 * std20) / (ma20 + 1e-9)

    low_14 = ls.rolling(14).min()
    high_14 = hs.rolling(14).max()
    f["stochastic_k"] = ((cs - low_14) / (high_14 - low_14 + 1e-9)) * 100
    f["stochastic_d"] = f["stochastic_k"].rolling(3).mean()

    # ═══════════════════════════════════════════════════════
    # SUPERTREND & VWAP
    # ═══════════════════════════════════════════════════════
    tr = pd.concat([hs - ls, (hs - cs.shift()).abs(), (ls - cs.shift()).abs()], axis=1).max(axis=1)
    atr10 = tr.rolling(10).mean()
    hl2 = (hs + ls) / 2
    ub_arr = (hl2 + 3 * atr10).values
    lb_arr = (hl2 - 3 * atr10).values
    dir_arr = np.ones(n, dtype=np.float64)
    st_arr = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > ub_arr[i - 1]:
            dir_arr[i] = 1
        elif close[i] < lb_arr[i - 1]:
            dir_arr[i] = -1
        else:
            dir_arr[i] = dir_arr[i - 1]
        st_arr[i] = lb_arr[i] if dir_arr[i] == 1 else ub_arr[i]
    f["supertrend_signal"] = dir_arr
    f["supertrend_dist"] = (close - st_arr) / (close + 1e-9)

    typical = (hs + ls + cs) / 3
    cum_tp_vol = (typical * vs).rolling(20).sum()
    cum_vol = vs.rolling(20).sum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    f["vwap_dist"] = (cs - vwap) / (cs + 1e-9)

    # ═══════════════════════════════════════════════════════
    # ADX + TREND SLOPES
    # ═══════════════════════════════════════════════════════
    plus_dm = pd.Series(np.where((hs.diff() > 0) & (hs.diff() > -ls.diff()), hs.diff(), 0), index=df.index)
    minus_dm = pd.Series(np.where((-ls.diff() > 0) & (-ls.diff() > hs.diff()), -ls.diff(), 0), index=df.index)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14.replace(0, np.nan))
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    f["adx_14"] = dx.rolling(14).mean()

    def _slope(x):
        return np.polyfit(np.arange(len(x)), x, 1)[0]
    f["trend_slope_10"] = cs.rolling(10).apply(_slope, raw=True) / cs
    f["trend_slope_20"] = cs.rolling(20).apply(_slope, raw=True) / cs

    # ═══════════════════════════════════════════════════════
    # VOLATILITY
    # ═══════════════════════════════════════════════════════
    f["vol_10d"] = daily_ret.rolling(10).std()
    f["vol_21d"] = daily_ret.rolling(21).std()
    f["atr_ratio"] = atr14 / cs
    f["intraday_range"] = (hs - ls) / cs
    f["vol_of_vol"] = daily_ret.rolling(21).std().rolling(21).std()
    atr_fast = tr.rolling(7).mean()
    atr_slow = tr.rolling(21).mean()
    f["atr_expansion"] = atr_fast / atr_slow.replace(0, np.nan)

    # ═══════════════════════════════════════════════════════
    # VOLUME & FLOW
    # ═══════════════════════════════════════════════════════
    avg_vol = vs.rolling(20).mean()
    f["rel_volume"] = vs / avg_vol.replace(0, np.nan)
    f["vol_trend"] = vs.rolling(5).mean() / avg_vol.replace(0, np.nan)
    mf = typical * vs
    pos_mf = pd.Series(np.where(typical > typical.shift(1), mf, 0), index=df.index)
    neg_mf = pd.Series(np.where(typical < typical.shift(1), mf, 0), index=df.index)
    mfr = pos_mf.rolling(14).sum() / neg_mf.rolling(14).sum().replace(0, np.nan)
    f["mfi_14"] = 100 - (100 / (1 + mfr))
    clv = ((cs - ls) - (hs - cs)) / (hs - ls).replace(0, np.nan)
    ad = (clv.fillna(0) * vs).cumsum()
    ad_ema5 = ad.ewm(span=5).mean()
    ad_ema20 = ad.ewm(span=20).mean()
    f["ad_momentum"] = np.where(ad_ema20 != 0, (ad_ema5 / ad_ema20 - 1) * 100, 0).astype(float)
    up_vol = pd.Series(np.where(daily_ret > 0, volume, 0), index=df.index)
    dn_vol = pd.Series(np.where(daily_ret < 0, volume, 0), index=df.index)
    f["up_down_vol_ratio"] = up_vol.rolling(20).sum() / dn_vol.rolling(20).sum().replace(0, np.nan)
    vol_surge = vs.rolling(5).mean() / avg_vol.replace(0, np.nan)
    f["vol_price_confirm"] = np.where(
        (ret5 > 0) & (vol_surge > 1.5), 1.0,
        np.where((ret5 < 0) & (vol_surge > 1.5), -1.0, 0.0))
    obv = (np.sign(daily_ret.fillna(0)) * vs).cumsum()
    f["obv_slope"] = obv.pct_change(10)

    # ═══════════════════════════════════════════════════════
    # S/R & STRUCTURE
    # ═══════════════════════════════════════════════════════
    high_20 = hs.rolling(20).max()
    low_20 = ls.rolling(20).min()
    range_20 = high_20 - low_20
    f["sr_proximity"] = np.where(range_20 > 0, (cs - low_20) / range_20, 0.5)
    high_52 = hs.rolling(252).max()
    low_52 = ls.rolling(252).min()
    range_52 = high_52 - low_52
    f["fib_level"] = np.where(range_52 > 0, (high_52 - cs) / range_52, 0.5)
    f["range_pos_20"] = (cs - low_20) / (high_20 - low_20 + 1e-9)
    f["dist_52w_high"] = cs / high_52 - 1
    f["dist_52w_low"] = cs / low_52 - 1

    # ═══════════════════════════════════════════════════════
    # RISK
    # ═══════════════════════════════════════════════════════
    ret_60 = cs.pct_change(60)
    vol_60 = daily_ret.rolling(60).std()
    f["sharpe_60d"] = np.where(vol_60 > 0, ret_60 / vol_60, 0).astype(float)
    f["stretch_atr"] = (cs - ema20) / atr14.replace(0, np.nan)
    max_dd_60 = (cs / cs.rolling(60).max() - 1).rolling(60).min()
    f["calmar_ratio"] = np.where(max_dd_60 < -0.01, ret_60.fillna(0) / abs(max_dd_60), 0).astype(float)

    # ═══════════════════════════════════════════════════════
    # 30 CANDLESTICK PATTERNS — VECTORIZED, EVERY BAR
    # ═══════════════════════════════════════════════════════
    pat = _vectorized_patterns(opn, high, low, close)
    for k, v in pat.items():
        f[k] = v

    # ═══════════════════════════════════════════════════════
    # FUNDAMENTALS (20 — static per stock)
    # ═══════════════════════════════════════════════════════
    fund = _load_fundamentals()
    sym_fund = fund.get(symbol, {}) if symbol else {}

    def _fval(key, default=0.0):
        v = sym_fund.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return float(v)

    f["pe_trailing"] = _fval("trailingPE", 25)
    f["pe_forward"] = _fval("forwardPE", 25)
    f["pb_ratio"] = _fval("priceToBook", 3)
    f["peg_ratio"] = _fval("pegRatio", 1.5)
    f["ev_ebitda"] = _fval("enterpriseToEbitda", 15)
    f["ps_ratio"] = _fval("priceToSalesTrailing12Months", 5)
    f["roe"] = _fval("returnOnEquity", 0.1)
    f["roa"] = _fval("returnOnAssets", 0.05)
    f["profit_margin"] = _fval("profitMargins", 0.1)
    f["operating_margin"] = _fval("operatingMargins", 0.1)
    f["gross_margin"] = _fval("grossMargins", 0.3)
    f["earnings_growth"] = _fval("earningsGrowth", 0)
    f["revenue_growth"] = _fval("revenueGrowth", 0)
    f["quarterly_earnings_growth"] = _fval("earningsQuarterlyGrowth", 0)
    f["debt_to_equity"] = _fval("debtToEquity", 50)
    f["current_ratio"] = _fval("currentRatio", 1.5)
    f["quick_ratio"] = _fval("quickRatio", 1.0)
    f["dividend_yield"] = _fval("dividendYield", 0)
    fcf = _fval("freeCashflow", 0)
    mcap = _fval("marketCap", 1e10)
    f["free_cashflow_yield"] = fcf / mcap if mcap > 0 else 0
    f["market_cap_log"] = np.log10(max(mcap, 1))

    # ═══════════════════════════════════════════════════════
    # INTERACTION FEATURES: FUNDAMENTAL x TIMING
    # These change daily because they combine static fundamentals
    # with dynamic technical signals — THIS is what finds "best trade TODAY"
    # ═══════════════════════════════════════════════════════
    pe = f["pe_trailing"]
    roe_val = f["roe"]
    eg = f["earnings_growth"]
    pb = f["pb_ratio"]
    pm = f["profit_margin"]

    # Value + momentum: cheap stock that's also going up
    f["value_momentum"] = np.where(pe < 20, 1.0, 0.0) * f["ret_5d"].clip(-0.2, 0.2) * 10

    # Quality + breakout: high ROE stock breaking above 200 MA
    f["quality_breakout"] = np.where(roe_val > 0.15, 1.0, 0.0) * f["above_200ma"] * f["ema20_slope"].clip(-0.05, 0.05) * 20

    # Growth + momentum: growing earnings AND rising price
    f["growth_momentum"] = np.where(eg > 0.1, 1.0, 0.0) * np.where(f["ret_5d"] > 0, 1.0, 0.0)

    # Cheap reversal: low PB + RSI oversold + near support
    f["cheap_reversal"] = (
        np.where(pb < 2, 1.0, 0.0) *
        np.where(f["rsi_14"] < 35, 1.0, 0.0) *
        np.where(f["sr_proximity"] < 0.2, 1.0, 0.0)
    )

    # Quality + trend: high margin + supertrend UP + volume surge
    f["quality_trend"] = (
        np.where(pm > 0.15, 1.0, 0.0) *
        np.where(f["supertrend_signal"] > 0, 1.0, 0.0) *
        np.where(f["rel_volume"] > 1.2, 1.0, 0.0)
    )

    # Value + vol squeeze: cheap stock in low vol (about to break)
    f["value_vol_squeeze"] = (
        np.where(pe < 20, 1.0, 0.0) *
        np.where(f["bb_width"] < f["bb_width"].rolling(50).quantile(0.2), 1.0, 0.0)
    )

    # ═══════════════════════════════════════════════════════
    # INTERMARKET — global macro context (same for all stocks on a date)
    # ═══════════════════════════════════════════════════════
    im = _load_intermarket()
    fii_net = _load_fii_net()
    if not im.empty:
        im_aligned = im.reindex(df.index, method="ffill")
        for col, src in [
            ("vix_level", "vix_level"), ("vix_regime", "vix_regime"),
            ("ivix_level", "ivix_level"),
            ("crude_ret_5d", "crude_ret_5d"), ("gold_ret_5d", "gold_ret_5d"),
            ("dxy_ret_5d", "dxy_ret_5d"), ("usdinr_ret_5d", "usdinr_ret_5d"),
            ("sp500_ret_5d", "sp500_ret_5d"), ("us10y_level", "us10y_level"),
        ]:
            f[col] = im_aligned[src].values if src in im_aligned.columns else 0.0
    else:
        for col in ["vix_level", "vix_regime", "ivix_level", "crude_ret_5d",
                     "gold_ret_5d", "dxy_ret_5d", "usdinr_ret_5d",
                     "sp500_ret_5d", "us10y_level"]:
            f[col] = 0.0

    if not fii_net.empty:
        fii_aligned = fii_net.reindex(df.index, method="ffill").fillna(0)
        f["fii_net_5d"] = fii_aligned.values
    else:
        f["fii_net_5d"] = 0.0

    # Cross-sectional (placeholder)
    f["mom_rank"] = 0.0
    f["vol_rank"] = 0.0
    f["rs_score"] = 0.0

    return f


def build_full_store(max_stocks=None):
    """One-time: compute features for ALL stocks × ALL dates, store in SQLite.
    Run locally (needs RAM). Takes ~5-10 min for 434 stocks."""
    init_db()
    src = _get_data_dir()
    import glob
    files = glob.glob(os.path.join(src, "*.csv"))
    if max_stocks:
        files = files[:max_stocks]

    conn = _db()
    total = 0
    t0 = time.time()

    for i, path in enumerate(files):
        name = os.path.basename(path).replace(".csv", "").replace("_daily", "")
        if name.lower() in ("manifest",) or name in EXCLUDE:
            continue

        try:
            df = pd.read_csv(path, index_col="Date", parse_dates=True,
                             usecols=["Date", "Open", "High", "Low", "Close", "Volume"])
            if len(df) < 60:
                continue
            df = df.sort_index()

            feat = compute_stock_features(df, symbol=name)
            feat = feat.dropna(subset=["ret_21d", "rsi_14", "vol_21d"])

            now = datetime.now().isoformat()
            rows = []
            for date, row in feat.iterrows():
                fdict = {}
                for col in FEATURE_COLS:
                    val = row.get(col, 0.0)
                    fdict[col] = round(float(val), 6) if pd.notna(val) else 0.0
                price = float(df.loc[date, "Close"]) if date in df.index else 0.0
                rows.append((name, str(date.date()), json.dumps(fdict), price, now))

            conn.executemany(
                "INSERT OR REPLACE INTO features (symbol, date, features, price, computed_at) "
                "VALUES (?, ?, ?, ?, ?)", rows
            )
            conn.commit()
            total += len(rows)

            if (i + 1) % 20 == 0:
                log.warning("Feature store: %d/%d stocks, %d rows, %.0fs",
                            i + 1, len(files), total, time.time() - t0)
                gc.collect()

        except Exception as e:
            log.warning("Feature store skip %s: %s", name, e)

    # Store metadata
    conn.execute("INSERT OR REPLACE INTO store_meta VALUES (?, ?)",
                 ("last_full_build", datetime.now().isoformat()))
    conn.execute("INSERT OR REPLACE INTO store_meta VALUES (?, ?)",
                 ("total_rows", str(total)))
    conn.commit()
    conn.close()

    elapsed = time.time() - t0
    log.warning("Feature store built: %d stocks, %d rows in %.0fs", i + 1, total, elapsed)
    return {"stocks": i + 1, "rows": total, "elapsed": round(elapsed, 1)}


def update_daily():
    """Append today's features for all stocks. Fast — only reads last 260 rows per CSV."""
    init_db()
    src = _get_data_dir()
    import glob
    files = glob.glob(os.path.join(src, "*.csv"))

    conn = _db()
    updated = 0
    today = datetime.now().strftime("%Y-%m-%d")

    for path in files:
        name = os.path.basename(path).replace(".csv", "").replace("_daily", "")
        if name.lower() in ("manifest",) or name in EXCLUDE:
            continue
        try:
            # Only read tail
            with open(path, 'r') as f:
                n_lines = sum(1 for _ in f) - 1
            skip = max(0, n_lines - 260)
            df = pd.read_csv(path, index_col="Date", parse_dates=True,
                             usecols=["Date", "Open", "High", "Low", "Close", "Volume"],
                             skiprows=range(1, skip + 1) if skip > 0 else None)
            if len(df) < 60:
                continue
            df = df.sort_index()

            feat = compute_stock_features(df, symbol=name)
            last_row = feat.iloc[-1]
            last_date = str(feat.index[-1].date())
            price = float(df["Close"].iloc[-1])

            fdict = {}
            for col in FEATURE_COLS:
                val = last_row.get(col, 0.0)
                fdict[col] = round(float(val), 6) if pd.notna(val) else 0.0

            conn.execute(
                "INSERT OR REPLACE INTO features VALUES (?, ?, ?, ?, ?)",
                (name, last_date, json.dumps(fdict), price, datetime.now().isoformat())
            )
            updated += 1
        except Exception:
            pass

    conn.commit()
    conn.close()
    return updated


def load_training_panel(min_date=None, max_date=None, recent_years=5):
    """Load features from SQLite into a training-ready DataFrame.
    Defaults to last 5 years to avoid MemoryError on large stores.
    Returns: DataFrame with columns [symbol, date, price] + FEATURE_COLS."""
    conn = _db()
    conditions = []
    params = []

    if min_date:
        conditions.append("date >= ?")
        params.append(min_date)
    elif recent_years:
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=recent_years * 365)).strftime("%Y-%m-%d")
        conditions.append("date >= ?")
        params.append(cutoff)
    if max_date:
        conditions.append("date <= ?")
        params.append(max_date)

    query = "SELECT symbol, date, features, price FROM features"
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY date, symbol"

    records = []
    for row in conn.execute(query, params):
        sym, date, feat_json, price = row
        feat = json.loads(feat_json)
        feat["symbol"] = sym
        feat["date"] = date
        feat["price"] = price
        records.append(feat)
    conn.close()

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_latest_features():
    """Load the most recent feature row per stock. For inference."""
    conn = _db()
    rows = conn.execute("""
        SELECT f.symbol, f.features, f.price, f.date
        FROM features f
        INNER JOIN (SELECT symbol, MAX(date) as max_date FROM features GROUP BY symbol) m
        ON f.symbol = m.symbol AND f.date = m.max_date
    """).fetchall()
    conn.close()

    records = []
    for sym, feat_json, price, date in rows:
        feat = json.loads(feat_json)
        feat["symbol"] = sym
        feat["price"] = price
        feat["date"] = date
        records.append(feat)

    return pd.DataFrame(records) if records else pd.DataFrame()


def get_store_stats():
    """Return stats about the feature store."""
    try:
        conn = _db()
        total = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
        symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM features").fetchone()[0]
        dates = conn.execute("SELECT COUNT(DISTINCT date) FROM features").fetchone()[0]
        latest = conn.execute("SELECT MAX(date) FROM features").fetchone()[0]
        meta_row = conn.execute("SELECT value FROM store_meta WHERE key='last_full_build'").fetchone()
        conn.close()
        return {
            "total_rows": total,
            "symbols": symbols,
            "dates": dates,
            "latest_date": latest,
            "last_full_build": meta_row[0] if meta_row else None,
        }
    except Exception:
        return {"total_rows": 0, "symbols": 0, "error": "store not initialized"}


SNAPSHOT_PATH = os.path.join(ROOT, "data", "latest_features.json.gz")


def export_latest_snapshot():
    """Export latest features to a compact gzipped JSON (~1MB).
    This is what gets deployed to the 1GB server — no 5GB SQLite needed."""
    latest = load_latest_features()
    if latest.empty:
        return {"error": "no features"}
    records = latest.to_dict(orient="records")
    import gzip
    with gzip.open(SNAPSHOT_PATH, "wt") as f:
        json.dump(records, f)
    size_mb = os.path.getsize(SNAPSHOT_PATH) / 1e6
    log.warning("Exported %d stocks to %s (%.1f MB)", len(records), SNAPSHOT_PATH, size_mb)
    return {"stocks": len(records), "path": SNAPSHOT_PATH, "size_mb": round(size_mb, 2)}


def load_latest_features_fast():
    """Load features from gzipped snapshot (for 1GB server — no SQLite needed)."""
    if os.path.exists(SNAPSHOT_PATH):
        import gzip
        with gzip.open(SNAPSHOT_PATH, "rt") as f:
            records = json.load(f)
        return pd.DataFrame(records)
    return load_latest_features()


if __name__ == "__main__":
    import sys as _sys
    logging.basicConfig(level=logging.WARNING)
    if "--update" in _sys.argv:
        n = update_daily()
        print(f"Updated {n} stocks")
    elif "--export" in _sys.argv:
        result = export_latest_snapshot()
        print(f"Exported: {result}")
    else:
        print("Building full feature store...")
        result = build_full_store()
        print(f"Done: {result}")

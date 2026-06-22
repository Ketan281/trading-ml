import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(
       os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA_DIR    = os.path.join(ROOT, "data",
                            "historical")
FEATURE_DIR = os.path.join(ROOT, "data",
                            "features")
os.makedirs(FEATURE_DIR, exist_ok=True)

# ── Load Symbol Data ──────────────────────────────────
def load_symbol(name):
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(
        path,
        index_col="Date",
        parse_dates=True
    )
    df = df.dropna()
    return df

# ── Trend Features ────────────────────────────────────
def add_trend_features(df):
    close = df["Close"]

    # EMAs
    for span in [5, 9, 20, 50, 100, 200]:
        df[f"ema_{span}"] = close.ewm(
            span=span
        ).mean()

    # Price vs EMAs (normalized)
    for span in [9, 20, 50, 200]:
        df[f"price_vs_ema{span}"] = (
            close / df[f"ema_{span}"] - 1
        ) * 100

    # EMA crossovers
    df["ema9_vs_21"]  = (
        df["ema_9"] / df["ema_20"] - 1
    ) * 100
    df["ema20_vs_50"] = (
        df["ema_20"] / df["ema_50"] - 1
    ) * 100
    df["ema50_vs_200"] = (
        df["ema_50"] / df["ema_200"] - 1
    ) * 100

    # Higher highs / lower lows
    high = df["High"]
    low  = df["Low"]
    df["hh_10"] = (
        high.rolling(10).max() ==
        high
    ).astype(int)
    df["ll_10"] = (
        low.rolling(10).min() ==
        low
    ).astype(int)

    # ADX
    df = add_adx(df)

    return df

# ── ADX ───────────────────────────────────────────────
def add_adx(df, period=14):
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]

    plus_dm  = high.diff()
    minus_dm = low.diff().abs()
    plus_dm[plus_dm  < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr      = tr.rolling(period).mean()
    plus_di  = 100 * (
        plus_dm.rolling(period).mean() / atr
    )
    minus_di = 100 * (
        minus_dm.rolling(period).mean() / atr
    )
    dx       = (
        100 * (plus_di - minus_di).abs() /
        (plus_di + minus_di)
    )

    df["adx"]      = dx.rolling(period).mean()
    df["plus_di"]  = plus_di
    df["minus_di"] = minus_di
    df["di_diff"]  = plus_di - minus_di

    return df

# ── Momentum Features ─────────────────────────────────
def add_momentum_features(df):
    close = df["Close"]

    # RSI
    for period in [7, 14, 21]:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(
            period
        ).mean()
        loss  = (-delta.clip(upper=0)).rolling(
            period
        ).mean()
        rs    = gain / loss
        df[f"rsi_{period}"] = 100 - (
            100 / (1 + rs)
        )

    # MACD
    ema12       = close.ewm(span=12).mean()
    ema26       = close.ewm(span=26).mean()
    macd        = ema12 - ema26
    signal      = macd.ewm(span=9).mean()
    df["macd"]        = macd
    df["macd_signal"] = signal
    df["macd_hist"]   = macd - signal
    df["macd_cross"]  = (
        (macd > signal) &
        (macd.shift(1) <= signal.shift(1))
    ).astype(int)

    # Rate of Change
    for period in [5, 10, 20, 60]:
        df[f"roc_{period}"] = (
            close.pct_change(period) * 100
        )

    # Stochastic
    for k in [14, 21]:
        low_k  = df["Low"].rolling(k).min()
        high_k = df["High"].rolling(k).max()
        df[f"stoch_k_{k}"] = (
            (close - low_k) /
            (high_k - low_k) * 100
        )
        df[f"stoch_d_{k}"] = df[
            f"stoch_k_{k}"
        ].rolling(3).mean()

    # Williams %R
    df["williams_r"] = (
        (df["High"].rolling(14).max() - close) /
        (df["High"].rolling(14).max() -
         df["Low"].rolling(14).min()) * -100
    )

    # CCI
    tp         = (
        df["High"] + df["Low"] + close
    ) / 3
    mad        = tp.rolling(20).apply(
        lambda x: np.abs(
            x - x.mean()
        ).mean()
    )
    df["cci"]  = (
        tp - tp.rolling(20).mean()
    ) / (0.015 * mad)

    return df

# ── Volatility Features ───────────────────────────────
def add_volatility_features(df):
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]

    # ATR
    for period in [7, 14, 21]:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs()
        ], axis=1).max(axis=1)
        df[f"atr_{period}"] = tr.rolling(
            period
        ).mean()
        df[f"atr_pct_{period}"] = (
            df[f"atr_{period}"] / close * 100
        )

    # Historical Volatility
    returns = close.pct_change()
    for period in [10, 20, 60]:
        df[f"hv_{period}"] = (
            returns.rolling(period).std() *
            np.sqrt(252) * 100
        )

    # Bollinger Bands
    for period in [20]:
        ma  = close.rolling(period).mean()
        std = close.rolling(period).std()
        df[f"bb_upper_{period}"] = ma + 2 * std
        df[f"bb_lower_{period}"] = ma - 2 * std
        df[f"bb_width_{period}"] = (
            4 * std / ma * 100
        )
        df[f"bb_pct_{period}"]   = (
            (close - ma) / (2 * std)
        )
        df[f"bb_position_{period}"] = np.where(
            close > ma + 2 * std, 1,
            np.where(
                close < ma - 2 * std, -1, 0
            )
        )

    # Keltner Channels
    ema20      = close.ewm(span=20).mean()
    atr14      = df["atr_14"]
    df["kc_upper"] = ema20 + 2 * atr14
    df["kc_lower"] = ema20 - 2 * atr14
    df["kc_pct"]   = (
        (close - ema20) / (2 * atr14)
    )

    # Volatility regime
    hv20 = df["hv_20"]
    df["vol_regime"] = pd.cut(
        hv20,
        bins=[0, 10, 15, 20, 30, 999],
        labels=[0, 1, 2, 3, 4]
    ).astype(float)

    return df

# ── Volume Features ───────────────────────────────────
def add_volume_features(df):
    close  = df["Close"]
    volume = df["Volume"]

    if volume.sum() == 0:
        # Indices don't have real volume
        df["volume_ratio_20"] = 1.0
        df["obv_trend"]       = 0.0
        df["vwap_dev"]        = 0.0
        return df

    # Volume ratios
    for period in [5, 10, 20]:
        df[f"volume_ratio_{period}"] = (
            volume / volume.rolling(period).mean()
        )

    # OBV
    obv = (
        np.sign(close.diff()) * volume
    ).cumsum()
    df["obv"]       = obv
    df["obv_trend"] = (
        obv / obv.rolling(20).mean() - 1
    ) * 100

    # VWAP deviation
    typical = (
        df["High"] + df["Low"] + close
    ) / 3
    vwap    = (
        typical * volume
    ).rolling(20).sum() / volume.rolling(20).sum()
    df["vwap_dev"] = (
        (close - vwap) / vwap * 100
    )

    # Volume price trend
    df["vpt"] = (
        close.pct_change() * volume
    ).cumsum()

    # Force index
    df["force_index"] = close.diff() * volume

    # Accumulation/Distribution
    clv = (
        (close - df["Low"]) -
        (df["High"] - close)
    ) / (df["High"] - df["Low"])
    df["ad"] = (clv * volume).cumsum()

    return df

# ── Price Action Features ─────────────────────────────
def add_price_action_features(df):
    open_  = df["Open"]
    high   = df["High"]
    low    = df["Low"]
    close  = df["Close"]

    # Candle body
    df["body_size"]  = (
        (close - open_).abs() /
        open_ * 100
    )
    df["upper_wick"] = (
        high - close.clip(lower=open_)
    ) / open_ * 100
    df["lower_wick"] = (
        open_.clip(upper=close) - low
    ) / open_ * 100
    df["is_bullish"] = (
        close > open_
    ).astype(int)

    # Gap
    df["gap_pct"] = (
        (open_ - close.shift(1)) /
        close.shift(1) * 100
    )
    df["gap_up"]   = (
        df["gap_pct"] > 0.5
    ).astype(int)
    df["gap_down"] = (
        df["gap_pct"] < -0.5
    ).astype(int)

    # Inside bar
    df["inside_bar"] = (
        (high < high.shift(1)) &
        (low  > low.shift(1))
    ).astype(int)

    # Outside bar
    df["outside_bar"] = (
        (high > high.shift(1)) &
        (low  < low.shift(1))
    ).astype(int)

    # Doji
    df["doji"] = (
        df["body_size"] < 0.1
    ).astype(int)

    # Returns
    for period in [1, 2, 3, 5, 10, 20]:
        df[f"return_{period}d"] = (
            close.pct_change(period) * 100
        )

    # Rolling max/min distance
    for period in [20, 52]:
        df[f"dist_from_high_{period}"] = (
            (close - high.rolling(period).max()) /
            high.rolling(period).max() * 100
        )
        df[f"dist_from_low_{period}"] = (
            (close - low.rolling(period).min()) /
            low.rolling(period).min() * 100
        )

    return df

# ── Calendar Features ─────────────────────────────────
def add_calendar_features(df):
    idx = df.index

    df["day_of_week"]  = idx.dayofweek
    df["week_of_year"] = idx.isocalendar(
    ).week.astype(int)
    df["month"]        = idx.month
    df["quarter"]      = idx.quarter
    df["year"]         = idx.year

    # Is Monday / Friday
    df["is_monday"]  = (
        idx.dayofweek == 0
    ).astype(int)
    df["is_friday"]  = (
        idx.dayofweek == 4
    ).astype(int)

    # Month start/end
    df["is_month_start"] = (
        idx.is_month_start
    ).astype(int)
    df["is_month_end"]   = (
        idx.is_month_end
    ).astype(int)

    # Results season
    df["results_season"] = idx.month.isin(
        [1, 2, 4, 5, 7, 8, 10, 11]
    ).astype(int)

    return df

# ── IV Proxy Features ─────────────────────────────────
# Since we don't have historical options data
# we estimate IV from price volatility
def add_iv_proxy_features(df):
    close   = df["Close"]
    returns = close.pct_change()

    # Short term vol (proxy for ATM IV)
    df["iv_proxy_5d"]  = (
        returns.rolling(5).std() *
        np.sqrt(252) * 100
    )
    df["iv_proxy_10d"] = (
        returns.rolling(10).std() *
        np.sqrt(252) * 100
    )
    df["iv_proxy_20d"] = (
        returns.rolling(20).std() *
        np.sqrt(252) * 100
    )

    # IV percentile (rank in last 252 days)
    df["iv_percentile"] = (
        df["iv_proxy_20d"].rolling(252).rank(
            pct=True
        ) * 100
    )

    # IV regime
    df["iv_regime"] = pd.cut(
        df["iv_proxy_20d"],
        bins=[0, 8, 12, 18, 25, 999],
        labels=["very_low", "low",
                "normal", "high", "extreme"]
    )

    # Vol of vol (vol regime change signal)
    df["vol_of_vol"] = (
        df["iv_proxy_20d"].rolling(20).std()
    )

    # IV mean reversion signal
    iv_mean = df["iv_proxy_20d"].rolling(60).mean()
    iv_std  = df["iv_proxy_20d"].rolling(60).std()
    df["iv_zscore"] = (
        (df["iv_proxy_20d"] - iv_mean) / iv_std
    )

    return df

# ── Market Regime Features ────────────────────────────
def add_regime_features(df):
    close = df["Close"]

    # Trend regime (encoded)
    ema9  = close.ewm(span=9).mean()
    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()

    conditions = [
        (close > ema9) & (ema9 > ema20) & (ema20 > ema50),
        (close > ema20) & (ema20 > ema50),
        (close > ema50),
        (close < ema9) & (ema9 < ema20) & (ema20 < ema50),
        (close < ema20) & (ema20 < ema50),
        (close < ema50)
    ]
    choices = [3, 2, 1, -3, -2, -1]
    df["trend_regime"] = np.select(
        conditions, choices, default=0
    )

    # Momentum regime
    roc20 = close.pct_change(20) * 100
    df["momentum_regime"] = pd.cut(
        roc20,
        bins=[-999, -10, -5, -1,
              1, 5, 10, 999],
        labels=[-3, -2, -1, 0, 1, 2, 3]
    ).astype(float)

    # Combined regime score
    df["regime_score"] = (
        df["trend_regime"].fillna(0) +
        df["momentum_regime"].fillna(0)
    )

    return df

# ── Real Intermarket Features (from actual market data) ──
INTEL_DIR = os.path.join(ROOT, "data", "market_intel")
_intermarket_cache = None

def _load_intermarket_matrix():
    """Load pre-built intermarket feature matrix (real data from
    crude, DXY, VIX, gold, US indices, FII/DII, USD/INR).
    Built by training/fetch_real_market_data.py."""
    global _intermarket_cache
    if _intermarket_cache is not None:
        return _intermarket_cache
    path = os.path.join(INTEL_DIR, "intermarket_features.csv")
    if os.path.exists(path):
        _intermarket_cache = pd.read_csv(path, index_col="Date", parse_dates=True)
        return _intermarket_cache
    return None


def add_intermarket_features(df):
    """Join REAL intermarket data (crude, DXY, VIX, gold, US indices,
    FII/DII flows, USD/INR) into the stock feature set.

    Data comes from training/fetch_real_market_data.py which downloads
    20 years of actual global market data via yfinance + NSE APIs.
    Falls back to price-derived features only when real data is missing.
    """
    close = df["Close"]
    volume = df["Volume"]

    # ── Always compute price-derived institutional signals ──
    # MFI — real money flow from price + volume (not a proxy)
    typical = (df["High"] + df["Low"] + close) / 3
    mf = typical * volume
    pos_mf = pd.Series(np.where(typical > typical.shift(1), mf, 0), index=df.index)
    neg_mf = pd.Series(np.where(typical < typical.shift(1), mf, 0), index=df.index)
    mfr = pos_mf.rolling(14).sum() / neg_mf.rolling(14).sum().replace(0, np.nan)
    df["mfi_14"] = 100 - (100 / (1 + mfr))

    # A/D line momentum — real accumulation/distribution signal
    clv = ((close - df["Low"]) - (df["High"] - close)) / \
          (df["High"] - df["Low"]).replace(0, np.nan)
    ad = (clv.fillna(0) * volume).cumsum()
    ad_ema5 = ad.ewm(span=5).mean()
    ad_ema20 = ad.ewm(span=20).mean()
    df["ad_momentum"] = np.where(
        ad_ema20 != 0, (ad_ema5 / ad_ema20 - 1) * 100, 0
    )

    # S/R proximity — real support/resistance from price structure
    high_20 = df["High"].rolling(20).max()
    low_20 = df["Low"].rolling(20).min()
    high_52 = df["High"].rolling(252).max()
    low_52 = df["Low"].rolling(252).min()
    range_20 = high_20 - low_20
    range_52 = high_52 - low_52
    df["sr_proximity_20"] = np.where(range_20 > 0, (close - low_20) / range_20, 0.5)
    df["sr_proximity_52"] = np.where(range_52 > 0, (close - low_52) / range_52, 0.5)
    df["fib_level"] = np.where(range_52 > 0, (high_52 - close) / range_52, 0.5)

    # Volume-weighted RSI — real institutional conviction
    delta = close.diff()
    vol_gain = (delta.clip(lower=0) * volume).rolling(14).sum()
    vol_loss = (-delta.clip(upper=0) * volume).rolling(14).sum()
    df["vol_rsi"] = np.where(vol_loss != 0, 100 - (100 / (1 + vol_gain / vol_loss)), 50)

    # Volume-price confirmation — real delivery signal
    ret_5d = close.pct_change(5)
    vol_5d = volume.rolling(5).mean()
    vol_20d = volume.rolling(20).mean()
    vol_surge = vol_5d / vol_20d
    df["vol_price_confirm"] = np.where(
        (ret_5d > 0) & (vol_surge > 1.5), 1,
        np.where((ret_5d < 0) & (vol_surge > 1.5), -1, 0)
    ).astype(float)

    # ── Join REAL intermarket data ──
    im = _load_intermarket_matrix()
    if im is not None and len(im) > 0:
        # Align by date — forward-fill intermarket data for Indian holidays
        joined_cols = []
        for col in im.columns:
            if col not in df.columns:
                aligned = im[col].reindex(df.index, method="ffill")
                if aligned.notna().sum() > len(df) * 0.3:
                    df[col] = aligned
                    joined_cols.append(col)
        if joined_cols:
            print(f"[+{len(joined_cols)} real intermarket] ", end="")
    else:
        # Fallback: derive what we can from own price action
        ma200 = close.rolling(200).mean()
        df["rs_vs_ma200"] = (close / ma200 - 1) * 100
        hv_5 = close.pct_change().rolling(5).std() * np.sqrt(252) * 100
        hv_20 = close.pct_change().rolling(20).std() * np.sqrt(252) * 100
        df["fear_gauge"] = np.where(hv_20 > 0, (hv_5 / hv_20 - 1) * 100, 0)

    return df


def add_fundamental_proxy_features(df):
    """Quality signals derived from price/volume behavior.

    These are NOT proxies — they're real signals that reveal fundamental
    quality: drawdown depth, recovery speed, Sharpe ratio, volume
    asymmetry. Professional quant funds use exactly these features.
    """
    close = df["Close"]
    volume = df["Volume"]

    rolling_max = close.rolling(252).max()
    drawdown = (close / rolling_max - 1) * 100
    df["max_drawdown_252"] = drawdown.rolling(252).min()
    df["current_drawdown"] = drawdown

    ret_60 = close.pct_change(60)
    vol_60 = close.pct_change().rolling(60).std()
    df["sharpe_60d"] = np.where(vol_60 > 0, ret_60 / vol_60, 0)

    ret_1d = close.pct_change()
    up_vol = pd.Series(np.where(ret_1d > 0, volume, 0), index=df.index)
    dn_vol = pd.Series(np.where(ret_1d < 0, volume, 0), index=df.index)
    df["up_down_vol_ratio"] = (
        up_vol.rolling(20).sum() /
        dn_vol.rolling(20).sum().replace(0, np.nan)
    )

    return df


# ── Build Full Feature Set ────────────────────────────
def build_features(name):
    print(f"  Building features: {name}... ",
          end="")

    df = load_symbol(name)
    if df is None or len(df) < 200:
        print(f"❌ Insufficient data")
        return None

    try:
        df = add_trend_features(df)
        df = add_momentum_features(df)
        df = add_volatility_features(df)
        df = add_volume_features(df)
        df = add_price_action_features(df)
        df = add_calendar_features(df)
        df = add_iv_proxy_features(df)
        df = add_regime_features(df)
        df = add_intermarket_features(df)
        df = add_fundamental_proxy_features(df)

        # Drop raw OHLCV — keep features only
        feature_cols = [
            c for c in df.columns
            if c not in [
                "Open", "High", "Low", "Volume"
            ]
        ]
        df_features = df[feature_cols].copy()

        # Drop early rows with NaN
        df_features = df_features.dropna(
            subset=["adx", "rsi_14",
                    "macd", "atr_14"]
        )

        # Save
        path = os.path.join(
            FEATURE_DIR, f"{name}_features.csv"
        )
        df_features.to_csv(path)

        print(
            f"✅ {len(df_features)} rows | "
            f"{len(df_features.columns)} features"
        )
        return df_features

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

# ── Build Features for All Symbols ───────────────────
def build_all_features():
    print("=" * 60)
    print("  Trading AI — Feature Engineering")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load manifest
    manifest_path = os.path.join(
        DATA_DIR, "manifest.json"
    )
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        symbols = [
            s["name"]
            for s in manifest["symbols"]
        ]
    else:
        # Use all CSV files
        symbols = [
            f.replace(".csv", "")
            for f in os.listdir(DATA_DIR)
            if f.endswith(".csv")
            and f != "manifest.json"
        ]

    print(f"\n  Processing {len(symbols)} symbols...")
    print()

    success = []
    failed  = []

    for i, name in enumerate(symbols, 1):
        print(f"  [{i:02d}/{len(symbols)}] ",
              end="")
        df = build_features(name)

        if df is not None:
            success.append({
                "name"    : name,
                "rows"    : len(df),
                "features": len(df.columns)
            })
        else:
            failed.append(name)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  FEATURE ENGINEERING SUMMARY")
    print(f"{'=' * 60}")
    print(f"  ✅ Success : {len(success)}")
    print(f"  ❌ Failed  : {len(failed)}")

    if success:
        total_rows = sum(s["rows"] for s in success)
        avg_feat   = sum(
            s["features"] for s in success
        ) // len(success)
        print(f"  Total rows : {total_rows:,}")
        print(f"  Avg features: {avg_feat}")

    # Feature catalog
    if success:
        sample_path = os.path.join(
            FEATURE_DIR,
            f"{success[0]['name']}_features.csv"
        )
        sample = pd.read_csv(
            sample_path,
            index_col="Date",
            parse_dates=True,
            nrows=1
        )
        feature_list = list(sample.columns)

        catalog = {
            "total_features": len(feature_list),
            "features"      : feature_list,
            "categories"    : {
                "trend"       : [
                    f for f in feature_list
                    if any(x in f for x in [
                        "ema", "adx", "di",
                        "hh", "ll"
                    ])
                ],
                "momentum"    : [
                    f for f in feature_list
                    if any(x in f for x in [
                        "rsi", "macd", "roc",
                        "stoch", "williams",
                        "cci"
                    ])
                ],
                "volatility"  : [
                    f for f in feature_list
                    if any(x in f for x in [
                        "atr", "hv", "bb",
                        "kc", "vol"
                    ])
                ],
                "volume"      : [
                    f for f in feature_list
                    if any(x in f for x in [
                        "volume", "obv",
                        "vwap", "vpt",
                        "force", "ad"
                    ])
                ],
                "price_action": [
                    f for f in feature_list
                    if any(x in f for x in [
                        "body", "wick", "gap",
                        "inside", "outside",
                        "doji", "return",
                        "dist"
                    ])
                ],
                "calendar"    : [
                    f for f in feature_list
                    if any(x in f for x in [
                        "day", "week", "month",
                        "quarter", "year",
                        "monday", "friday",
                        "results"
                    ])
                ],
                "iv_proxy"    : [
                    f for f in feature_list
                    if "iv" in f
                ],
                "regime"      : [
                    f for f in feature_list
                    if "regime" in f
                ]
            }
        }

        cat_path = os.path.join(
            FEATURE_DIR, "feature_catalog.json"
        )
        with open(cat_path, "w") as f:
            json.dump(catalog, f, indent=2)

        print(f"\n  Feature Categories:")
        for cat, feats in catalog[
            "categories"
        ].items():
            print(f"     {cat:<15}: "
                  f"{len(feats)} features")

        print(
            f"\n  Total features per symbol: "
            f"{catalog['total_features']}"
        )
        print(
            f"  ✅ Catalog saved → {cat_path}"
        )

    print(
        f"\n  ✅ Features saved → {FEATURE_DIR}"
    )
    return success

# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    build_all_features()

    print(f"\n  🎯 Next step: Build labels")
    print(
        f"  Command: "
        f"python training/build_labels.py"
    )
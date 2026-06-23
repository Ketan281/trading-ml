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
    # Momentum (multi-horizon)
    "ret_1d", "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    # Mean reversion
    "rsi_14", "bb_pctb", "dist_ma20", "dist_ma50", "dist_ma200",
    # Trend
    "above_20ema", "above_50ema", "above_200ma", "ma_slope_20", "macd_hist",
    "trend_slope_10",
    # Volatility
    "vol_10d", "vol_21d", "atr_ratio", "intraday_range",
    # Volume / institutional flow
    "rel_volume", "vol_trend", "mfi_14", "ad_momentum", "up_down_vol_ratio",
    "vol_price_confirm",
    # S/R and structure
    "sr_proximity", "fib_level", "range_pos_20",
    # Quality / risk
    "sharpe_60d", "stretch_atr",
    # Cross-sectional (computed at panel level, not per-stock)
    "mom_rank", "vol_rank", "rs_score",
]


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


def compute_stock_features(df):
    """Compute all features from an OHLCV DataFrame. Returns DataFrame with FEATURE_COLS."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    f = pd.DataFrame(index=df.index)

    # Momentum
    for d in [1, 5, 10, 21, 63, 126, 252]:
        col = f"ret_{d}d"
        f[col] = close.pct_change(d) if len(df) > d else 0.0

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # Bollinger %B
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    f["bb_pctb"] = (close - ma20) / (2 * std20 + 1e-9)

    # Distance from MAs
    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()
    ma200 = close.rolling(200).mean()
    f["dist_ma20"] = close / ema20 - 1
    f["dist_ma50"] = close / ema50 - 1
    f["dist_ma200"] = (close / ma200 - 1) if len(df) > 200 else 0.0

    # Trend
    f["above_20ema"] = (close > ema20).astype(float)
    f["above_50ema"] = (close > ema50).astype(float)
    f["above_200ma"] = (close > ma200).astype(float) if len(df) > 200 else 0.5
    f["ma_slope_20"] = ema20.pct_change(5)

    # MACD histogram
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    f["macd_hist"] = macd - signal

    # Trend slope (normalized 10d linear regression)
    def _slope(x):
        return np.polyfit(np.arange(len(x)), x, 1)[0]
    f["trend_slope_10"] = close.rolling(10).apply(_slope, raw=True) / close

    # Volatility
    daily_ret = close.pct_change()
    f["vol_10d"] = daily_ret.rolling(10).std()
    f["vol_21d"] = daily_ret.rolling(21).std()
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    f["atr_ratio"] = atr14 / close
    f["intraday_range"] = (high - low) / close

    # Volume
    avg_vol = volume.rolling(20).mean()
    f["rel_volume"] = volume / avg_vol.replace(0, np.nan)
    f["vol_trend"] = volume.rolling(5).mean() / avg_vol.replace(0, np.nan)

    # MFI 14
    typical = (high + low + close) / 3
    mf = typical * volume
    pos_mf = pd.Series(np.where(typical > typical.shift(1), mf, 0), index=df.index)
    neg_mf = pd.Series(np.where(typical < typical.shift(1), mf, 0), index=df.index)
    mfr = pos_mf.rolling(14).sum() / neg_mf.rolling(14).sum().replace(0, np.nan)
    f["mfi_14"] = 100 - (100 / (1 + mfr))

    # A/D momentum
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    ad = (clv.fillna(0) * volume).cumsum()
    ad_ema5 = ad.ewm(span=5).mean()
    ad_ema20 = ad.ewm(span=20).mean()
    f["ad_momentum"] = np.where(ad_ema20 != 0, (ad_ema5 / ad_ema20 - 1) * 100, 0).astype(float)

    # Up/down volume ratio
    ret_1d = close.pct_change()
    up_vol = pd.Series(np.where(ret_1d > 0, volume, 0), index=df.index)
    dn_vol = pd.Series(np.where(ret_1d < 0, volume, 0), index=df.index)
    f["up_down_vol_ratio"] = up_vol.rolling(20).sum() / dn_vol.rolling(20).sum().replace(0, np.nan)

    # Volume-price confirmation
    ret_5d = close.pct_change(5)
    vol_5d = volume.rolling(5).mean()
    vol_surge = vol_5d / avg_vol.replace(0, np.nan)
    f["vol_price_confirm"] = np.where(
        (ret_5d > 0) & (vol_surge > 1.5), 1.0,
        np.where((ret_5d < 0) & (vol_surge > 1.5), -1.0, 0.0)
    )

    # S/R proximity
    high_20 = high.rolling(20).max()
    low_20 = low.rolling(20).min()
    range_20 = high_20 - low_20
    f["sr_proximity"] = np.where(range_20 > 0, (close - low_20) / range_20, 0.5)

    # Fibonacci level (52-week)
    high_52 = high.rolling(252).max()
    low_52 = low.rolling(252).min()
    range_52 = high_52 - low_52
    f["fib_level"] = np.where(range_52 > 0, (high_52 - close) / range_52, 0.5)

    # Range position
    roll_high = high.rolling(20).max()
    roll_low = low.rolling(20).min()
    f["range_pos_20"] = (close - roll_low) / (roll_high - roll_low + 1e-9)

    # Sharpe 60d
    ret_60 = close.pct_change(60)
    vol_60 = daily_ret.rolling(60).std()
    f["sharpe_60d"] = np.where(vol_60 > 0, ret_60 / vol_60, 0).astype(float)

    # Stretch from EMA in ATR units
    f["stretch_atr"] = (close - ema20) / atr14.replace(0, np.nan)

    # Cross-sectional features (placeholder — filled at panel level)
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

            feat = compute_stock_features(df)
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

            feat = compute_stock_features(df)
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


def load_training_panel(min_date=None, max_date=None):
    """Load features from SQLite into a training-ready DataFrame.
    Returns: DataFrame with columns [symbol, date, price] + FEATURE_COLS."""
    conn = _db()
    query = "SELECT symbol, date, features, price FROM features"
    conditions = []
    params = []
    if min_date:
        conditions.append("date >= ?")
        params.append(min_date)
    if max_date:
        conditions.append("date <= ?")
        params.append(max_date)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY date, symbol"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame()

    records = []
    for sym, date, feat_json, price in rows:
        feat = json.loads(feat_json)
        feat["symbol"] = sym
        feat["date"] = date
        feat["price"] = price
        records.append(feat)

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


if __name__ == "__main__":
    import sys as _sys
    logging.basicConfig(level=logging.WARNING)
    if "--update" in _sys.argv:
        n = update_daily()
        print(f"Updated {n} stocks")
    else:
        print("Building full feature store...")
        result = build_full_store()
        print(f"Done: {result}")

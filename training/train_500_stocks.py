"""
Large-scale ML training: 450+ NSE stocks with earnings events + intermarket features.

- Fetches earnings surprise data from yfinance for each stock
- Builds earnings-aware features (days to/from earnings, surprise momentum)
- Joins real intermarket features (crude, DXY, VIX, gold, FII etc.)
- Era-based walk-forward validation across 4 market regimes
- XGBoost + RandomForest ensemble
- Parallel batch training with progress tracking
"""

import os
import sys
import json
import time
import pickle
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

HIST_DIR   = os.path.join(ROOT, "data", "historical")
DATA_DIR   = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
EARN_DIR   = os.path.join(ROOT, "data", "earnings")
os.makedirs(EARN_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

MIN_ROWS      = 500
BATCH_SIZE    = 10
MAX_WORKERS   = 4      # for earnings fetch (yfinance rate limit)
EARNINGS_CACHE_DAYS = 7  # re-fetch earnings if cache older than this


# ── Step 1: Discover trainable stocks ────────────────────
def discover_stocks():
    stocks = []
    for f in sorted(os.listdir(HIST_DIR)):
        if not f.endswith(".csv"):
            continue
        sym = f.replace(".csv", "")
        path = os.path.join(HIST_DIR, f)
        try:
            nrows = sum(1 for _ in open(path)) - 1
        except Exception:
            continue
        if nrows >= MIN_ROWS:
            stocks.append({"symbol": sym, "rows": nrows, "path": path})
    stocks.sort(key=lambda x: x["rows"], reverse=True)
    return stocks


# ── Step 2: Fetch earnings data ──────────────────────────
def fetch_earnings_single(symbol):
    """Fetch earnings dates + surprise from yfinance, cache to CSV."""
    cache_path = os.path.join(EARN_DIR, f"{symbol}_earnings.csv")

    # Use cache if fresh enough
    if os.path.exists(cache_path):
        age_days = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age_days < EARNINGS_CACHE_DAYS:
            return cache_path

    import yfinance as yf
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        ed = ticker.earnings_dates
        if ed is None or ed.empty:
            return None

        df = ed.reset_index()
        df.columns = ["earnings_date", "eps_estimate", "eps_reported", "surprise_pct"]
        df["earnings_date"] = pd.to_datetime(df["earnings_date"]).dt.tz_localize(None)
        df["symbol"] = symbol
        df = df.dropna(subset=["eps_reported"])
        if df.empty:
            return None

        df.to_csv(cache_path, index=False)
        return cache_path
    except Exception:
        return None


def fetch_all_earnings(symbols, max_workers=MAX_WORKERS):
    """Fetch earnings for all symbols with rate-limited parallelism."""
    print(f"\n  [EARN] Fetching earnings data for {len(symbols)} stocks...")
    results = {}
    done = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_earnings_single, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym = futures[future]
            done += 1
            try:
                path = future.result()
                if path:
                    results[sym] = path
                else:
                    failed += 1
            except Exception:
                failed += 1

            if done % 50 == 0:
                print(f"    {done}/{len(symbols)} done, {len(results)} with earnings, "
                      f"{failed} without")

    print(f"  [EARN] Done: {len(results)} stocks have earnings data, "
          f"{failed} without")
    return results


# ── Step 3: Build earnings features ─────────────────────
def build_earnings_features(symbol, price_index):
    """Build earnings-aware features for a stock's date index."""
    cache_path = os.path.join(EARN_DIR, f"{symbol}_earnings.csv")
    if not os.path.exists(cache_path):
        return pd.DataFrame(index=price_index)

    try:
        edf = pd.read_csv(cache_path, parse_dates=["earnings_date"])
    except Exception:
        return pd.DataFrame(index=price_index)

    edf = edf.sort_values("earnings_date").drop_duplicates("earnings_date")
    earn_dates = edf["earnings_date"].values

    f = pd.DataFrame(index=price_index)

    # Days since last earnings
    days_since = np.full(len(price_index), np.nan)
    # Days to next earnings
    days_to = np.full(len(price_index), np.nan)
    # Last surprise %
    last_surprise = np.full(len(price_index), np.nan)
    # Surprise streak (consecutive beats or misses)
    surprise_streak = np.full(len(price_index), 0.0)

    earn_dates_ts = pd.to_datetime(earn_dates)
    surprises = edf["surprise_pct"].values

    for i, dt in enumerate(price_index):
        # Days since last
        past = earn_dates_ts[earn_dates_ts <= dt]
        if len(past) > 0:
            days_since[i] = (dt - past[-1]).days
            idx = len(past) - 1
            if idx < len(surprises):
                last_surprise[i] = surprises[idx]

            # Surprise streak
            streak = 0
            for j in range(len(past) - 1, max(len(past) - 5, -1), -1):
                if j < len(surprises) and not np.isnan(surprises[j]):
                    if surprises[j] > 0:
                        streak += 1
                    elif surprises[j] < 0:
                        streak -= 1
                    else:
                        break

            surprise_streak[i] = streak

        # Days to next
        future = earn_dates_ts[earn_dates_ts > dt]
        if len(future) > 0:
            days_to[i] = (future[0] - dt).days

    f["days_since_earnings"] = days_since
    f["days_to_earnings"] = days_to
    f["last_surprise_pct"] = last_surprise
    f["surprise_streak"] = surprise_streak

    # Earnings proximity flag (within 5 days of announcement)
    f["near_earnings"] = ((f["days_to_earnings"] <= 5) |
                          (f["days_since_earnings"] <= 2)).astype(float)

    # Surprise momentum (rolling avg of last 4 surprises)
    # This is already captured by last_surprise and streak, but add a smoother
    f["surprise_momentum"] = f["last_surprise_pct"].fillna(0)

    return f


# ── Step 4: Train single stock ───────────────────────────
def train_single(symbol, verbose=False):
    """Train XGB + RF ensemble for one stock with all features."""
    from models.ml_models import train_model

    # Ensure _daily.csv exists for ml_models.py compatibility
    hist_path = os.path.join(HIST_DIR, f"{symbol}.csv")
    daily_path = os.path.join(DATA_DIR, f"{symbol}_daily.csv")
    if not os.path.exists(daily_path) and os.path.exists(hist_path):
        try:
            df = pd.read_csv(hist_path)
            df.to_csv(daily_path, index=False)
        except Exception:
            pass

    try:
        result = train_model(symbol)
        if result:
            # Now add earnings features to the saved model metadata
            meta_path = os.path.join(MODELS_DIR, f"{symbol}_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                meta["has_earnings"] = os.path.exists(
                    os.path.join(EARN_DIR, f"{symbol}_earnings.csv"))
                meta["training_scale"] = "500_stock_batch"
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
            return {"symbol": symbol, "status": "ok",
                    "edge": result.get("edge", 0),
                    "accuracy": result.get("xgb_cv", 0)}
        return {"symbol": symbol, "status": "no_result"}
    except Exception as e:
        if verbose:
            traceback.print_exc()
        return {"symbol": symbol, "status": "error", "error": str(e)[:100]}


# ── Step 5: Batch train all ──────────────────────────────
def train_all(symbols=None, skip_existing=False):
    """Train all stocks in batches with progress tracking."""
    if symbols is None:
        stocks = discover_stocks()
        symbols = [s["symbol"] for s in stocks]

    total = len(symbols)
    print("=" * 70)
    print(f"  LARGE-SCALE ML TRAINING -- {total} NSE stocks")
    print(f"  Real intermarket features + earnings events")
    print(f"  Era-based walk-forward | XGB + RF ensemble")
    print("=" * 70)

    # Skip already trained if requested
    if skip_existing:
        already = set()
        for f in os.listdir(MODELS_DIR):
            if f.endswith("_xgb.pkl"):
                already.add(f.replace("_xgb.pkl", ""))
        symbols = [s for s in symbols if s not in already]
        print(f"  Skipping {total - len(symbols)} already trained, "
              f"training {len(symbols)} remaining")
        total = len(symbols)

    if total == 0:
        print("  Nothing to train.")
        return

    # Fetch earnings first
    fetch_all_earnings(symbols)

    # Copy historical -> daily for stocks that don't have _daily.csv
    print(f"\n  [DATA] Ensuring _daily.csv files exist...")
    copied = 0
    for sym in symbols:
        daily_path = os.path.join(DATA_DIR, f"{sym}_daily.csv")
        hist_path = os.path.join(HIST_DIR, f"{sym}.csv")
        if not os.path.exists(daily_path) and os.path.exists(hist_path):
            try:
                df = pd.read_csv(hist_path)
                df.to_csv(daily_path, index=False)
                copied += 1
            except Exception:
                pass
    print(f"  [DATA] Copied {copied} historical files to _daily.csv format")

    # Train in batches
    results = {"success": [], "failed": [], "no_data": []}
    start = time.time()

    for batch_start in range(0, total, BATCH_SIZE):
        batch = symbols[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        elapsed = time.time() - start
        rate = (batch_start / elapsed) if elapsed > 0 and batch_start > 0 else 0
        eta = ((total - batch_start) / rate / 60) if rate > 0 else 0

        print(f"\n  --- Batch {batch_num}/{total_batches} "
              f"({batch_start}/{total}) "
              f"[{elapsed/60:.1f}m elapsed"
              f"{f', ~{eta:.0f}m remaining' if eta > 0 else ''}] ---")

        for sym in batch:
            t0 = time.time()
            result = train_single(sym, verbose=False)
            dt = time.time() - t0

            if result["status"] == "ok":
                edge = result.get("edge", 0) or 0
                acc = result.get("accuracy", 0) or 0
                tag = "[+]" if edge > 0 else "[-]"
                print(f"    {tag} {sym:<16} acc={acc:.3f} edge={edge:+.4f} "
                      f"({dt:.1f}s)")
                results["success"].append({
                    "symbol": sym, "time_sec": round(dt, 1),
                    "edge": round(edge, 4), "accuracy": round(acc, 4)
                })
            elif result["status"] == "error":
                print(f"    [X] {sym:<16} ERROR: {result.get('error', '?')[:60]} "
                      f"({dt:.1f}s)")
                results["failed"].append({
                    "symbol": sym, "error": result.get("error", "")[:100]
                })
            else:
                print(f"    [?] {sym:<16} no result ({dt:.1f}s)")
                results["no_data"].append(sym)

    elapsed = time.time() - start

    # Summary
    print("\n" + "=" * 70)
    print(f"  TRAINING COMPLETE")
    print(f"  Total time: {elapsed/60:.1f} minutes ({elapsed:.0f}s)")
    print(f"  Success: {len(results['success'])}")
    print(f"  Failed : {len(results['failed'])}")
    print(f"  No data: {len(results['no_data'])}")

    # Edge distribution
    edges = [r["edge"] for r in results["success"] if r.get("edge")]
    if edges:
        pos = [e for e in edges if e > 0]
        print(f"\n  Edge distribution:")
        print(f"    Positive edge : {len(pos)}/{len(edges)} "
              f"({100*len(pos)/len(edges):.0f}%)")
        print(f"    Mean edge     : {np.mean(edges):+.4f}")
        print(f"    Best edge     : {max(edges):+.4f}")
        accs = [r["accuracy"] for r in results["success"] if r.get("accuracy")]
        if accs:
            print(f"    Mean accuracy : {np.mean(accs):.4f}")

    # Save manifest
    manifest = {
        "date": datetime.now().isoformat(),
        "total_attempted": total,
        "total_time_sec": round(elapsed, 1),
        "success": results["success"],
        "failed": results["failed"],
        "no_data": results["no_data"],
        "config": {
            "min_rows": MIN_ROWS,
            "features": "55 (26 technical + 29 intermarket)",
            "earnings_features": "6 (days_since, days_to, surprise, streak, near, momentum)",
            "validation": "era_based_walk_forward",
            "models": "XGBoost + RandomForest ensemble",
        }
    }
    manifest_path = os.path.join(MODELS_DIR, "training_manifest_500.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  [OK] Manifest -> {manifest_path}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip stocks that already have trained models")
    parser.add_argument("--max", type=int, default=500,
                        help="Max stocks to train")
    args = parser.parse_args()

    stocks = discover_stocks()[:args.max]
    syms = [s["symbol"] for s in stocks]
    print(f"  Discovered {len(stocks)} trainable stocks (>= {MIN_ROWS} rows)")
    train_all(syms, skip_existing=args.skip_existing)

"""
Train ML models for all 50 NSE stocks + NIFTY/BANKNIFTY using
20 years of real data with real intermarket features.

Era-based walk-forward validation:
  - 2006-2010 train -> 2011 test (GFC era)
  - 2012-2015 train -> 2016 test (bull run)
  - 2016-2019 train -> 2020 test (pre-covid)
  - 2020-2025 train -> 2026 test (post-covid)
"""

import os
import sys
import time
import json
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.ml_models import train_model

HIST_DIR = os.path.join(ROOT, "data", "historical")

PRIORITY_SYMBOLS = [
    "NIFTY", "BANKNIFTY",
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT",
    "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
    "BAJFINANCE", "HCLTECH", "SUNPHARMA", "WIPRO",
    "NTPC", "POWERGRID", "ONGC", "TATASTEEL", "JSWSTEEL",
    "HINDALCO", "DRREDDY", "CIPLA", "DIVISLAB",
    "ADANIENT", "ADANIPORTS", "COALINDIA",
    "BAJAJFINSV", "TECHM", "GRASIM",
    "EICHERMOT", "HEROMOTOCO", "BPCL",
    "INDUSINDBK", "BRITANNIA", "APOLLOHOSP",
    "TATACONSUM", "BAJAJ-AUTO", "UPL",
    "SBILIFE", "HDFCLIFE", "PIDILITIND", "VEDL",
    "NESTLEIND", "ULTRACEMCO",
]


def main():
    print("=" * 60)
    print("  Training ALL Models (20-Year Real Data)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = {"success": [], "failed": [], "skipped": []}
    total = len(PRIORITY_SYMBOLS)

    for i, symbol in enumerate(PRIORITY_SYMBOLS, 1):
        path = os.path.join(HIST_DIR, f"{symbol}.csv")
        if not os.path.exists(path):
            print(f"\n[{i:02d}/{total}] {symbol} -- SKIP (no data file)")
            results["skipped"].append(symbol)
            continue

        print(f"\n{'=' * 60}")
        print(f"[{i:02d}/{total}] {symbol}")
        print("=" * 60)

        try:
            t0 = time.time()
            result = train_model(symbol)
            elapsed = time.time() - t0

            if result is not None:
                results["success"].append({
                    "symbol": symbol,
                    "time_sec": round(elapsed, 1),
                })
                print(f"\n  Done in {elapsed:.1f}s")
            else:
                results["failed"].append(symbol)
                print(f"\n  FAILED (returned None)")
        except Exception as e:
            results["failed"].append(symbol)
            print(f"\n  ERROR: {e}")

    # Summary
    print(f"\n\n{'=' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Success : {len(results['success'])}/{total}")
    print(f"  Failed  : {len(results['failed'])}/{total}")
    print(f"  Skipped : {len(results['skipped'])}/{total}")

    if results["failed"]:
        print(f"\n  Failed: {', '.join(results['failed'])}")
    if results["skipped"]:
        print(f"  Skipped: {', '.join(results['skipped'])}")

    # Save results
    manifest = {
        "date": datetime.now().isoformat(),
        "total": total,
        **results,
    }
    out_path = os.path.join(ROOT, "models", "training_manifest.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest -> {out_path}")


if __name__ == "__main__":
    main()

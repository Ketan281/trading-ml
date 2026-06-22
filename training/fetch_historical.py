import os
import sys
import time
import pandas as pd
import yfinance as yf
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(
       os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "data", "historical")
os.makedirs(DATA_DIR, exist_ok=True)

# ── All Symbols ───────────────────────────────────────
SYMBOLS = {
    # Indices
    "NIFTY"      : "^NSEI",
    "BANKNIFTY"  : "^NSEBANK",

    # Top 50 NSE Stocks
    "RELIANCE"   : "RELIANCE.NS",
    "TCS"        : "TCS.NS",
    "HDFCBANK"   : "HDFCBANK.NS",
    "INFY"       : "INFY.NS",
    "ICICIBANK"  : "ICICIBANK.NS",
    "HINDUNILVR" : "HINDUNILVR.NS",
    "ITC"        : "ITC.NS",
    "SBIN"       : "SBIN.NS",
    "BHARTIARTL" : "BHARTIARTL.NS",
    "KOTAKBANK"  : "KOTAKBANK.NS",
    "LT"         : "LT.NS",
    "AXISBANK"   : "AXISBANK.NS",
    "ASIANPAINT" : "ASIANPAINT.NS",
    "MARUTI"     : "MARUTI.NS",
    "TITAN"      : "TITAN.NS",
    "NESTLEIND"  : "NESTLEIND.NS",
    "WIPRO"      : "WIPRO.NS",
    "ULTRACEMCO" : "ULTRACEMCO.NS",
    "BAJFINANCE" : "BAJFINANCE.NS",
    "HCLTECH"    : "HCLTECH.NS",
    "SUNPHARMA"  : "SUNPHARMA.NS",
    "TATAMOTORS" : "TATAMOTORS.NS",
    "POWERGRID"  : "POWERGRID.NS",
    "NTPC"       : "NTPC.NS",
    "ONGC"       : "ONGC.NS",
    "JSWSTEEL"   : "JSWSTEEL.NS",
    "TATASTEEL"  : "TATASTEEL.NS",
    "ADANIENT"   : "ADANIENT.NS",
    "ADANIPORTS" : "ADANIPORTS.NS",
    "COALINDIA"  : "COALINDIA.NS",
    "BAJAJFINSV" : "BAJAJFINSV.NS",
    "TECHM"      : "TECHM.NS",
    "DIVISLAB"   : "DIVISLAB.NS",
    "DRREDDY"    : "DRREDDY.NS",
    "CIPLA"      : "CIPLA.NS",
    "GRASIM"     : "GRASIM.NS",
    "HINDALCO"   : "HINDALCO.NS",
    "EICHERMOT"  : "EICHERMOT.NS",
    "HEROMOTOCO" : "HEROMOTOCO.NS",
    "BPCL"       : "BPCL.NS",
    "INDUSINDBK" : "INDUSINDBK.NS",
    "BRITANNIA"  : "BRITANNIA.NS",
    "APOLLOHOSP" : "APOLLOHOSP.NS",
    "TATACONSUM" : "TATACONSUM.NS",
    "BAJAJ-AUTO" : "BAJAJ-AUTO.NS",
    "UPL"        : "UPL.NS",
    "SBILIFE"    : "SBILIFE.NS",
    "HDFCLIFE"   : "HDFCLIFE.NS",
    "PIDILITIND" : "PIDILITIND.NS",
    "VEDL"       : "VEDL.NS",
}

# ── Date Range ────────────────────────────────────────
START_DATE = "2006-01-01"
END_DATE   = "2026-06-22"

# ── Fetch Single Symbol ───────────────────────────────
def fetch_symbol(name, ticker,
                  start=START_DATE,
                  end=END_DATE,
                  retries=3):
    for attempt in range(retries):
        try:
            print(f"  Fetching {name} "
                  f"({ticker})... ", end="")

            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                progress=False
            )

            if df.empty:
                print(f"❌ No data")
                return None

            # Flatten columns if MultiIndex
            if isinstance(
                df.columns, pd.MultiIndex
            ):
                df.columns = [
                    c[0] for c in df.columns
                ]

            # Keep OHLCV
            df = df[
                ["Open", "High", "Low",
                 "Close", "Volume"]
            ]
            df.index.name = "Date"
            df = df.dropna()

            # Save
            path = os.path.join(
                DATA_DIR, f"{name}.csv"
            )
            df.to_csv(path)

            rows = len(df)
            start_d = df.index[0].date()
            end_d   = df.index[-1].date()

            print(
                f"✅ {rows} rows "
                f"({start_d} → {end_d})"
            )
            return df

        except Exception as e:
            if attempt < retries - 1:
                print(
                    f"⚠ Retry {attempt+1}... "
                )
                time.sleep(2)
            else:
                print(f"❌ Failed: {e}")
                return None

# ── Fetch All Symbols ─────────────────────────────────
def fetch_all():
    print("=" * 60)
    print("  Trading AI — 10Y Historical Data Fetch")
    print(f"  Period: {START_DATE} → {END_DATE}")
    print(f"  Symbols: {len(SYMBOLS)}")
    print("=" * 60)

    success = []
    failed  = []
    total   = len(SYMBOLS)

    for i, (name, ticker) in enumerate(
        SYMBOLS.items(), 1
    ):
        print(f"  [{i:02d}/{total}] ", end="")
        df = fetch_symbol(name, ticker)

        if df is not None:
            success.append({
                "name"  : name,
                "ticker": ticker,
                "rows"  : len(df),
                "start" : str(df.index[0].date()),
                "end"   : str(df.index[-1].date())
            })
        else:
            failed.append(name)

        # Rate limiting
        time.sleep(0.5)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  FETCH SUMMARY")
    print(f"{'=' * 60}")
    print(f"  ✅ Success : {len(success)}/{total}")
    print(f"  ❌ Failed  : {len(failed)}/{total}")

    if failed:
        print(f"\n  Failed symbols:")
        for s in failed:
            print(f"     → {s}")

    # Save manifest
    import json
    manifest = {
        "fetch_date" : datetime.now().isoformat(),
        "start_date" : START_DATE,
        "end_date"   : END_DATE,
        "total"      : total,
        "success"    : len(success),
        "failed"     : len(failed),
        "symbols"    : success,
        "failed_list": failed
    }

    manifest_path = os.path.join(
        DATA_DIR, "manifest.json"
    )
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  ✅ Data saved → {DATA_DIR}")
    print(f"  ✅ Manifest  → {manifest_path}")
    return manifest

# ── Verify Data Quality ───────────────────────────────
def verify_data():
    print(f"\n{'=' * 60}")
    print(f"  DATA QUALITY CHECK")
    print(f"{'=' * 60}")

    files  = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".csv")
    ]

    issues = []

    print(f"\n  {'SYMBOL':<15} {'ROWS':<8} "
          f"{'START':<12} {'END':<12} {'STATUS'}")
    print("  " + "─" * 55)

    for file in sorted(files):
        name = file.replace(".csv", "")
        path = os.path.join(DATA_DIR, file)

        try:
            df   = pd.read_csv(
                path,
                index_col="Date",
                parse_dates=True
            )
            rows = len(df)
            s    = df.index[0].date()
            e    = df.index[-1].date()

            # Check minimum rows
            # (~2500 trading days in 10Y)
            if rows < 1000:
                status = "⚠ LOW"
                issues.append(
                    f"{name}: only {rows} rows"
                )
            elif rows < 2000:
                status = "⚠ PARTIAL"
            else:
                status = "✅ GOOD"

            print(
                f"  {name:<15} {rows:<8} "
                f"{str(s):<12} {str(e):<12} "
                f"{status}"
            )

        except Exception as e:
            print(
                f"  {name:<15} ❌ Read error: {e}"
            )
            issues.append(f"{name}: read error")

    if issues:
        print(f"\n  ⚠ Issues found:")
        for issue in issues:
            print(f"     → {issue}")
    else:
        print(f"\n  ✅ All data looks good!")

    print(f"\n  Total files: {len(files)}")
    return len(issues) == 0

# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    # Fetch all data
    manifest = fetch_all()

    # Verify quality
    verify_data()

    print(f"\n  🎯 Next step: Run feature engineering")
    print(
        f"  Command: "
        f"python training/build_features.py"
    )
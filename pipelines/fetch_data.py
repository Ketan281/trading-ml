import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# ── Symbols ──────────────────────────────────────────
SYMBOLS = {
    "NIFTY":     "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "RELIANCE":  "RELIANCE.NS",
    "TCS":       "TCS.NS"
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Fetch & Save ─────────────────────────────────────
def fetch_and_save(name, ticker, period="6mo", interval="1d"):
    print(f"Fetching {name} ({ticker})...")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)

    if df.empty:
        print(f"  ⚠ No data for {name}")
        return None

    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "Close", "High", "Low", "Volume"]]
    df.index.name = "Date"

    path = os.path.join(DATA_DIR, f"{name}_daily.csv")
    df.to_csv(path)
    print(f"  ✅ Saved {len(df)} rows → {path}")
    return df

# ── Main ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Trading AI — Market Data Fetch")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    for name, ticker in SYMBOLS.items():
        fetch_and_save(name, ticker)

    print("\n✅ All data fetched and saved to /data folder!")
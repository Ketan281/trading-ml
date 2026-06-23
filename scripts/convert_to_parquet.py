"""
Convert CSV price files to Parquet for faster I/O and lower memory (rule #7).

Run once:  python scripts/convert_to_parquet.py
Then set  PARQUET_MODE=1  in systemd env to use parquet files.

Parquet files are ~3-5x smaller and ~10x faster to read than CSV for pandas.
On a 1GB instance this matters: 434 stock files × multiple reads per pipeline.
"""

import os
import sys
import glob
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd


def convert_dir(src_dir, out_dir=None):
    if out_dir is None:
        out_dir = src_dir.rstrip("/\\") + "_parquet"
    os.makedirs(out_dir, exist_ok=True)

    files = glob.glob(os.path.join(src_dir, "*.csv"))
    print(f"Converting {len(files)} CSV files to Parquet...")
    print(f"  Source: {src_dir}")
    print(f"  Output: {out_dir}")

    t0 = time.time()
    converted = 0
    errors = 0
    saved_bytes = 0

    for path in files:
        name = os.path.basename(path).replace(".csv", "")
        out = os.path.join(out_dir, f"{name}.parquet")
        try:
            df = pd.read_csv(path, index_col="Date", parse_dates=True)
            for col in df.select_dtypes("float64").columns:
                df[col] = df[col].astype("float32")
            df.to_parquet(out, engine="pyarrow", compression="snappy")

            csv_size = os.path.getsize(path)
            pq_size = os.path.getsize(out)
            saved_bytes += csv_size - pq_size
            converted += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error: {name} — {e}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s — {converted} converted, {errors} errors")
    print(f"Space saved: {saved_bytes / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    data_dir = os.path.join(ROOT, "data")
    hist_dir = os.path.join(ROOT, "data", "historical")

    if os.path.isdir(hist_dir) and glob.glob(os.path.join(hist_dir, "*.csv")):
        convert_dir(hist_dir)
    if glob.glob(os.path.join(data_dir, "*_daily.csv")):
        convert_dir(data_dir, os.path.join(data_dir, "parquet"))

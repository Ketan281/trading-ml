"""
Backfill FO Bhavcopy from NSE archives -- strike-level OI/volume/price for ALL history.

NSE publishes daily derivatives settlement data (free, public):
  - Every option contract: strike, OI, change in OI, volume, OHLC, settlement price
  - Available from 2015+ in consistent CSV format
  - ~2500 trading days = massive dataset for strike-level ML

Downloads, parses, and converts to the same format as collect_option_chain.py raw CSVs
so the strike ranker ML can train on years of data instead of waiting 60 days.

Output: data/option_chain/raw/{NIFTY,BANKNIFTY}/{date}.csv (same schema as live collector)

Usage:
    python training/backfill_fo_bhavcopy.py                    # last 6 months
    python training/backfill_fo_bhavcopy.py --years 3          # last 3 years
    python training/backfill_fo_bhavcopy.py --from 2020-01-01  # from specific date
"""

import os
import sys
import io
import time
import zipfile
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

log = logging.getLogger("backfill_fo")

OUT_RAW = os.path.join(ROOT, "data", "option_chain", "raw")
OUT_AGG = os.path.join(ROOT, "data", "option_chain", "agg")
os.makedirs(OUT_RAW, exist_ok=True)
os.makedirs(OUT_AGG, exist_ok=True)

INDICES = ["NIFTY", "BANKNIFTY"]
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.nseindia.com/",
}


def _download_bhavcopy(date):
    """Download FO bhavcopy CSV for a given date from NSE archives."""
    d = date
    month_str = MONTHS[d.month - 1]
    year_str = str(d.year)

    # Try multiple URL patterns (NSE changed format over the years)
    urls = [
        # New format (2024+)
        f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{year_str}/{month_str}/fo{d.strftime('%d')}{month_str}{year_str}bhav.csv.zip",
        # Alternate format
        f"https://www1.nseindia.com/content/historical/DERIVATIVES/{year_str}/{month_str}/fo{d.strftime('%d')}{month_str}{year_str}bhav.csv.zip",
    ]

    for url in urls:
        try:
            req = Request(url, headers=HEADERS)
            resp = urlopen(req, timeout=30)
            data = resp.read()
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    df = pd.read_csv(f)
                    return df
        except (HTTPError, URLError, zipfile.BadZipFile):
            continue
        except Exception as e:
            log.debug(f"Failed {url}: {e}")
            continue

    return None


def _compute_iv_approx(spot, strike, premium, is_call, dte_years):
    """Approximate IV using Brenner-Subrahmanyam formula.
    IV ≈ premium * sqrt(2π) / (spot * sqrt(T))
    Good enough for ML features -- not for trading."""
    if dte_years <= 0 or spot <= 0 or premium <= 0:
        return 0
    iv = premium * np.sqrt(2 * np.pi) / (spot * np.sqrt(dte_years))
    return min(max(iv * 100, 1), 200)  # cap at 1-200%


def _parse_bhavcopy_to_strikes(bhavcopy_df, date, symbol):
    """Convert FO bhavcopy into strike-level format matching live collector."""
    # Filter for index options only
    mask = (
        (bhavcopy_df["INSTRUMENT"] == "OPTIDX") &
        (bhavcopy_df["SYMBOL"] == symbol)
    )
    opts = bhavcopy_df[mask].copy()

    if opts.empty:
        return pd.DataFrame(), None, None

    # Parse expiry dates
    opts["EXPIRY_DT"] = pd.to_datetime(opts["EXPIRY_DT"], format="%d-%b-%Y", errors="coerce")
    opts = opts.dropna(subset=["EXPIRY_DT"])

    if opts.empty:
        return pd.DataFrame(), None, None

    # Get nearest expiry
    nearest_expiry = opts["EXPIRY_DT"].min()
    opts = opts[opts["EXPIRY_DT"] == nearest_expiry].copy()

    # Get spot from futures
    fut_mask = (
        (bhavcopy_df["INSTRUMENT"] == "FUTIDX") &
        (bhavcopy_df["SYMBOL"] == symbol)
    )
    futs = bhavcopy_df[fut_mask]
    if not futs.empty:
        futs_parsed = futs.copy()
        futs_parsed["EXPIRY_DT"] = pd.to_datetime(futs_parsed["EXPIRY_DT"], format="%d-%b-%Y", errors="coerce")
        nearest_fut = futs_parsed.loc[futs_parsed["EXPIRY_DT"].idxmin()]
        spot = float(nearest_fut["CLOSE"])
    else:
        # Approximate from ATM options
        calls = opts[opts["OPTION_TYP"] == "CE"]
        if not calls.empty:
            spot = float(calls.iloc[len(calls)//2]["STRIKE_PR"])
        else:
            return pd.DataFrame(), None, None

    # DTE for IV calculation
    dte_days = (nearest_expiry - pd.Timestamp(date)).days
    dte_years = max(dte_days / 365, 1/365)

    # Pivot into strike rows
    calls = opts[opts["OPTION_TYP"] == "CE"].set_index("STRIKE_PR")
    puts = opts[opts["OPTION_TYP"] == "PE"].set_index("STRIKE_PR")

    all_strikes = sorted(set(calls.index) | set(puts.index))
    step = 50 if symbol == "NIFTY" else 100
    atm = round(spot / step) * step

    # Keep ATM ± 10 strikes
    strike_range = step * 10
    all_strikes = [s for s in all_strikes if atm - strike_range <= s <= atm + strike_range]

    rows = []
    for strike in all_strikes:
        ce = calls.loc[strike] if strike in calls.index else None
        pe = puts.loc[strike] if strike in puts.index else None

        # Handle duplicate strikes (multiple series) -- take first
        if ce is not None and isinstance(ce, pd.DataFrame):
            ce = ce.iloc[0]
        if pe is not None and isinstance(pe, pd.DataFrame):
            pe = pe.iloc[0]

        ce_oi = int(ce["OPEN_INT"]) if ce is not None else 0
        ce_chg = int(ce["CHG_IN_OI"]) if ce is not None else 0
        ce_vol = int(ce["CONTRACTS"]) if ce is not None else 0
        ce_ltp = float(ce["CLOSE"]) if ce is not None else 0
        pe_oi = int(pe["OPEN_INT"]) if pe is not None else 0
        pe_chg = int(pe["CHG_IN_OI"]) if pe is not None else 0
        pe_vol = int(pe["CONTRACTS"]) if pe is not None else 0
        pe_ltp = float(pe["CLOSE"]) if pe is not None else 0

        ce_iv = _compute_iv_approx(spot, strike, ce_ltp, True, dte_years) if ce_ltp > 0 else 0
        pe_iv = _compute_iv_approx(spot, strike, pe_ltp, False, dte_years) if pe_ltp > 0 else 0

        rows.append({
            "timestamp": f"{date} 15:30:00",
            "expiry": nearest_expiry.strftime("%d-%b-%Y"),
            "strike": int(strike),
            "ce_oi": ce_oi, "ce_chg_oi": ce_chg,
            "ce_vol": ce_vol, "ce_iv": round(ce_iv, 2),
            "ce_ltp": ce_ltp,
            "pe_oi": pe_oi, "pe_chg_oi": pe_chg,
            "pe_vol": pe_vol, "pe_iv": round(pe_iv, 2),
            "pe_ltp": pe_ltp,
        })

    df = pd.DataFrame(rows)
    return df, spot, nearest_expiry


def _compute_agg_from_strikes(strikes_df, symbol, spot, expiry, date, vix=None):
    """Compute aggregate ML row from strike data (same as live collector)."""
    if strikes_df.empty or spot <= 0:
        return None

    step = 50 if symbol == "NIFTY" else 100
    atm = round(spot / step) * step

    tot_ce_oi = strikes_df["ce_oi"].sum()
    tot_pe_oi = strikes_df["pe_oi"].sum()
    tot_ce_vol = strikes_df["ce_vol"].sum()
    tot_pe_vol = strikes_df["pe_vol"].sum()

    # Max Pain
    strikes_arr = strikes_df["strike"].values
    pains = []
    for k in strikes_arr:
        ce_pain = ((k - strikes_df["strike"]).clip(lower=0) * strikes_df["ce_oi"]).sum()
        pe_pain = ((strikes_df["strike"] - k).clip(lower=0) * strikes_df["pe_oi"]).sum()
        pains.append(ce_pain + pe_pain)
    max_pain = int(strikes_arr[int(pd.Series(pains).idxmin())]) if pains else atm

    atm_row = strikes_df.iloc[(strikes_df["strike"] - atm).abs().argmin()]
    otm_call = strikes_df[strikes_df["strike"] > atm]
    otm_put = strikes_df[strikes_df["strike"] < atm]

    return {
        "timestamp": f"{date} 15:30:00",
        "symbol": symbol,
        "expiry": expiry.strftime("%d-%b-%Y") if expiry else "",
        "spot": round(spot, 2),
        "atm": atm,
        "pcr_oi": round(tot_pe_oi / max(tot_ce_oi, 1), 3),
        "pcr_vol": round(tot_pe_vol / max(tot_ce_vol, 1), 3),
        "tot_ce_oi": int(tot_ce_oi), "tot_pe_oi": int(tot_pe_oi),
        "ce_chg_oi": int(strikes_df["ce_chg_oi"].sum()),
        "pe_chg_oi": int(strikes_df["pe_chg_oi"].sum()),
        "tot_ce_vol": int(tot_ce_vol), "tot_pe_vol": int(tot_pe_vol),
        "atm_ce_iv": round(float(atm_row["ce_iv"]), 2),
        "atm_pe_iv": round(float(atm_row["pe_iv"]), 2),
        "atm_iv": round(float((atm_row["ce_iv"] + atm_row["pe_iv"]) / 2), 2),
        "max_pain": max_pain,
        "max_pain_dist_pct": round((spot - max_pain) / spot * 100, 2),
        "call_writing_strike": int(otm_call.loc[otm_call["ce_oi"].idxmax(), "strike"]) if len(otm_call) else atm,
        "put_writing_strike": int(otm_put.loc[otm_put["pe_oi"].idxmax(), "strike"]) if len(otm_put) else atm,
        "otm_call_oi": int(otm_call["ce_oi"].sum()),
        "otm_put_oi": int(otm_put["pe_oi"].sum()),
        "india_vix": vix,
    }


def backfill(start_date, end_date=None):
    """Download and process FO bhavcopy for date range."""
    if end_date is None:
        end_date = datetime.now() - timedelta(days=1)
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    print("=" * 64)
    print(f"  FO BHAVCOPY BACKFILL")
    print(f"  {start_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')}")
    print("=" * 64)

    # Generate all trading dates
    all_dates = pd.bdate_range(start_date, end_date)
    # Remove known Indian holidays (approximate -- weekend-only filter)
    dates = [d.date() for d in all_dates]

    total = len(dates)
    downloaded = 0
    skipped = 0
    failed = 0
    agg_rows = {sym: [] for sym in INDICES}

    for i, date in enumerate(dates):
        date_str = date.strftime("%Y-%m-%d")

        # Skip if already have data
        already_done = all(
            os.path.exists(os.path.join(OUT_RAW, sym, f"{date_str}.csv"))
            for sym in INDICES
        )
        if already_done:
            skipped += 1
            continue

        # Download
        bhavcopy = _download_bhavcopy(date)
        if bhavcopy is None:
            failed += 1
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{total}] {date_str} -- no data (holiday?)")
            continue

        downloaded += 1

        for sym in INDICES:
            strikes_df, spot, expiry = _parse_bhavcopy_to_strikes(bhavcopy, date_str, sym)
            if strikes_df.empty:
                continue

            # Save raw strikes
            out_dir = os.path.join(OUT_RAW, sym)
            os.makedirs(out_dir, exist_ok=True)
            strikes_df.to_csv(os.path.join(out_dir, f"{date_str}.csv"), index=False)

            # Compute aggregate
            agg = _compute_agg_from_strikes(strikes_df, sym, spot, expiry, date_str)
            if agg:
                agg_rows[sym].append(agg)

        if (i + 1) % 20 == 0 or i == total - 1:
            print(f"  [{i+1}/{total}] {date_str}  "
                  f"downloaded={downloaded} skipped={skipped} failed={failed}")

        # Rate limit: NSE blocks rapid requests
        time.sleep(0.5)

    # Merge aggregate data with existing
    for sym in INDICES:
        if not agg_rows[sym]:
            continue
        new_df = pd.DataFrame(agg_rows[sym])
        agg_path = os.path.join(OUT_AGG, f"{sym}.csv")
        if os.path.exists(agg_path):
            old_df = pd.read_csv(agg_path)
            merged = pd.concat([old_df, new_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=["timestamp"], keep="last")
            merged = merged.sort_values("timestamp")
        else:
            merged = new_df
        merged.to_csv(agg_path, index=False)
        print(f"  {sym} aggregate: {len(merged)} total rows")

    # Final count
    for sym in INDICES:
        raw_dir = os.path.join(OUT_RAW, sym)
        if os.path.exists(raw_dir):
            n_files = len([f for f in os.listdir(raw_dir) if f.endswith(".csv")])
            print(f"  {sym} raw: {n_files} days total")

    print(f"\n  Done: {downloaded} new, {skipped} existing, {failed} unavailable")
    return downloaded


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description="Backfill FO Bhavcopy from NSE")
    parser.add_argument("--years", type=float, default=0,
                        help="Years of history to download (default: 0.5)")
    parser.add_argument("--months", type=int, default=0,
                        help="Months of history")
    parser.add_argument("--from", dest="from_date", type=str, default=None,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="to_date", type=str, default=None,
                        help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.from_date:
        start = args.from_date
    elif args.years > 0:
        start = (datetime.now() - timedelta(days=int(args.years * 365))).strftime("%Y-%m-%d")
    elif args.months > 0:
        start = (datetime.now() - timedelta(days=args.months * 30)).strftime("%Y-%m-%d")
    else:
        # Default: max possible -- 5 years
        start = (datetime.now() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")

    end = args.to_date if args.to_date else None
    backfill(start, end)

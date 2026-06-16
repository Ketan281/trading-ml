"""
Live option-chain collector — NIFTY / BANKNIFTY (builds the ML dataset).

WHY THIS IS THE KEYSTONE
------------------------
A professional options ML system needs HISTORICAL intraday option-chain
features: PCR, OI change / build-up, IV, Max Pain, call/put writing levels,
ATM/OTM OI concentration — sampled through the day, for years. That data is
NOT available for free anywhere (yfinance has zero Indian options history).
The only way to own it is to START RECORDING IT. This collector snapshots the
live NSE chain every few minutes and appends:

  • data/option_chain/agg/<SYMBOL>.csv   — one ML-ready row per snapshot
        (spot, PCR, total/▵ OI, ATM IV, Max Pain, writing levels, …)
  • data/option_chain/raw/<SYMBOL>/<date>.csv — full strike-level snapshot
        (for IV-smile / per-strike OI-buildup features later)

Run it every ~3–5 min during market hours (9:15–15:30 IST). After a few
months you have the genuine dataset the walk-forward chain model trains on —
the same honest "collect the data first" discipline as the intraday-bar
collector. Idempotent: snapshots are de-duped by timestamp.
"""

import os
import sys
import json
from datetime import datetime

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from jugaad_data.nse import NSELive

OUT_AGG = os.path.join(ROOT, "data", "option_chain", "agg")
OUT_RAW = os.path.join(ROOT, "data", "option_chain", "raw")
os.makedirs(OUT_AGG, exist_ok=True)
os.makedirs(OUT_RAW, exist_ok=True)

INDICES   = ["NIFTY", "BANKNIFTY"]
ATM_WINDOW = 10          # strikes each side of ATM kept in the raw snapshot
nse = NSELive()


def _india_vix():
    try:
        d = nse.live_index("INDIA VIX")
        return float(d["last"]) if d and "last" in d else None
    except Exception:
        try:
            return float(nse.live_index("INDIA VIX")["metadata"]["last"])
        except Exception:
            return None


def _parse_chain(symbol):
    """Return (spot, nearest_expiry, list-of-strike-rows) for the nearest
    expiry, or None on failure."""
    raw = nse.index_option_chain(symbol)
    recs = raw["records"]
    spot = recs.get("underlyingValue")
    expiries = recs.get("expiryDates", [])
    if not spot or not expiries:
        return None
    nearest = expiries[0]
    rows = []
    for item in recs["data"]:
        if nearest not in item.get("expiryDates", []):
            continue
        strike = item.get("strikePrice", 0)
        if not strike:
            continue
        ce, pe = item.get("CE", {}), item.get("PE", {})
        rows.append({
            "strike": strike,
            "ce_oi": ce.get("openInterest", 0), "ce_chg_oi": ce.get("changeinOpenInterest", 0),
            "ce_vol": ce.get("totalTradedVolume", 0), "ce_iv": ce.get("impliedVolatility", 0),
            "ce_ltp": ce.get("lastPrice", 0),
            "pe_oi": pe.get("openInterest", 0), "pe_chg_oi": pe.get("changeinOpenInterest", 0),
            "pe_vol": pe.get("totalTradedVolume", 0), "pe_iv": pe.get("impliedVolatility", 0),
            "pe_ltp": pe.get("lastPrice", 0),
        })
    return spot, nearest, rows


def _aggregate(symbol, spot, expiry, rows, vix):
    """Collapse the strike chain into one ML-ready feature row."""
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    step = 50 if symbol == "NIFTY" else 100
    atm = round(spot / step) * step

    tot_ce_oi, tot_pe_oi = df["ce_oi"].sum(), df["pe_oi"].sum()
    tot_ce_vol, tot_pe_vol = df["ce_vol"].sum(), df["pe_vol"].sum()

    # Max Pain: strike minimising total writer payout.
    strikes = df["strike"].values
    pains = []
    for k in strikes:
        ce_pain = ((k - df["strike"]).clip(lower=0) * df["ce_oi"]).sum()
        pe_pain = ((df["strike"] - k).clip(lower=0) * df["pe_oi"]).sum()
        pains.append(ce_pain + pe_pain)
    max_pain = int(strikes[int(pd.Series(pains).idxmin())]) if pains else atm

    atm_row = df.iloc[(df["strike"] - atm).abs().argmin()]
    otm_call = df[df["strike"] > atm]      # calls above spot = resistance writers
    otm_put  = df[df["strike"] < atm]      # puts below spot = support writers

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol, "expiry": expiry,
        "spot": round(float(spot), 2), "atm": atm,
        "pcr_oi":  round(tot_pe_oi / tot_ce_oi, 3) if tot_ce_oi else 0,
        "pcr_vol": round(tot_pe_vol / tot_ce_vol, 3) if tot_ce_vol else 0,
        "tot_ce_oi": int(tot_ce_oi), "tot_pe_oi": int(tot_pe_oi),
        "ce_chg_oi": int(df["ce_chg_oi"].sum()), "pe_chg_oi": int(df["pe_chg_oi"].sum()),
        "tot_ce_vol": int(tot_ce_vol), "tot_pe_vol": int(tot_pe_vol),
        "atm_ce_iv": round(float(atm_row["ce_iv"]), 2),
        "atm_pe_iv": round(float(atm_row["pe_iv"]), 2),
        "atm_iv": round(float((atm_row["ce_iv"] + atm_row["pe_iv"]) / 2), 2),
        "max_pain": max_pain,
        "max_pain_dist_pct": round((spot - max_pain) / spot * 100, 2),
        # Writing levels = strike with the most OI on each side.
        "call_writing_strike": int(otm_call.loc[otm_call["ce_oi"].idxmax(), "strike"])
                               if len(otm_call) else atm,
        "put_writing_strike":  int(otm_put.loc[otm_put["pe_oi"].idxmax(), "strike"])
                               if len(otm_put) else atm,
        "otm_call_oi": int(otm_call["ce_oi"].sum()),
        "otm_put_oi":  int(otm_put["pe_oi"].sum()),
        "india_vix": vix,
    }


def _append_csv(path, row_or_df, dedup_cols):
    df_new = row_or_df if isinstance(row_or_df, pd.DataFrame) else pd.DataFrame([row_or_df])
    if os.path.exists(path):
        old = pd.read_csv(path)
        merged = pd.concat([old, df_new], ignore_index=True)
    else:
        merged = df_new
    merged = merged.drop_duplicates(subset=dedup_cols, keep="last")
    merged.to_csv(path, index=False)
    return len(df_new)


def collect_once():
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 60)
    print(f"  OPTION-CHAIN SNAPSHOT  {stamp} IST")
    print("=" * 60)
    vix = _india_vix()
    for sym in INDICES:
        try:
            parsed = _parse_chain(sym)
            if not parsed:
                print(f"  {sym:<10} ✗ no chain"); continue
            spot, expiry, rows = parsed
            agg = _aggregate(sym, spot, expiry, rows, vix)
            if not agg:
                print(f"  {sym:<10} ✗ empty"); continue

            _append_csv(os.path.join(OUT_AGG, f"{sym}.csv"), agg, ["timestamp"])

            # raw strike-level snapshot (ATM window) for richer future features
            df_raw = pd.DataFrame(rows)
            df_raw = df_raw[(df_raw["strike"] >= agg["atm"] - ATM_WINDOW * (50 if sym == "NIFTY" else 100)) &
                            (df_raw["strike"] <= agg["atm"] + ATM_WINDOW * (50 if sym == "NIFTY" else 100))]
            df_raw.insert(0, "timestamp", agg["timestamp"])
            df_raw.insert(1, "expiry", expiry)
            day = datetime.now().strftime("%Y-%m-%d")
            os.makedirs(os.path.join(OUT_RAW, sym), exist_ok=True)
            _append_csv(os.path.join(OUT_RAW, sym, f"{day}.csv"), df_raw,
                        ["timestamp", "strike"])

            print(f"  {sym:<10} spot {agg['spot']:>10} | PCR {agg['pcr_oi']:>5} | "
                  f"ATM IV {agg['atm_iv']:>5} | MaxPain {agg['max_pain']:>6} | "
                  f"VIX {vix if vix else '-'}")
        except Exception as e:
            print(f"  {sym:<10} ✗ {type(e).__name__}: {str(e)[:80]}")
    print(f"\n  Archive → data/option_chain/  (agg = ML rows, raw = strike-level)")


if __name__ == "__main__":
    collect_once()

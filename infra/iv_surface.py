"""
Historical IV surface database — query the volatility the market is pricing.

Built on the option-chain collector's archive (agg/ = ATM IV per snapshot,
raw/ = strike-level IV smile per snapshot). It turns those snapshots into the
reads an options desk lives on:

  • atm_iv_history(symbol)  — ATM IV term series (is vol rising/falling?)
  • iv_percentile(symbol)   — current IV vs its own history (rich or cheap?)
  • smile(symbol)           — IV by strike right now (the skew shape)

HONEST DEPTH NOTE: this grows as the 5-min collector runs. Today it is shallow
(collection just started); the engine is correct and the surface deepens daily.
Term structure across expiries arrives once we collect multiple expiries.
"""

import os
import sys
import glob

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGG_DIR = os.path.join(ROOT, "data", "option_chain", "agg")
RAW_DIR = os.path.join(ROOT, "data", "option_chain", "raw")


def atm_iv_history(symbol):
    path = os.path.join(AGG_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df[["timestamp", "spot", "atm_iv", "pcr_oi", "max_pain"]].dropna(subset=["atm_iv"])


def iv_percentile(symbol):
    h = atm_iv_history(symbol)
    if h is None or len(h) < 3:
        return {"symbol": symbol, "note": "insufficient IV history (collector accumulating)",
                "snapshots": 0 if h is None else len(h)}
    cur = float(h["atm_iv"].iloc[-1])
    pct = float((h["atm_iv"] < cur).mean() * 100)
    return {"symbol": symbol, "snapshots": len(h), "current_atm_iv": round(cur, 2),
            "iv_percentile": round(pct, 1),
            "iv_min": round(float(h["atm_iv"].min()), 2),
            "iv_max": round(float(h["atm_iv"].max()), 2),
            "read": ("IV rich — favour selling premium" if pct > 70 else
                     "IV cheap — favour buying premium" if pct < 30 else
                     "IV mid-range")}


def smile(symbol):
    """Latest strike-level IV smile (CE & PE IV vs strike)."""
    files = sorted(glob.glob(os.path.join(RAW_DIR, symbol, "*.csv")))
    if not files:
        return None
    df = pd.read_csv(files[-1])
    df = df[df["timestamp"] == df["timestamp"].max()]
    sm = df.groupby("strike").agg(ce_iv=("ce_iv", "last"),
                                  pe_iv=("pe_iv", "last")).reset_index()
    return sm[(sm["ce_iv"] > 0) | (sm["pe_iv"] > 0)]


def report(symbol):
    print("=" * 60)
    print(f"  IV SURFACE — {symbol}")
    print("=" * 60)
    p = iv_percentile(symbol)
    if p.get("note"):
        print(f"  {p['note']} ({p['snapshots']} snapshots so far)")
    else:
        print(f"  Snapshots       : {p['snapshots']}")
        print(f"  Current ATM IV  : {p['current_atm_iv']}%  "
              f"(percentile {p['iv_percentile']} of {p['iv_min']}–{p['iv_max']}%)")
        print(f"  Read            : {p['read']}")
    sm = smile(symbol)
    if sm is not None and len(sm):
        atm_i = (sm["ce_iv"].replace(0, np.nan).mean())
        otm_put = sm[sm["strike"] < sm["strike"].median()]["pe_iv"].replace(0, np.nan).mean()
        otm_call = sm[sm["strike"] > sm["strike"].median()]["ce_iv"].replace(0, np.nan).mean()
        print(f"  Smile (now)     : OTM-put {otm_put:.1f}% | ~ATM {atm_i:.1f}% | "
              f"OTM-call {otm_call:.1f}%  → "
              f"{'put skew (fear)' if otm_put > otm_call else 'call skew'}")
    return p


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["NIFTY", "BANKNIFTY"]):
        report(s); print()

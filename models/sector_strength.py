"""
Sector-relative strength — which SECTORS lead, and each stock's RS vs its peers.

Overall relative-strength ranking (the cross-sectional model) answers "which
stocks are strong". This adds the dimension a desk actually rotates on:

  • Sector RS   : rank the 20 NSE sectors by blended multi-horizon momentum,
                  so capital concentrates where leadership is.
  • Intra-sector RS : each stock's momentum MINUS its sector's median — is it a
                  leader or laggard WITHIN its own group (removes the sector beta).

A stock that is strong both absolutely AND vs a leading sector is the highest-
quality long; a strong stock in a dead sector is often a value trap.
"""

import os
import sys
import json

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.cross_sectional import load_prices

IND_PATH = os.path.join(ROOT, "data", "historical", "industries.json")

# Blended momentum horizons (trading days) and weights.
HORIZONS = {21: 0.45, 63: 0.35, 126: 0.20}


def load_industries():
    if os.path.exists(IND_PATH):
        with open(IND_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _blended_momentum(df):
    c = df["Close"]
    m = 0.0
    for h, w in HORIZONS.items():
        if len(c) > h:
            m += w * (c.iloc[-1] / c.iloc[-h - 1] - 1)
    return m


def sector_strength(prices=None):
    """Return (sector_table, stock_table). sector_table ranks sectors by RS;
    stock_table gives each stock its sector, momentum, and intra-sector RS."""
    if prices is None:
        prices = load_prices()
    ind = load_industries()
    if not ind:
        return None, None

    rows = []
    for sym, df in prices.items():
        sec = ind.get(sym)
        if not sec or len(df) < 130:
            continue
        rows.append({"symbol": sym, "sector": sec,
                     "momentum": _blended_momentum(df)})
    stocks = pd.DataFrame(rows)
    if stocks.empty:
        return None, None

    # Sector aggregate = median member momentum (robust to outliers).
    sec_g = stocks.groupby("sector")["momentum"]
    sector = pd.DataFrame({
        "sector_momentum": sec_g.median(),
        "members": sec_g.size(),
        "breadth_pos": sec_g.apply(lambda x: (x > 0).mean() * 100),
    }).reset_index()
    sector = sector[sector["members"] >= 3]
    sector["rs_rank"] = sector["sector_momentum"].rank(ascending=False).astype(int)
    sector["rs_score"] = (sector["sector_momentum"].rank(pct=True) * 100).round(1)
    sector = sector.sort_values("sector_momentum", ascending=False).reset_index(drop=True)

    # Intra-sector RS = stock momentum − its sector median.
    med = sector.set_index("sector")["sector_momentum"]
    stocks["sector_median"] = stocks["sector"].map(med)
    stocks["intra_sector_rs"] = stocks["momentum"] - stocks["sector_median"]
    stocks["sector_rs_rank"] = stocks["sector"].map(
        sector.set_index("sector")["rs_rank"])
    stocks = stocks.sort_values("intra_sector_rs", ascending=False).reset_index(drop=True)
    return sector, stocks


def attach_sector_rs(ranked):
    """Merge sector RS onto a ranked dataframe (expects a 'symbol' column).
    Adds: sector, sector_rs_rank, sector_rs_score, intra_sector_rs."""
    sector, stocks = sector_strength()
    if stocks is None:
        ranked = ranked.copy()
        for c in ("sector", "sector_rs_rank", "sector_rs_score", "intra_sector_rs"):
            ranked[c] = np.nan
        return ranked
    srs = stocks[["symbol", "sector", "sector_rs_rank", "intra_sector_rs"]]
    sec_score = sector.set_index("sector")["rs_score"]
    out = ranked.merge(srs, on="symbol", how="left")
    out["sector_rs_score"] = out["sector"].map(sec_score)
    return out


if __name__ == "__main__":
    sector, stocks = sector_strength()
    if sector is None:
        print("  ⚠ No industries.json / price data.")
    else:
        print("=" * 60)
        print("  SECTOR RELATIVE STRENGTH  (blended 1-6 month momentum)")
        print("=" * 60)
        print(f"  {'RANK':<5}{'SECTOR':<34}{'MOM%':>7}{'BREADTH':>9}{'N':>5}")
        for r in sector.itertuples():
            print(f"  {r.rs_rank:<5}{r.sector:<34}{r.sector_momentum*100:>6.1f}%"
                  f"{r.breadth_pos:>8.0f}%{r.members:>5}")
        print("\n  Top intra-sector leaders (strong WITHIN a strong sector):")
        lead = stocks[stocks["sector_rs_rank"] <= 5].head(10)
        for r in lead.itertuples():
            print(f"     {r.symbol:<12} {r.sector:<28} "
                  f"intra-RS {r.intra_sector_rs*100:+.1f}% (sector #{r.sector_rs_rank})")

"""
Sector leadership and rotation framework.

Ranks each of the 20 NSE sectors (and 9 tracked groups) on five dimensions
-- momentum, relative strength vs NIFTY, breadth, volatility, trend strength --
then combines them into a composite leadership score, rotation phase, and a
per-sector conviction multiplier that downstream modules use to size trades.
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

NIFTY_PATHS = [
    os.path.join(ROOT, "data", "historical", "NIFTY.csv"),
    os.path.join(ROOT, "data", "NIFTY_daily.csv"),
]

SECTOR_GROUPS = {
    "Banking": ["Financial Services"],
    "IT": ["Information Technology"],
    "Pharma": ["Healthcare"],
    "FMCG": ["Fast Moving Consumer Goods"],
    "Auto": ["Automobile and Auto Components"],
    "Metals": ["Metals & Mining"],
    "Energy": ["Oil Gas & Consumable Fuels", "Power"],
    "Realty": ["Realty", "Construction", "Construction Materials"],
    "FinServices": ["Financial Services"],
}

MOMENTUM_WEIGHTS = {5: 0.2, 20: 0.4, 60: 0.4}

COMPOSITE_WEIGHTS = {
    "momentum": 0.35,
    "rs": 0.25,
    "breadth": 0.25,
    "trend_strength": 0.15,
}

MIN_MEMBERS = 3


def _load_industries():
    if os.path.exists(IND_PATH):
        with open(IND_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load_nifty():
    for p in NIFTY_PATHS:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, index_col="Date", parse_dates=True)
                if "Close" in df.columns and len(df) > 60:
                    return df.sort_index()
            except Exception:
                pass
    return None


def _close_matrix(prices):
    closes = {s: df["Close"] for s, df in prices.items() if len(df) > 60}
    mat = pd.DataFrame(closes).sort_index()
    return mat.dropna(how="all")


def _ema(series, span):
    return series.ewm(span=span, min_periods=span).mean()


def _sector_members(industries):
    """Map raw sector name -> list of symbols."""
    mapping = {}
    for sym, sec in industries.items():
        mapping.setdefault(sec, []).append(sym)
    return mapping


def _group_members(industries):
    """Map group name -> list of symbols (may overlap across groups)."""
    sec_to_syms = _sector_members(industries)
    mapping = {}
    for grp, sectors in SECTOR_GROUPS.items():
        syms = []
        for sec in sectors:
            syms.extend(sec_to_syms.get(sec, []))
        if syms:
            mapping[grp] = syms
    return mapping


def _compute_sector_metrics(name, symbols, mat, nifty_close):
    """Compute all five metrics for one sector/group given a close matrix."""
    cols = [s for s in symbols if s in mat.columns]
    if len(cols) < MIN_MEMBERS:
        return None

    sub = mat[cols].dropna(how="all")
    if len(sub) < 65:
        return None

    sector_mean = sub.mean(axis=1)

    # 1. Momentum -- blended 5/20/60 day return
    momentum = 0.0
    for h, w in MOMENTUM_WEIGHTS.items():
        if len(sector_mean) > h:
            momentum += w * (sector_mean.iloc[-1] / sector_mean.iloc[-h - 1] - 1)

    # 2. Relative strength vs NIFTY at 20d
    rs = 0.0
    if nifty_close is not None and len(nifty_close) > 20 and len(sector_mean) > 20:
        nifty_aligned = nifty_close.reindex(sector_mean.index, method="ffill")
        if len(nifty_aligned.dropna()) > 20:
            sec_20d = sector_mean.iloc[-1] / sector_mean.iloc[-21] - 1
            nif_20d = nifty_aligned.iloc[-1] / nifty_aligned.iloc[-21] - 1
            rs = sec_20d - nif_20d

    # 3. Breadth -- % of members above their 50 EMA
    ema50 = sub.apply(lambda c: _ema(c, 50))
    latest_above = (sub.iloc[-1] > ema50.iloc[-1]).sum()
    breadth = latest_above / len(cols) * 100

    # 4. Volatility -- median 20-day realized vol of members
    member_rets = sub.pct_change()
    vol_20 = member_rets.rolling(20).std().iloc[-1]
    volatility = float(vol_20.median()) if not vol_20.isna().all() else 0.0

    # 5. Trend strength -- % of members with close > EMA20 AND EMA20 > EMA50
    ema20 = sub.apply(lambda c: _ema(c, 20))
    trend_count = 0
    for c in cols:
        if c in sub.columns and c in ema20.columns and c in ema50.columns:
            close_val = sub[c].iloc[-1]
            e20 = ema20[c].iloc[-1]
            e50 = ema50[c].iloc[-1]
            if not (np.isnan(close_val) or np.isnan(e20) or np.isnan(e50)):
                if close_val > e20 and e20 > e50:
                    trend_count += 1
    trend_strength = trend_count / len(cols) * 100

    # Breadth 20 days ago for rotation phase detection
    breadth_20d_ago = 0.0
    if len(sub) > 20 and len(ema50) > 20:
        past_above = (sub.iloc[-21] > ema50.iloc[-21]).sum()
        breadth_20d_ago = past_above / len(cols) * 100

    # Momentum 20 days ago for rotation phase detection
    momentum_20d_ago = 0.0
    if len(sector_mean) > 40:
        for h, w in MOMENTUM_WEIGHTS.items():
            idx = -21
            ref_idx = idx - h - 1
            if abs(ref_idx) < len(sector_mean):
                momentum_20d_ago += w * (sector_mean.iloc[idx] / sector_mean.iloc[ref_idx] - 1)

    return {
        "name": name,
        "members": len(cols),
        "momentum": round(momentum, 6),
        "rs": round(rs, 6),
        "breadth": round(breadth, 1),
        "volatility": round(volatility, 6),
        "trend_strength": round(trend_strength, 1),
        "breadth_20d_ago": round(breadth_20d_ago, 1),
        "momentum_20d_ago": round(momentum_20d_ago, 6),
    }


def _assign_ranks_and_scores(records):
    """Add leadership_rank, rotation_score, rotation_phase, conviction_multiplier."""
    if not records:
        return []

    df = pd.DataFrame(records)

    mom_rank = df["momentum"].rank(ascending=False)
    rs_rank = df["rs"].rank(ascending=False)
    breadth_rank = df["breadth"].rank(ascending=False)
    trend_rank = df["trend_strength"].rank(ascending=False)

    n = len(df)
    mom_pct = (n + 1 - mom_rank) / n * 100
    rs_pct = (n + 1 - rs_rank) / n * 100
    breadth_pct = (n + 1 - breadth_rank) / n * 100
    trend_pct = (n + 1 - trend_rank) / n * 100

    composite = (COMPOSITE_WEIGHTS["momentum"] * mom_pct +
                 COMPOSITE_WEIGHTS["rs"] * rs_pct +
                 COMPOSITE_WEIGHTS["breadth"] * breadth_pct +
                 COMPOSITE_WEIGHTS["trend_strength"] * trend_pct)

    df["leadership_rank"] = composite.rank(ascending=False).astype(int)
    df["weakness_rank"] = composite.rank(ascending=True).astype(int)

    cmin, cmax = composite.min(), composite.max()
    if cmax > cmin:
        df["rotation_score"] = ((composite - cmin) / (cmax - cmin) * 100).round(1)
    else:
        df["rotation_score"] = 50.0

    phases = []
    for _, row in df.iterrows():
        mom_improving = row["momentum"] > row["momentum_20d_ago"]
        breadth_improving = row["breadth"] > row["breadth_20d_ago"]
        if mom_improving and breadth_improving:
            phases.append("Leading")
        elif mom_improving and not breadth_improving:
            phases.append("Improving")
        elif not mom_improving and breadth_improving:
            phases.append("Weakening")
        else:
            phases.append("Lagging")
    df["rotation_phase"] = phases

    multipliers = []
    for _, row in df.iterrows():
        rank = row["leadership_rank"]
        total = n
        pct = rank / total
        if rank <= 3 and total >= 6:
            m = 1.5 - (rank - 1) * 0.1
        elif pct <= 0.33:
            m = 1.2 + (0.33 - pct) / 0.33 * 0.3
        elif pct <= 0.5:
            m = 1.0 + (0.5 - pct) / 0.17 * 0.15
        elif pct <= 0.7:
            m = 0.85 + (0.7 - pct) / 0.2 * 0.15
        else:
            m = 0.5 + (1.0 - pct) / 0.3 * 0.35
        multipliers.append(round(min(1.5, max(0.5, m)), 2))
    df["conviction_multiplier"] = multipliers

    df = df.drop(columns=["breadth_20d_ago", "momentum_20d_ago"])
    df = df.sort_values("leadership_rank").reset_index(drop=True)
    return df.to_dict("records")


def sector_rotation_read(prices=None):
    """Full sector rotation analysis.

    Returns dict with:
    - sectors: list of dicts (raw 20 sectors), each with name, momentum, rs,
      breadth, volatility, trend_strength, leadership_rank, rotation_score,
      rotation_phase, conviction_multiplier
    - groups: list of dicts (9 tracked groups), same fields
    - leaders: top 3 sectors
    - laggards: bottom 3 sectors
    - conviction_multiplier: dict mapping sector/group name -> float (0.5 to 1.5)
    """
    if prices is None:
        prices = load_prices()

    industries = _load_industries()
    if not industries:
        return {"sectors": [], "groups": [], "leaders": [], "laggards": [],
                "conviction_multiplier": {}}

    nifty_df = _load_nifty()
    nifty_close = nifty_df["Close"] if nifty_df is not None else None

    mat = _close_matrix(prices)

    sec_members = _sector_members(industries)
    raw_records = []
    for sec_name, syms in sec_members.items():
        r = _compute_sector_metrics(sec_name, syms, mat, nifty_close)
        if r is not None:
            raw_records.append(r)

    sectors = _assign_ranks_and_scores(raw_records)

    grp_members = _group_members(industries)
    grp_records = []
    for grp_name, syms in grp_members.items():
        r = _compute_sector_metrics(grp_name, syms, mat, nifty_close)
        if r is not None:
            grp_records.append(r)

    groups = _assign_ranks_and_scores(grp_records)

    leaders = [s["name"] for s in sectors[:3]] if len(sectors) >= 3 else [s["name"] for s in sectors]
    laggards = [s["name"] for s in sectors[-3:]] if len(sectors) >= 3 else []

    conviction = {}
    for s in sectors:
        conviction[s["name"]] = s["conviction_multiplier"]
    for g in groups:
        conviction[g["name"]] = g["conviction_multiplier"]

    return {
        "sectors": sectors,
        "groups": groups,
        "leaders": leaders,
        "laggards": laggards,
        "conviction_multiplier": conviction,
    }


def sector_rotation_series(prices=None, days=120):
    """Time series of sector rotation scores for charting.

    Returns dict mapping sector_name -> pd.Series of rotation scores indexed
    by date. Sampled at ~30 points over the window to keep compute reasonable.
    """
    if prices is None:
        prices = load_prices()

    industries = _load_industries()
    if not industries:
        return {}

    nifty_df = _load_nifty()
    nifty_close = nifty_df["Close"] if nifty_df is not None else None

    mat = _close_matrix(prices)
    sec_members = _sector_members(industries)

    dates = mat.index[-days:] if len(mat) > days else mat.index
    step = max(1, len(dates) // 30)
    sample_dates = dates[::step]
    if len(dates) > 0 and dates[-1] not in sample_dates:
        sample_dates = sample_dates.append(pd.DatetimeIndex([dates[-1]]))

    result = {}
    for dt in sample_dates:
        mat_slice = mat.loc[:dt]
        if len(mat_slice) < 65:
            continue

        nifty_slice = None
        if nifty_close is not None:
            nifty_slice = nifty_close.loc[:dt]
            if len(nifty_slice) < 25:
                nifty_slice = None

        records = []
        for sec_name, syms in sec_members.items():
            r = _compute_sector_metrics(sec_name, syms, mat_slice, nifty_slice)
            if r is not None:
                records.append(r)

        scored = _assign_ranks_and_scores(records)
        for s in scored:
            if s["name"] not in result:
                result[s["name"]] = {}
            result[s["name"]][dt] = s["rotation_score"]

    series = {}
    for name, pts in result.items():
        if pts:
            s = pd.Series(pts, dtype=float).sort_index()
            s.name = name
            series[name] = s

    return series


def get_sector_conviction(symbol, prices=None):
    """Get the sector rotation conviction multiplier for a specific stock.

    Returns float between 0.5 and 1.5. Defaults to 1.0 if the stock's sector
    is not found or data is insufficient.
    """
    industries = _load_industries()
    sector = industries.get(symbol)
    if not sector:
        return 1.0

    data = sector_rotation_read(prices)
    conv = data.get("conviction_multiplier", {})

    if sector in conv:
        return conv[sector]

    for grp_name, grp_sectors in SECTOR_GROUPS.items():
        if sector in grp_sectors and grp_name in conv:
            return conv[grp_name]

    return 1.0


if __name__ == "__main__":
    data = sector_rotation_read()
    sectors = data["sectors"]
    groups = data["groups"]

    if not sectors:
        print("  No data -- check industries.json and price files.")
        sys.exit(1)

    print("=" * 90)
    print("  SECTOR ROTATION ANALYSIS")
    print("=" * 90)

    print("\n  --- RAW SECTORS (20) ---\n")
    header = (f"  {'RK':<4}{'SECTOR':<38}{'MOM%':>7}{'RS%':>7}{'BRTH':>6}"
              f"{'VOL':>7}{'TRND':>6}{'SCORE':>7}{'PHASE':<13}{'CONV':>6}")
    print(header)
    print("  " + "-" * 88)
    for s in sectors:
        print(f"  {s['leadership_rank']:<4}{s['name']:<38}"
              f"{s['momentum']*100:>6.1f}%{s['rs']*100:>6.1f}%"
              f"{s['breadth']:>5.0f}%{s['volatility']*100:>6.1f}%"
              f"{s['trend_strength']:>5.0f}%{s['rotation_score']:>6.1f}"
              f"  {s['rotation_phase']:<13}{s['conviction_multiplier']:>5.2f}")

    print(f"\n  Leaders  : {', '.join(data['leaders'])}")
    print(f"  Laggards : {', '.join(data['laggards'])}")

    if groups:
        print("\n  --- TRACKED GROUPS (9) ---\n")
        header = (f"  {'RK':<4}{'GROUP':<20}{'MOM%':>7}{'RS%':>7}{'BRTH':>6}"
                  f"{'VOL':>7}{'TRND':>6}{'SCORE':>7}{'PHASE':<13}{'CONV':>6}")
        print(header)
        print("  " + "-" * 75)
        for g in groups:
            print(f"  {g['leadership_rank']:<4}{g['name']:<20}"
                  f"{g['momentum']*100:>6.1f}%{g['rs']*100:>6.1f}%"
                  f"{g['breadth']:>5.0f}%{g['volatility']*100:>6.1f}%"
                  f"{g['trend_strength']:>5.0f}%{g['rotation_score']:>6.1f}"
                  f"  {g['rotation_phase']:<13}{g['conviction_multiplier']:>5.2f}")

    print("\n  --- CONVICTION MULTIPLIERS ---\n")
    conv = data["conviction_multiplier"]
    for name in sorted(conv, key=lambda k: conv[k], reverse=True):
        bar = "#" * int(conv[name] * 20)
        print(f"  {name:<38} {conv[name]:.2f}  {bar}")

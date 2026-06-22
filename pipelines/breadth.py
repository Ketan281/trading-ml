"""
Market-breadth engine -- measures how broadly the market is participating in
any move, not just what the index-level price says.

A rising index on narrow breadth (a handful of heavyweights) is fragile; a
rising index on broad breadth is durable. This reads the internals of the full
~500-stock universe plus sector/index slices:

  Advance/Decline        count and volume-weighted
  DMA participation      % above 20 EMA, 50 DMA, 200 DMA
  New highs / lows       52-week new highs vs new lows
  Breadth trend          5-day change in participation
  McClellan oscillator   EMA(19) - EMA(39) of daily AD ratio
  Breadth thrust         sudden expansion (>80% advancing in one session)
  Sector breadth         per-sector AD + participation scores
  Index breadth          NIFTY 50 and BANKNIFTY constituent breadth
  Regime classification  Strong Bullish / Bullish / Neutral / Bearish / Strong Bearish

Combined into a 0-100 composite breadth score. Computed from price+volume
history alone (leak-free) and returns a time series, so it doubles as a
feature for the regime classifier and the ensemble.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.cross_sectional import load_prices


# -- Constituent lists --------------------------------------------------------

NIFTY_50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BHARTIARTL", "BPCL",
    "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK",
    "INFY", "ITC", "JSWSTEEL", "KOTAKBANK", "LT",
    "LTIM", "M&M", "MARUTI", "NESTLEIND", "NTPC",
    "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SBIN",
    "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM", "TATAMOTORS", "TATASTEEL",
    "TCS", "TECHM", "TITAN", "ULTRACEMCO", "WIPRO",
]

BANK_STOCKS = [
    "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK", "SBIN",
    "INDUSINDBK", "BANKBARODA", "PNB", "IDFCFIRSTB", "FEDERALBNK",
    "BANDHANBNK", "AUBANK",
]


def _load_sector_map():
    path = os.path.join(ROOT, "data", "historical", "industries.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -- Matrix builders ----------------------------------------------------------

def _close_matrix(prices):
    closes = {s: df["Close"] for s, df in prices.items() if len(df) > 220}
    mat = pd.DataFrame(closes).sort_index()
    return mat.dropna(how="all")


def _volume_matrix(prices):
    vols = {}
    for s, df in prices.items():
        if len(df) > 220 and "Volume" in df.columns:
            vols[s] = df["Volume"]
    mat = pd.DataFrame(vols).sort_index()
    return mat.dropna(how="all")


# -- Core breadth computations ------------------------------------------------

def _advancing_mask(mat):
    return mat.diff() > 0


def _declining_mask(mat):
    return (mat.diff() < 0) & mat.notna()


def _pct_above(mat, ref):
    """Percentage of non-NaN stocks above a reference matrix (MA, etc.)."""
    above = (mat > ref).sum(axis=1)
    total = mat.notna().sum(axis=1)
    return (above / total.replace(0, np.nan)) * 100


def _ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


# -- McClellan-style oscillator -----------------------------------------------

def _mcclellan_oscillator(ad_ratio_series):
    """EMA(19) - EMA(39) of the daily AD ratio."""
    ema19 = _ema(ad_ratio_series, 19)
    ema39 = _ema(ad_ratio_series, 39)
    return ema19 - ema39


# -- Breadth series (full universe) -------------------------------------------

def breadth_series(prices=None, days=250):
    """Time series of all breadth internals (last `days` sessions)."""
    if prices is None:
        prices = load_prices()

    mat = _close_matrix(prices)
    vol_mat = _volume_matrix(prices)

    # Align volume matrix to close matrix dates
    common_dates = mat.index.intersection(vol_mat.index)
    vol_aligned = vol_mat.reindex(index=common_dates, columns=mat.columns)

    # Moving averages
    ema20 = mat.ewm(span=20, adjust=False).mean()
    ma50 = mat.rolling(50).mean()
    ma200 = mat.rolling(200).mean()

    # 52-week extremes
    hi52 = mat.rolling(252).max()
    lo52 = mat.rolling(252).min()

    # Advance / decline masks
    adv_mask = _advancing_mask(mat)
    dec_mask = _declining_mask(mat)

    out = pd.DataFrame(index=mat.index)

    # DMA participation
    out["pct_above_20ema"] = _pct_above(mat, ema20)
    out["pct_above_50dma"] = _pct_above(mat, ma50)
    out["pct_above_200dma"] = _pct_above(mat, ma200)

    # Advance / decline counts
    out["advances"] = adv_mask.sum(axis=1)
    out["declines"] = dec_mask.sum(axis=1)
    total_counted = out["advances"] + out["declines"]
    out["ad_ratio"] = out["advances"] / out["declines"].replace(0, np.nan)
    out["pct_advancing"] = (out["advances"] / total_counted.replace(0, np.nan)) * 100

    # Volume-weighted breadth
    adv_vol = (vol_aligned * adv_mask.reindex(
        index=common_dates, columns=mat.columns)).sum(axis=1)
    dec_vol = (vol_aligned * dec_mask.reindex(
        index=common_dates, columns=mat.columns)).sum(axis=1)
    total_vol = adv_vol + dec_vol
    vol_breadth = (adv_vol / total_vol.replace(0, np.nan)) * 100
    out["volume_breadth"] = vol_breadth.reindex(mat.index)

    # New highs / lows
    out["new_highs"] = (mat >= hi52).sum(axis=1)
    out["new_lows"] = (mat <= lo52).sum(axis=1)
    out["nh_nl_diff"] = out["new_highs"] - out["new_lows"]
    nh_nl_total = out["new_highs"] + out["new_lows"]
    out["nh_nl_ratio"] = (
        out["new_highs"] / nh_nl_total.replace(0, np.nan)
    ) * 100

    # McClellan oscillator
    out["mcclellan_osc"] = _mcclellan_oscillator(out["ad_ratio"].fillna(1.0))

    # Breadth thrust flag: >80% advancing in a single session
    out["breadth_thrust"] = out["pct_advancing"] >= 80.0

    return out.tail(days)


# -- Composite score ----------------------------------------------------------

def _composite_score(row):
    """
    Weighted composite:
      25% AD ratio percentile (mapped: ratio 0.5->0, 1.0->50, 2.0->100)
      20% % above 50 DMA
      20% % above 200 DMA
      15% NH/NL ratio (already 0-100 or NaN)
      10% breadth trend (5-day change in % above 50 DMA, scaled)
      10% volume breadth (already 0-100)
    """
    ad = row.get("ad_ratio_score", 50)
    a50 = row.get("pct_above_50dma", 50)
    a200 = row.get("pct_above_200dma", 50)
    nhnl = row.get("nh_nl_ratio_score", 50)
    trend = row.get("breadth_trend_score", 50)
    vb = row.get("volume_breadth", 50)

    score = (0.25 * ad + 0.20 * a50 + 0.20 * a200 +
             0.15 * nhnl + 0.10 * trend + 0.10 * vb)
    return float(np.clip(score, 0, 100))


def _classify_regime(score):
    if score >= 75:
        return "Strong Bullish"
    elif score >= 60:
        return "Bullish"
    elif score >= 40:
        return "Neutral"
    elif score >= 25:
        return "Bearish"
    else:
        return "Strong Bearish"


_REGIME_SIGNALS = {
    "Strong Bullish": "broad & strengthening -- healthy participation across the board",
    "Bullish": "constructive -- majority participating, breadth supportive",
    "Neutral": "mixed / narrowing -- selective market, watch for divergences",
    "Bearish": "weak -- few stocks holding up, rally is fragile",
    "Strong Bearish": "very weak -- broad distribution, risk-off posture warranted",
}


def _ad_ratio_to_score(ratio):
    """Map AD ratio to 0-100 scale. ratio=0.5->0, 1.0->50, 2.0->100."""
    if ratio != ratio:  # NaN check
        return 50.0
    return float(np.clip((ratio - 0.5) / 1.5 * 100, 0, 100))


# -- Sector breadth -----------------------------------------------------------

def sector_breadth(prices=None, days=1):
    """
    Per-sector breadth: AD ratio and % above 50 DMA for each sector.
    Returns dict of {sector: {ad_ratio, pct_above_50dma, score, stocks_counted}}.
    Uses the last `days` sessions (default=1 for latest only).
    """
    if prices is None:
        prices = load_prices()
    sector_map = _load_sector_map()
    if not sector_map:
        return {}

    # Invert to {sector: [symbols]}
    sectors = {}
    for sym, sector in sector_map.items():
        sectors.setdefault(sector, []).append(sym)

    mat = _close_matrix(prices)
    ma50 = mat.rolling(50).mean()
    tail = mat.tail(days)
    ma50_tail = ma50.reindex(tail.index)

    results = {}
    for sector, symbols in sorted(sectors.items()):
        cols = [s for s in symbols if s in mat.columns]
        if len(cols) < 3:
            continue

        sec_close = tail[cols]
        sec_ma50 = ma50_tail[cols]

        # Latest row
        last_close = sec_close.iloc[-1]
        last_ma50 = sec_ma50.iloc[-1]
        prev_close = mat[cols].iloc[-2] if len(mat) >= 2 else last_close

        change = last_close - prev_close
        valid = change.notna()
        advances = (change[valid] > 0).sum()
        declines = (change[valid] <= 0).sum()
        ad_ratio = advances / max(declines, 1)

        above_50 = (last_close > last_ma50).sum()
        total_valid = last_close.notna().sum()
        pct_above = (above_50 / max(total_valid, 1)) * 100

        # Sector score: 50% AD ratio score + 50% participation
        ad_score = _ad_ratio_to_score(ad_ratio)
        sector_score = 0.5 * ad_score + 0.5 * pct_above
        sector_score = float(np.clip(sector_score, 0, 100))

        results[sector] = {
            "ad_ratio": round(ad_ratio, 2),
            "advances": int(advances),
            "declines": int(declines),
            "pct_above_50dma": round(pct_above, 1),
            "score": round(sector_score, 1),
            "stocks_counted": int(total_valid),
        }

    return results


# -- Index breadth (NIFTY 50, BANKNIFTY) -------------------------------------

def _index_breadth_calc(prices, constituents, label):
    """Compute breadth for a specific index constituent list."""
    mat = _close_matrix(prices)
    available = [s for s in constituents if s in mat.columns]
    if len(available) < 5:
        return {"label": label, "score": None, "signal": "insufficient data",
                "constituents_found": len(available)}

    sub = mat[available]
    ma50 = sub.rolling(50).mean()
    ma200 = sub.rolling(200).mean()

    last = sub.iloc[-1]
    prev = sub.iloc[-2] if len(sub) >= 2 else last
    last_ma50 = ma50.iloc[-1]
    last_ma200 = ma200.iloc[-1]

    change = last - prev
    valid = change.notna()
    advances = int((change[valid] > 0).sum())
    declines = int((change[valid] <= 0).sum())
    ad_ratio = advances / max(declines, 1)

    total = int(last.notna().sum())
    pct_above_50 = float((last > last_ma50).sum() / max(total, 1) * 100)
    pct_above_200 = float((last > last_ma200).sum() / max(total, 1) * 100)

    # Simple score for the index: 40% AD + 30% pct_above_50 + 30% pct_above_200
    ad_score = _ad_ratio_to_score(ad_ratio)
    score = 0.40 * ad_score + 0.30 * pct_above_50 + 0.30 * pct_above_200
    score = float(round(np.clip(score, 0, 100), 1))

    return {
        "label": label,
        "score": score,
        "regime": _classify_regime(score),
        "advances": advances,
        "declines": declines,
        "ad_ratio": round(ad_ratio, 2),
        "pct_above_50dma": round(pct_above_50, 1),
        "pct_above_200dma": round(pct_above_200, 1),
        "constituents_found": len(available),
        "constituents_total": len(constituents),
    }


def index_breadth(prices=None):
    """Breadth for NIFTY 50 and BANKNIFTY constituents."""
    if prices is None:
        prices = load_prices()
    return {
        "nifty50": _index_breadth_calc(prices, NIFTY_50, "NIFTY 50"),
        "banknifty": _index_breadth_calc(prices, BANK_STOCKS, "BANKNIFTY"),
    }


# -- Main read: composite for latest session ----------------------------------

def breadth_read(prices=None):
    """
    Composite breadth snapshot for the latest session.
    Returns a dict with score, regime, signal, all internals, sector and index
    breadth, McClellan oscillator, and breadth thrust status.
    """
    if prices is None:
        prices = load_prices()

    s = breadth_series(prices, days=60)
    if s.empty:
        return {"score": None, "signal": "no data", "regime": "Neutral"}

    last = s.iloc[-1]

    # Core readings
    a20 = last["pct_above_20ema"]
    a50 = last["pct_above_50dma"]
    a200 = last["pct_above_200dma"]
    a50_5d = s["pct_above_50dma"].iloc[-5] if len(s) >= 5 else a50
    trend = a50 - a50_5d

    # Sub-scores for composite
    ad_ratio_score = _ad_ratio_to_score(last["ad_ratio"])
    nh_nl_ratio_val = last.get("nh_nl_ratio", 50)
    nh_nl_ratio_score = float(nh_nl_ratio_val) if nh_nl_ratio_val == nh_nl_ratio_val else 50.0
    breadth_trend_score = float(np.clip(50 + trend * 5, 0, 100))
    vol_breadth = last.get("volume_breadth", 50)
    vol_breadth_val = float(vol_breadth) if vol_breadth == vol_breadth else 50.0

    score = _composite_score({
        "ad_ratio_score": ad_ratio_score,
        "pct_above_50dma": a50,
        "pct_above_200dma": a200,
        "nh_nl_ratio_score": nh_nl_ratio_score,
        "breadth_trend_score": breadth_trend_score,
        "volume_breadth": vol_breadth_val,
    })
    score = round(score, 1)

    regime = _classify_regime(score)
    signal = _REGIME_SIGNALS[regime]

    # McClellan
    mcclellan = last.get("mcclellan_osc", 0.0)
    mcclellan_val = float(mcclellan) if mcclellan == mcclellan else 0.0

    # Breadth thrust: check last 5 sessions for any thrust event
    recent_thrusts = s["breadth_thrust"].tail(5)
    thrust_active = bool(recent_thrusts.any())
    thrust_today = bool(last.get("breadth_thrust", False))

    # Sector breadth
    sec_breadth = sector_breadth(prices, days=1)

    # Index breadth
    idx_breadth = index_breadth(prices)

    as_of = str(s.index[-1].date())

    return {
        "score": score,
        "regime": regime,
        "signal": signal,
        "as_of": as_of,

        # DMA participation
        "pct_above_20ema": round(float(a20), 1) if a20 == a20 else None,
        "pct_above_50dma": round(float(a50), 1),
        "pct_above_200dma": round(float(a200), 1),
        "participation_trend_5d": round(float(trend), 1),

        # Advance / decline
        "advances": int(last["advances"]),
        "declines": int(last["declines"]),
        "ad_ratio": round(float(last["ad_ratio"]), 2) if last["ad_ratio"] == last["ad_ratio"] else None,
        "pct_advancing": round(float(last["pct_advancing"]), 1) if last["pct_advancing"] == last["pct_advancing"] else None,

        # Volume-weighted breadth
        "volume_breadth": round(vol_breadth_val, 1),

        # New highs / lows
        "new_highs": int(last["new_highs"]),
        "new_lows": int(last["new_lows"]),
        "nh_nl_ratio": round(float(nh_nl_ratio_val), 1) if nh_nl_ratio_val == nh_nl_ratio_val else None,

        # McClellan oscillator
        "mcclellan_oscillator": round(mcclellan_val, 3),

        # Breadth thrust
        "breadth_thrust_today": thrust_today,
        "breadth_thrust_recent_5d": thrust_active,

        # Sub-scores (for transparency / debugging)
        "sub_scores": {
            "ad_ratio_score": round(ad_ratio_score, 1),
            "nh_nl_ratio_score": round(nh_nl_ratio_score, 1),
            "breadth_trend_score": round(breadth_trend_score, 1),
            "volume_breadth_score": round(vol_breadth_val, 1),
        },

        # Sector and index breakdowns
        "sector_breadth": sec_breadth,
        "index_breadth": idx_breadth,
    }


# -- CLI ----------------------------------------------------------------------

if __name__ == "__main__":
    r = breadth_read()
    if r["score"] is None:
        print("No breadth data available.")
        raise SystemExit(1)

    print("=" * 68)
    print(f"  MARKET BREADTH ENGINE  ({r['as_of']})")
    print("=" * 68)
    print(f"  Composite Score : {r['score']}/100")
    print(f"  Regime          : {r['regime']}")
    print(f"  Signal          : {r['signal']}")
    print("-" * 68)
    print(f"  % above 20-EMA  : {r['pct_above_20ema']}%")
    print(f"  % above 50-DMA  : {r['pct_above_50dma']}%  "
          f"(5d trend {r['participation_trend_5d']:+})")
    print(f"  % above 200-DMA : {r['pct_above_200dma']}%")
    print(f"  Adv/Dec         : {r['advances']}/{r['declines']}  "
          f"(AD ratio {r['ad_ratio']})")
    print(f"  Volume breadth  : {r['volume_breadth']}%")
    print(f"  New highs/lows  : {r['new_highs']} / {r['new_lows']}  "
          f"(NH/NL ratio {r['nh_nl_ratio']})")
    print(f"  McClellan Osc   : {r['mcclellan_oscillator']}")
    thrust_str = "YES" if r["breadth_thrust_today"] else "no"
    print(f"  Breadth thrust  : {thrust_str}  "
          f"(recent 5d: {'yes' if r['breadth_thrust_recent_5d'] else 'no'})")

    # Index breadth
    print("-" * 68)
    for key in ("nifty50", "banknifty"):
        ib = r["index_breadth"].get(key, {})
        label = ib.get("label", key)
        sc = ib.get("score")
        if sc is not None:
            print(f"  {label:12s} : score {sc}/100  regime={ib['regime']}  "
                  f"A/D={ib['advances']}/{ib['declines']}  "
                  f">50DMA={ib['pct_above_50dma']}%  "
                  f"({ib['constituents_found']}/{ib['constituents_total']} found)")
        else:
            print(f"  {label:12s} : {ib.get('signal', 'n/a')}")

    # Top/bottom sectors
    sec = r.get("sector_breadth", {})
    if sec:
        ranked = sorted(sec.items(), key=lambda x: x[1]["score"], reverse=True)
        print("-" * 68)
        print("  SECTOR BREADTH (top 5 / bottom 5):")
        for name, data in ranked[:5]:
            print(f"    {name:30s}  score={data['score']:5.1f}  "
                  f">50DMA={data['pct_above_50dma']:5.1f}%  "
                  f"A/D={data['advances']}/{data['declines']}")
        if len(ranked) > 10:
            print("    ...")
        for name, data in ranked[-5:]:
            print(f"    {name:30s}  score={data['score']:5.1f}  "
                  f">50DMA={data['pct_above_50dma']:5.1f}%  "
                  f"A/D={data['advances']}/{data['declines']}")
    print("=" * 68)

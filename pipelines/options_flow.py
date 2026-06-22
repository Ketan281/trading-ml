"""
Options flow analytics engine -- composite sentiment, build-up
classification, IV regime, zone detection, vol expansion probability.

Pulls a live option-chain snapshot from NSELive (via options_chain.py)
and layers every read an index-options desk cares about into a single
scored output.
"""

import os
import sys
import json
import glob

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipelines.options.options_chain import (
    fetch_options_chain,
    get_strike_step,
    round_to_strike,
)
from pipelines.options.chain_live_intel import bs_greeks, STEP

OUTPUT_DIR = os.path.join(ROOT, "data", "options")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HISTORICAL_DIR = os.path.join(ROOT, "data", "options")
AGG_DIR = os.path.join(ROOT, "data", "option_chain", "agg")


# ---- helpers --------------------------------------------------------

def _safe_div(num, den, default=0.0):
    return round(num / den, 4) if den else default


def _clamp(value, lo=0.0, hi=100.0):
    return max(lo, min(hi, value))


def _fetch_chain(symbol):
    """Fetch chain and return (df, spot, atm, expiry, dte) or Nones."""
    result = fetch_options_chain(symbol)
    if not result or result["df"].empty:
        return None, None, None, None, None
    df = result["df"]
    spot = result["spot"]
    atm = result["atm"]
    expiry = result["expiry"]
    from datetime import datetime
    try:
        exp_date = datetime.strptime(expiry, "%d-%b-%Y")
        dte = max((exp_date - datetime.now()).days, 1)
    except Exception:
        dte = 7
    return df, spot, atm, expiry, dte


# ---- 1. OI change analysis -----------------------------------------

def _oi_change_analysis(df):
    total_ce_chng = int(df["ce_chng_oi"].sum())
    total_pe_chng = int(df["pe_chng_oi"].sum())

    per_strike_ce = (
        df[["strike", "ce_chng_oi"]]
        .sort_values("ce_chng_oi", ascending=False)
        .head(5)
        .to_dict("records")
    )
    per_strike_pe = (
        df[["strike", "pe_chng_oi"]]
        .sort_values("pe_chng_oi", ascending=False)
        .head(5)
        .to_dict("records")
    )

    if total_pe_chng > total_ce_chng:
        direction = "bullish"
    elif total_ce_chng > total_pe_chng:
        direction = "bearish"
    else:
        direction = "neutral"

    return {
        "total_ce_oi_change": total_ce_chng,
        "total_pe_oi_change": total_pe_chng,
        "net_oi_change_direction": direction,
        "top_ce_oi_change_strikes": per_strike_ce,
        "top_pe_oi_change_strikes": per_strike_pe,
    }


# ---- 2. Build-up classification ------------------------------------

def classify_buildup(chain_df, spot):
    """Classify OI build-up type per strike.

    Returns a dict with per-strike classification, aggregate counts,
    and the dominant pattern across all strikes.
    """
    rows = []
    for _, r in chain_df.iterrows():
        ce_oi_chg = r.get("ce_chng_oi", 0)
        pe_oi_chg = r.get("pe_chng_oi", 0)

        ce_buildup = _classify_single(r["ce_ltp"], ce_oi_chg)
        pe_buildup = _classify_single(r["pe_ltp"], pe_oi_chg)

        rows.append({
            "strike": r["strike"],
            "ce_buildup": ce_buildup,
            "pe_buildup": pe_buildup,
            "ce_oi_change": ce_oi_chg,
            "pe_oi_change": pe_oi_chg,
        })

    buildup_df = pd.DataFrame(rows)

    agg = {}
    for col in ("ce_buildup", "pe_buildup"):
        counts = buildup_df[col].value_counts().to_dict()
        agg[col] = counts

    dominant = _dominant_buildup(buildup_df)

    return {
        "per_strike": buildup_df.to_dict("records"),
        "aggregate": agg,
        "dominant_pattern": dominant,
    }


def _classify_single(ltp, oi_chg):
    """Classify build-up from price and OI change.

    Long Build-Up:   price up + OI up
    Short Build-Up:  price down + OI up
    Short Covering:  price up + OI down
    Long Unwinding:  price down + OI down

    NSE snapshot does not carry previous LTP explicitly; we treat
    nonzero LTP as a proxy for active premium (positive price signal).
    """
    if oi_chg > 0:
        if ltp > 0:
            return "long_buildup"
        return "short_buildup"
    elif oi_chg < 0:
        if ltp > 0:
            return "short_covering"
        return "long_unwinding"
    return "neutral"


def _dominant_buildup(buildup_df):
    all_labels = list(buildup_df["ce_buildup"]) + list(buildup_df["pe_buildup"])
    counts = pd.Series(all_labels).value_counts()
    if counts.empty:
        return "neutral"
    top = counts.index[0]
    return top


# ---- 3. PCR analytics ----------------------------------------------

def _pcr_analytics(df, symbol):
    total_ce_oi = df["ce_oi"].sum()
    total_pe_oi = df["pe_oi"].sum()
    total_ce_vol = df["ce_volume"].sum()
    total_pe_vol = df["pe_volume"].sum()
    total_ce_chng = df["ce_chng_oi"].sum()
    total_pe_chng = df["pe_chng_oi"].sum()

    pcr_oi = _safe_div(total_pe_oi, total_ce_oi)
    pcr_vol = _safe_div(total_pe_vol, total_ce_vol)
    pcr_oi_chng = _safe_div(total_pe_chng, total_ce_chng)

    pcr_5d_avg = _pcr_5d_average(symbol)

    if pcr_5d_avg is not None:
        pcr_trend = "rising" if pcr_oi > pcr_5d_avg else "falling"
    else:
        pcr_trend = None

    return {
        "pcr_oi": round(pcr_oi, 3),
        "pcr_volume": round(pcr_vol, 3),
        "pcr_oi_change": round(pcr_oi_chng, 3),
        "pcr_5d_avg": round(pcr_5d_avg, 3) if pcr_5d_avg else None,
        "pcr_trend": pcr_trend,
    }


def _pcr_5d_average(symbol):
    """Try loading PCR from collected snapshots or saved OI analysis files."""
    agg_path = os.path.join(AGG_DIR, f"{symbol}.csv")
    if os.path.exists(agg_path):
        try:
            agg = pd.read_csv(agg_path)
            if "pcr_oi" in agg.columns and len(agg) >= 5:
                return float(agg["pcr_oi"].tail(5).mean())
        except Exception:
            pass

    pattern = os.path.join(HISTORICAL_DIR, f"{symbol}_oi_analysis_*.json")
    files = sorted(glob.glob(pattern))
    if len(files) < 2:
        return None
    pcrs = []
    for f in files[-5:]:
        try:
            with open(f) as fh:
                data = json.load(fh)
            pcr_val = data.get("pcr", {}).get("pcr_oi")
            if pcr_val is not None:
                pcrs.append(pcr_val)
        except Exception:
            continue
    return float(np.mean(pcrs)) if pcrs else None


# ---- 4. IV analytics -----------------------------------------------

def compute_iv_analytics(chain_df, symbol="NIFTY"):
    """IV rank, percentile, skew, ATM IV."""
    spot = chain_df["spot"].iloc[0]
    step = get_strike_step(symbol)
    atm = round_to_strike(spot, step)

    atm_row = chain_df[chain_df["strike"] == atm]
    if atm_row.empty:
        atm_row = chain_df.iloc[
            (chain_df["strike"] - atm).abs().argsort()[:1]
        ]

    atm_ce_iv = float(atm_row["ce_iv"].values[0])
    atm_pe_iv = float(atm_row["pe_iv"].values[0])
    atm_iv = round((atm_ce_iv + atm_pe_iv) / 2, 2)

    otm_call_strike = atm + 3 * step
    otm_put_strike = atm - 3 * step
    otm_ce_row = chain_df[chain_df["strike"] == otm_call_strike]
    otm_pe_row = chain_df[chain_df["strike"] == otm_put_strike]

    otm_ce_iv = (
        float(otm_ce_row["ce_iv"].values[0])
        if not otm_ce_row.empty else 0
    )
    otm_pe_iv = (
        float(otm_pe_row["pe_iv"].values[0])
        if not otm_pe_row.empty else 0
    )

    skew = (
        round(otm_pe_iv - otm_ce_iv, 2)
        if otm_pe_iv and otm_ce_iv else 0
    )

    iv_rank, iv_percentile_val = _iv_rank_percentile(symbol, atm_iv)

    return {
        "atm_iv": atm_iv,
        "atm_ce_iv": atm_ce_iv,
        "atm_pe_iv": atm_pe_iv,
        "otm_call_iv": otm_ce_iv,
        "otm_put_iv": otm_pe_iv,
        "iv_skew": skew,
        "iv_rank": iv_rank,
        "iv_percentile": iv_percentile_val,
    }


def _iv_rank_percentile(symbol, current_iv):
    """IV Rank and Percentile from historical data.

    IV Rank = (current - min) / (max - min) * 100
    IV Percentile = % of days where IV was below current
    """
    iv_history = _load_iv_history(symbol)

    if iv_history is None or len(iv_history) < 5:
        return None, None

    iv_min = float(iv_history.min())
    iv_max = float(iv_history.max())

    if iv_max == iv_min:
        return 50.0, 50.0

    iv_rank = round(
        (current_iv - iv_min) / (iv_max - iv_min) * 100, 1
    )
    iv_rank = _clamp(iv_rank)

    iv_pct = round(float((iv_history < current_iv).mean()) * 100, 1)

    return iv_rank, iv_pct


def _load_iv_history(symbol):
    """Load historical ATM IV series from available sources."""
    agg_path = os.path.join(AGG_DIR, f"{symbol}.csv")
    if os.path.exists(agg_path):
        try:
            agg = pd.read_csv(agg_path)
            if "atm_iv" in agg.columns and len(agg) >= 5:
                return agg["atm_iv"].dropna()
        except Exception:
            pass

    pattern = os.path.join(
        HISTORICAL_DIR, f"{symbol}_iv_analysis_*.json"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        return None

    ivs = []
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            iv_val = data.get("iv_surface", {}).get("atm_iv")
            if iv_val is not None and iv_val > 0:
                ivs.append(iv_val)
        except Exception:
            continue

    return pd.Series(ivs) if len(ivs) >= 2 else None


# ---- 5. Zone detection ---------------------------------------------

def detect_zones(chain_df, spot):
    """Support, resistance, OI walls, gamma risk zones."""

    support = (
        chain_df.nlargest(3, "pe_oi")[["strike", "pe_oi"]]
        .to_dict("records")
    )

    resistance = (
        chain_df.nlargest(3, "ce_oi")[["strike", "ce_oi"]]
        .to_dict("records")
    )

    avg_oi = (chain_df["ce_oi"].mean() + chain_df["pe_oi"].mean()) / 2
    threshold = avg_oi * 2 if avg_oi > 0 else 0

    oi_walls = []
    for _, r in chain_df.iterrows():
        if r["ce_oi"] > threshold or r["pe_oi"] > threshold:
            wall_type = []
            if r["ce_oi"] > threshold:
                wall_type.append("resistance")
            if r["pe_oi"] > threshold:
                wall_type.append("support")
            oi_walls.append({
                "strike": int(r["strike"]),
                "ce_oi": int(r["ce_oi"]),
                "pe_oi": int(r["pe_oi"]),
                "type": "+".join(wall_type),
            })

    gamma_zones = _gamma_risk_zones(chain_df, spot)

    return {
        "support_zones": support,
        "resistance_zones": resistance,
        "oi_walls": oi_walls,
        "gamma_risk_zones": gamma_zones,
    }


def _gamma_risk_zones(chain_df, spot):
    """Strikes with highest gamma exposure near spot."""
    if "ce_gamma" not in chain_df.columns:
        return []

    nearby = chain_df[
        (chain_df["strike"] >= spot * 0.97)
        & (chain_df["strike"] <= spot * 1.03)
    ].copy()

    if nearby.empty:
        return []

    nearby["total_gamma"] = (
        nearby["ce_gamma"].abs() * nearby["ce_oi"]
        + nearby["pe_gamma"].abs() * nearby["pe_oi"]
    )

    top = nearby.nlargest(3, "total_gamma")
    return [
        {
            "strike": int(r["strike"]),
            "gamma_exposure": round(float(r["total_gamma"]), 2),
        }
        for _, r in top.iterrows()
    ]


# ---- 6. Volatility expansion probability ---------------------------

def vol_expansion_probability(symbol="NIFTY"):
    """Probability of volatility expansion 0-100.

    Based on IV rank + OI build-up pattern + historical vol regime.
    Low IV rank with aggressive positioning = expansion likely.
    High IV rank with unwinding = contraction likely.
    """
    df, spot, atm, expiry, dte = _fetch_chain(symbol)
    if df is None:
        return {"probability": None, "note": "chain unavailable"}

    iv_data = compute_iv_analytics(df, symbol)
    buildup_data = classify_buildup(df, spot)

    iv_rank = iv_data.get("iv_rank")
    dominant = buildup_data.get("dominant_pattern", "neutral")

    # IV rank component: low rank = room to expand
    if iv_rank is not None:
        iv_component = _clamp(100 - iv_rank, 0, 100)
    else:
        iv_component = 50

    # Build-up component
    buildup_map = {
        "long_buildup": 60,
        "short_buildup": 70,
        "short_covering": 40,
        "long_unwinding": 55,
        "neutral": 45,
    }
    buildup_component = buildup_map.get(dominant, 50)

    # OI change velocity -- large net OI additions signal positioning
    total_oi_chng = (
        abs(df["ce_chng_oi"].sum()) + abs(df["pe_chng_oi"].sum())
    )
    total_oi = df["ce_oi"].sum() + df["pe_oi"].sum()
    oi_ratio = (
        _safe_div(total_oi_chng, total_oi) * 100 if total_oi else 0
    )
    oi_component = _clamp(oi_ratio * 10, 0, 100)

    probability = round(
        iv_component * 0.45
        + buildup_component * 0.30
        + oi_component * 0.25,
        1,
    )
    probability = _clamp(probability)

    return {
        "probability": probability,
        "iv_rank": iv_rank,
        "iv_component": round(iv_component, 1),
        "buildup_component": buildup_component,
        "oi_velocity_component": round(oi_component, 1),
        "dominant_buildup": dominant,
    }


# ---- 7. Options sentiment score -------------------------------------

def options_sentiment_score(symbol="NIFTY"):
    """Composite options sentiment 0-100.

    Weights: PCR signal (30%) + build-up pattern (25%) + IV regime (20%)
           + OI wall positioning (15%) + skew (10%)

    > 70  Bullish | 50-70 Mildly bullish | 30-50 Mildly bearish | < 30 Bearish
    """
    df, spot, atm, expiry, dte = _fetch_chain(symbol)
    if df is None:
        return {
            "score": None,
            "label": "unavailable",
            "note": "chain unavailable",
        }

    pcr_data = _pcr_analytics(df, symbol)
    buildup_data = classify_buildup(df, spot)
    iv_data = compute_iv_analytics(df, symbol)
    zone_data = detect_zones(df, spot)

    # -- PCR signal (30%) --
    pcr = pcr_data["pcr_oi"]
    if pcr >= 1.5:
        pcr_score = 90
    elif pcr >= 1.2:
        pcr_score = 75
    elif pcr >= 0.8:
        pcr_score = 50
    elif pcr >= 0.5:
        pcr_score = 30
    else:
        pcr_score = 10

    # -- Build-up pattern (25%) --
    dominant = buildup_data.get("dominant_pattern", "neutral")
    buildup_scores = {
        "long_buildup": 80,
        "short_covering": 70,
        "neutral": 50,
        "long_unwinding": 30,
        "short_buildup": 20,
    }
    buildup_score = buildup_scores.get(dominant, 50)

    # -- IV regime (20%) --
    atm_iv = iv_data["atm_iv"]
    iv_rank = iv_data.get("iv_rank")
    if iv_rank is not None:
        if iv_rank < 20:
            iv_score = 60
        elif iv_rank < 40:
            iv_score = 55
        elif iv_rank < 60:
            iv_score = 50
        elif iv_rank < 80:
            iv_score = 40
        else:
            iv_score = 25
    else:
        if atm_iv < 12:
            iv_score = 60
        elif atm_iv < 18:
            iv_score = 50
        else:
            iv_score = 30

    # -- OI wall positioning (15%) --
    support_strikes = [
        z["strike"] for z in zone_data["support_zones"]
    ]
    resist_strikes = [
        z["strike"] for z in zone_data["resistance_zones"]
    ]

    if support_strikes and resist_strikes:
        below = [s for s in support_strikes if s <= spot]
        above = [s for s in resist_strikes if s >= spot]
        nearest_support = max(below) if below else min(support_strikes)
        nearest_resist = min(above) if above else max(resist_strikes)

        dist_support = abs(spot - nearest_support)
        dist_resist = abs(nearest_resist - spot)

        if dist_support < dist_resist:
            wall_score = 65
        elif dist_resist < dist_support:
            wall_score = 35
        else:
            wall_score = 50
    else:
        wall_score = 50

    # -- IV skew (10%) --
    # Positive skew (put IV > call IV) = fear = bearish pressure
    skew = iv_data.get("iv_skew", 0)
    if skew > 3:
        skew_score = 25
    elif skew > 1:
        skew_score = 40
    elif skew > -1:
        skew_score = 55
    elif skew > -3:
        skew_score = 65
    else:
        skew_score = 80

    # -- composite --
    score = round(
        pcr_score * 0.30
        + buildup_score * 0.25
        + iv_score * 0.20
        + wall_score * 0.15
        + skew_score * 0.10,
        1,
    )
    score = _clamp(score)

    if score > 70:
        label = "bullish"
    elif score > 50:
        label = "mildly_bullish"
    elif score > 30:
        label = "mildly_bearish"
    else:
        label = "bearish"

    return {
        "score": score,
        "label": label,
        "components": {
            "pcr_score": pcr_score,
            "buildup_score": buildup_score,
            "iv_score": iv_score,
            "wall_score": wall_score,
            "skew_score": skew_score,
        },
        "weights": {
            "pcr": 0.30,
            "buildup": 0.25,
            "iv_regime": 0.20,
            "oi_wall": 0.15,
            "skew": 0.10,
        },
        "inputs": {
            "pcr_oi": pcr_data["pcr_oi"],
            "dominant_buildup": dominant,
            "atm_iv": atm_iv,
            "iv_rank": iv_rank,
            "iv_skew": skew,
        },
    }


# ---- Full analysis --------------------------------------------------

def options_flow_analysis(symbol="NIFTY"):
    """Complete options flow analysis. Returns dict with all metrics."""
    from datetime import datetime

    df, spot, atm, expiry, dte = _fetch_chain(symbol)
    if df is None:
        print(f"  [options_flow] {symbol}: chain fetch failed")
        return None

    print(f"\n{'=' * 60}")
    print(f"  OPTIONS FLOW ANALYSIS -- {symbol}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Spot: {spot} | ATM: {atm} | Expiry: {expiry} | DTE: {dte}")
    print(f"{'=' * 60}")

    oi_change = _oi_change_analysis(df)
    buildup = classify_buildup(df, spot)
    pcr = _pcr_analytics(df, symbol)
    iv = compute_iv_analytics(df, symbol)
    zones = detect_zones(df, spot)
    vol_prob = vol_expansion_probability(symbol)
    sentiment = options_sentiment_score(symbol)

    print(f"\n  OI Change:")
    print(f"    CE OI change (total) : {oi_change['total_ce_oi_change']:,}")
    print(f"    PE OI change (total) : {oi_change['total_pe_oi_change']:,}")
    print(f"    Direction            : "
          f"{oi_change['net_oi_change_direction'].upper()}")

    print(f"\n  Build-Up:")
    print(f"    Dominant pattern     : "
          f"{buildup['dominant_pattern'].upper()}")

    print(f"\n  PCR:")
    print(f"    PCR (OI)             : {pcr['pcr_oi']}")
    print(f"    PCR (Volume)         : {pcr['pcr_volume']}")
    print(f"    PCR (OI Change)      : {pcr['pcr_oi_change']}")
    print(f"    PCR 5-day avg        : {pcr['pcr_5d_avg']}")
    print(f"    PCR trend            : {pcr['pcr_trend']}")

    print(f"\n  IV Analytics:")
    print(f"    ATM IV               : {iv['atm_iv']}%")
    print(f"    OTM Call IV          : {iv['otm_call_iv']}%")
    print(f"    OTM Put IV           : {iv['otm_put_iv']}%")
    print(f"    IV Skew (put-call)   : {iv['iv_skew']}%")
    print(f"    IV Rank              : {iv['iv_rank']}")
    print(f"    IV Percentile        : {iv['iv_percentile']}")

    print(f"\n  Zones:")
    print(f"    Support (PE OI)      : "
          f"{[z['strike'] for z in zones['support_zones']]}")
    print(f"    Resistance (CE OI)   : "
          f"{[z['strike'] for z in zones['resistance_zones']]}")
    print(f"    OI walls             : {len(zones['oi_walls'])} strikes")
    print(f"    Gamma risk zones     : "
          f"{[z['strike'] for z in zones['gamma_risk_zones']]}")

    print(f"\n  Vol Expansion Prob     : {vol_prob['probability']}/100")

    print(f"\n  SENTIMENT SCORE        : "
          f"{sentiment['score']}/100 -- {sentiment['label'].upper()}")

    print(f"\n  Components:")
    for k, v in sentiment.get("components", {}).items():
        print(f"    {k:<20} : {v}")

    print(f"{'=' * 60}\n")

    report = {
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "spot": spot,
        "atm": atm,
        "expiry": expiry,
        "dte": dte,
        "oi_change": oi_change,
        "buildup": {
            "dominant_pattern": buildup["dominant_pattern"],
            "aggregate": buildup["aggregate"],
        },
        "pcr": pcr,
        "iv": iv,
        "zones": {
            "support_zones": zones["support_zones"],
            "resistance_zones": zones["resistance_zones"],
            "oi_walls_count": len(zones["oi_walls"]),
            "gamma_risk_zones": zones["gamma_risk_zones"],
        },
        "vol_expansion_probability": vol_prob["probability"],
        "sentiment": sentiment,
    }

    path = os.path.join(
        OUTPUT_DIR,
        f"{symbol}_flow_analysis_"
        f"{datetime.now().strftime('%Y%m%d_%H%M')}.json",
    )
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Saved -> {path}")

    return report


# ---- main -----------------------------------------------------------

if __name__ == "__main__":
    from datetime import datetime

    print("=" * 60)
    print("  Trading AI -- Options Flow Analytics")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    for sym in ["NIFTY", "BANKNIFTY"]:
        try:
            result = options_flow_analysis(sym)
            if result is None:
                print(f"  {sym}: analysis failed (chain unavailable)")
        except Exception as e:
            print(f"  {sym}: error -- {e}")

    print("\n  Options flow analytics complete.")

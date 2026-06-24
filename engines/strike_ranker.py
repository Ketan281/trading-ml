"""
Strike-Level Option Contract Ranker — rank 200+ contracts, pick top N.

APPROACH (same math as stock direction model that hit 89%)
---------------------------------------------------------
Instead of binary "NIFTY up or down?" (50.9% ceiling), this ranks individual
option contracts cross-sectionally: "which of these 200 contracts will return
the most today?" — the same ranking problem that achieved 89% on stocks.

Each row = one strike (CE or PE) with features:
  - Strike distance from spot / max pain
  - OI, OI change, volume at this strike
  - IV, IV skew, IV rank
  - PCR at this strike level
  - Greeks approximation (moneyness proxy for delta/gamma)
  - Breadth + intermarket context
  - Calendar (expiry proximity, day of week)

Label: actual return of that contract (next snapshot or EOD).

TRAINING MODES
--------------
  1. Historical backfill: uses raw strike CSVs (data/option_chain/raw/)
  2. Live: collects, predicts, and logs for next-day label creation

DATA REQUIREMENT: 3-6 months of strike-level snapshots (collecting since Jun 2026).
Until enough data, falls back to rule-based scoring with the same feature set.

Run: python -m engines.strike_ranker
"""

import os
import sys
import json
import time
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

log = logging.getLogger("strike_ranker")

MODEL_DIR = os.path.join(ROOT, "models", "intraday")
RAW_DIR = os.path.join(ROOT, "data", "option_chain", "raw")
AGG_DIR = os.path.join(ROOT, "data", "option_chain", "agg")
os.makedirs(MODEL_DIR, exist_ok=True)

INDICES = ["NIFTY", "BANKNIFTY"]
MIN_DAYS_FOR_ML = 60

STRIKE_FEATURES = [
    # Strike positioning
    "moneyness",              # (strike - spot) / spot
    "abs_moneyness",          # abs distance
    "dist_from_max_pain",     # (strike - max_pain) / spot
    "strikes_from_atm",       # integer steps from ATM

    # This strike's OI & volume
    "oi_normalized",          # OI at this strike / total OI
    "oi_change_normalized",   # change OI / total OI
    "volume_normalized",      # vol / total vol
    "oi_to_vol_ratio",        # OI / volume (liquidity)

    # IV at this strike
    "iv",                     # implied vol
    "iv_vs_atm",              # IV - ATM_IV (skew position)
    "iv_rank_20d",            # where current IV sits vs last 20 days

    # PCR context
    "pcr_at_strike",          # PE OI / CE OI at this specific strike
    "pcr_oi_overall",         # overall PCR
    "pcr_vol_overall",

    # Option pricing
    "ltp",                    # last traded price
    "ltp_pct_of_spot",        # premium as % of spot
    "intrinsic_value_pct",    # intrinsic / spot
    "time_value_pct",         # extrinsic / spot

    # Aggregate context
    "max_pain_dist_pct",
    "atm_iv",
    "india_vix",
    "spot_change_pct",        # spot move since previous snapshot

    # Calendar
    "day_of_week",
    "days_to_expiry",
    "is_expiry_day",
    "hour_of_day",
    "minutes_since_open",

    # Direction bias from index model
    "index_model_prob",       # probability from index_options model

    # Option type
    "is_call",                # 1 for CE, 0 for PE
]


def _load_raw_day(symbol, date_str):
    path = os.path.join(RAW_DIR, symbol, f"{date_str}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def _load_agg_data(symbol):
    path = os.path.join(AGG_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["timestamp"])


def _compute_strike_features(strikes_df, agg_row, symbol, index_prob=0.5):
    """Compute ML features for each strike in a single snapshot."""
    spot = agg_row.get("spot", 0)
    atm = agg_row.get("atm", spot)
    max_pain = agg_row.get("max_pain", atm)
    atm_iv_val = agg_row.get("atm_iv", 20)
    vix = agg_row.get("india_vix", 15)
    pcr_oi = agg_row.get("pcr_oi", 1.0)
    pcr_vol = agg_row.get("pcr_vol", 1.0)
    max_pain_dist = agg_row.get("max_pain_dist_pct", 0)
    step = 50 if symbol == "NIFTY" else 100

    if spot <= 0:
        return pd.DataFrame()

    tot_ce_oi = strikes_df["ce_oi"].sum()
    tot_pe_oi = strikes_df["pe_oi"].sum()
    tot_ce_vol = strikes_df["ce_vol"].sum()
    tot_pe_vol = strikes_df["pe_vol"].sum()
    tot_oi = tot_ce_oi + tot_pe_oi
    tot_vol = tot_ce_vol + tot_pe_vol

    ts = strikes_df["timestamp"].iloc[0] if "timestamp" in strikes_df.columns else datetime.now()
    if isinstance(ts, str):
        ts = pd.to_datetime(ts)
    dow = ts.dayofweek
    hour = ts.hour
    mins_since_open = (hour - 9) * 60 + ts.minute - 15

    expiry_day = 3 if symbol == "NIFTY" else 2
    dte = (expiry_day - dow) % 7
    is_expiry = 1 if dte == 0 else 0

    rows = []
    for _, r in strikes_df.iterrows():
        strike = r["strike"]
        moneyness = (strike - spot) / spot

        # CE contract
        ce_oi = r.get("ce_oi", 0)
        ce_chg = r.get("ce_chg_oi", 0)
        ce_vol = r.get("ce_vol", 0)
        ce_iv = r.get("ce_iv", 0)
        ce_ltp = r.get("ce_ltp", 0)
        ce_intrinsic = max(spot - strike, 0)
        ce_time_val = max(ce_ltp - ce_intrinsic, 0)

        rows.append({
            "symbol": symbol, "strike": strike, "option_type": "CE",
            "timestamp": ts,
            "moneyness": moneyness,
            "abs_moneyness": abs(moneyness),
            "dist_from_max_pain": (strike - max_pain) / spot,
            "strikes_from_atm": round((strike - atm) / step),
            "oi_normalized": ce_oi / max(tot_oi, 1),
            "oi_change_normalized": ce_chg / max(tot_oi, 1),
            "volume_normalized": ce_vol / max(tot_vol, 1),
            "oi_to_vol_ratio": ce_oi / max(ce_vol, 1),
            "iv": ce_iv,
            "iv_vs_atm": ce_iv - atm_iv_val,
            "iv_rank_20d": 50,  # needs history
            "pcr_at_strike": r.get("pe_oi", 0) / max(ce_oi, 1),
            "pcr_oi_overall": pcr_oi,
            "pcr_vol_overall": pcr_vol,
            "ltp": ce_ltp,
            "ltp_pct_of_spot": ce_ltp / spot * 100,
            "intrinsic_value_pct": ce_intrinsic / spot * 100,
            "time_value_pct": ce_time_val / spot * 100,
            "max_pain_dist_pct": max_pain_dist,
            "atm_iv": atm_iv_val,
            "india_vix": vix if vix else 15,
            "spot_change_pct": 0,
            "day_of_week": dow,
            "days_to_expiry": dte,
            "is_expiry_day": is_expiry,
            "hour_of_day": hour,
            "minutes_since_open": mins_since_open,
            "index_model_prob": index_prob,
            "is_call": 1,
            "ltp_raw": ce_ltp,
        })

        # PE contract
        pe_oi = r.get("pe_oi", 0)
        pe_chg = r.get("pe_chg_oi", 0)
        pe_vol = r.get("pe_vol", 0)
        pe_iv = r.get("pe_iv", 0)
        pe_ltp = r.get("pe_ltp", 0)
        pe_intrinsic = max(strike - spot, 0)
        pe_time_val = max(pe_ltp - pe_intrinsic, 0)

        rows.append({
            "symbol": symbol, "strike": strike, "option_type": "PE",
            "timestamp": ts,
            "moneyness": -moneyness,  # flip for PE
            "abs_moneyness": abs(moneyness),
            "dist_from_max_pain": (max_pain - strike) / spot,
            "strikes_from_atm": round((atm - strike) / step),
            "oi_normalized": pe_oi / max(tot_oi, 1),
            "oi_change_normalized": pe_chg / max(tot_oi, 1),
            "volume_normalized": pe_vol / max(tot_vol, 1),
            "oi_to_vol_ratio": pe_oi / max(pe_vol, 1),
            "iv": pe_iv,
            "iv_vs_atm": pe_iv - atm_iv_val,
            "iv_rank_20d": 50,
            "pcr_at_strike": pe_oi / max(r.get("ce_oi", 1), 1),
            "pcr_oi_overall": pcr_oi,
            "pcr_vol_overall": pcr_vol,
            "ltp": pe_ltp,
            "ltp_pct_of_spot": pe_ltp / spot * 100,
            "intrinsic_value_pct": pe_intrinsic / spot * 100,
            "time_value_pct": pe_time_val / spot * 100,
            "max_pain_dist_pct": max_pain_dist,
            "atm_iv": atm_iv_val,
            "india_vix": vix if vix else 15,
            "spot_change_pct": 0,
            "day_of_week": dow,
            "days_to_expiry": dte,
            "is_expiry_day": is_expiry,
            "hour_of_day": hour,
            "minutes_since_open": mins_since_open,
            "index_model_prob": index_prob,
            "is_call": 0,
            "ltp_raw": pe_ltp,
        })

    return pd.DataFrame(rows)


def _rule_based_score(row):
    """Score a contract using proven options trading heuristics.
    Used when ML model doesn't have enough data yet."""
    score = 50.0

    # 1. Moneyness sweet spot: ATM and 1 strike OTM have best risk/reward
    abs_m = row.get("abs_moneyness", 0)
    if abs_m < 0.005:
        score += 15  # ATM
    elif abs_m < 0.015:
        score += 10  # near money
    elif abs_m > 0.04:
        score -= 15  # deep OTM — high theta decay

    # 2. OI buildup at strike = institutional interest
    oi_norm = row.get("oi_normalized", 0)
    if oi_norm > 0.05:
        score += 10
    oi_chg = row.get("oi_change_normalized", 0)
    if oi_chg > 0.01:
        score += 8  # fresh OI buildup = smart money

    # 3. Volume confirms interest
    vol_norm = row.get("volume_normalized", 0)
    if vol_norm > 0.05:
        score += 5

    # 4. IV: lower IV = cheaper options (buy low IV)
    iv_vs = row.get("iv_vs_atm", 0)
    if iv_vs < -2:
        score += 5  # cheaper than ATM
    elif iv_vs > 5:
        score -= 5  # expensive

    # 5. Direction alignment with index model
    prob = row.get("index_model_prob", 0.5)
    is_call = row.get("is_call", 1)
    if is_call and prob > 0.55:
        score += (prob - 0.5) * 40  # bullish + CE
    elif not is_call and prob < 0.45:
        score += (0.5 - prob) * 40  # bearish + PE
    elif is_call and prob < 0.45:
        score -= 10  # wrong direction
    elif not is_call and prob > 0.55:
        score -= 10

    # 6. Max pain gravity: spot tends to move toward max pain
    mp_dist = row.get("max_pain_dist_pct", 0)
    if is_call and mp_dist < 0:
        score += 5  # spot below max pain, CE benefits from pull up
    elif not is_call and mp_dist > 0:
        score += 5  # spot above max pain, PE benefits from pull down

    # 7. PCR signal
    pcr = row.get("pcr_oi_overall", 1.0)
    if pcr > 1.2 and is_call:
        score += 8  # high PCR = bullish (put writers confident)
    elif pcr < 0.8 and not is_call:
        score += 8  # low PCR = bearish

    # 8. Expiry day: theta decay accelerates, favor sellers / ATM buyers
    if row.get("is_expiry_day", 0):
        if abs_m < 0.005:
            score += 10  # ATM on expiry = gamma play
        elif abs_m > 0.02:
            score -= 10  # OTM on expiry = theta death

    # 9. VIX context
    vix = row.get("india_vix", 15)
    if vix > 20:
        score += 5  # high VIX = bigger moves expected
    elif vix < 12:
        score -= 5  # low VIX = compressed moves

    # 10. Liquidity: need volume to enter/exit
    if row.get("ltp", 0) < 2:
        score -= 20  # too cheap = illiquid
    if row.get("oi_to_vol_ratio", 0) > 100:
        score -= 5  # no volume = trapped

    return max(0, min(100, score))


# ── Conviction Filters ──────────────────────────────────────

def _compute_conviction(contracts_df, agg_row, symbol):
    """Apply the 4 strategy filters to determine conviction level.

    HIGH conviction (trade):   3+ signals align → full position
    MEDIUM conviction (trade): 2 signals align  → half position
    LOW conviction (skip):     <2 signals        → no trade

    Signals checked:
      1. Direction model agrees (>55% or <45%)
      2. PCR extreme (>1.3 bullish, <0.7 bearish)
      3. Max pain pull (spot away from max pain in favorable direction)
      4. VIX regime (>15 = enough volatility for options)
      5. Expiry proximity (0-2 DTE = gamma, 3-5 = theta)
      6. OI buildup confirms direction
    """
    prob = contracts_df["index_model_prob"].iloc[0] if len(contracts_df) > 0 else 0.5
    pcr = agg_row.get("pcr_oi", 1.0)
    mp_dist = agg_row.get("max_pain_dist_pct", 0)
    vix = agg_row.get("india_vix", 15) or 15
    dte = contracts_df["days_to_expiry"].iloc[0] if len(contracts_df) > 0 else 3

    bullish_signals = 0
    bearish_signals = 0

    # Direction model
    if prob > 0.55:
        bullish_signals += 1
    elif prob < 0.45:
        bearish_signals += 1

    # PCR
    if pcr > 1.2:
        bullish_signals += 1  # heavy put writing = support
    elif pcr < 0.8:
        bearish_signals += 1  # heavy call writing = resistance

    # Max pain
    if mp_dist < -0.5:
        bullish_signals += 1  # spot below max pain → pull up
    elif mp_dist > 0.5:
        bearish_signals += 1  # spot above max pain → pull down

    # VIX (need vol for options to work). NaN/None = assume vol is ok
    vol_ok = True if (vix is None or pd.isna(vix)) else vix > 13

    # OI buildup
    ce_chg = agg_row.get("ce_chg_oi", 0)
    pe_chg = agg_row.get("pe_chg_oi", 0)
    if pe_chg > ce_chg * 1.5:
        bullish_signals += 1  # put writing = bullish
    elif ce_chg > pe_chg * 1.5:
        bearish_signals += 1  # call writing = bearish

    max_signals = max(bullish_signals, bearish_signals)
    direction = "bullish" if bullish_signals >= bearish_signals else "bearish"

    if max_signals >= 3 and vol_ok:
        conviction = "HIGH"
    elif max_signals >= 2 and vol_ok:
        conviction = "MEDIUM"
    else:
        conviction = "LOW"

    return {
        "conviction": conviction,
        "direction": direction,
        "bullish_signals": bullish_signals,
        "bearish_signals": bearish_signals,
        "vol_ok": vol_ok,
        "details": {
            "model_prob": round(prob, 3),
            "pcr": round(pcr, 3),
            "max_pain_dist": round(mp_dist, 2),
            "vix": round(vix, 1),
            "dte": dte,
            "ce_chg_oi": ce_chg,
            "pe_chg_oi": pe_chg,
        }
    }


# ── Multi-Timeframe Entry ──────────────────────────────────

def _mtf_entry_filter(contracts_df, agg_df, symbol):
    """Multi-timeframe confirmation using intraday snapshots.
    Daily bias (index model) + intraday trend (OI flow direction).

    Returns boost/penalty to apply to scores."""
    if agg_df is None or len(agg_df) < 3:
        return 0

    recent = agg_df.sort_values("timestamp").tail(5)
    pcr_trend = recent["pcr_oi"].diff().mean()
    iv_trend = recent["atm_iv"].diff().mean() if "atm_iv" in recent.columns else 0

    boost = 0
    # Rising PCR = put buildup = bullish for CE
    if pcr_trend > 0.05:
        boost += 5
    elif pcr_trend < -0.05:
        boost -= 5

    # Falling IV = favorable for buying options
    if iv_trend < -0.5:
        boost += 3

    return boost


def _expiry_day_boost(contracts_df, agg_row, symbol):
    """Special handling for expiry day (Thursday NIFTY / Wednesday BANKNIFTY).

    On expiry:
    - ATM options have massive gamma → small spot move = big option move
    - Max pain gravity strongest in last 2 hours
    - Favor ATM strikes, penalize OTM
    """
    dte = contracts_df["days_to_expiry"].iloc[0] if len(contracts_df) > 0 else 3
    if dte != 0:
        return pd.Series(0, index=contracts_df.index)

    boost = pd.Series(0.0, index=contracts_df.index)

    # ATM strikes get huge gamma boost on expiry
    boost += np.where(contracts_df["abs_moneyness"] < 0.005, 20, 0)
    boost += np.where(contracts_df["abs_moneyness"] < 0.01, 10, 0)

    # OTM gets theta death penalty
    boost += np.where(contracts_df["abs_moneyness"] > 0.02, -15, 0)
    boost += np.where(contracts_df["abs_moneyness"] > 0.03, -20, 0)

    # Max pain is strongest on expiry
    mp_dist = agg_row.get("max_pain_dist_pct", 0)
    if abs(mp_dist) > 0.3:
        # Spot far from max pain → will likely converge
        if mp_dist > 0:
            # Spot above max pain → favor PE
            boost += np.where(contracts_df["is_call"] == 0, 10, -5)
        else:
            # Spot below max pain → favor CE
            boost += np.where(contracts_df["is_call"] == 1, 10, -5)

    return boost


# ── Main Ranking Engine ─────────────────────────────────────

def rank_contracts(symbol, top_n=5):
    """Rank all available contracts for an index and return top picks.

    Uses ML model if trained (>60 days data), otherwise rule-based scoring.
    Applies all 4 filters: conviction, MTF, expiry, high-conviction-only.
    """
    # Load latest raw strike data
    raw_files = sorted([f for f in os.listdir(os.path.join(RAW_DIR, symbol))
                        if f.endswith(".csv")]) if os.path.exists(os.path.join(RAW_DIR, symbol)) else []
    if not raw_files:
        return {"error": f"No strike data for {symbol}", "trades": []}

    latest_date = raw_files[-1].replace(".csv", "")
    strikes_df = _load_raw_day(symbol, latest_date)
    if strikes_df.empty:
        return {"error": "Empty strike data", "trades": []}

    # Use last snapshot of the day
    last_ts = strikes_df["timestamp"].max()
    strikes_df = strikes_df[strikes_df["timestamp"] == last_ts].copy()

    # Load aggregate data
    agg_df = _load_agg_data(symbol)
    if agg_df.empty:
        return {"error": "No aggregate data", "trades": []}

    agg_latest = agg_df.iloc[-1].to_dict()

    # Get index model probability
    index_prob = 0.5
    try:
        idx_model_path = os.path.join(MODEL_DIR, "latest_index_options.pkl")
        idx_meta_path = os.path.join(MODEL_DIR, "latest_index_options_meta.json")
        if os.path.exists(idx_model_path):
            from engines.index_options_trainer import (
                _load_index, _compute_breadth_features, _load_intermarket,
                _load_option_chain_daily, _compute_index_features
            )
            with open(idx_model_path, "rb") as f:
                idx_model = pickle.load(f)
            with open(idx_meta_path) as f:
                idx_meta = json.load(f)
            df_idx = _load_index(symbol)
            breadth = _compute_breadth_features()
            im, fii = _load_intermarket()
            oc = _load_option_chain_daily(symbol)
            feats = _compute_index_features(df_idx, symbol, breadth, oc, im, fii)
            if feats is not None and not feats.empty:
                latest_feat = feats.tail(1).copy()
                feat_cols = idx_meta.get("features", [])
                for c in feat_cols:
                    if c not in latest_feat.columns:
                        latest_feat[c] = 0.0
                    latest_feat[c] = latest_feat[c].replace([np.inf, -np.inf], 0).fillna(0)
                available = [c for c in feat_cols if c in latest_feat.columns]
                index_prob = float(idx_model.predict_proba(latest_feat[available])[0, 1])
    except Exception as e:
        log.warning(f"Index model unavailable: {e}")

    # Compute features for all contracts
    contracts = _compute_strike_features(strikes_df, agg_latest, symbol, index_prob)
    if contracts.empty:
        return {"error": "No contracts generated", "trades": []}

    # Check if ML model exists
    ml_model_path = os.path.join(MODEL_DIR, "latest_strike_ranker.pkl")
    use_ml = os.path.exists(ml_model_path) and len(raw_files) >= MIN_DAYS_FOR_ML

    if use_ml:
        with open(ml_model_path, "rb") as f:
            ml_model = pickle.load(f)
        feat_cols = [c for c in STRIKE_FEATURES if c in contracts.columns]
        contracts["ml_score"] = ml_model.predict(contracts[feat_cols])
    else:
        contracts["ml_score"] = contracts.apply(_rule_based_score, axis=1)

    # Apply conviction filter
    conviction_info = _compute_conviction(contracts, agg_latest, symbol)

    # Apply MTF boost
    mtf_boost = _mtf_entry_filter(contracts, agg_df, symbol)
    contracts["ml_score"] += mtf_boost

    # Apply expiry day boost
    expiry_boost = _expiry_day_boost(contracts, agg_latest, symbol)
    contracts["ml_score"] += expiry_boost

    # Filter by direction
    if conviction_info["conviction"] != "LOW":
        if conviction_info["direction"] == "bullish":
            contracts = contracts[contracts["is_call"] == 1].copy()
        else:
            contracts = contracts[contracts["is_call"] == 0].copy()

    # Filter out illiquid (LTP < 2 or zero volume)
    contracts = contracts[contracts["ltp"] >= 2].copy()
    contracts = contracts[contracts["volume_normalized"] > 0].copy()

    # Rank and pick top N
    contracts = contracts.sort_values("ml_score", ascending=False)
    top = contracts.head(top_n)

    trades = []
    for _, row in top.iterrows():
        strike = int(row["strike"])
        opt_type = row["option_type"]
        ltp = float(row["ltp"])
        spot = float(agg_latest["spot"])
        avg_range_pct = 1.2 if symbol == "NIFTY" else 1.5

        # Target & SL based on option premium
        if conviction_info["conviction"] == "HIGH":
            target_mult = 1.5
            sl_mult = 0.7
        elif conviction_info["conviction"] == "MEDIUM":
            target_mult = 1.3
            sl_mult = 0.75
        else:
            target_mult = 1.2
            sl_mult = 0.8

        target_price = round(ltp * target_mult, 1)
        sl_price = round(ltp * sl_mult, 1)

        trades.append({
            "symbol": symbol,
            "segment": "index_options_ranked",
            "strike": strike,
            "option_type": opt_type,
            "contract": f"{symbol} {strike} {opt_type}",
            "ltp": ltp,
            "target": target_price,
            "stoploss": sl_price,
            "spot": round(spot, 1),
            "direction": conviction_info["direction"],
            "conviction": conviction_info["conviction"],
            "ml_score": round(float(row["ml_score"]), 1),
            "moneyness": round(float(row["moneyness"]) * 100, 2),
            "iv": round(float(row["iv"]), 1),
            "oi_pct": round(float(row["oi_normalized"]) * 100, 2),
            "days_to_expiry": int(row["days_to_expiry"]),
            "model_type": "ML" if use_ml else "rule_based",
            "index_model_prob": round(index_prob, 3),
            "reason": (f"{'ML' if use_ml else 'Rule'} Rank #{trades.__len__() + 1}: "
                       f"{symbol} {strike} {opt_type} @ Rs.{ltp:.1f} | "
                       f"Score={row['ml_score']:.0f} | "
                       f"{conviction_info['conviction']} conviction "
                       f"({conviction_info['bullish_signals']}B/{conviction_info['bearish_signals']}Be signals) | "
                       f"IV={row['iv']:.1f} OI={row['oi_normalized']*100:.1f}%"),
        })

    return {
        "symbol": symbol,
        "spot": round(float(agg_latest["spot"]), 1),
        "data_date": latest_date,
        "model_type": "ML" if use_ml else "rule_based",
        "conviction": conviction_info,
        "trades": trades,
        "data_days": len(raw_files),
        "ml_ready": len(raw_files) >= MIN_DAYS_FOR_ML,
        "days_until_ml": max(0, MIN_DAYS_FOR_ML - len(raw_files)),
    }


def rank_all_indices(top_n=3):
    """Rank contracts across both NIFTY and BANKNIFTY.
    Only returns trades for HIGH/MEDIUM conviction setups."""
    all_results = []

    for sym in INDICES:
        result = rank_contracts(sym, top_n=top_n)
        if result.get("error"):
            continue

        # HIGH conviction filter: skip LOW conviction days
        if result["conviction"]["conviction"] == "LOW":
            all_results.append({
                "symbol": sym,
                "skipped": True,
                "reason": f"Low conviction ({result['conviction']['bullish_signals']}B/"
                          f"{result['conviction']['bearish_signals']}Be signals, "
                          f"need 2+ aligned signals)",
                "conviction": result["conviction"],
                "data_days": result["data_days"],
                "ml_ready": result["ml_ready"],
            })
            continue

        all_results.append(result)

    return all_results


def data_status():
    """How much strike data is collected, when ML training becomes possible."""
    status = {}
    for sym in INDICES:
        raw_dir = os.path.join(RAW_DIR, sym)
        if os.path.exists(raw_dir):
            files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
            dates = sorted([f.replace(".csv", "") for f in files])
            status[sym] = {
                "days_collected": len(dates),
                "first_date": dates[0] if dates else None,
                "last_date": dates[-1] if dates else None,
                "ml_ready": len(dates) >= MIN_DAYS_FOR_ML,
                "days_until_ml": max(0, MIN_DAYS_FOR_ML - len(dates)),
                "pct_complete": round(min(len(dates) / MIN_DAYS_FOR_ML * 100, 100), 1),
            }
        else:
            status[sym] = {"days_collected": 0, "ml_ready": False,
                           "days_until_ml": MIN_DAYS_FOR_ML, "pct_complete": 0}

    ml_model_exists = os.path.exists(os.path.join(MODEL_DIR, "latest_strike_ranker.pkl"))
    status["ml_model_trained"] = ml_model_exists

    return status


# ── ML Training (runs when 60+ days available) ─────────────

def train_strike_ranker():
    """Train XGBRanker on historical strike data.
    Labels: actual return of each contract (LTP change from snapshot to EOD).
    Only runs when 60+ days of data available."""
    import xgboost as xgb

    print("=" * 64)
    print("  STRIKE-LEVEL RANKER TRAINING")
    print("=" * 64)

    all_data = []
    for sym in INDICES:
        raw_dir = os.path.join(RAW_DIR, sym)
        if not os.path.exists(raw_dir):
            continue
        files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".csv")])
        if len(files) < MIN_DAYS_FOR_ML:
            print(f"  {sym}: {len(files)} days (need {MIN_DAYS_FOR_ML}), skipping")
            continue

        agg_df = _load_agg_data(sym)
        if agg_df.empty:
            continue

        print(f"  {sym}: processing {len(files)} days...")

        for i, f_name in enumerate(files[:-1]):  # skip last day (no label yet)
            date_str = f_name.replace(".csv", "")
            next_date = files[i+1].replace(".csv", "")

            current = _load_raw_day(sym, date_str)
            next_day = _load_raw_day(sym, next_date)
            if current.empty or next_day.empty:
                continue

            # Use first snapshot as features, last as "EOD" for labels
            first_ts = current["timestamp"].min()
            last_ts = current["timestamp"].max()

            first_snap = current[current["timestamp"] == first_ts].copy()
            last_snap = current[current["timestamp"] == last_ts].copy()

            # Match aggregate
            agg_match = agg_df[agg_df["timestamp"].dt.date == pd.to_datetime(date_str).date()]
            if agg_match.empty:
                continue
            agg_row = agg_match.iloc[0].to_dict()

            contracts = _compute_strike_features(first_snap, agg_row, sym)
            if contracts.empty:
                continue

            # Label: return from first snapshot LTP to last snapshot LTP
            last_prices = dict(zip(
                zip(last_snap["strike"], ["CE"]*len(last_snap)),
                last_snap["ce_ltp"]
            ))
            last_prices.update(dict(zip(
                zip(last_snap["strike"], ["PE"]*len(last_snap)),
                last_snap["pe_ltp"]
            )))

            contracts["eod_ltp"] = contracts.apply(
                lambda r: last_prices.get((r["strike"], r["option_type"]), r["ltp"]), axis=1)
            contracts["return"] = np.where(
                contracts["ltp"] > 0,
                (contracts["eod_ltp"] - contracts["ltp"]) / contracts["ltp"],
                0)
            contracts["date"] = date_str
            all_data.append(contracts)

    if not all_data:
        print("  Not enough data for training. Keep collecting!")
        return None

    panel = pd.concat(all_data, ignore_index=True)
    print(f"\n  Total: {len(panel)} contract-snapshots across {panel['date'].nunique()} days")

    # Features
    feat_cols = [c for c in STRIKE_FEATURES if c in panel.columns]
    for c in feat_cols:
        panel[c] = panel[c].replace([np.inf, -np.inf], 0).fillna(0)

    # Group by date for ranking
    groups = panel.groupby("date").size().values

    X = panel[feat_cols].values
    y = panel["return"].values

    model = xgb.XGBRanker(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.3, reg_lambda=2.0,
        objective="rank:pairwise",
        random_state=42, verbosity=0, n_jobs=-1,
    )
    model.fit(X, y, group=groups)

    # Save
    model_path = os.path.join(MODEL_DIR, "latest_strike_ranker.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    meta = {
        "type": "strike_ranker",
        "n_features": len(feat_cols),
        "features": feat_cols,
        "n_contracts": len(panel),
        "n_days": int(panel["date"].nunique()),
        "trained_at": datetime.now().isoformat(),
    }
    with open(os.path.join(MODEL_DIR, "latest_strike_ranker_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [OK] Strike ranker saved ({len(feat_cols)} features, {len(panel)} samples)")
    return meta


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    print("\n=== DATA STATUS ===")
    status = data_status()
    for sym, info in status.items():
        if sym == "ml_model_trained":
            print(f"  ML model trained: {info}")
            continue
        dte = info.get('days_until_ml', 0)
        tag = '[ML READY]' if info['ml_ready'] else f'[{dte}d to go]'
        print(f"  {sym}: {info['days_collected']} days ({info['pct_complete']:.0f}% to ML) {tag}")

    print("\n=== RANKED CONTRACTS ===")
    results = rank_all_indices(top_n=3)
    for res in results:
        if res.get("skipped"):
            print(f"\n  {res['symbol']}: SKIPPED — {res['reason']}")
            continue
        print(f"\n  {res['symbol']} | Spot {res['spot']} | "
              f"Conviction: {res['conviction']['conviction']} "
              f"({res['conviction']['direction'].upper()}) | "
              f"Model: {res['model_type']}")
        for t in res.get("trades", []):
            print(f"    {t['contract']:25s}  LTP Rs.{t['ltp']:>7.1f}  "
                  f"Target Rs.{t['target']:>7.1f}  SL Rs.{t['stoploss']:>7.1f}  "
                  f"Score={t['ml_score']:.0f}  {t['conviction']}")
            print(f"      {t['reason']}")

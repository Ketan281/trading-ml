"""
Intraday ML Inference — predict today's best intraday trades.

Two models:
  1. Direction model: which stocks will move UP most today (equity trades)
  2. Range model: which stocks will have largest range today (options trades)

Uses daily features + real 15m session data when available.
"""

import os
import sys
import json
import time
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

log = logging.getLogger("intraday_inference")

MODEL_DIR = os.path.join(ROOT, "models", "intraday")
HIST_DIR = os.path.join(ROOT, "data", "historical")
INTRADAY_DIR = os.path.join(ROOT, "data", "intraday", "15m")

_models = {}
_metas = {}


def _load_model(target="direction"):
    global _models, _metas
    if target in _models:
        return _models[target], _metas[target]

    model_path = os.path.join(MODEL_DIR, f"latest_{target}.pkl")
    meta_path = os.path.join(MODEL_DIR, f"latest_{target}_meta.json")

    if not os.path.exists(model_path):
        log.warning("No intraday %s model at %s", target, model_path)
        return None, None

    with open(model_path, "rb") as f:
        _models[target] = pickle.load(f)
    with open(meta_path, "r") as f:
        _metas[target] = json.load(f)

    log.warning("Intraday %s model loaded: %s (IC: %s, edge: %s)",
                target, _metas[target].get("model_id"),
                _metas[target].get("validation", {}).get("mean_rank_ic"),
                _metas[target].get("edge"))
    return _models[target], _metas[target]


def reload_models():
    global _models, _metas
    _models = {}
    _metas = {}


def _get_latest_features():
    """Build today's features for all stocks from daily CSVs."""
    from engines.intraday_trainer import (
        _list_symbols, _load_daily, _load_15m_session_features,
        compute_intraday_features, ALL_RAW_FEATURES
    )

    symbols = _list_symbols()
    frames = []
    for sym in symbols:
        df = _load_daily(sym)
        if df is None or len(df) < 30:
            continue
        session_df = _load_15m_session_features(sym)
        feats = compute_intraday_features(df, sym, session_df)
        if feats is not None and len(feats) > 0:
            frames.append(feats.tail(1))

    if not frames:
        return pd.DataFrame()

    latest = pd.concat(frames, ignore_index=True)
    return latest


def predict_intraday(target="direction", top_n=20):
    """Score all stocks for intraday trading.

    target='direction': best stocks for directional equity intraday
    target='range': best stocks for options (high expected range)
    """
    model, meta = _load_model(target)
    if model is None:
        return {"error": f"No intraday {target} model. Run: python -m engines.intraday_trainer",
                "picks": []}

    t0 = time.time()
    latest = _get_latest_features()
    if latest.empty:
        return {"error": "No features available", "picks": []}

    from engines.intraday_trainer import ALL_RAW_FEATURES

    # Z-score normalization (cross-sectional)
    for feat in ALL_RAW_FEATURES:
        zfeat = feat + "_z"
        if feat in latest.columns:
            mean = latest[feat].mean()
            std = latest[feat].std() + 1e-9
            latest[zfeat] = (latest[feat] - mean) / std
        else:
            latest[zfeat] = 0.0

    features = meta.get("features", [])
    available = [f for f in features if f in latest.columns]
    if not available:
        return {"error": "No matching features", "picks": []}

    for col in available:
        latest[col] = latest[col].replace([np.inf, -np.inf], 0).fillna(0)

    scores = model.predict_proba(latest[available])[:, 1]
    latest["ml_score"] = scores
    latest["ml_rank"] = latest["ml_score"].rank(ascending=False).astype(int)
    latest["confidence"] = latest["ml_score"].rank(pct=True).mul(100).round(1)

    ranked = latest.sort_values("ml_score", ascending=False).reset_index(drop=True)

    picks = []
    for _, row in ranked.head(top_n).iterrows():
        pick = {
            "rank": int(row["ml_rank"]),
            "symbol": row["symbol"],
            "price": round(float(row.get("price", 0)), 2),
            "open": round(float(row.get("open", 0)), 2),
            "ml_score": round(float(row["ml_score"]), 4),
            "confidence": float(row["confidence"]),
            "target": target,
            "features": {},
        }
        for feat in ["prev_intraday_ret", "prev_intraday_range", "overnight_ret",
                      "avg_intraday_range_5d", "rsi_14", "supertrend_signal",
                      "vwap_dist", "rel_volume", "adx_14", "vix_level",
                      "range_expansion", "bb_width", "dist_ema20",
                      "pat_net_score", "day_of_week"]:
            if feat in row.index:
                val = row[feat]
                pick["features"][feat] = round(float(val), 4) if pd.notna(val) else 0
        picks.append(pick)

    elapsed = time.time() - t0

    return {
        "picks": picks,
        "model_id": meta.get("model_id"),
        "edge": meta.get("edge"),
        "target": target,
        "total_scored": len(latest),
        "inference_time_ms": round(elapsed * 1000, 1),
        "scored_at": datetime.now().isoformat(),
        "validation": meta.get("validation", {}),
    }


def get_intraday_equity_trades(capital=100000, max_picks=5):
    """Actionable intraday equity trades with entry/stop/target."""
    result = predict_intraday(target="direction", top_n=30)
    if result.get("error"):
        return []

    trades = []
    per_trade = capital / max_picks

    for pick in result["picks"]:
        if len(trades) >= max_picks:
            break
        if pick["confidence"] < 40:
            continue

        sym = pick["symbol"]
        price = pick["price"]
        if price <= 0:
            continue

        feats = pick["features"]
        avg_range = feats.get("avg_intraday_range_5d", 0.015)
        if avg_range < 0.005:
            avg_range = 0.015

        expected_move = price * avg_range * 0.6
        rsi = feats.get("rsi_14", 50)
        supertrend = feats.get("supertrend_signal", 0)
        overnight = feats.get("overnight_ret", 0)
        pat = feats.get("pat_net_score", 0)

        bullish = sum([
            1 if overnight >= 0 else 0,
            1 if rsi < 65 else 0,
            1 if supertrend > 0 else 0,
            1 if pat > 0 else 0,
        ])
        action = "buy" if bullish >= 2 else "sell"

        if action == "buy":
            stop = round(price - expected_move * 0.7, 2)
            target = round(price + expected_move, 2)
        else:
            stop = round(price + expected_move * 0.7, 2)
            target = round(price - expected_move, 2)

        risk = abs(price - stop)
        qty = max(1, int(per_trade * 0.02 / risk)) if risk > 0 else 1
        rr = round(abs(target - price) / risk, 1) if risk > 0 else 0

        trades.append({
            "symbol": sym,
            "segment": "equity_intraday",
            "action": action,
            "entry": price,
            "stop": stop,
            "target": target,
            "qty": qty,
            "confidence": pick["confidence"],
            "ml_score": pick["ml_score"],
            "ml_rank": pick["rank"],
            "reward_risk": rr,
            "expected_range_pct": round(avg_range * 100, 2),
            "reason": (f"Intraday ML #{pick['rank']} (score {pick['ml_score']:.3f}, "
                       f"conf {pick['confidence']:.0f}%) — "
                       f"RSI {rsi:.0f}, range {avg_range*100:.1f}%, "
                       f"gap {'UP' if overnight > 0 else 'DN'} {abs(overnight)*100:.2f}%"),
        })

    return trades


def _predict_index_options():
    """Predict NIFTY/BANKNIFTY direction using the index options model."""
    model_path = os.path.join(MODEL_DIR, "latest_index_options.pkl")
    meta_path = os.path.join(MODEL_DIR, "latest_index_options_meta.json")

    if not os.path.exists(model_path):
        return []

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    from engines.index_options_trainer import (
        _load_index, _compute_breadth_features, _load_intermarket,
        _load_option_chain_daily, _compute_index_features, FEATURE_COLS, INDICES
    )

    breadth_df = _compute_breadth_features()
    im_df, fii_series = _load_intermarket()

    predictions = []
    features_list = meta.get("features", [])

    for sym in INDICES:
        df = _load_index(sym)
        if len(df) < 30:
            continue
        oc_df = _load_option_chain_daily(sym)
        feats = _compute_index_features(df, sym, breadth_df, oc_df, im_df, fii_series)
        if feats is None or feats.empty:
            continue

        latest = feats.tail(1).copy()
        for col in features_list:
            if col in latest.columns:
                latest[col] = latest[col].replace([np.inf, -np.inf], 0).fillna(0)
            else:
                latest[col] = 0.0

        available = [f for f in features_list if f in latest.columns]
        prob = model.predict_proba(latest[available])[0, 1]

        row = latest.iloc[0]
        spot = float(row.get("price", 0))
        open_price = float(row.get("open_price", spot))
        avg_range = float(row.get("avg_range_5d", 0.012))
        if avg_range < 0.005:
            avg_range = 0.012

        direction = "bullish" if prob > 0.5 else "bearish"
        confidence = round(abs(prob - 0.5) * 200, 1)

        expected_move = spot * avg_range * 0.6
        # Strike selection: round to nearest 50 for NIFTY, 100 for BANKNIFTY
        strike_step = 50 if sym == "NIFTY" else 100
        atm_strike = round(spot / strike_step) * strike_step

        if direction == "bullish":
            strike = atm_strike
            option_type = "CE"
            target_strike = atm_strike + strike_step * 2
            sl_points = expected_move * 0.5
        else:
            strike = atm_strike
            option_type = "PE"
            target_strike = atm_strike - strike_step * 2
            sl_points = expected_move * 0.5

        # Strategy based on confidence
        if confidence > 70:
            strategy = f"buy_{sym}_{atm_strike}_{option_type}"
            strategy_name = f"Buy {sym} {atm_strike} {option_type}"
        elif confidence > 50:
            if direction == "bullish":
                strategy = f"bull_call_spread_{sym}"
                strategy_name = f"Bull Call Spread {sym} {atm_strike}/{atm_strike + strike_step}"
            else:
                strategy = f"bear_put_spread_{sym}"
                strategy_name = f"Bear Put Spread {sym} {atm_strike}/{atm_strike - strike_step}"
        else:
            strategy = f"iron_condor_{sym}"
            strategy_name = f"Iron Condor {sym} (low conviction)"

        predictions.append({
            "symbol": sym,
            "segment": "index_options",
            "spot": round(spot, 1),
            "direction": direction,
            "probability": round(prob, 4),
            "confidence": confidence,
            "option_type": option_type,
            "strike": atm_strike,
            "strategy": strategy_name,
            "expected_move": round(expected_move, 1),
            "expected_range_pct": round(avg_range * 100, 2),
            "target": round(spot + expected_move if direction == "bullish" else spot - expected_move, 1),
            "stoploss": round(spot - sl_points if direction == "bullish" else spot + sl_points, 1),
            "model_id": meta.get("model_id"),
            "edge": meta.get("edge"),
            "reason": (f"Index ML: {sym} {direction.upper()} (prob {prob:.1%}, conf {confidence:.0f}%) — "
                       f"range {avg_range*100:.1f}%, "
                       f"Buy {atm_strike} {option_type}"),
        })

    return predictions


def get_intraday_options_trades(capital=100000, max_picks=5):
    """Combined options trades: strike-ranked index options first, then stocks.

    Priority:
      1. Strike-ranked NIFTY/BANKNIFTY contracts (conviction-filtered)
      2. Index direction model fallback (if no strike data)
      3. Stock options (from stock direction model + range filter)
    """
    trades = []

    # === PRIMARY: V2 OI-flow based index options (75-79% win rate) ===
    try:
        from engines.index_options_v2 import predict_index_options_v2
        v2_trades = predict_index_options_v2()
        for t in v2_trades:
            if t.get("tier") != "NO_TRADE":
                trades.append(t)
    except Exception as e:
        log.warning(f"Index options V2 unavailable: {e}")

    # === FALLBACK: Strike ranker if V2 returned nothing ===
    if not trades:
        try:
            from engines.strike_ranker import rank_all_indices
            ranked = rank_all_indices(top_n=3)
            for res in ranked:
                if res.get("skipped"):
                    continue
                for t in res.get("trades", []):
                    trades.append(t)
        except Exception as e:
            log.warning(f"Strike ranker unavailable: {e}")

    # === FALLBACK: Index direction model ===
    if not trades:
        index_trades = _predict_index_options()
        for t in index_trades:
            trades.append(t)

    # === SECONDARY: Stock options (fill remaining slots) ===
    remaining = max_picks - len(trades)
    if remaining <= 0:
        return trades

    dir_result = predict_intraday(target="direction", top_n=30)
    if dir_result.get("error"):
        return trades

    candidates = [p for p in dir_result["picks"]
                  if p["features"].get("avg_intraday_range_5d", 0) > 0.01]
    if not candidates:
        candidates = dir_result["picks"]

    for pick in candidates:
        if len(trades) >= max_picks:
            break
        if pick["confidence"] < 50:
            continue

        sym = pick["symbol"]
        price = pick["price"]
        if price <= 0:
            continue

        feats = pick["features"]
        avg_range = feats.get("avg_intraday_range_5d", 0.015)
        if avg_range < 0.005:
            avg_range = 0.015

        rsi = feats.get("rsi_14", 50)
        supertrend = feats.get("supertrend_signal", 0)
        overnight = feats.get("overnight_ret", 0)

        bullish = sum([1 if overnight >= 0 else 0,
                       1 if rsi < 65 else 0,
                       1 if supertrend > 0 else 0])
        direction = "bullish" if bullish >= 2 else "bearish"

        if pick["confidence"] > 80 and avg_range > 0.02:
            strategy = "buy_call" if direction == "bullish" else "buy_put"
        elif pick["confidence"] > 60:
            strategy = "bull_call_spread" if direction == "bullish" else "bear_put_spread"
        else:
            strategy = "long_straddle" if avg_range > 0.025 else "long_strangle"

        expected_move = price * avg_range * 0.6
        if direction == "bullish":
            eq_target = round(price + expected_move, 2)
            eq_stop = round(price - expected_move * 0.5, 2)
        else:
            eq_target = round(price - expected_move, 2)
            eq_stop = round(price + expected_move * 0.5, 2)

        trades.append({
            "symbol": sym,
            "segment": "stock_options",
            "direction": direction,
            "option_strategy": strategy,
            "equity_price": price,
            "equity_target": eq_target,
            "equity_stop": eq_stop,
            "expected_range_pct": round(avg_range * 100, 2),
            "ml_score": pick["ml_score"],
            "ml_rank": pick["rank"],
            "confidence": pick["confidence"],
            "reason": (f"Stock Dir ML #{pick['rank']} (conf {pick['confidence']:.0f}%) — "
                       f"{'BULL' if direction == 'bullish' else 'BEAR'}, "
                       f"range {avg_range*100:.1f}%, "
                       f"strategy: {strategy}"),
        })

    return trades


def model_status():
    """Status of all intraday models."""
    status = {}
    for target in ["direction", "range"]:
        _, meta = _load_model(target)
        if meta:
            status[target] = {
                "loaded": True,
                "model_id": meta.get("model_id"),
                "edge": meta.get("edge"),
                "validation": meta.get("validation"),
                "trained_at": meta.get("trained_at"),
            }
        else:
            status[target] = {"loaded": False}

    # Index options model
    idx_meta_path = os.path.join(MODEL_DIR, "latest_index_options_meta.json")
    if os.path.exists(idx_meta_path):
        with open(idx_meta_path) as f:
            idx_meta = json.load(f)
        status["index_options"] = {
            "loaded": True,
            "model_id": idx_meta.get("model_id"),
            "edge": idx_meta.get("edge"),
            "validation": idx_meta.get("validation"),
            "trained_at": idx_meta.get("trained_at"),
        }
    else:
        status["index_options"] = {"loaded": False}

    # Strike ranker + data collection status
    try:
        from engines.strike_ranker import data_status
        status["strike_ranker"] = data_status()
    except Exception:
        status["strike_ranker"] = {"error": "unavailable"}

    return status


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    print("\n=== INTRADAY EQUITY TRADES ===")
    eq = get_intraday_equity_trades(capital=100000, max_picks=5)
    for t in eq:
        print(f"  {t['action'].upper():4s} {t['symbol']:12s} @ Rs.{t['entry']:8.1f}  "
              f"Stop={t['stop']:8.1f}  Target={t['target']:8.1f}  "
              f"R:R={t['reward_risk']:.1f}  Conf={t['confidence']:.0f}%")
        print(f"       {t['reason']}")

    print("\n=== INTRADAY OPTIONS TRADES (INDEX + STOCK) ===")
    opt = get_intraday_options_trades(capital=100000, max_picks=5)
    for t in opt:
        if t["segment"] == "index_options":
            print(f"  [{t['segment'].upper()}] {t['direction'].upper():7s} {t['symbol']:10s}  "
                  f"Spot={t['spot']:>8.1f}  Strike={t['strike']}  {t['option_type']}  "
                  f"Conf={t['confidence']:.0f}%")
            print(f"       {t['strategy']}")
            print(f"       {t['reason']}")
        else:
            print(f"  [{t['segment'].upper()}] {t['direction'].upper():7s} {t['symbol']:10s}  "
                  f"@ Rs.{t['equity_price']:8.1f}  "
                  f"Strategy={t.get('option_strategy',''):18s}  "
                  f"Range={t['expected_range_pct']:.1f}%  Conf={t['confidence']:.0f}%")
            print(f"       {t['reason']}")

"""
ML Inference Engine — fast prediction from pre-trained model.

Loads the trained model ONCE at startup (~5MB), then scores all stocks
in <1 second using features from the feature store.

This is what runs on the 1GB server. Zero training, just prediction.
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

log = logging.getLogger("ml_inference")

MODEL_DIR = os.path.join(ROOT, "models", "ml_v2")

_model = None
_meta = None
_loaded_at = None


def _load_model():
    """Load the latest trained model. Called once at startup."""
    global _model, _meta, _loaded_at

    latest_pkl = os.path.join(MODEL_DIR, "latest.pkl")
    latest_meta = os.path.join(MODEL_DIR, "latest_meta.json")

    if not os.path.exists(latest_pkl):
        log.warning("No trained model found at %s", latest_pkl)
        return False

    with open(latest_pkl, "rb") as f:
        _model = pickle.load(f)
    with open(latest_meta, "r") as f:
        _meta = json.load(f)

    _loaded_at = datetime.now()
    log.warning("ML model loaded: %s (Rank IC: %s, edge: %s)",
                _meta.get("model_id", "unknown"),
                _meta.get("validation", {}).get("mean_rank_ic", "?"),
                _meta.get("edge", "?"))
    return True


def get_model():
    if _model is None:
        _load_model()
    return _model


def get_meta():
    if _meta is None:
        _load_model()
    return _meta


def predict_all(top_n=20):
    """Score ALL stocks using the latest features and trained model.
    Returns ranked list of picks with confidence scores.

    This is the main inference call — runs in <1 second."""
    model = get_model()
    meta = get_meta()
    if model is None or meta is None:
        return {"error": "No trained model available", "picks": []}

    t0 = time.time()

    from engines.feature_store import load_latest_features_fast
    latest = load_latest_features_fast()
    if latest.empty:
        return {"error": "Feature store empty", "picks": []}

    raw_features = meta.get("raw_features", [])
    z_features = meta.get("features", [])
    zscore_stats = meta.get("zscore_stats", {})

    # Cross-sectional z-score normalization (same as training)
    for feat in raw_features:
        zfeat = feat + "_z"
        if feat in latest.columns:
            mean = latest[feat].mean()
            std = latest[feat].std() + 1e-9
            latest[zfeat] = (latest[feat] - mean) / std
        else:
            latest[zfeat] = 0.0

    # Replace inf/nan
    for col in z_features:
        if col in latest.columns:
            latest[col] = latest[col].replace([np.inf, -np.inf], 0).fillna(0)

    available = [f for f in z_features if f in latest.columns]
    if not available:
        return {"error": "No matching features", "picks": []}

    # Predict
    scores = model.predict_proba(latest[available])[:, 1]
    latest["ml_score"] = scores
    latest["ml_rank"] = latest["ml_score"].rank(ascending=False).astype(int)

    # Confidence: rank-based percentile within today's cross-section
    # Top stock = 100, median = 50, bottom = 0
    latest["confidence"] = latest["ml_score"].rank(pct=True).mul(100).round(1)

    # Sort by score
    ranked = latest.sort_values("ml_score", ascending=False).reset_index(drop=True)

    # Build picks
    picks = []
    for _, row in ranked.head(top_n).iterrows():
        pick = {
            "rank": int(row["ml_rank"]),
            "symbol": row["symbol"],
            "price": round(float(row.get("price", 0)), 2),
            "ml_score": round(float(row["ml_score"]), 4),
            "confidence": float(row["confidence"]),
            "features": {},
        }
        # Include key feature values for explainability
        for feat in ["ret_5d", "ret_21d", "ret_63d", "rsi_14", "mfi_14",
                      "rel_volume", "dist_ema20", "sharpe_60d", "sr_proximity",
                      "adx_14", "supertrend_signal", "vwap_dist", "roe",
                      "earnings_growth", "pattern_score", "bb_pctb"]:
            if feat in row.index:
                val = row[feat]
                pick["features"][feat] = round(float(val), 4) if pd.notna(val) else 0
        picks.append(pick)

    elapsed = time.time() - t0

    return {
        "picks": picks,
        "model_id": meta.get("model_id", "unknown"),
        "model_edge": meta.get("edge", "unknown"),
        "horizon": meta.get("horizon", "21d"),
        "total_scored": len(latest),
        "inference_time_ms": round(elapsed * 1000, 1),
        "scored_at": datetime.now().isoformat(),
        "validation": meta.get("validation", {}),
    }


def predict_stock(symbol):
    """Score a single stock. Returns detailed prediction."""
    result = predict_all(top_n=500)
    if result.get("error"):
        return result
    for pick in result["picks"]:
        if pick["symbol"] == symbol:
            return {"prediction": pick, "model": result["model_id"]}
    return {"prediction": None, "note": f"{symbol} not in universe or no features"}


def get_top_picks_for_trading(capital=100000, max_picks=5):
    """Get actionable trading picks with entry/stop/target levels.

    Combines ML ranking with price structure to generate trade specs
    that the auto-trading system can execute."""
    result = predict_all(top_n=30)
    if result.get("error"):
        return []

    from engines.feature_store import load_latest_features_fast
    latest = load_latest_features_fast()
    price_map = dict(zip(latest["symbol"], latest["price"]))
    feat_map = {}
    for _, row in latest.iterrows():
        feat_map[row["symbol"]] = row

    trades = []
    per_trade = capital / max_picks

    for pick in result["picks"]:
        if len(trades) >= max_picks:
            break
        if pick["confidence"] < 30:
            continue

        sym = pick["symbol"]
        price = pick["price"]
        if price <= 0:
            continue

        row = feat_map.get(sym)
        if row is None:
            continue

        atr_ratio = float(row.get("atr_ratio", 0.02)) if row is not None else 0.02
        atr = price * atr_ratio

        # Entry at current price, stop at 2 ATR, target at 3 ATR (1.5:1 R:R)
        stop = round(price - 2 * atr, 2)
        target = round(price + 3 * atr, 2)
        risk_per_share = price - stop
        qty = max(1, int(per_trade * 0.02 / risk_per_share)) if risk_per_share > 0 else 1

        # Determine action from momentum + supertrend + VWAP
        ret_5d = pick["features"].get("ret_5d", 0)
        rsi = pick["features"].get("rsi_14", 50)
        supertrend = pick["features"].get("supertrend_signal", 1)
        vwap = pick["features"].get("vwap_dist", 0)
        bullish_signals = sum([
            1 if ret_5d >= 0 else 0,
            1 if rsi < 65 else 0,
            1 if supertrend > 0 else 0,
            1 if vwap > 0 else 0,
        ])
        action = "buy" if bullish_signals >= 2 else "sell"
        side = "long" if action == "buy" else "short"
        if side == "short":
            stop = round(price + 2 * atr, 2)
            target = round(price - 3 * atr, 2)

        reward_risk = round(abs(target - price) / abs(price - stop), 1) if abs(price - stop) > 0 else 0

        trades.append({
            "symbol": sym,
            "segment": "equity_intraday",
            "action": action,
            "side": side,
            "entry": price,
            "stop": stop,
            "target": target,
            "qty": qty,
            "confidence": pick["confidence"],
            "ml_score": pick["ml_score"],
            "ml_rank": pick["rank"],
            "reward_risk": reward_risk,
            "reason": (f"ML rank #{pick['rank']} (score {pick['ml_score']:.3f}, "
                       f"confidence {pick['confidence']:.0f}%) -- "
                       f"RSI {rsi:.0f}, 5d ret {ret_5d*100:.1f}%, "
                       f"ST {'UP' if supertrend > 0 else 'DN'}, "
                       f"VWAP {'above' if vwap > 0 else 'below'}"),
        })

    return trades


def get_options_trades(capital=100000, max_picks=3):
    """ML-ranked stocks -> options trade recommendations.
    Uses ML to pick the stock, then options pipeline for strike/strategy."""
    equity_trades = get_top_picks_for_trading(capital=capital, max_picks=max_picks * 2)
    if not equity_trades:
        return []

    options_trades = []
    for trade in equity_trades:
        if len(options_trades) >= max_picks:
            break
        sym = trade["symbol"]
        try:
            from pipelines.options.strategy_selector import select_strategy
            strategy = select_strategy(sym, trade["side"], trade["confidence"])
            if strategy and strategy.get("strategy"):
                options_trades.append({
                    "symbol": sym,
                    "segment": "options_intraday",
                    "equity_action": trade["action"],
                    "equity_entry": trade["entry"],
                    "equity_stop": trade["stop"],
                    "equity_target": trade["target"],
                    "ml_score": trade["ml_score"],
                    "ml_rank": trade["ml_rank"],
                    "confidence": trade["confidence"],
                    "option_strategy": strategy.get("strategy", "buy_call" if trade["side"] == "long" else "buy_put"),
                    "option_details": strategy,
                    "reason": trade["reason"],
                })
        except Exception:
            # Fallback: simple call/put based on ML direction
            options_trades.append({
                "symbol": sym,
                "segment": "options_intraday",
                "equity_action": trade["action"],
                "equity_entry": trade["entry"],
                "equity_stop": trade["stop"],
                "equity_target": trade["target"],
                "ml_score": trade["ml_score"],
                "ml_rank": trade["ml_rank"],
                "confidence": trade["confidence"],
                "option_strategy": "buy_call" if trade["side"] == "long" else "buy_put",
                "option_details": None,
                "reason": trade["reason"],
            })

    return options_trades


def model_status():
    """Return current model status for the API."""
    meta = get_meta()
    if meta is None:
        return {
            "loaded": False,
            "error": "No trained model found. Run: python -m engines.ml_trainer",
        }
    return {
        "loaded": True,
        "model_id": meta.get("model_id"),
        "edge": meta.get("edge"),
        "horizon": meta.get("horizon"),
        "trained_at": meta.get("trained_at"),
        "validation": meta.get("validation"),
        "backtest": meta.get("backtest"),
        "loaded_at": _loaded_at.isoformat() if _loaded_at else None,
    }


def reload_model():
    """Force reload the model (after retraining)."""
    global _model, _meta, _loaded_at
    _model = None
    _meta = None
    _loaded_at = None
    return _load_model()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    result = predict_all(top_n=15)
    if result.get("error"):
        print(f"Error: {result['error']}")
    else:
        print(f"\nML Predictions (model: {result['model_id']}, edge: {result['model_edge']})")
        print(f"Scored {result['total_scored']} stocks in {result['inference_time_ms']:.0f}ms\n")
        for p in result["picks"]:
            print(f"  #{p['rank']:>3}  {p['symbol']:<15} "
                  f"score={p['ml_score']:.3f}  conf={p['confidence']:.0f}%  "
                  f"price=Rs{p['price']:.0f}")

"""
Multi-timeframe confluence scoring — the experienced-trader judgment layer.

Runs the 19-indicator technical engine on each timeframe (15m, 1H, 4H, Daily),
then aggregates. A trade is only recommended when enough indicators across
enough timeframes agree — like an experienced trader who waits for "everything
to line up" before pulling the trigger.

Scoring:
  Each indicator on each TF votes +1 (bullish), -1 (bearish), or 0 (neutral).
  Higher TFs carry more weight (Daily > 4H > 1H > 15m) because the trend on a
  higher timeframe is more reliable than noise on a lower one.
  Final score: weighted sum normalised to [-1, +1].
  |score| > threshold = trade; sign = direction.
"""

from pipelines.forex import data as fxdata
from pipelines.forex import technical as tech

TF_WEIGHTS = {"15m": 1.0, "1h": 1.5, "4h": 2.0, "1d": 3.0}
TRADE_THRESHOLD = 0.30
HIGH_CONFIDENCE = 0.50


def _signal_to_score(sig: str) -> int:
    return {"bullish": 1, "bearish": -1}.get(sig, 0)


def score_pair(pair: str) -> dict:
    """Analyse one pair across all timeframes. Returns a full signal report."""
    mtf = fxdata.fetch_multi_timeframe(pair)
    if not mtf:
        return {"pair": pair, "error": "no data", "score": 0, "direction": "none",
                "confidence": "none", "tf_signals": {}, "indicators": {}}

    tf_signals = {}
    all_indicator_results = {}
    weighted_sum = 0.0
    weight_total = 0.0

    for tf, df in mtf.items():
        w = TF_WEIGHTS.get(tf, 1.0)
        indicators = tech.analyse(df)
        if not indicators:
            continue
        scores = []
        for name, result in indicators.items():
            s = _signal_to_score(result.get("signal", "neutral"))
            scores.append(s)
        tf_score = sum(scores) / len(scores) if scores else 0
        bullish = sum(1 for s in scores if s > 0)
        bearish = sum(1 for s in scores if s < 0)
        neutral = sum(1 for s in scores if s == 0)
        tf_signals[tf] = {
            "score": round(tf_score, 3),
            "bullish": bullish, "bearish": bearish, "neutral": neutral,
            "total": len(scores), "weight": w,
            "bias": "bullish" if tf_score > 0.1 else ("bearish" if tf_score < -0.1 else "neutral"),
        }
        all_indicator_results[tf] = indicators
        weighted_sum += tf_score * w
        weight_total += w

    final_score = weighted_sum / weight_total if weight_total else 0

    if final_score > TRADE_THRESHOLD:
        direction = "buy"
    elif final_score < -TRADE_THRESHOLD:
        direction = "sell"
    else:
        direction = "none"

    if abs(final_score) >= HIGH_CONFIDENCE:
        confidence = "high"
    elif abs(final_score) >= TRADE_THRESHOLD:
        confidence = "medium"
    else:
        confidence = "low"

    # Count how many TFs agree with the direction
    agreeing_tfs = sum(1 for tf_data in tf_signals.values()
                       if tf_data["bias"] == ("bullish" if direction == "buy" else "bearish"))

    atr_val = None
    entry = fxdata.current_price(pair)
    pip_info = fxdata.PIP_INFO.get(pair, {"pip": 0.0001})
    pip_size = pip_info["pip"]

    if "1h" in all_indicator_results and "ATR" in all_indicator_results["1h"]:
        atr_val = all_indicator_results["1h"]["ATR"]["value"]

    sl_pips = 30
    tp_pips = 60
    if atr_val:
        sl_pips = max(15, round(atr_val / pip_size * 1.5))
        tp_pips = max(30, round(atr_val / pip_size * 3.0))

    trade_plan = None
    if direction != "none" and entry:
        sign = 1 if direction == "buy" else -1
        sl = round(entry - sign * sl_pips * pip_size, 5)
        tp = round(entry + sign * tp_pips * pip_size, 5)
        trade_plan = {
            "pair": pair, "direction": direction, "entry": entry,
            "stop_loss": sl, "take_profit": tp,
            "sl_pips": sl_pips, "tp_pips": tp_pips,
            "risk_reward": round(tp_pips / sl_pips, 2),
        }

    return {
        "pair": pair,
        "score": round(final_score, 3),
        "direction": direction,
        "confidence": confidence,
        "agreeing_timeframes": agreeing_tfs,
        "total_timeframes": len(tf_signals),
        "tf_signals": tf_signals,
        "trade_plan": trade_plan,
        "indicators": {tf: {name: {"signal": ind["signal"],
                                    "value": ind.get("value")}
                            for name, ind in inds.items()}
                       for tf, inds in all_indicator_results.items()},
    }


def scan_all_pairs() -> list:
    """Score all supported pairs and rank by absolute confluence strength."""
    results = []
    for pair in fxdata.list_pairs():
        try:
            r = score_pair(pair)
            results.append(r)
        except Exception:
            pass
    results.sort(key=lambda x: abs(x.get("score", 0)), reverse=True)
    return results


def best_trade() -> dict | None:
    """Return the single best tradeable opportunity across all pairs,
    or None if nothing meets the threshold."""
    for r in scan_all_pairs():
        if r.get("direction") != "none" and r.get("trade_plan"):
            return r
    return None

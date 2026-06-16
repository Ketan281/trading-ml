"""
Deterministic trading decision engine.

This module makes the actual trade decision from indicator data using
fixed, rule-based logic — NOT a language model. Every number that matters
(action, confidence, entry, stop-loss, target) is computed here so the
output is repeatable and cannot be hallucinated.

The LLM layer (intelligence.py) only writes the natural-language reasoning
on top of the decision this engine produces.
"""

# ── Signal Voting ─────────────────────────────────────
# Each signal returns a vote in [-1, +1] and a weight.
# Positive = bullish, negative = bearish.

def _vote_trend(ind):
    trend = ind.get("trend", "sideways")
    if trend == "uptrend":
        return 1.0
    if trend == "downtrend":
        return -1.0
    return 0.0

def _vote_ema_stack(ind):
    price = ind.get("price")
    ema20 = ind.get("ema_20")
    ema50 = ind.get("ema_50")
    if price is None or ema20 is None or ema50 is None:
        return 0.0
    if price > ema20 > ema50:
        return 1.0
    if price < ema20 < ema50:
        return -1.0
    # Price above both but EMAs not stacked, or mixed
    if price > ema20 and price > ema50:
        return 0.5
    if price < ema20 and price < ema50:
        return -0.5
    return 0.0

def _vote_macd(ind):
    macd = ind.get("macd", "neutral")
    if macd == "bullish":
        return 1.0
    if macd == "bearish":
        return -1.0
    return 0.0

def _vote_rsi(ind):
    rsi = float(ind.get("rsi", 50))
    # Overbought / oversold reduce conviction in the trend direction.
    if rsi >= 70:
        return -0.5          # too hot to chase longs
    if rsi <= 30:
        return 0.5           # washed out, lean long
    if 50 < rsi < 70:
        return 0.5           # healthy bullish momentum
    if 30 < rsi < 50:
        return -0.5          # healthy bearish momentum
    return 0.0

def _vote_vwap(ind):
    vwap = ind.get("vwap", "")
    if vwap == "above_vwap":
        return 0.5
    if vwap == "below_vwap":
        return -0.5
    return 0.0

def _vote_bollinger(ind):
    b = ind.get("bollinger", "inside_bands")
    if b == "above_upper":
        return -0.5          # stretched, mean-reversion risk
    if b == "below_lower":
        return 0.5
    return 0.0

# (vote_fn, weight)
SIGNALS = [
    (_vote_trend,     2.0),
    (_vote_ema_stack, 2.0),
    (_vote_macd,      1.5),
    (_vote_rsi,       1.0),
    (_vote_vwap,      1.0),
    (_vote_bollinger, 1.0),
]

# Weight given to the ML model's learned prediction when available.
# It is the single heaviest signal because it is trained on history,
# but its vote is scaled by the model's own confidence so an unsure
# model pulls the score toward neutral rather than dominating it.
ML_WEIGHT = 3.0


def _vote_ml(ml_signal):
    """Convert an ML prediction into a (vote, weight) pair.
    vote in [-1, +1] scaled by model confidence."""
    regime = ml_signal.get("ml_regime", "sideways")
    conf   = float(ml_signal.get("confidence", 0.5))
    base   = {"uptrend": 1.0, "downtrend": -1.0}.get(regime, 0.0)
    return base * conf, ML_WEIGHT

# Weight given to fresh candlestick / price-action patterns. Moderate —
# patterns sharpen TIMING and confirm direction, but shouldn't outvote the
# trend + EMA structure on their own.
PATTERN_WEIGHT = 1.5


def _vote_pattern(patterns):
    """Convert detected price-action into a (vote, weight) pair.
    Uses the recency-weighted net pattern score in [-1, +1]."""
    return float(patterns.get("pattern_score", 0.0)), PATTERN_WEIGHT

# Risk:reward and stop-loss settings
ATR_STOP_MULT   = 1.5    # stop = entry -/+ 1.5 * ATR
RISK_REWARD     = 2.0    # target distance = 2x stop distance
ENTRY_PULLBACK  = 0.3    # allow entry within 0.3*ATR of current price


# ── Helpers ───────────────────────────────────────────
def _score(ind, ml_signal=None, patterns=None):
    """Weighted net signal score in [-1, +1].
    ML (when supplied) is the heaviest vote; price-action patterns add a
    moderate timing/confirmation vote."""
    weighted_sum = 0.0
    total_weight = 0.0
    for vote_fn, weight in SIGNALS:
        weighted_sum += vote_fn(ind) * weight
        total_weight += weight
    if ml_signal:
        vote, weight = _vote_ml(ml_signal)
        weighted_sum += vote * weight
        total_weight += weight
    if patterns:
        vote, weight = _vote_pattern(patterns)
        weighted_sum += vote * weight
        total_weight += weight
    return weighted_sum / total_weight


def _risk_level(ind):
    vol = ind.get("volatility", "medium")
    return {
        "low":    "low",
        "medium": "medium",
        "high":   "high",
    }.get(vol, "medium")


def _position_size(confidence, risk_level):
    if risk_level == "high":
        if confidence >= 0.75:
            return "half"
        return "quarter"
    if confidence >= 0.75:
        return "full"
    if confidence >= 0.55:
        return "half"
    return "quarter"


def _market_condition(ind, score):
    rsi = float(ind.get("rsi", 50))
    if score >= 0.5:
        return "overextended_uptrend" if rsi >= 75 else "strong_uptrend"
    if score >= 0.2:
        return "weak_uptrend"
    if score <= -0.5:
        return "overextended_downtrend" if rsi <= 25 else "strong_downtrend"
    if score <= -0.2:
        return "weak_downtrend"
    return "sideways"


def _round(x):
    return round(float(x), 2)


# ── Main Decision ─────────────────────────────────────
def decide(ind, ml_signal=None, patterns=None):
    """
    Build a complete, deterministic trade decision from indicator data.

    `ind` is the dict produced by indicators.analyze() — it must contain
    at least: symbol, price, atr. Other keys default sensibly if missing.

    `ml_signal` is an optional dict {"ml_regime", "confidence"} from a
    trained model (models.ml_models.predict). When present it is folded
    into the signal vote as the heaviest input, scaled by its confidence.

    `patterns` is an optional dict from pipelines.patterns.detect_patterns().
    When present it adds a moderate vote AND drives the entry-timing signal:
    a "buy" only becomes a fired trigger when a fresh bullish price-action
    pattern confirms it at a sensible location.

    Returns a decision dict shaped to match what the downstream
    validation / memory / portfolio / print layers already expect.
    """
    symbol = ind.get("symbol", "?")
    price  = float(ind.get("price", 0) or 0)
    atr    = float(ind.get("atr", 0) or 0)

    score      = _score(ind, ml_signal, patterns)
    confidence = round(abs(score), 3)          # 0..1, strength of agreement
    risk_level = _risk_level(ind)
    condition  = _market_condition(ind, score)

    # Decide direction from score thresholds.
    if score >= 0.2:
        action = "buy"
    elif score <= -0.2:
        action = "sell"
    else:
        action = "hold"

    # ── Entry timing from price action ────────────────
    # The directional call (buy/sell) says WHAT; the pattern trigger says
    # WHETHER the entry is clean right now.
    pat_trigger = (patterns or {}).get("entry_trigger")
    pat_primary = (patterns or {}).get("primary")
    pat_context = (patterns or {}).get("context", "mid")
    if patterns:
        if action == "buy":
            entry_signal = "trigger" if pat_trigger == "long" else "wait"
        elif action == "sell":
            entry_signal = "trigger" if pat_trigger == "short" else "wait"
        else:
            entry_signal = "wait"
    else:
        entry_signal = "trigger" if action in ("buy", "sell") else "wait"

    # Compute price levels deterministically from ATR.
    entry_zone = "wait"
    stop_loss  = "N/A"
    target     = "N/A"

    if action in ("buy", "sell") and price > 0 and atr > 0:
        stop_dist   = ATR_STOP_MULT * atr
        target_dist = RISK_REWARD * stop_dist
        pullback    = ENTRY_PULLBACK * atr

        if action == "buy":
            entry_lo = _round(price - pullback)
            entry_hi = _round(price)
            entry_zone = f"{entry_lo}-{entry_hi}"
            stop_loss  = str(_round(price - stop_dist))
            target     = str(_round(price + target_dist))
        else:  # sell
            entry_lo = _round(price)
            entry_hi = _round(price + pullback)
            entry_zone = f"{entry_lo}-{entry_hi}"
            stop_loss  = str(_round(price + stop_dist))
            target     = str(_round(price - target_dist))

    position_size = (
        _position_size(confidence, risk_level)
        if action in ("buy", "sell") else "avoid"
    )

    # Plain, factual reasoning from the actual votes (no LLM needed).
    reasoning = _build_reasoning(ind, score, ml_signal)

    return {
        "symbol":           symbol,
        "market_condition": condition,
        "regime_alignment": "neutral",
        "action":           action,
        "strategy":         _strategy_hint(action, risk_level),
        "confidence":       confidence,
        "risk_level":       risk_level,
        "entry_zone":       entry_zone,
        "stop_loss":        stop_loss,
        "target":           target,
        "position_size":    position_size,
        "reasoning":        reasoning,
        "regime_notes":     "",
        "memory_notes":     "",
        "event_notes":      "",
        "warnings":         [],
        # price-action / entry timing
        "entry_signal":     entry_signal,        # "trigger" | "wait"
        "pattern":          pat_primary,         # headline pattern, if any
        "pattern_context":  pat_context,         # pullback / breakout / ...
        # engine internals, useful for narration / debugging
        "engine_score":     round(score, 3),
        "ml_signal":        ml_signal,
        "decided_by":       _decided_by(ml_signal, patterns),
    }


def _decided_by(ml_signal, patterns):
    parts = ["rules"]
    if ml_signal:
        parts.append("ml")
    if patterns:
        parts.append("patterns")
    return "+".join(parts)


def _strategy_hint(action, risk_level):
    if action == "hold":
        return "stay_flat"
    if risk_level == "high":
        return "directional_with_spread"
    return "trend_following"


def _build_reasoning(ind, score, ml_signal=None):
    bits = []
    if ml_signal:
        bits.append(
            f"ML model predicts {ml_signal.get('ml_regime', 'sideways')} "
            f"(confidence {ml_signal.get('confidence', 0)})."
        )
    trend = ind.get("trend", "sideways")
    rsi   = float(ind.get("rsi", 50))
    macd  = ind.get("macd", "neutral")
    bits.append(f"Trend is {trend} (net signal score {round(score, 2)}).")
    bits.append(f"RSI={rsi}, MACD={macd}.")
    price = ind.get("price")
    ema20 = ind.get("ema_20")
    ema50 = ind.get("ema_50")
    if None not in (price, ema20, ema50):
        rel = "above" if price > ema20 else "below"
        bits.append(f"Price {price} is {rel} EMA20 {ema20} (EMA50 {ema50}).")
    bits.append(
        f"Volatility {ind.get('volatility', 'medium')}, "
        f"ATR {ind.get('atr', 'N/A')}."
    )
    return bits


# ── Self Test ─────────────────────────────────────────
if __name__ == "__main__":
    import json
    sample = {
        "symbol": "NIFTY", "price": 23719.3, "trend": "uptrend",
        "rsi": 58.0, "macd": "bullish", "ema_20": 23600.0,
        "ema_50": 23400.0, "bollinger": "inside_bands",
        "vwap": "above_vwap", "atr": 180.0, "volatility": "medium",
    }
    print("── rules only ──")
    print(json.dumps(decide(sample), indent=2))
    print("\n── rules + ML (bearish model overrides bullish indicators) ──")
    ml = {"ml_regime": "downtrend", "confidence": 0.9}
    print(json.dumps(decide(sample, ml), indent=2))

"""
Conviction Intelligence V2 — institutional trade grading.

Computes 7 sub-scores and assigns a letter grade. Hard disqualifiers
ensure the majority of signals become No Trade.

Grades: A+ (≥85) | A (≥70) | B (≥55) | C (≥40) | NO_TRADE (<40)
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

GRADE_THRESHOLDS = {"A+": 85, "A": 70, "B": 55, "C": 40}

SCORE_WEIGHTS = {
    "probability":       0.20,
    "conviction":        0.20,
    "risk":              0.15,
    "reward":            0.15,
    "market_quality":    0.10,
    "liquidity":         0.10,
    "regime_alignment":  0.10,
}

# Hard disqualifiers — instant No Trade
DISQUALIFIERS = {
    "market_quality":    30,
    "liquidity":         20,
    "risk_max":          80,
    "conviction_min":    30,
    "regime_alignment":  20,
}

NIFTY_50 = {
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
}


# ── Sub-Score Computations ─────────────────────────────

def compute_probability_score(signal, enrichment):
    """0-100. Statistical probability of trade success based on signal
    agreement across modules."""
    score = 50.0

    confidence = signal.get("fused_confidence", signal.get("confidence", 0.5))
    score = confidence * 80

    agreement = signal.get("agreement", "neutral")
    if agreement == "strong":
        score += 15
    elif agreement == "moderate":
        score += 5
    elif agreement == "weak":
        score -= 10

    ml_conf = signal.get("ml_confidence", 0)
    if isinstance(ml_conf, (int, float)) and ml_conf > 0.7:
        score += 10

    breadth = enrichment.get("breadth", {})
    bs = breadth.get("composite_score", breadth.get("score", 50))
    action = signal.get("final_action", "hold")
    if action == "buy" and bs > 60:
        score += 5
    elif action == "buy" and bs < 35:
        score -= 10

    return float(np.clip(score, 0, 100))


def compute_conviction_score(signal, enrichment):
    """0-100. How many signal dimensions agree?"""
    agreements = 0
    total_dims = 0

    action = signal.get("final_action", "hold")
    is_long = action == "buy"
    is_short = action == "sell"

    checks = [
        ("breadth", enrichment.get("breadth", {}).get("regime", "neutral"),
         lambda r: r in ("strong_bullish", "bullish") if is_long
         else r in ("strong_bearish", "bearish") if is_short else False),
        ("rs", enrichment.get("relative_strength", {}).get("composite_percentile", 50),
         lambda p: p > 60 if is_long else p < 40 if is_short else False),
        ("sector", enrichment.get("sector_rotation", {}).get("phase", "neutral"),
         lambda p: p.lower() in ("leading", "improving") if is_long
         else p.lower() in ("lagging", "weakening") if is_short else False),
        ("mtf", enrichment.get("multi_timeframe", {}).get("alignment", 50),
         lambda a: a > 60 if (is_long or is_short) else False),
        ("options_flow", enrichment.get("options_flow", {}).get("sentiment_score", 50),
         lambda s: s > 60 if is_long else s < 40 if is_short else False),
    ]

    for name, value, check_fn in checks:
        total_dims += 1
        try:
            if check_fn(value):
                agreements += 1
        except (TypeError, AttributeError):
            pass

    regime_v2 = enrichment.get("regime_v2", signal.get("regime_v2", {}))
    day_type = regime_v2.get("day_type", "")
    if day_type:
        total_dims += 1
        if is_long and day_type in ("trend_day", "breakout_day", "risk_on"):
            agreements += 1
        elif is_short and day_type in ("panic_selling", "risk_off"):
            agreements += 1

    if total_dims == 0:
        return 50.0

    agreement_pct = agreements / total_dims
    score = agreement_pct * 100

    if agreement_pct >= 0.8:
        score = min(score + 10, 100)
    elif agreement_pct <= 0.2:
        score = max(score - 10, 0)

    return float(np.clip(score, 0, 100))


def compute_risk_score(signal, enrichment):
    """0-100. Higher = MORE risky. High risk = bad for grade."""
    risk = 30.0

    vol = signal.get("volatility", "medium")
    vol_map = {"low": 0, "medium": 10, "high": 25, "extreme": 45}
    risk += vol_map.get(vol.lower() if isinstance(vol, str) else "medium", 10)

    atr = signal.get("atr")
    price = signal.get("price")
    if atr and price and price > 0:
        atr_pct = (atr / price) * 100
        if atr_pct > 3.0:
            risk += 20
        elif atr_pct > 2.0:
            risk += 10

    portfolio_risk = enrichment.get("portfolio_risk", {})
    risk_regime = portfolio_risk.get("risk_regime", "normal")
    regime_risk = {"very_low": -10, "low": 0, "normal": 5,
                   "elevated": 15, "extreme": 30}
    risk += regime_risk.get(risk_regime, 5)

    dd = enrichment.get("drawdown", {})
    current_dd = dd.get("current", 0)
    if isinstance(current_dd, (int, float)) and current_dd < -0.1:
        risk += 15

    return float(np.clip(risk, 0, 100))


def compute_reward_score(signal):
    """0-100. Risk-reward quality."""
    entry = signal.get("price", signal.get("entry_zone"))
    stop = signal.get("stop_loss")
    target = signal.get("target")

    for val_name in ("entry", "stop", "target"):
        val = locals()[val_name]
        if isinstance(val, str):
            try:
                locals()[val_name] = float(val.split("-")[0])
            except (ValueError, IndexError):
                locals()[val_name] = None

    entry = float(entry) if entry and not isinstance(entry, str) else None
    stop = float(stop) if stop and not isinstance(stop, str) else None
    target = float(target) if target and not isinstance(target, str) else None

    if not all([entry, stop, target]):
        return 50.0

    risk_dist = abs(entry - stop)
    reward_dist = abs(target - entry)
    if risk_dist == 0:
        return 50.0

    rr = reward_dist / risk_dist
    if rr >= 4.0:
        return 95.0
    elif rr >= 3.0:
        return 85.0
    elif rr >= 2.5:
        return 75.0
    elif rr >= 2.0:
        return 65.0
    elif rr >= 1.5:
        return 50.0
    elif rr >= 1.0:
        return 35.0
    return 15.0


def compute_market_quality_score(enrichment):
    """0-100. Is the overall market environment conducive?"""
    score = 50.0

    breadth = enrichment.get("breadth", {})
    bs = breadth.get("composite_score", breadth.get("score", 50))
    score = bs * 0.5

    regime = breadth.get("regime", "neutral")
    clarity = {
        "strong_bullish": 30, "strong_bearish": 25,
        "bullish": 20, "bearish": 15,
        "neutral": 5,
    }
    score += clarity.get(regime, 5)

    regime_v2 = enrichment.get("regime_v2", {})
    confidence = regime_v2.get("regime_confidence", regime_v2.get("confidence", 0.5))
    if isinstance(confidence, (int, float)):
        score += confidence * 15

    mtf = enrichment.get("multi_timeframe", {})
    alignment = mtf.get("alignment", 50)
    if isinstance(alignment, (int, float)) and alignment > 70:
        score += 10

    return float(np.clip(score, 0, 100))


def compute_liquidity_score(symbol, prices=None):
    """0-100. Based on market cap tier and index membership."""
    if symbol in ("NIFTY", "BANKNIFTY", "FINNIFTY"):
        return 95.0
    if symbol in NIFTY_50:
        return 85.0
    return 40.0


def compute_regime_alignment_score(signal, enrichment):
    """0-100. Does the trade direction match the regime?"""
    score = 50.0
    action = signal.get("final_action", "hold")

    breadth = enrichment.get("breadth", {})
    regime = breadth.get("regime", "neutral")

    if action == "buy":
        alignment = {
            "strong_bullish": 95, "bullish": 80, "neutral": 50,
            "bearish": 20, "strong_bearish": 5,
        }
    elif action == "sell":
        alignment = {
            "strong_bearish": 90, "bearish": 75, "neutral": 50,
            "bullish": 20, "strong_bullish": 5,
        }
    else:
        return 50.0

    score = alignment.get(regime, 50)

    regime_v2 = enrichment.get("regime_v2", {})
    day_type = regime_v2.get("day_type", "")
    if action == "buy" and day_type in ("panic_selling", "short_covering_rally"):
        score = max(score - 30, 0)
    elif action == "buy" and day_type in ("trend_day", "risk_on"):
        score = min(score + 10, 100)
    elif action == "sell" and day_type in ("panic_selling", "risk_off"):
        score = min(score + 10, 100)

    return float(np.clip(score, 0, 100))


# ── Grading ────────────────────────────────────────────

def grade_signal(signal, enrichment=None):
    """THE GRADING FUNCTION.

    Returns {grade, composite_score, all 7 sub-scores, reasons, disqualifiers}.
    """
    if enrichment is None:
        enrichment = {}

    symbol = signal.get("symbol", "")

    scores = {
        "probability":      compute_probability_score(signal, enrichment),
        "conviction":       compute_conviction_score(signal, enrichment),
        "risk":             compute_risk_score(signal, enrichment),
        "reward":           compute_reward_score(signal),
        "market_quality":   compute_market_quality_score(enrichment),
        "liquidity":        compute_liquidity_score(symbol),
        "regime_alignment": compute_regime_alignment_score(signal, enrichment),
    }

    # Risk is inverted: high risk = bad for grade, so use (100 - risk)
    adjusted = dict(scores)
    adjusted["risk"] = 100 - scores["risk"]

    composite = sum(adjusted[k] * SCORE_WEIGHTS[k] for k in SCORE_WEIGHTS)

    # Hard disqualifiers
    disqualifiers = []
    if scores["market_quality"] < DISQUALIFIERS["market_quality"]:
        disqualifiers.append(
            f"Market quality too low: {scores['market_quality']:.0f} < {DISQUALIFIERS['market_quality']}")
    if scores["liquidity"] < DISQUALIFIERS["liquidity"]:
        disqualifiers.append(
            f"Liquidity too low: {scores['liquidity']:.0f} < {DISQUALIFIERS['liquidity']}")
    if scores["risk"] > DISQUALIFIERS["risk_max"]:
        disqualifiers.append(
            f"Risk too high: {scores['risk']:.0f} > {DISQUALIFIERS['risk_max']}")
    if scores["conviction"] < DISQUALIFIERS["conviction_min"]:
        disqualifiers.append(
            f"Conviction too low: {scores['conviction']:.0f} < {DISQUALIFIERS['conviction_min']}")
    if scores["regime_alignment"] < DISQUALIFIERS["regime_alignment"]:
        disqualifiers.append(
            f"Fighting the regime: alignment {scores['regime_alignment']:.0f} < {DISQUALIFIERS['regime_alignment']}")

    if disqualifiers:
        grade = "NO_TRADE"
    elif composite >= GRADE_THRESHOLDS["A+"]:
        grade = "A+"
    elif composite >= GRADE_THRESHOLDS["A"]:
        grade = "A"
    elif composite >= GRADE_THRESHOLDS["B"]:
        grade = "B"
    elif composite >= GRADE_THRESHOLDS["C"]:
        grade = "C"
    else:
        grade = "NO_TRADE"

    reasons = []
    if grade != "NO_TRADE":
        top = sorted(adjusted.items(), key=lambda x: -x[1])[:3]
        reasons = [f"{k} strong at {v:.0f}" for k, v in top]
    else:
        if not disqualifiers:
            reasons = [f"Composite {composite:.0f} below C threshold ({GRADE_THRESHOLDS['C']})"]

    return {
        "grade": grade,
        "composite_score": round(composite, 1),
        "probability": round(scores["probability"], 1),
        "conviction": round(scores["conviction"], 1),
        "risk": round(scores["risk"], 1),
        "reward": round(scores["reward"], 1),
        "market_quality": round(scores["market_quality"], 1),
        "liquidity": round(scores["liquidity"], 1),
        "regime_alignment": round(scores["regime_alignment"], 1),
        "reasons": reasons,
        "disqualifiers": disqualifiers,
    }


# ── CLI ────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  CONVICTION INTELLIGENCE V2")
    print("=" * 60)

    # A+ signal: everything aligned
    strong = {
        "symbol": "RELIANCE", "final_action": "buy",
        "fused_confidence": 0.82, "agreement": "strong",
        "ml_confidence": 0.78, "price": 2800, "atr": 40,
        "volatility": "medium", "stop_loss": 2755, "target": 2920,
    }
    strong_enr = {
        "breadth": {"composite_score": 72, "regime": "bullish"},
        "relative_strength": {"composite_percentile": 85, "rs_regime": "Leader"},
        "sector_rotation": {"phase": "leading"},
        "multi_timeframe": {"alignment": 78},
        "options_flow": {"sentiment_score": 72},
        "regime_v2": {"day_type": "trend_day", "regime_confidence": 0.7},
    }
    r = grade_signal(strong, strong_enr)
    print(f"\n  Strong signal:")
    print(f"    Grade: {r['grade']} (composite {r['composite_score']})")
    print(f"    Probability={r['probability']}, Conviction={r['conviction']}, "
          f"Risk={r['risk']}, Reward={r['reward']}")
    print(f"    Reasons: {r['reasons']}")

    # Weak signal: fighting regime, low conviction
    weak = {
        "symbol": "SMALLCAP", "final_action": "buy",
        "fused_confidence": 0.52, "agreement": "weak",
        "price": 100, "atr": 8, "volatility": "high",
        "stop_loss": 92, "target": 105,
    }
    weak_enr = {
        "breadth": {"composite_score": 28, "regime": "bearish"},
        "relative_strength": {"composite_percentile": 25},
        "multi_timeframe": {"alignment": 30},
        "regime_v2": {"day_type": "risk_off"},
    }
    r2 = grade_signal(weak, weak_enr)
    print(f"\n  Weak signal:")
    print(f"    Grade: {r2['grade']} (composite {r2['composite_score']})")
    print(f"    Disqualifiers: {r2['disqualifiers']}")

    # No enrichment
    bare = {"symbol": "TCS", "final_action": "buy",
            "fused_confidence": 0.65, "price": 3500,
            "stop_loss": 3400, "target": 3700}
    r3 = grade_signal(bare, {})
    print(f"\n  Bare signal (no enrichment):")
    print(f"    Grade: {r3['grade']} (composite {r3['composite_score']})")

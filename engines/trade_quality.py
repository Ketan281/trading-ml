"""
Trade Quality Engine — scores every setup across 8 dimensions.

Reads enrichment data produced by institutional_engine.py and computes
a composite Trade Quality Score (0-100) and Opportunity Score (0-100).

Component Weights:
  breadth 15%, rs 15%, sector 10%, options_flow 10%,
  mtf 15%, volatility 10%, liquidity 10%, risk_reward 15%
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

WEIGHTS = {
    "breadth":      0.15,
    "rs":           0.15,
    "sector":       0.10,
    "options_flow": 0.10,
    "mtf":          0.15,
    "volatility":   0.10,
    "liquidity":    0.10,
    "risk_reward":  0.15,
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


# ── Component Scorers ──────────────────────────────────

def score_breadth_quality(enrichment):
    """0-100. Breadth regime strength and market participation."""
    b = enrichment.get("breadth")
    if not b:
        return 50.0
    score = 0.0
    cs = b.get("composite_score", b.get("score", 50))
    score += cs * 0.6
    regime = b.get("regime", "neutral")
    regime_bonus = {
        "strong_bullish": 30, "bullish": 20, "neutral": 10,
        "bearish": 0, "strong_bearish": -10,
    }
    score += regime_bonus.get(regime, 10)
    participation = b.get("pct_above_ema20", b.get("participation", 50))
    if participation > 70:
        score += 10
    elif participation > 50:
        score += 5
    return float(np.clip(score, 0, 100))


def score_rs_quality(enrichment):
    """0-100. Relative strength percentile and momentum."""
    rs = enrichment.get("relative_strength")
    if not rs:
        return 50.0
    pctile = rs.get("composite_percentile", rs.get("percentile", 50))
    score = pctile * 0.7
    regime = rs.get("rs_regime", rs.get("regime", "neutral"))
    regime_bonus = {"leader": 25, "strong": 15, "neutral": 5,
                    "weak": -5, "laggard": -15}
    score += regime_bonus.get(regime.lower() if isinstance(regime, str) else "neutral", 5)
    momentum = rs.get("momentum", 0)
    if momentum > 0:
        score += min(momentum * 5, 10)
    return float(np.clip(score, 0, 100))


def score_sector_quality(enrichment):
    """0-100. Sector phase and rotation momentum."""
    sec = enrichment.get("sector_rotation", enrichment.get("sector"))
    if not sec:
        return 50.0
    score = 50.0
    phase = sec.get("phase", sec.get("sector_phase", "neutral"))
    phase_map = {"leading": 30, "improving": 15, "neutral": 0,
                 "weakening": -10, "lagging": -20}
    score += phase_map.get(phase.lower() if isinstance(phase, str) else "neutral", 0)
    rank = sec.get("rank", sec.get("sector_rank", 50))
    if isinstance(rank, (int, float)):
        if rank <= 3:
            score += 15
        elif rank <= 5:
            score += 5
        elif rank > 10:
            score -= 10
    conv = sec.get("conviction_multiplier", 1.0)
    if conv > 1.2:
        score += 10
    elif conv < 0.8:
        score -= 10
    return float(np.clip(score, 0, 100))


def score_options_flow_quality(enrichment):
    """0-100. Options sentiment alignment, PCR, IV rank."""
    of = enrichment.get("options_flow", enrichment.get("options_sentiment"))
    if not of:
        return 50.0
    score = 50.0
    sentiment = of.get("sentiment_score", of.get("sentiment", 50))
    if isinstance(sentiment, (int, float)):
        score = sentiment * 0.6 + 20
    pcr = of.get("pcr", of.get("put_call_ratio"))
    if isinstance(pcr, (int, float)):
        if 0.8 < pcr < 1.2:
            score += 10
        elif pcr > 1.5:
            score += 15
        elif pcr < 0.5:
            score -= 10
    iv_rank = of.get("iv_rank", of.get("iv_percentile"))
    if isinstance(iv_rank, (int, float)):
        if iv_rank < 30:
            score += 10
        elif iv_rank > 70:
            score -= 5
    return float(np.clip(score, 0, 100))


def score_mtf_quality(enrichment):
    """0-100. Multi-timeframe alignment and consensus."""
    mtf = enrichment.get("multi_timeframe", enrichment.get("mtf"))
    if not mtf:
        return 50.0
    alignment = mtf.get("alignment", mtf.get("alignment_score", 50))
    if isinstance(alignment, (int, float)):
        score = alignment * 0.8
    else:
        score = 50.0
    consensus = mtf.get("consensus", mtf.get("direction_agreement"))
    if isinstance(consensus, str):
        if consensus.lower() in ("strong_bullish", "strong_bearish"):
            score += 20
        elif consensus.lower() in ("bullish", "bearish"):
            score += 10
    elif isinstance(consensus, (int, float)):
        score += consensus * 0.2
    return float(np.clip(score, 0, 100))


def score_volatility_conditions(signal):
    """0-100. Volatility regime favorability. Moderate vol is ideal."""
    atr = signal.get("atr")
    price = signal.get("price", signal.get("entry_zone"))
    if isinstance(price, str):
        try:
            price = float(price.split("-")[0])
        except (ValueError, IndexError):
            price = None
    if not atr or not price:
        return 50.0
    atr_pct = (atr / price) * 100
    if 0.8 < atr_pct < 2.0:
        score = 80.0
    elif 0.5 < atr_pct < 3.0:
        score = 60.0
    elif atr_pct > 4.0:
        score = 25.0
    elif atr_pct <= 0.3:
        score = 30.0
    else:
        score = 50.0
    vol = signal.get("volatility", "medium")
    if isinstance(vol, str):
        vol_map = {"low": 10, "medium": 0, "high": -10, "extreme": -25}
        score += vol_map.get(vol.lower(), 0)
    return float(np.clip(score, 0, 100))


def score_liquidity(symbol, signal=None):
    """0-100. Large-cap index members score highest."""
    score = 50.0
    if symbol in NIFTY_50:
        score = 85.0
    elif symbol in ("NIFTY", "BANKNIFTY", "FINNIFTY"):
        score = 95.0
    else:
        score = 40.0
    if signal:
        vol = signal.get("volume", signal.get("avg_volume"))
        if isinstance(vol, (int, float)):
            if vol > 5_000_000:
                score = min(score + 10, 100)
            elif vol < 100_000:
                score = max(score - 20, 10)
    return float(np.clip(score, 0, 100))


def score_risk_reward(signal):
    """0-100. Quality of risk/reward ratio."""
    entry = signal.get("entry_zone", signal.get("price"))
    stop = signal.get("stop_loss")
    target = signal.get("target")

    if isinstance(entry, str):
        try:
            parts = entry.split("-")
            entry = float(parts[0])
        except (ValueError, IndexError):
            entry = None
    if isinstance(stop, str):
        try:
            stop = float(stop)
        except ValueError:
            stop = None
    if isinstance(target, str):
        try:
            target = float(target)
        except ValueError:
            target = None

    if not all([entry, stop, target]):
        return 50.0

    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk == 0:
        return 50.0

    rr = reward / risk
    if rr >= 4.0:
        score = 95.0
    elif rr >= 3.0:
        score = 85.0
    elif rr >= 2.5:
        score = 75.0
    elif rr >= 2.0:
        score = 65.0
    elif rr >= 1.5:
        score = 50.0
    elif rr >= 1.0:
        score = 35.0
    else:
        score = 15.0

    return float(score)


# ── Main Computation ───────────────────────────────────

def compute_trade_quality(signal, enrichment=None):
    """Compute composite Trade Quality Score and Opportunity Score.

    Args:
        signal: dict from decision_engine (symbol, price, action, stop, target, etc.)
        enrichment: dict from institutional_engine (breadth, rs, sector, mtf, etc.)

    Returns:
        {trade_quality_score, opportunity_score, component_scores,
         quality_grade, flags}
    """
    if enrichment is None:
        enrichment = {}

    symbol = signal.get("symbol", "")

    components = {
        "breadth":      score_breadth_quality(enrichment),
        "rs":           score_rs_quality(enrichment),
        "sector":       score_sector_quality(enrichment),
        "options_flow": score_options_flow_quality(enrichment),
        "mtf":          score_mtf_quality(enrichment),
        "volatility":   score_volatility_conditions(signal),
        "liquidity":    score_liquidity(symbol, signal),
        "risk_reward":  score_risk_reward(signal),
    }

    tqs = sum(components[k] * WEIGHTS[k] for k in WEIGHTS)

    market_env = (components["breadth"] + components["mtf"]) / 2
    opportunity = tqs * (market_env / 100)

    flags = []
    for k, v in components.items():
        if v < 30:
            flags.append(f"weak_{k}")
    if components["risk_reward"] < 40:
        flags.append("poor_risk_reward")
    if components["liquidity"] < 30:
        flags.append("illiquid")

    if tqs >= 80:
        grade = "excellent"
    elif tqs >= 60:
        grade = "good"
    elif tqs >= 40:
        grade = "acceptable"
    else:
        grade = "poor"

    return {
        "trade_quality_score": round(tqs, 1),
        "opportunity_score": round(opportunity, 1),
        "component_scores": {k: round(v, 1) for k, v in components.items()},
        "quality_grade": grade,
        "flags": flags,
    }


# ── CLI ────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  TRADE QUALITY ENGINE")
    print("=" * 60)

    test_signal = {
        "symbol": "RELIANCE",
        "price": 2800,
        "atr": 45,
        "volatility": "medium",
        "stop_loss": "2755",
        "target": "2890",
        "entry_zone": "2800-2810",
    }

    test_enrichment = {
        "breadth": {"composite_score": 65, "regime": "bullish", "pct_above_ema20": 62},
        "relative_strength": {"composite_percentile": 78, "rs_regime": "Strong", "momentum": 2},
        "sector_rotation": {"phase": "leading", "rank": 2, "conviction_multiplier": 1.3},
        "options_flow": {"sentiment_score": 70, "pcr": 1.1, "iv_rank": 35},
        "multi_timeframe": {"alignment": 75, "consensus": "bullish"},
    }

    result = compute_trade_quality(test_signal, test_enrichment)
    print(f"\n  Trade Quality Score : {result['trade_quality_score']}")
    print(f"  Opportunity Score   : {result['opportunity_score']}")
    print(f"  Quality Grade       : {result['quality_grade']}")
    print(f"\n  Component Scores:")
    for k, v in result["component_scores"].items():
        bar = "#" * int(v / 5)
        print(f"    {k:<15} {v:5.1f}  {bar}")
    if result["flags"]:
        print(f"\n  Flags: {', '.join(result['flags'])}")

    test2 = {"symbol": "SMALLCAP", "price": 100, "atr": 8,
             "volatility": "high", "stop_loss": "92", "target": "105"}
    r2 = compute_trade_quality(test2, {})
    print(f"\n  Weak signal test: TQS={r2['trade_quality_score']}, "
          f"Grade={r2['quality_grade']}, Flags={r2['flags']}")

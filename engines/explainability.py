"""
Explainability Engine — "Why This Trade?" for every recommendation.

Generates human-readable explanations decomposing each trade decision
into bullish/bearish factors, component reads, sizing logic, and
risk warnings. Also explains why signals were rejected.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _factor(name, score, threshold_good=60, threshold_bad=40):
    """Classify a factor as bullish, bearish, or neutral."""
    if not isinstance(score, (int, float)):
        return {"factor": name, "score": 0, "weight": "neutral"}
    if score >= threshold_good:
        return {"factor": name, "score": round(score, 1), "weight": "bullish"}
    elif score <= threshold_bad:
        return {"factor": name, "score": round(score, 1), "weight": "bearish"}
    return {"factor": name, "score": round(score, 1), "weight": "neutral"}


def _regime_read(enrichment):
    regime_v2 = enrichment.get("regime_v2", {})
    day_type = regime_v2.get("day_type", "unknown")
    macro = regime_v2.get("macro_regime", "unknown")
    implication = {
        "trend_day": "Strong trend — follow direction, trail stops wide",
        "range_day": "Rangebound — fade extremes, tight stops",
        "breakout_day": "Breakout in play — follow with measured targets",
        "mean_reversion_day": "Mean reversion — fade extremes cautiously",
        "vol_expansion": "Volatility expanding — reduce size, quick profits",
        "vol_contraction": "Low volatility — breakout may be imminent",
        "panic_selling": "PANIC — no new longs, protect capital",
        "short_covering_rally": "Relief rally — likely unsustainable",
        "risk_on": "Risk-on — broad participation, full sizing",
        "risk_off": "Risk-off — favour defensives, reduce exposure",
    }.get(day_type, "No strong day-type signal")
    return {"regime": macro, "day_type": day_type, "implication": implication}


def _breadth_read(enrichment):
    b = enrichment.get("breadth", {})
    score = b.get("composite_score", b.get("score", 50))
    regime = b.get("regime", "neutral")
    implications = {
        "strong_bullish": "Broad market participation — bullish confirmation",
        "bullish": "Healthy breadth — supports long positions",
        "neutral": "Mixed breadth — no strong directional bias",
        "bearish": "Narrow participation — longs are risky",
        "strong_bearish": "Very weak breadth — avoid new longs entirely",
    }
    return {"score": score, "regime": regime,
            "implication": implications.get(regime, "Unclear breadth signal")}


def _rs_read(enrichment):
    rs = enrichment.get("relative_strength", {})
    pctile = rs.get("composite_percentile", rs.get("percentile", 50))
    regime = rs.get("rs_regime", rs.get("regime", "neutral"))
    if pctile > 80:
        impl = "Top-tier relative strength — stock is leading the market"
    elif pctile > 60:
        impl = "Above-average strength — reasonable candidate"
    elif pctile > 40:
        impl = "Average strength — no edge from momentum"
    else:
        impl = "Lagging relative strength — weakest names lose more"
    return {"percentile": pctile, "regime": regime, "implication": impl}


def _sector_read(enrichment):
    sec = enrichment.get("sector_rotation", enrichment.get("sector", {}))
    phase = sec.get("phase", sec.get("sector_phase", "neutral"))
    rank = sec.get("rank", sec.get("sector_rank", "N/A"))
    implications = {
        "leading": "Sector is leading rotation — strong tailwind",
        "improving": "Sector improving — rotating into strength",
        "neutral": "Sector neutral — no rotation edge",
        "weakening": "Sector weakening — headwind for longs",
        "lagging": "Sector lagging — avoid or go short",
    }
    return {"phase": phase, "rank": rank,
            "implication": implications.get(
                phase.lower() if isinstance(phase, str) else "neutral",
                "No clear sector signal")}


def _options_flow_read(enrichment):
    of = enrichment.get("options_flow", enrichment.get("options_sentiment", {}))
    sentiment = of.get("sentiment_score", of.get("sentiment", 50))
    pcr = of.get("pcr", of.get("put_call_ratio", "N/A"))
    if isinstance(sentiment, (int, float)):
        if sentiment > 65:
            impl = "Bullish options flow — smart money positioning long"
        elif sentiment < 35:
            impl = "Bearish options flow — put buying dominant"
        else:
            impl = "Neutral options flow — no clear positioning"
    else:
        impl = "Options flow data unavailable"
    return {"sentiment": sentiment, "pcr": pcr, "implication": impl}


def _mtf_read(enrichment):
    mtf = enrichment.get("multi_timeframe", enrichment.get("mtf", {}))
    alignment = mtf.get("alignment", mtf.get("alignment_score", 50))
    consensus = mtf.get("consensus", "neutral")
    if isinstance(alignment, (int, float)):
        if alignment > 70:
            impl = "Strong multi-timeframe alignment — high conviction"
        elif alignment > 50:
            impl = "Moderate alignment — some timeframes agree"
        else:
            impl = "Poor alignment — timeframes disagree, expect whipsaws"
    else:
        impl = "MTF data unavailable"
    return {"alignment": alignment, "consensus": consensus, "implication": impl}


def _sizing_logic(sizing_result):
    if not sizing_result:
        return "Position sizing not computed"
    method = sizing_result.get("sizing_method", "unknown")
    mults = sizing_result.get("multipliers", {})
    parts = [f"Method: {method}"]
    if mults.get("conviction"):
        parts.append(f"conviction {mults['conviction']:.2f}x")
    if mults.get("drawdown"):
        parts.append(f"drawdown {mults['drawdown']:.2f}x")
    if mults.get("portfolio"):
        parts.append(f"portfolio {mults['portfolio']:.2f}x")
    if mults.get("quality"):
        parts.append(f"quality {mults['quality']:.2f}x")
    return " | ".join(parts)


def _stop_logic(signal):
    stop = signal.get("stop_loss")
    price = signal.get("price")
    if stop and price:
        try:
            s, p = float(stop) if isinstance(stop, str) else stop, \
                   float(price) if isinstance(price, str) else price
            dist = abs(p - s) / p * 100
            return f"Stop at {s:,.1f} ({dist:.1f}% from entry {p:,.1f})"
        except (ValueError, TypeError):
            pass
    return "Stop loss details unavailable"


def _target_logic(signal):
    target = signal.get("target")
    price = signal.get("price")
    if target and price:
        try:
            t, p = float(target) if isinstance(target, str) else target, \
                   float(price) if isinstance(price, str) else price
            dist = abs(t - p) / p * 100
            return f"Target at {t:,.1f} ({dist:.1f}% from entry {p:,.1f})"
        except (ValueError, TypeError):
            pass
    return "Target details unavailable"


# ── Main Explanation Functions ─────────────────────────

def explain_trade(signal, enrichment=None, grade_result=None,
                  quality_result=None, sizing_result=None,
                  regime_result=None):
    """Generate complete 'Why This Trade?' explanation."""
    enrichment = enrichment or {}
    grade_result = grade_result or {}
    quality_result = quality_result or {}
    sizing_result = sizing_result or {}

    symbol = signal.get("symbol", "?")
    action = signal.get("final_action", "hold")
    confidence = signal.get("fused_confidence", signal.get("confidence", 0))
    grade = grade_result.get("grade", "?")
    quality = quality_result.get("trade_quality_score", 0)

    # Build factors list
    all_factors = []
    if grade_result:
        for key in ("probability", "conviction", "reward", "market_quality",
                     "liquidity", "regime_alignment"):
            val = grade_result.get(key)
            if val is not None:
                all_factors.append(_factor(key, val))
        risk_val = grade_result.get("risk")
        if risk_val is not None:
            f = _factor("risk", 100 - risk_val)
            f["factor"] = "risk (inverted)"
            all_factors.append(f)

    if quality_result:
        for key, val in quality_result.get("component_scores", {}).items():
            if key not in [f["factor"] for f in all_factors]:
                all_factors.append(_factor(key, val))

    bullish = [f for f in all_factors if f["weight"] == "bullish"]
    bearish = [f for f in all_factors if f["weight"] == "bearish"]
    neutral = [f for f in all_factors if f["weight"] == "neutral"]

    direction = "LONG" if action == "buy" else "SHORT" if action == "sell" else "HOLD"
    summary = (f"{direction} {symbol} — Grade {grade}, "
               f"confidence {confidence:.0%}, quality {quality:.0f}/100. "
               f"{len(bullish)} bullish, {len(bearish)} bearish factors.")

    entry = signal.get("price", signal.get("entry_zone"))
    stop = signal.get("stop_loss")
    t1 = signal.get("target")
    rr = None
    if entry and stop and t1:
        try:
            e = float(str(entry).split("-")[0])
            s = float(str(stop))
            t = float(str(t1))
            risk_d = abs(e - s)
            rew_d = abs(t - e)
            rr = round(rew_d / risk_d, 2) if risk_d > 0 else None
        except (ValueError, TypeError):
            pass

    risk_factors = []
    if grade_result.get("risk", 0) > 60:
        risk_factors.append("Elevated risk score")
    for f in quality_result.get("flags", []):
        risk_factors.append(f.replace("_", " ").title())

    what_wrong = []
    if grade_result.get("regime_alignment", 100) < 40:
        what_wrong.append("Fighting the regime — direction misaligned with market")
    if grade_result.get("market_quality", 100) < 40:
        what_wrong.append("Poor market quality — environment not conducive")
    if quality_result.get("component_scores", {}).get("volatility", 100) < 30:
        what_wrong.append("Extreme volatility conditions")
    if not what_wrong:
        what_wrong.append("No major red flags identified")

    return {
        "summary": summary,
        "confidence": round(confidence, 4),
        "conviction": grade_result.get("conviction", 0),
        "grade": grade,
        "quality_score": quality,
        "risk_reward": {
            "entry": entry, "stop": stop,
            "target": t1, "expected_r": rr,
        },
        "bullish_factors": bullish,
        "bearish_factors": bearish,
        "neutral_factors": neutral,
        "reasons_selected": grade_result.get("reasons", []),
        "regime_context": _regime_read(enrichment),
        "breadth_read": _breadth_read(enrichment),
        "rs_read": _rs_read(enrichment),
        "sector_read": _sector_read(enrichment),
        "options_flow_read": _options_flow_read(enrichment),
        "mtf_read": _mtf_read(enrichment),
        "position_sizing_logic": _sizing_logic(sizing_result),
        "stop_logic": _stop_logic(signal),
        "target_logic": _target_logic(signal),
        "risk_factors": risk_factors,
        "what_could_go_wrong": what_wrong,
    }


def explain_no_trade(signal, enrichment=None, grade_result=None,
                     psychology_result=None):
    """Explain why a signal was rejected."""
    grade_result = grade_result or {}
    psychology_result = psychology_result or {}

    reasons = []
    if grade_result.get("disqualifiers"):
        reasons.extend(grade_result["disqualifiers"])
    if grade_result.get("grade") == "NO_TRADE" and not grade_result.get("disqualifiers"):
        reasons.append(f"Composite score {grade_result.get('composite_score', 0):.0f} "
                       f"below minimum threshold")
    if psychology_result.get("allowed") is False:
        reasons.append(f"Psychology gate: {psychology_result.get('reason', 'blocked')}")

    what_would_help = []
    if grade_result.get("conviction", 100) < 40:
        what_would_help.append("Need more signal dimensions to agree (breadth + RS + sector + MTF)")
    if grade_result.get("market_quality", 100) < 30:
        what_would_help.append("Wait for broader market improvement")
    if grade_result.get("regime_alignment", 100) < 20:
        what_would_help.append("Trade direction must align with market regime")
    if grade_result.get("risk", 0) > 80:
        what_would_help.append("Volatility needs to normalize")
    if not what_would_help:
        what_would_help.append("Multiple factors need improvement simultaneously")

    return {
        "symbol": signal.get("symbol", "?"),
        "action_attempted": signal.get("final_action", "hold"),
        "reason_category": "disqualified" if grade_result.get("disqualifiers")
                           else "low_score" if grade_result.get("grade") == "NO_TRADE"
                           else "psychology_gate",
        "specific_reasons": reasons,
        "what_would_make_it_tradeable": what_would_help,
        "nearest_grade": _nearest_grade(grade_result),
        "composite_score": grade_result.get("composite_score", 0),
    }


def _nearest_grade(grade_result):
    score = grade_result.get("composite_score", 0)
    if score >= 40:
        return "C"
    elif score >= 30:
        return "C (close)"
    return "far from tradeable"


def comparative_explanation(signals, selected_idx=0):
    """Explain why one signal was selected over others."""
    if not signals:
        return {"explanation": "No signals to compare"}

    selected = signals[selected_idx] if selected_idx < len(signals) else signals[0]
    others = [s for i, s in enumerate(signals) if i != selected_idx]

    reasons_others_rejected = []
    for s in others:
        sym = s.get("symbol", "?")
        grade = s.get("grade", s.get("grade_result", {}).get("grade", "?"))
        score = s.get("composite_score",
                      s.get("grade_result", {}).get("composite_score", 0))
        if grade == "NO_TRADE":
            reasons_others_rejected.append(f"{sym}: No Trade (score {score:.0f})")
        else:
            reasons_others_rejected.append(
                f"{sym}: Grade {grade} (score {score:.0f}) — lower conviction")

    return {
        "selected": selected.get("symbol", "?"),
        "selected_grade": selected.get("grade",
                                       selected.get("grade_result", {}).get("grade", "?")),
        "reasons_selected": [
            f"Highest composite score among candidates",
            f"Best risk-adjusted conviction",
        ],
        "reasons_others_rejected": reasons_others_rejected,
    }


# ── CLI ────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  EXPLAINABILITY ENGINE")
    print("=" * 60)

    sig = {
        "symbol": "RELIANCE", "final_action": "buy",
        "fused_confidence": 0.78, "price": 2800,
        "stop_loss": 2755, "target": 2920,
    }
    enr = {
        "breadth": {"composite_score": 68, "regime": "bullish"},
        "relative_strength": {"composite_percentile": 82, "rs_regime": "Leader"},
        "sector_rotation": {"phase": "leading", "rank": 2},
        "options_flow": {"sentiment_score": 70, "pcr": 1.1},
        "multi_timeframe": {"alignment": 75, "consensus": "bullish"},
        "regime_v2": {"day_type": "trend_day", "macro_regime": "bull"},
    }
    grade_r = {
        "grade": "A", "composite_score": 74,
        "probability": 72, "conviction": 68, "risk": 35,
        "reward": 75, "market_quality": 65, "liquidity": 85,
        "regime_alignment": 80, "reasons": ["reward strong at 75"],
    }
    quality_r = {
        "trade_quality_score": 72,
        "component_scores": {"breadth": 65, "rs": 78, "volatility": 60},
        "flags": [],
    }

    expl = explain_trade(sig, enr, grade_r, quality_r)
    print(f"\n  {expl['summary']}")
    print(f"\n  Bullish: {len(expl['bullish_factors'])}")
    for f in expl["bullish_factors"]:
        print(f"    + {f['factor']}: {f['score']}")
    print(f"  Bearish: {len(expl['bearish_factors'])}")
    for f in expl["bearish_factors"]:
        print(f"    - {f['factor']}: {f['score']}")
    print(f"\n  Regime: {expl['regime_context']['implication']}")
    print(f"  Breadth: {expl['breadth_read']['implication']}")
    print(f"  Stop: {expl['stop_logic']}")
    print(f"  Target: {expl['target_logic']}")

    no_trade = explain_no_trade(
        {"symbol": "WEAK", "final_action": "buy"},
        grade_result={"grade": "NO_TRADE", "composite_score": 28,
                      "conviction": 20, "market_quality": 25,
                      "disqualifiers": ["Market quality too low: 25 < 30"]})
    print(f"\n  No Trade: {no_trade['specific_reasons']}")
    print(f"  Would help: {no_trade['what_would_make_it_tradeable']}")

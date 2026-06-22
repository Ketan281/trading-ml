"""
Institutional Recommendation Engine — the Phase 2 orchestrator.

Wraps the existing institutional_engine (Phase 1) with conviction grading,
psychology gating, quality scoring, capital allocation, strategy selection,
and full explainability. Every recommendation is institutional-grade.

Signal flow:
  Phase 1 enrichment → regime_v2 → conviction_v2 → trade_quality →
  psychology gate → capital_allocation → strategy selection →
  explainability → final recommendation
"""

import os
import sys
import json
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STRATEGY_TYPES = [
    "buy_call", "buy_put",
    "bull_call_spread", "bear_put_spread",
    "iron_condor", "iron_butterfly",
    "long_straddle", "long_strangle",
    "covered_call", "protective_put",
    "equity_long", "equity_short",
    "no_trade",
]


def _safe(fn, *a, module="unknown", **kw):
    try:
        return fn(*a, **kw)
    except Exception as e:
        print(f"  [{module}] Error: {e}")
        return None


# ── Strategy Selection ─────────────────────────────────

def select_strategy(signal, regime=None, enrichment=None):
    """Choose optimal strategy type based on regime, IV, direction."""
    action = signal.get("final_action", "hold")
    enrichment = enrichment or {}
    regime = regime or {}

    day_type = regime.get("day_type", "range_day")
    forbidden = set(regime.get("forbidden_strategies", []))

    of = enrichment.get("options_flow", {})
    iv_rank = of.get("iv_rank", of.get("iv_percentile", 50))
    is_options = signal.get("segment") in ("options", "index_options")

    if action == "buy":
        if is_options:
            if isinstance(iv_rank, (int, float)) and iv_rank > 60:
                strat = "bull_call_spread"
            else:
                strat = "buy_call"
        else:
            strat = "equity_long"
    elif action == "sell":
        if is_options:
            if isinstance(iv_rank, (int, float)) and iv_rank > 60:
                strat = "bear_put_spread"
            else:
                strat = "buy_put"
        else:
            strat = "equity_short"
    elif action == "hold":
        if day_type == "vol_contraction":
            strat = "long_straddle" if "long_straddle" not in forbidden else "no_trade"
        elif day_type == "range_day" and isinstance(iv_rank, (int, float)) and iv_rank > 70:
            strat = "iron_condor" if "iron_condor" not in forbidden else "no_trade"
        else:
            strat = "no_trade"
    else:
        strat = "no_trade"

    if strat in forbidden:
        strat = "no_trade"

    return strat


# ── Full Pipeline ──────────────────────────────────────

def full_pipeline(signal, df=None, prices=None, positions=None,
                  capital=1_000_000, psychology_state=None):
    """Phase 2 wrapper around institutional_engine.

    1. Run Phase 1 enrichment (institutional_engine)
    2. Detect regime v2
    3. Grade with conviction_v2
    4. Score with trade_quality
    5. Gate with psychology_engine
    6. Size with capital_allocation
    7. Select strategy
    8. Generate explanation
    9. Return complete recommendation
    """
    enrichment = {}

    # 1. Phase 1 enrichment
    try:
        from pipelines.institutional_engine import enrich_signal
        signal = enrich_signal(signal, prices)
        enrichment = signal.get("enrichment", {})
        for key in ("breadth", "relative_strength", "sector_rotation",
                     "multi_timeframe", "options_flow", "intraday_features",
                     "portfolio_risk"):
            if key in signal:
                enrichment[key] = signal[key]
    except Exception as e:
        print(f"  [Phase1] Enrichment partial: {e}")

    # 2. Regime V2
    from engines.regime_v2 import classify_regime_v2, adapt_signal
    regime = _safe(classify_regime_v2, module="regime_v2") or {}
    enrichment["regime_v2"] = regime
    signal = _safe(adapt_signal, signal, regime, module="regime_adapt") or signal

    # 3. Conviction grading
    from engines.conviction_v2 import grade_signal
    grade_result = _safe(grade_signal, signal, enrichment, module="conviction_v2")
    if not grade_result:
        grade_result = {"grade": "NO_TRADE", "composite_score": 0,
                        "disqualifiers": ["Conviction engine failed"]}

    # 4. Trade quality
    from engines.trade_quality import compute_trade_quality
    quality_result = _safe(compute_trade_quality, signal, enrichment,
                           module="trade_quality")
    if not quality_result:
        quality_result = {"trade_quality_score": 0, "opportunity_score": 0,
                          "component_scores": {}, "quality_grade": "poor", "flags": []}

    # 5. Psychology gate
    from engines.psychology_engine import can_trade, load_state
    if psychology_state is None:
        psychology_state = _safe(load_state, module="psychology") or {}
    psych_result = _safe(can_trade, capital, signal, psychology_state,
                         module="psychology")
    if not psych_result:
        psych_result = {"allowed": False, "reason": "Psychology engine unavailable"}

    # Check gates
    grade = grade_result.get("grade", "NO_TRADE")
    is_no_trade = grade == "NO_TRADE" or not psych_result.get("allowed", False)

    if is_no_trade and grade != "NO_TRADE":
        grade_result["grade"] = "NO_TRADE"
        grade_result.setdefault("disqualifiers", []).append(
            f"Psychology gate: {psych_result.get('reason', 'blocked')}")

    # 6. Capital allocation
    sizing_result = {"shares": 0, "capital_allocated": 0, "reason": "No trade"}
    if not is_no_trade:
        from engines.capital_allocation import compute_position_size
        portfolio = {"positions": positions or [], "current_drawdown": 0}
        sizing_result = _safe(
            compute_position_size,
            signal, grade, grade_result.get("conviction", 50),
            quality_result.get("trade_quality_score", 50),
            regime.get("macro_regime", "unknown"),
            portfolio, capital, psychology_state,
            module="capital_alloc") or sizing_result

    # 7. Strategy selection
    strategy = "no_trade"
    if not is_no_trade:
        strategy = select_strategy(signal, regime, enrichment)

    # 8. Explainability
    from engines.explainability import explain_trade, explain_no_trade
    if is_no_trade:
        explanation = _safe(explain_no_trade, signal, enrichment,
                            grade_result, psych_result,
                            module="explain") or {}
    else:
        explanation = _safe(explain_trade, signal, enrichment,
                            grade_result, quality_result,
                            sizing_result, regime,
                            module="explain") or {}

    return {
        "symbol": signal.get("symbol", ""),
        "action": signal.get("final_action", "hold"),
        "strategy": strategy,
        "grade": grade_result.get("grade", "NO_TRADE"),
        "composite_score": grade_result.get("composite_score", 0),
        "confidence": round(signal.get("fused_confidence",
                                       signal.get("confidence", 0)), 4),
        "conviction": round(grade_result.get("conviction", 0), 1),
        "trade_quality": quality_result.get("trade_quality_score", 0),
        "opportunity_score": quality_result.get("opportunity_score", 0),
        "quality_grade": quality_result.get("quality_grade", "poor"),
        "position_size": sizing_result,
        "regime": {
            "macro": regime.get("macro_regime", "unknown"),
            "day_type": regime.get("day_type", "unknown"),
            "playbook": regime.get("playbook", ""),
        },
        "psychology": {
            "allowed": psych_result.get("allowed", False),
            "risk_state": psych_result.get("risk_state", "unknown"),
            "psychology_score": psych_result.get("psychology_score", 0),
            "discipline_score": psych_result.get("discipline_score", 0),
        },
        "grade_detail": grade_result,
        "quality_detail": quality_result,
        "explanation": explanation,
        "signal": {
            "entry": signal.get("price", signal.get("entry_zone")),
            "stop_loss": signal.get("stop_loss"),
            "target": signal.get("target"),
        },
        "timestamp": datetime.now().isoformat(),
    }


# ── Batch Recommendations ─────────────────────────────

def generate_recommendations(signals, capital=1_000_000, portfolio=None):
    """Process multiple signals through the full pipeline.

    Returns {timestamp, regime, psychology, recommendations, no_trade_reasons,
             market_summary, stats}.
    """
    from engines.regime_v2 import classify_regime_v2
    from engines.psychology_engine import load_state

    regime = _safe(classify_regime_v2, module="regime_v2") or {}
    psych_state = _safe(load_state, module="psychology") or {}
    positions = portfolio.get("positions", []) if isinstance(portfolio, dict) else []

    recommendations = []
    no_trade_reasons = []

    for sig in signals:
        result = full_pipeline(sig, positions=positions, capital=capital,
                               psychology_state=psych_state)
        if result["grade"] == "NO_TRADE":
            no_trade_reasons.append({
                "symbol": result["symbol"],
                "reason": (result.get("explanation", {}).get("specific_reasons", ["No trade"])
                           if isinstance(result.get("explanation"), dict)
                           else ["No trade"]),
                "composite_score": result["composite_score"],
                "nearest_grade": result.get("explanation", {}).get("nearest_grade", "?"),
            })
        else:
            recommendations.append(result)

    recommendations.sort(key=lambda r: r["composite_score"], reverse=True)
    for i, rec in enumerate(recommendations):
        rec["rank"] = i + 1

    total = len(signals)
    traded = len(recommendations)
    filtered = len(no_trade_reasons)

    return {
        "timestamp": datetime.now().isoformat(),
        "regime": {
            "macro": regime.get("macro_regime", "unknown"),
            "day_type": regime.get("day_type", "unknown"),
            "playbook": regime.get("playbook", ""),
        },
        "psychology": {
            "risk_state": psych_state.get("risk_state", "unknown"),
            "psychology_score": psych_state.get("psychology_score", 100),
        },
        "recommendations": recommendations,
        "no_trade_reasons": no_trade_reasons,
        "market_summary": (
            f"Regime: {regime.get('macro_regime', '?')} | "
            f"Day: {regime.get('day_type', '?')} | "
            f"Signals: {total} → Traded: {traded} | Filtered: {filtered} "
            f"({filtered/total*100:.0f}% rejection rate)" if total > 0 else "No signals"
        ),
        "stats": {
            "total_signals": total,
            "recommendations": traded,
            "filtered": filtered,
            "rejection_rate": round(filtered / total, 3) if total > 0 else 0,
        },
    }


def rank_recommendations(recs):
    """Rank by composite of grade, conviction, quality, risk-reward."""
    def score(r):
        grade_pts = {"A+": 100, "A": 80, "B": 60, "C": 40}.get(r.get("grade"), 0)
        return grade_pts + r.get("conviction", 0) * 0.3 + r.get("trade_quality", 0) * 0.2
    return sorted(recs, key=score, reverse=True)


# ── CLI ────────────────────────────────────────────────

if __name__ == "__main__":
    from memory.phase2_schema import migrate
    migrate()

    print("=" * 60)
    print("  INSTITUTIONAL RECOMMENDATION ENGINE")
    print("=" * 60)

    test_signals = [
        {"symbol": "RELIANCE", "final_action": "buy",
         "fused_confidence": 0.78, "price": 2800,
         "stop_loss": 2755, "target": 2920, "atr": 40,
         "volatility": "medium", "agreement": "strong"},
        {"symbol": "TCS", "final_action": "buy",
         "fused_confidence": 0.62, "price": 3500,
         "stop_loss": 3400, "target": 3650, "atr": 55,
         "volatility": "medium", "agreement": "moderate"},
        {"symbol": "SMALLCAP", "final_action": "buy",
         "fused_confidence": 0.45, "price": 100,
         "stop_loss": 92, "target": 108, "atr": 8,
         "volatility": "high", "agreement": "weak"},
    ]

    result = generate_recommendations(test_signals)
    print(f"\n  {result['market_summary']}")
    print(f"\n  Recommendations ({len(result['recommendations'])}):")
    for r in result["recommendations"]:
        print(f"    #{r['rank']} {r['symbol']} — {r['strategy']} "
              f"Grade {r['grade']} (score {r['composite_score']:.0f})")

    print(f"\n  No Trade ({len(result['no_trade_reasons'])}):")
    for nt in result["no_trade_reasons"]:
        print(f"    {nt['symbol']}: {nt['reason']}")

    out_path = os.path.join(OUTPUT_DIR, "phase2_recommendations.json")
    json.dump(result, open(out_path, "w"), indent=2, default=str)
    print(f"\n  Saved → {out_path}")

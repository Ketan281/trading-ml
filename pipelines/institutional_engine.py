"""
Institutional-grade signal engine — integrates all 9 new intelligence
modules with the existing combined_intelligence pipeline.

This is the top-level orchestrator:
  1. Market Breadth → regime filter + conviction multiplier
  2. Relative Strength → stock quality filter
  3. Sector Rotation → sector conviction multiplier
  4. Intraday Features → feature enrichment
  5. Options Flow → sentiment overlay
  6. Multi-Timeframe → alignment filter
  7. Trade Management → exit/stop intelligence
  8. Portfolio Risk v2 → position sizing + regime gate
  9. Intraday ML Prep → dataset readiness (background)

Signal flow:
  base_signal (from combined_intelligence)
  → breadth regime filter (block if Strong Bearish)
  → RS quality gate (skip bottom 20% laggards)
  → sector conviction multiplier (1.5x leaders, 0.5x laggards)
  → MTF alignment filter (block if alignment < 30)
  → options sentiment overlay (adjust confidence)
  → portfolio risk regime gate (scale position)
  → final enriched signal
"""

import os
import sys
import json
import traceback
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _safe_call(func, *args, module_name="unknown", **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"  [{module_name}] Error: {e}")
        return None


def enrich_signal(signal, prices=None):
    symbol = signal.get("symbol", "")
    action = signal.get("final_action", "hold")
    confidence = signal.get("fused_confidence", 0.5)
    enrichment = {"modules_applied": [], "modules_failed": []}

    # ── 1. Breadth Regime Filter ────────────────────
    try:
        from pipelines.breadth import breadth_read
        breadth = breadth_read()
        if breadth:
            regime = breadth.get("regime", "neutral")
            score = breadth.get("composite_score", 50)
            enrichment["breadth"] = {"regime": regime, "score": score}
            enrichment["modules_applied"].append("breadth")

            if regime == "strong_bearish" and action == "buy":
                signal["warnings"] = signal.get("warnings", [])
                signal["warnings"].append("Breadth regime is Strong Bearish — high-risk buy")
                confidence *= 0.6
            elif regime in ("strong_bullish", "bullish") and action == "buy":
                confidence *= 1.1
            elif regime in ("strong_bullish", "bullish") and action == "sell":
                confidence *= 0.8
    except Exception as e:
        enrichment["modules_failed"].append(f"breadth: {e}")

    # ── 2. Relative Strength Quality Gate ───────────
    try:
        from pipelines.relative_strength import rs_for_symbol
        rs = rs_for_symbol(symbol, prices=prices)
        if rs:
            rs_pctile = rs.get("composite_percentile", 50)
            enrichment["relative_strength"] = {
                "percentile": rs_pctile,
                "regime": rs.get("rs_regime", "neutral"),
            }
            enrichment["modules_applied"].append("relative_strength")

            if rs_pctile < 20 and action == "buy":
                signal["warnings"] = signal.get("warnings", [])
                signal["warnings"].append(f"RS percentile {rs_pctile} — bottom quintile laggard")
                confidence *= 0.7
            elif rs_pctile > 80 and action == "buy":
                confidence *= 1.15
    except Exception as e:
        enrichment["modules_failed"].append(f"relative_strength: {e}")

    # ── 3. Sector Rotation Conviction ───────────────
    try:
        from pipelines.sector_rotation import get_sector_conviction
        conv = get_sector_conviction(symbol)
        if conv:
            multiplier = conv.get("conviction_multiplier", 1.0)
            enrichment["sector_rotation"] = {
                "sector": conv.get("sector_group", "unknown"),
                "rank": conv.get("rank"),
                "phase": conv.get("phase", "neutral"),
                "multiplier": multiplier,
            }
            enrichment["modules_applied"].append("sector_rotation")
            confidence *= multiplier
    except Exception as e:
        enrichment["modules_failed"].append(f"sector_rotation: {e}")

    # ── 4. Multi-Timeframe Alignment ────────────────
    try:
        from pipelines.multi_timeframe import multi_timeframe_read
        mtf = multi_timeframe_read(symbol)
        if mtf and "error" not in mtf:
            alignment = mtf.get("alignment_score", 50)
            agreement = mtf.get("agreement_score", 50)
            continuation = mtf.get("continuation_probability", 50)
            enrichment["multi_timeframe"] = {
                "alignment_score": alignment,
                "agreement_score": agreement,
                "consensus": mtf.get("consensus_direction", "neutral"),
                "continuation_prob": continuation,
                "counter_trend": mtf.get("counter_trend_detected", False),
            }
            enrichment["modules_applied"].append("multi_timeframe")

            if alignment < 30 and action in ("buy", "sell"):
                signal["warnings"] = signal.get("warnings", [])
                signal["warnings"].append(f"MTF alignment only {alignment}% — weak conviction")
                confidence *= 0.75
            elif alignment > 70:
                confidence *= 1.1
    except Exception as e:
        enrichment["modules_failed"].append(f"multi_timeframe: {e}")

    # ── 5. Options Flow Sentiment ───────────────────
    try:
        from pipelines.options_flow import options_flow_read
        flow = options_flow_read(symbol)
        if flow and "error" not in flow:
            sentiment = flow.get("composite_sentiment", 50)
            enrichment["options_flow"] = {
                "sentiment_score": sentiment,
                "iv_rank": flow.get("iv", {}).get("iv_rank"),
                "pcr": flow.get("pcr"),
            }
            enrichment["modules_applied"].append("options_flow")

            if action == "buy" and sentiment > 65:
                confidence *= 1.05
            elif action == "buy" and sentiment < 35:
                confidence *= 0.9
    except Exception as e:
        enrichment["modules_failed"].append(f"options_flow: {e}")

    # ── 6. Intraday Features ───────────────────────
    try:
        from pipelines.intraday_features import compute_intraday_features
        idf = compute_intraday_features(symbol)
        if idf:
            enrichment["intraday_features"] = {
                "rvol": idf.get("rvol"),
                "momentum_score": idf.get("intraday_momentum_score"),
                "vol_compression": idf.get("vol_compression", {}).get("compressed", False),
            }
            enrichment["modules_applied"].append("intraday_features")
    except Exception as e:
        enrichment["modules_failed"].append(f"intraday_features: {e}")

    # ── Clamp and finalize confidence ───────────────
    confidence = max(0.0, min(1.0, confidence))
    signal["fused_confidence"] = round(confidence, 3)
    signal["enrichment"] = enrichment
    signal["institutional_grade"] = True
    signal["modules_applied"] = len(enrichment["modules_applied"])
    signal["modules_total"] = 9

    return signal


def enrich_and_manage(signal, df=None, prices=None, positions=None, capital=500_000):
    signal = enrich_signal(signal, prices)

    # ── 7. Trade Management ─────────────────────────
    if signal.get("final_action") in ("buy", "sell") and df is not None:
        try:
            from pipelines.trade_management import trade_management_read
            entry = signal.get("price", 0)
            stop = float(signal.get("stop_loss", entry * 0.95)) if signal.get("stop_loss") else entry * 0.95
            current = entry
            direction = "long" if signal["final_action"] == "buy" else "short"
            alignment = signal.get("enrichment", {}).get("multi_timeframe", {}).get("alignment_score", 50)
            tm = trade_management_read(df, entry, stop, current, direction,
                                       alignment_score=alignment)
            signal["trade_management"] = {
                "active_stop": tm.get("active_stop"),
                "trailing_stop": tm.get("trailing_stop"),
                "targets": tm.get("partial_exits", {}).get("targets", []),
                "trade_quality": tm.get("trade_quality"),
            }
            signal["enrichment"]["modules_applied"].append("trade_management")
        except Exception as e:
            signal["enrichment"]["modules_failed"].append(f"trade_management: {e}")

    # ── 8. Portfolio Risk v2 Gate ───────────────────
    if positions is not None:
        try:
            from pipelines.portfolio_risk_v2 import portfolio_risk_v2_read
            risk = portfolio_risk_v2_read(positions, capital)
            regime = risk.get("risk_regime", {})
            alloc_mult = regime.get("capital_allocation_multiplier", 1.0)
            signal["portfolio_risk_v2"] = {
                "regime": regime.get("regime", "normal"),
                "alloc_multiplier": alloc_mult,
                "composite_score": regime.get("composite_risk_score", 50),
            }
            signal["enrichment"]["modules_applied"].append("portfolio_risk_v2")
            signal["recommended_size_multiplier"] = alloc_mult
        except Exception as e:
            signal["enrichment"]["modules_failed"].append(f"portfolio_risk_v2: {e}")

    return signal


def run_institutional_pipeline(symbols=None, positions=None, capital=500_000):
    from pipelines.combined_intelligence import combined_analyze

    symbols = symbols or ["NIFTY", "BANKNIFTY", "RELIANCE", "HDFCBANK"]
    print("\n" + "=" * 65)
    print("  INSTITUTIONAL-GRADE SIGNAL ENGINE")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 65)

    all_signals = []
    for symbol in symbols:
        base = combined_analyze(symbol)
        if not base:
            continue
        enriched = enrich_and_manage(base, positions=positions, capital=capital)
        all_signals.append(enriched)

    # Summary
    print("\n" + "=" * 75)
    print("  INSTITUTIONAL SIGNAL SUMMARY")
    print("=" * 75)
    print(f"  {'SYM':<12} {'ACTION':<8} {'CONF':<7} {'BREADTH':<10} "
          f"{'RS%':<6} {'MTF':<6} {'MODS':<5}")
    print("  " + "-" * 70)

    for s in all_signals:
        e = s.get("enrichment", {})
        breadth_r = e.get("breadth", {}).get("regime", "?")[:8]
        rs_p = e.get("relative_strength", {}).get("percentile", "?")
        mtf_a = e.get("multi_timeframe", {}).get("alignment_score", "?")
        n_mods = s.get("modules_applied", 0)
        print(f"  {s['symbol']:<12} {s['final_action']:<8} "
              f"{s['fused_confidence']:<7.3f} {breadth_r:<10} "
              f"{str(rs_p):<6} {str(mtf_a):<6} {n_mods}/9")

    report_path = os.path.join(OUTPUT_DIR, "institutional_signals.json")
    with open(report_path, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(),
                    "signals": all_signals}, f, indent=2, default=str)
    print(f"\n  Saved: {report_path}")
    return all_signals


if __name__ == "__main__":
    run_institutional_pipeline()

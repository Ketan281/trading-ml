import json
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipelines.indicators   import analyze
from pipelines.intelligence import query_intelligence
from models.ml_models       import predict as ml_predict

OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYMBOLS = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS"]

# ── Signal Fusion Logic ───────────────────────────────
def fuse_signals(ai_decision, ml_prediction):
    ai_action     = ai_decision.get("action", "hold")
    ai_confidence = ai_decision.get("confidence", 0.5)
    ai_risk       = ai_decision.get("risk_level", "medium")
    ml_regime     = ml_prediction.get("ml_regime", "sideways")
    ml_confidence = ml_prediction.get("confidence", 0.5)

    # Agreement scoring
    agreement = False
    if ai_action == "buy"  and ml_regime == "uptrend":
        agreement = True
    if ai_action == "sell" and ml_regime == "downtrend":
        agreement = True
    if ai_action == "hold" and ml_regime == "sideways":
        agreement = True

    # Fused confidence
    if agreement:
        fused_confidence = round(
            (ai_confidence * 0.6) + (ml_confidence * 0.4), 3
        )
        signal_strength = "strong"
    else:
        fused_confidence = round(
            min(ai_confidence, ml_confidence) * 0.8, 3
        )
        signal_strength = "weak"

    # Risk override
    final_action = ai_action
    if ai_risk == "extreme":
        final_action     = "avoid"
        signal_strength  = "blocked"
        fused_confidence = 0.0

    # Build consensus
    if agreement:
        consensus = f"AGREE — Both ML and AI signal {ai_action}"
    else:
        consensus = (f"DISAGREE — AI says {ai_action} "
                     f"but ML sees {ml_regime}")

    return {
        "final_action"    : final_action,
        "fused_confidence": fused_confidence,
        "signal_strength" : signal_strength,
        "agreement"       : agreement,
        "consensus"       : consensus,
        "ai_action"       : ai_action,
        "ai_confidence"   : ai_confidence,
        "ml_regime"       : ml_regime,
        "ml_confidence"   : ml_confidence,
        "risk_level"      : ai_risk
    }

# ── Risk Gate ─────────────────────────────────────────
def apply_risk_gate(fused, indicators):
    warnings = []
    blocked  = False

    rsi         = indicators.get("rsi", 50)
    volatility  = indicators.get("volatility", "medium")
    bollinger   = indicators.get("bollinger", "inside_bands")
    confidence  = fused.get("fused_confidence", 0)
    action      = fused.get("final_action")

    # RSI extreme check
    if rsi > 78 and action == "buy":
        warnings.append("⚠ RSI > 78 — Overbought, risky to buy")
        blocked = True
    if rsi < 25 and action == "sell":
        warnings.append("⚠ RSI < 25 — Oversold, risky to sell")
        blocked = True

    # Volatility check
    if volatility == "high" and confidence < 0.6:
        warnings.append("⚠ High volatility with low confidence")
        blocked = True

    # Bollinger check
    if bollinger == "above_upper" and action == "buy":
        warnings.append("⚠ Price above Bollinger upper band")

    # Confidence check
    if confidence < 0.5:
        warnings.append("⚠ Low fused confidence — consider avoiding")
        blocked = True

    # Disagreement check
    if not fused.get("agreement") and action in ["buy", "sell"]:
        warnings.append("⚠ ML and AI disagree — reduce position size")

    return {
        "blocked" : blocked,
        "warnings": warnings,
        "final_action": "avoid" if blocked else fused["final_action"]
    }

# ── Analyze One Symbol ────────────────────────────────
def combined_analyze(symbol):
    print(f"\n{'─' * 55}")
    print(f"  🔍 Analyzing {symbol}")
    print(f"{'─' * 55}")

    # Step 1 — Get Indicators
    indicators = analyze(symbol)
    if not indicators:
        print(f"  ⚠ No indicator data for {symbol}")
        return None

    # Step 2 — Get AI Decision
    print(f"  🧠 Querying AI...")
    ai_decision = query_intelligence(indicators)
    if not ai_decision:
        print(f"  ⚠ No AI decision for {symbol}")
        return None

    # Step 3 — Get ML Prediction
    print(f"  🤖 Running ML model...")
    ml_prediction = ml_predict(symbol, indicators)
    if not ml_prediction:
        print(f"  ⚠ No ML prediction for {symbol}")
        ml_prediction = {
            "ml_regime" : "sideways",
            "confidence": 0.5
        }

    # Step 4 — Fuse Signals
    fused = fuse_signals(ai_decision, ml_prediction)

    # Step 5 — Apply Risk Gate
    risk_gate = apply_risk_gate(fused, indicators)

    # Step 6 — Build Final Signal
    final_signal = {
        "symbol"          : symbol,
        "timestamp"       : datetime.now().isoformat(),
        "price"           : indicators.get("price"),
        "final_action"    : risk_gate["final_action"],
        "fused_confidence": fused["fused_confidence"],
        "signal_strength" : fused["signal_strength"],
        "agreement"       : fused["agreement"],
        "consensus"       : fused["consensus"],
        "risk_level"      : fused["risk_level"],
        "entry_zone"      : ai_decision.get("entry_zone"),
        "stop_loss"       : ai_decision.get("stop_loss"),
        "warnings"        : risk_gate["warnings"],
        "blocked"         : risk_gate["blocked"],
        "components": {
            "ai": {
                "action"    : ai_decision.get("action"),
                "confidence": ai_decision.get("confidence"),
                "condition" : ai_decision.get("market_condition"),
                "reasoning" : ai_decision.get("reasoning", [])
            },
            "ml": {
                "regime"    : ml_prediction.get("ml_regime"),
                "confidence": ml_prediction.get("confidence")
            }
        }
    }

    # Print result
    status = "🔴 BLOCKED" if risk_gate["blocked"] else "🟢 ACTIVE"
    print(f"\n  {status} — {symbol}")
    print(f"  Final Action    : {final_signal['final_action'].upper()}")
    print(f"  Fused Confidence: {final_signal['fused_confidence']}")
    print(f"  Signal Strength : {final_signal['signal_strength']}")
    print(f"  Consensus       : {final_signal['consensus']}")
    print(f"  Entry Zone      : {final_signal['entry_zone']}")
    print(f"  Stop Loss       : {final_signal['stop_loss']}")

    if risk_gate["warnings"]:
        print(f"\n  Warnings:")
        for w in risk_gate["warnings"]:
            print(f"    {w}")

    # Save signal
    out_path = os.path.join(
        OUTPUT_DIR,
        f"{symbol}_combined_signal.json"
    )
    with open(out_path, "w") as f:
        json.dump(final_signal, f, indent=2)

    return final_signal

# ── Run All Symbols ───────────────────────────────────
def run_combined_intelligence():
    print("\n" + "=" * 55)
    print("  Trading AI — Combined Intelligence (ML + AI)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    all_signals = []

    for symbol in SYMBOLS:
        signal = combined_analyze(symbol)
        if signal:
            all_signals.append(signal)

    # Final Summary Table
    print("\n\n" + "=" * 65)
    print("  COMBINED INTELLIGENCE SUMMARY")
    print("=" * 65)
    print(f"  {'SYMBOL':<12} {'ACTION':<10} {'CONF':<8} "
          f"{'STRENGTH':<10} {'STATUS'}")
    print("  " + "-" * 60)

    for s in all_signals:
        status = "🔴 BLOCKED" if s["blocked"] else "🟢 ACTIVE"
        print(f"  {s['symbol']:<12} "
              f"{s['final_action'].upper():<10} "
              f"{s['fused_confidence']:<8} "
              f"{s['signal_strength']:<10} "
              f"{status}")

    # Best opportunity
    active = [s for s in all_signals
              if not s["blocked"]
              and s["final_action"] in ["buy", "sell"]]

    if active:
        best = max(active, key=lambda x: x["fused_confidence"])
        print(f"\n  🎯 Best Opportunity → {best['symbol']}")
        print(f"     Action     : {best['final_action'].upper()}")
        print(f"     Confidence : {best['fused_confidence']}")
        print(f"     Entry Zone : {best['entry_zone']}")
        print(f"     Stop Loss  : {best['stop_loss']}")
    else:
        print(f"\n  ⏳ No strong active signals right now — stay patient")

    # Save combined report
    report_path = os.path.join(
        OUTPUT_DIR, "combined_report.json"
    )
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "signals"  : all_signals
        }, f, indent=2)

    print(f"\n  ✅ Combined report → {report_path}\n")
    return all_signals

# ── Entry Point ───────────────────────────────────────
if __name__ == "__main__":
    run_combined_intelligence()
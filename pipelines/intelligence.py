import json
import os
import sys
from datetime import datetime

try:
    import ollama
except ImportError:
    ollama = None

ROOT = os.path.dirname(os.path.dirname(
       os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.regime_detector          import detect_full_regime
from pipelines.decision_engine        import decide
from pipelines.hallucination_control import validate_decision
from memory.memory_retrieval         import build_memory_context
from pipelines.event_awareness       import (
    build_event_context,
    apply_event_adjustments
)
from pipelines.portfolio_risk        import (
    check_portfolio_risk,
    run_portfolio_risk_all
)

OUTPUT_DIR     = os.path.join(ROOT, "outputs")
STRATEGIES_DIR = os.path.join(ROOT, "strategies")
os.makedirs(OUTPUT_DIR,     exist_ok=True)
os.makedirs(STRATEGIES_DIR, exist_ok=True)

# Minimum proven walk-forward edge (over the majority baseline) a model
# must have before we let it influence real decisions. A model with no
# edge is noise at best and harmful at worst, so we refuse to trust it.
MIN_ML_EDGE = 0.02   # +2 percentage points over baseline


# ── Learned ML Signal (optional, edge-gated) ──────────
def get_ml_signal(symbol):
    """Fetch the trained model's prediction for a symbol. Failure-safe and
    edge-gated: returns None if the ML stack is unavailable, no model is
    trained, or the model has not proven a positive out-of-sample edge —
    so the engine falls back to rules-only rather than trusting noise."""
    if not symbol:
        return None
    try:
        from models.ml_models import predict
        sig = predict(symbol)
    except Exception as e:
        print(f"  ⚠ ML signal skipped: {e}")
        return None

    if not sig:
        return None

    edge = sig.get("edge")
    if edge is None or edge < MIN_ML_EDGE:
        print(
            f"  🚫 ML signal ignored — no proven edge "
            f"(edge={edge}). Using rules only."
        )
        return None

    return sig


# ── Narration Prompt (explanation only) ──────────────
def build_narration_prompt(decision, data, regime=None):
    memory_ctx = data.get("memory_context", "No historical memory yet.")
    event_ctx  = data.get("event_context",  "No event data.")
    regime_bias = "unknown"
    if regime:
        regime_bias = regime.get("fusion", {}).get(
            "primary_bias", "neutral")

    return f"""You are a veteran Indian-markets trader writing a short note
explaining a trade decision that has ALREADY been made by a rules engine.

Do NOT change any numbers. Do NOT second-guess the action. Only explain
the decision clearly, like a 20-year desk trader briefing a junior.

THE DECISION (fixed — explain, do not alter):
  Symbol     : {decision['symbol']}
  Action     : {decision['action']}
  Confidence : {decision['confidence']}
  Condition  : {decision['market_condition']}
  Entry      : {decision['entry_zone']}
  Stop Loss  : {decision['stop_loss']}
  Target     : {decision['target']}
  Risk Level : {decision['risk_level']}

SUPPORTING DATA:
  Price={data.get('price')} RSI={data.get('rsi')} MACD={data.get('macd')}
  Trend={data.get('trend')} EMA20={data.get('ema_20')} EMA50={data.get('ema_50')}
  VWAP={data.get('vwap')} Bollinger={data.get('bollinger')}
  Volatility={data.get('volatility')} ATR={data.get('atr')}
  Regime bias={regime_bias}

CONTEXT:
{memory_ctx}
{event_ctx}

Return ONLY valid JSON, no extra text:
{{
  "reasoning":    ["short point 1", "short point 2", "short point 3"],
  "regime_notes": "<one sentence on regime fit, or empty>",
  "memory_notes": "<one sentence on what history suggests, or empty>",
  "event_notes":  "<one sentence on event impact, or empty>"
}}"""


# ── Decide (deterministic) + Narrate (LLM) ────────────
def query_intelligence(data, regime=None):
    """
    Make the trade decision with the deterministic rule engine, then ask
    the LLM ONLY to write the human-readable reasoning. The model never
    sets action / confidence / entry / stop-loss / target, so it cannot
    hallucinate the trade.
    """
    print(f"\n  🧠 Analyzing {data['symbol']}...")

    # ── 1. Learned ML signal (optional, failure-safe) ─
    ml_signal = get_ml_signal(data.get("symbol"))
    if ml_signal:
        print(
            f"  🤖 ML: {ml_signal['ml_regime']} "
            f"(conf {ml_signal['confidence']})"
        )

    # ── 2. Deterministic decision (the trade itself) ──
    decision = decide(data, ml_signal)

    # ── 3. Optional LLM narration (explanation only) ──
    narration = narrate_decision(decision, data, regime)
    if narration:
        # Only replace soft, text-only fields. Numbers stay untouched.
        if narration.get("reasoning"):
            decision["reasoning"] = narration["reasoning"]
        for key in ("regime_notes", "memory_notes", "event_notes"):
            if narration.get(key):
                decision[key] = narration[key]

    return decision


def narrate_decision(decision, data, regime=None):
    """Ask the LLM to explain a decision in plain English. Failure-safe:
    returns None if the model errors or returns unusable output, in which
    case the engine's own reasoning is kept."""
    if ollama is None:
        return None
    prompt = build_narration_prompt(decision, data, regime)
    try:
        response = ollama.chat(
            model    = "qwen2.5:1.5b",
            messages = [{"role": "user", "content": prompt}],
            options  = {"temperature": 0.2, "num_predict": 500},
        )
    except Exception as e:
        print(f"  ⚠ Narration skipped (LLM error): {e}")
        return None

    raw = response["message"]["content"].strip()
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                raw = part
                break

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print("  ⚠ Narration not JSON — keeping engine reasoning.")
        return None

# ── Validate Against Regime ───────────────────────────
def validate_against_regime(decision, regime):
    if not regime or not decision:
        return decision

    fusion     = regime.get("fusion",     {})
    volatility = regime.get("volatility", {})
    expiry     = regime.get("expiry",     {})
    warnings   = decision.get("warnings",   [])
    confidence = float(
        decision.get("confidence", 0.5)
    )
    action     = decision.get("action", "hold")

    # Rule 1 — Not tradeable
    if not fusion.get("tradeable", True):
        decision["action"]        = "avoid"
        decision["confidence"]    = 0.0
        decision["position_size"] = "avoid"
        warnings.append(
            "Market not tradeable"
        )

    # Rule 2 — Extreme caution
    if fusion.get("caution_level") == "extreme":
        decision["action"]     = "avoid"
        decision["confidence"] = max(
            0.0, confidence - 0.3
        )
        warnings.append(
            "Extreme caution — expiry + high vol"
        )

    # Rule 3 — Extreme volatility
    if volatility.get("vol_regime") == "extreme":
        if action in ["buy", "sell"]:
            decision["action"]        = "reduce_exposure"
            decision["confidence"]    = max(
                0.0, confidence - 0.2
            )
            decision["position_size"] = "quarter"
            warnings.append(
                "Extreme volatility — quarter position"
            )

    # Rule 4 — Expiry day
    if expiry.get("is_weekly_expiry"):
        decision["confidence"] = max(
            0.0,
            float(decision.get("confidence", 0.5))
            - 0.1
        )
        warnings.append(
            "Weekly expiry — avoid overnight"
        )

    # Rule 5 — Regime misalignment
    bias = fusion.get("primary_bias", "neutral")
    if action == "buy" and bias == "bearish":
        decision["confidence"]       = max(
            0.0,
            float(decision.get("confidence", 0.5))
            - 0.15
        )
        decision["regime_alignment"] = "misaligned"
        warnings.append(
            "BUY vs BEARISH regime"
        )
    elif action == "sell" and bias == "bullish":
        decision["confidence"]       = max(
            0.0,
            float(decision.get("confidence", 0.5))
            - 0.15
        )
        decision["regime_alignment"] = "misaligned"
        warnings.append(
            "SELL vs BULLISH regime"
        )

    # Rule 6 — Strategy avoid list
    avoid_list = fusion.get("avoid", [])
    strategy   = decision.get("strategy", "")
    for avoid in avoid_list:
        if avoid.replace("_", " ") in \
                strategy.replace("_", " "):
            warnings.append(
                f"Strategy in avoid list: {strategy}"
            )
            decision["confidence"] = max(
                0.0,
                float(decision.get("confidence", 0.5))
                - 0.2
            )

    decision["warnings"] = warnings
    return decision

# ── Apply Memory Adjustment ───────────────────────────
def apply_memory_adjustment(decision, memory_meta):
    if not memory_meta or not decision:
        return decision

    stats      = memory_meta.get("stats")
    if not stats:
        return decision

    confidence = float(
        decision.get("confidence", 0.5)
    )
    warnings   = decision.get("warnings", [])
    win_rate   = stats.get("win_rate",     50)
    sample     = stats.get("with_outcomes", 0)

    if sample < 3:
        return decision

    if win_rate >= 65:
        boost = min(0.10, (win_rate - 65) / 100)
        decision["confidence"] = min(
            0.95, confidence + boost
        )
        decision["memory_notes"] = (
            f"Memory boost: {win_rate}% win rate "
            f"in {sample} similar trades"
        )
    elif win_rate < 40:
        penalty = min(0.15, (40 - win_rate) / 100)
        decision["confidence"] = max(
            0.0, confidence - penalty
        )
        warnings.append(
            f"Memory warning: {win_rate}% win rate "
            f"in {sample} similar trades"
        )
        decision["memory_notes"] = (
            f"Memory penalty: {win_rate}% win rate"
        )

    decision["warnings"] = warnings
    return decision

# ── Print Decision ────────────────────────────────────
def print_decision(result):
    if not result:
        return

    symbol  = result.get("symbol", "?")
    vstatus = result.get(
        "validation_status", "UNKNOWN"
    )
    vicon   = {
        "CLEAN"       : "✅",
        "MINOR_ISSUES": "⚠️",
        "DEGRADED"    : "🟠",
        "HIGH_RISK"   : "🔴",
        "BLOCKED"     : "❌"
    }.get(vstatus, "❓")

    print(f"\n  {'─' * 55}")
    print(f"  {vicon} {symbol} — {vstatus}")
    print(f"  {'─' * 55}")
    print(f"  Condition    : "
          f"{result.get('market_condition')}")
    print(f"  Regime       : "
          f"{result.get('regime_alignment', 'N/A')}")
    print(f"  Action       : {result.get('action')}")
    print(f"  Strategy     : "
          f"{result.get('strategy', 'N/A')}")
    print(f"  Confidence   : "
          f"{result.get('confidence')}")
    print(f"  Val Penalty  : "
          f"{result.get('validation_penalty', 0)}")
    print(f"  Risk Level   : "
          f"{result.get('risk_level')}")
    print(f"  Position Size: "
          f"{result.get('position_size', 'N/A')}")
    print(f"  Entry Zone   : "
          f"{result.get('entry_zone')}")
    print(f"  Stop Loss    : "
          f"{result.get('stop_loss')}")
    print(f"  Target       : "
          f"{result.get('target', 'N/A')}")

    # Portfolio info
    port_status = result.get("portfolio_status")
    if port_status:
        picon = {
            "APPROVED": "✅",
            "REDUCED" : "⚠️",
            "BLOCKED" : "❌"
        }.get(port_status, "❓")
        print(f"  Portfolio    : "
              f"{picon} {port_status} | "
              f"Size: ₹{result.get('recommended_size', 0):,.0f}")

    # Event context
    ev = result.get("event_summary", {})
    if ev and ev.get("is_event_day"):
        print(f"  ⚠ EVENT DAY !")
    if ev and ev.get("risk", "normal") != "normal":
        print(f"  Event Risk   : "
              f"{ev.get('risk', 'normal').upper()}")

    # Warnings
    warnings = [
        w for w in result.get("warnings", []) if w
    ]
    if warnings:
        print(f"\n  ⚠ Warnings:")
        for w in warnings[:5]:
            print(f"    → {w}")

    # Reasoning
    print(f"\n  💭 Reasoning:")
    for r in result.get("reasoning", []):
        print(f"    - {r}")

    # Notes
    for key, label in [
        ("regime_notes", "🗺  Regime"),
        ("memory_notes", "🧠 Memory"),
        ("event_notes",  "📅 Event")
    ]:
        if result.get(key):
            print(f"\n  {label}: {result[key]}")

    # Summaries
    rs = result.get("regime_summary", {})
    if rs:
        print(f"\n  📊 Regime: "
              f"{rs.get('overall')} | "
              f"Bias: {rs.get('bias')} | "
              f"Vol: {rs.get('vol')} | "
              f"Score: {rs.get('score')}/100")

    ms = result.get("memory_summary", {})
    if ms and ms.get("similar_found", 0) > 0:
        wr = (ms.get("stats") or {}).get(
            "win_rate", "N/A"
        )
        print(f"  🔍 Memory: "
              f"{ms.get('similar_found')} similar | "
              f"WR: {wr}% | "
              f"Match: "
              f"{ms.get('top_similarity', 0):.0%}")

# ── Run Intelligence ──────────────────────────────────
def run_intelligence(oi_data_map=None,
                     portfolio_config=None):
    symbols = ["NIFTY", "BANKNIFTY",
               "RELIANCE", "TCS"]

    print("=" * 60)
    print("  Trading AI — Intelligence Core v6")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Indicators + Regime + Hallucination"
          " + Memory + Events + Portfolio")
    print("=" * 60)

    # ── Get Event Context Once ────────────────────────
    print("\n  📅 Loading event context...")
    try:
        event_ctx = build_event_context()
        print(
            f"  Event Risk    : "
            f"{event_ctx['event_risk'].upper()}"
        )
        print(
            f"  Is Event Day  : "
            f"{'🔴 YES' if event_ctx['is_event_day'] else '✅ NO'}"
        )
        if event_ctx.get("alerts"):
            for alert in event_ctx["alerts"]:
                print(f"  ⚠ {alert}")
    except Exception as e:
        print(f"  ⚠ Event context failed: {e}")
        event_ctx = None

    all_results = []

    for symbol in symbols:

        print(f"\n{'─' * 60}")
        print(f"  Processing {symbol}...")
        print(f"{'─' * 60}")

        # ── Load Indicators ───────────────────────────
        indicator_path = os.path.join(
            OUTPUT_DIR,
            f"{symbol}_indicators.json"
        )
        if not os.path.exists(indicator_path):
            print(f"  ⚠ No indicators for {symbol}")
            continue

        with open(indicator_path) as f:
            data = json.load(f)

        # ── Get Regime ────────────────────────────────
        regime = None
        if symbol in ["NIFTY", "BANKNIFTY"]:
            try:
                print(f"  📊 Detecting regime...")
                regime = detect_full_regime(symbol)
            except Exception as e:
                print(f"  ⚠ Regime failed: {e}")

        # ── Get Memory Context ────────────────────────
        memory_meta = {}
        try:
            print(f"  🧠 Retrieving memory...")
            memory_context, memory_meta = \
                build_memory_context(data, symbol)
            if memory_context:
                data["memory_context"] = memory_context
                print(
                    f"  ✅ Memory: "
                    f"{memory_meta.get('similar_found', 0)}"
                    f" similar found"
                )
        except Exception as e:
            print(f"  ⚠ Memory failed: {e}")

        # ── Add Event Context ─────────────────────────
        if event_ctx:
            data["event_context"] = \
                event_ctx["context_str"]

        # ── Query AI ──────────────────────────────────
        result = query_intelligence(data, regime)
        if not result:
            print(f"  ❌ AI failed for {symbol}")
            continue

        # ── Layer 1 — Regime Validation ───────────────
        if regime:
            result = validate_against_regime(
                result, regime
            )

        # ── Layer 2 — Memory Adjustment ───────────────
        if memory_meta:
            result = apply_memory_adjustment(
                result, memory_meta
            )

        # ── Layer 3 — Event Adjustment ────────────────
        if event_ctx:
            result = apply_event_adjustments(
                result, event_ctx
            )

        # ── Layer 4 — Hallucination Control ──────────
        oi_data = (oi_data_map or {}).get(symbol)
        try:
            result, validation = validate_decision(
                result, data, regime, oi_data
            )
            print(
                f"  🔍 Validation: "
                f"{result.get('validation_status')} "
                f"| Penalty: "
                f"{result.get('validation_penalty', 0)}"
            )
        except Exception as e:
            print(f"  ⚠ Validation failed: {e}")

        # ── Layer 5 — Portfolio Risk ───────────────────
        if result.get("action") in ["buy", "sell"]:
            try:
                port_result = check_portfolio_risk(
                    result, portfolio_config
                )
                result["portfolio_status"]  = \
                    port_result["portfolio_status"]
                result["recommended_size"]  = \
                    port_result["recommended_size"]
                result["portfolio_blocks"]  = \
                    port_result["blocks"]

                # Block if portfolio says no
                if port_result[
                    "portfolio_status"
                ] == "BLOCKED":
                    result["action"] = "avoid"
                    result["warnings"].append(
                        "Blocked by portfolio "
                        "risk engine"
                    )
                elif port_result[
                    "portfolio_status"
                ] == "REDUCED":
                    result["warnings"].append(
                        "Position size reduced "
                        "by portfolio engine"
                    )

            except Exception as e:
                print(f"  ⚠ Portfolio check failed: {e}")

        # ── Add Summaries ─────────────────────────────
        if regime:
            result["regime_summary"] = {
                "overall"  : regime["fusion"].get(
                                 "overall_regime"),
                "bias"     : regime["fusion"].get(
                                 "primary_bias"),
                "score"    : regime["fusion"].get(
                                 "regime_score"),
                "vol"      : regime["volatility"].get(
                                 "vol_regime"),
                "expiry"   : regime["expiry"].get(
                                 "expiry_type"),
                "tradeable": regime["fusion"].get(
                                 "tradeable")
            }

        if memory_meta:
            result["memory_summary"] = memory_meta

        if event_ctx:
            result["event_summary"] = {
                "risk"        : event_ctx["event_risk"],
                "is_event_day": event_ctx["is_event_day"],
                "alerts"      : event_ctx["alerts"]
            }

        # ── Print Decision ────────────────────────────
        print_decision(result)

        # ── Save Strategy ─────────────────────────────
        out_path = os.path.join(
            STRATEGIES_DIR,
            f"{symbol}_strategy.json"
        )
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2,
                      default=str)

        all_results.append(result)

    # ── Save Combined Report ──────────────────────────
    report_path = os.path.join(
        STRATEGIES_DIR, "daily_report.json"
    )
    with open(report_path, "w") as f:
        json.dump({
            "date"       : datetime.now().strftime(
                               "%Y-%m-%d %H:%M:%S"),
            "version"    : "v6",
            "event_risk" : event_ctx["event_risk"]
                           if event_ctx else "unknown",
            "analyses"   : all_results
        }, f, indent=2, default=str)

    # ── Final Summary ─────────────────────────────────
    print(f"\n{'=' * 75}")
    print(f"  INTELLIGENCE SUMMARY v6")
    print(f"{'=' * 75}")
    print(
        f"  {'SYMBOL':<12} {'ACTION':<10} "
        f"{'CONF':<8} {'SIZE':<10} "
        f"{'PORT':<10} {'STATUS'}"
    )
    print("  " + "─" * 70)

    for r in all_results:
        vstatus  = r.get(
            "validation_status", "UNKNOWN"
        )
        pstatus  = r.get(
            "portfolio_status", "N/A"
        )
        vicon    = {
            "CLEAN"       : "✅",
            "MINOR_ISSUES": "⚠️",
            "DEGRADED"    : "🟠",
            "HIGH_RISK"   : "🔴",
            "BLOCKED"     : "❌"
        }.get(vstatus, "❓")

        picon    = {
            "APPROVED": "✅",
            "REDUCED" : "⚠️",
            "BLOCKED" : "❌",
            "N/A"     : "─"
        }.get(pstatus, "─")

        print(
            f"  {r.get('symbol',''):<12} "
            f"{r.get('action',''):<10} "
            f"{str(r.get('confidence','')):<8} "
            f"{r.get('position_size','N/A'):<10} "
            f"{picon} {pstatus:<8} "
            f"{vicon} {vstatus}"
        )

    # ── Actionable Signals ────────────────────────────
    actionable = [
        r for r in all_results
        if r.get("action") in ["buy", "sell"]
        and r.get("validation_status") in [
            "CLEAN", "MINOR_ISSUES"
        ]
        and float(r.get("confidence", 0)) >= 0.6
        and r.get("position_size") != "avoid"
        and r.get("portfolio_status") in [
            "APPROVED", "REDUCED", None, "N/A"
        ]
    ]

    if actionable:
        print(f"\n  🎯 FINAL ACTIONABLE SIGNALS:")
        print(f"  {'─' * 70}")
        for r in actionable:
            ms     = r.get("memory_summary", {})
            mstats = ms.get("stats") or {}
            wr     = mstats.get("win_rate", "N/A")
            size   = r.get("recommended_size", 0)
            print(
                f"  ✅ {r['symbol']:<12} "
                f"{r['action'].upper():<6} | "
                f"Conf: {r['confidence']:<6} | "
                f"Size: ₹{size:>8,.0f} | "
                f"Entry: {r.get('entry_zone','N/A')} | "
                f"SL: {r.get('stop_loss','N/A')} | "
                f"Mem WR: {wr}%"
            )
            ev = r.get("event_summary", {})
            if ev.get("is_event_day"):
                print(
                    f"     ⚠ EVENT DAY — extra caution!"
                )
    else:
        print(
            f"\n  ⏳ No clean actionable signals — "
            f"stay patient"
        )

    print(f"\n  ✅ Report → {report_path}\n")
    return all_results

# ── Entry Point ───────────────────────────────────────
if __name__ == "__main__":
    run_intelligence()
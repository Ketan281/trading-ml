import os
import sys
import json
import numpy as np
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(
       os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Validation Rules ──────────────────────────────────
# Each rule checks one specific contradiction
# Returns (passed, penalty, reason)

# ── Rule 1 — RSI vs Action ────────────────────────────
def rule_rsi_action(decision, indicators):
    action = decision.get("action", "hold")
    rsi    = float(indicators.get("rsi", 50))
    issues = []
    penalty = 0

    if action == "buy" and rsi > 75:
        issues.append(
            f"BUY signal but RSI={rsi} is overbought"
        )
        penalty += 0.20

    if action == "buy" and rsi > 80:
        issues.append(
            f"STRONG BUY but RSI={rsi} extremely overbought"
        )
        penalty += 0.10

    if action == "sell" and rsi < 25:
        issues.append(
            f"SELL signal but RSI={rsi} is oversold"
        )
        penalty += 0.20

    if action == "sell" and rsi < 20:
        issues.append(
            f"STRONG SELL but RSI={rsi} extremely oversold"
        )
        penalty += 0.10

    return len(issues) == 0, penalty, issues

# ── Rule 2 — VWAP vs Action ───────────────────────────
def rule_vwap_action(decision, indicators):
    action = decision.get("action", "hold")
    vwap   = indicators.get("vwap", "above_vwap")
    issues = []
    penalty = 0

    if action == "buy" and vwap == "below_vwap":
        issues.append(
            "BUY signal but price is below VWAP"
        )
        penalty += 0.10

    if action == "sell" and vwap == "above_vwap":
        issues.append(
            "SELL signal but price is above VWAP"
        )
        penalty += 0.10

    return len(issues) == 0, penalty, issues

# ── Rule 3 — Trend vs Action ──────────────────────────
def rule_trend_action(decision, indicators):
    action     = decision.get("action",  "hold")
    trend      = indicators.get("trend", "sideways")
    condition  = decision.get(
        "market_condition", "sideways"
    )
    issues  = []
    penalty = 0

    # AI says bullish but trend says downtrend
    if (action == "buy" and
            "downtrend" in trend):
        issues.append(
            f"BUY signal but trend={trend}"
        )
        penalty += 0.15

    # AI says bearish but trend says uptrend
    if (action == "sell" and
            "uptrend" in trend):
        issues.append(
            f"SELL signal but trend={trend}"
        )
        penalty += 0.15

    # Condition contradicts trend
    if ("uptrend" in condition and
            "downtrend" in trend):
        issues.append(
            f"AI says {condition} but "
            f"data shows {trend}"
        )
        penalty += 0.10

    if ("downtrend" in condition and
            "uptrend" in trend):
        issues.append(
            f"AI says {condition} but "
            f"data shows {trend}"
        )
        penalty += 0.10

    return len(issues) == 0, penalty, issues

# ── Rule 4 — MACD vs Action ───────────────────────────
def rule_macd_action(decision, indicators):
    action = decision.get("action", "hold")
    macd   = indicators.get("macd", "neutral")
    issues = []
    penalty = 0

    if action == "buy" and macd == "bearish":
        issues.append(
            "BUY signal but MACD is bearish"
        )
        penalty += 0.10

    if action == "sell" and macd == "bullish":
        issues.append(
            "SELL signal but MACD is bullish"
        )
        penalty += 0.10

    return len(issues) == 0, penalty, issues

# ── Rule 5 — Bollinger vs Action ──────────────────────
def rule_bollinger_action(decision, indicators):
    action    = decision.get("action",    "hold")
    bollinger = indicators.get("bollinger","inside_bands")
    issues  = []
    penalty = 0

    if (action == "buy" and
            bollinger == "above_upper"):
        issues.append(
            "BUY signal but price above "
            "Bollinger upper band"
        )
        penalty += 0.10

    if (action == "sell" and
            bollinger == "below_lower"):
        issues.append(
            "SELL signal but price below "
            "Bollinger lower band"
        )
        penalty += 0.10

    return len(issues) == 0, penalty, issues

# ── Rule 6 — Confidence vs Risk ───────────────────────
def rule_confidence_risk(decision, indicators):
    confidence = float(
        decision.get("confidence", 0.5)
    )
    risk_level = decision.get(
        "risk_level", "medium"
    )
    issues  = []
    penalty = 0

    # High confidence but extreme risk
    if confidence > 0.8 and risk_level == "extreme":
        issues.append(
            f"Confidence={confidence} too high "
            f"for extreme risk"
        )
        penalty += 0.15

    # Low confidence but recommending action
    action = decision.get("action", "hold")
    if (confidence < 0.4 and
            action in ["buy", "sell"]):
        issues.append(
            f"Confidence={confidence} too low "
            f"for {action} action"
        )
        penalty += 0.10

    return len(issues) == 0, penalty, issues

# ── Rule 7 — Volatility vs Strategy ──────────────────
def rule_volatility_strategy(decision, indicators):
    volatility = indicators.get(
        "volatility", "medium"
    )
    strategy   = decision.get("strategy", "")
    action     = decision.get("action",   "hold")
    issues  = []
    penalty = 0

    # Naked options in high volatility
    if (volatility == "high" and
            action in ["buy", "sell"] and
            "spread" not in str(strategy).lower()):
        issues.append(
            "Directional trade in high volatility "
            "without spread protection"
        )
        penalty += 0.10

    # Selling premium in very low volatility
    if (volatility == "low" and
            "sell" in str(strategy).lower()):
        issues.append(
            "Selling premium when volatility is low "
            "— poor risk/reward"
        )
        penalty += 0.05

    return len(issues) == 0, penalty, issues

# ── Rule 8 — Numeric Consistency ─────────────────────
def rule_numeric_consistency(decision, indicators):
    issues  = []
    penalty = 0
    price   = float(indicators.get("price", 0))

    if price == 0:
        return True, 0, []

    # Entry zone vs current price
    entry_zone = str(
        decision.get("entry_zone", "")
    ).replace("₹", "").strip()

    try:
        if "-" in entry_zone:
            parts  = entry_zone.split("-")
            low_e  = float(parts[0].strip())
            high_e = float(parts[1].strip())

            # Entry zone way too far from price
            if low_e > price * 1.05:
                issues.append(
                    f"Entry zone {entry_zone} is "
                    f">5% above current price {price}"
                )
                penalty += 0.10
            if high_e < price * 0.95:
                issues.append(
                    f"Entry zone {entry_zone} is "
                    f">5% below current price {price}"
                )
                penalty += 0.10
    except Exception:
        pass

    # Stop loss vs entry
    stop_loss  = str(
        decision.get("stop_loss", "")
    ).replace("₹", "").strip()
    action     = decision.get("action", "hold")

    try:
        sl = float(stop_loss)
        if action == "buy" and sl > price:
            issues.append(
                f"Stop loss {sl} is above price "
                f"{price} for BUY trade"
            )
            penalty += 0.20

        if action == "sell" and sl < price:
            issues.append(
                f"Stop loss {sl} is below price "
                f"{price} for SELL trade"
            )
            penalty += 0.20
    except Exception:
        pass

    return len(issues) == 0, penalty, issues

# ── Rule 9 — OI Contradiction ─────────────────────────
def rule_oi_contradiction(decision,
                           oi_data=None):
    if not oi_data:
        return True, 0, []

    action  = decision.get("action", "hold")
    issues  = []
    penalty = 0

    pcr_signal = oi_data.get(
        "pcr", {}
    ).get("pcr_oi_signal", "neutral")

    overall_oi = oi_data.get(
        "overall_signal", "neutral"
    )

    if (action == "buy" and
            overall_oi in [
                "extremely_bearish", "bearish"
            ]):
        issues.append(
            f"BUY signal but OI shows {overall_oi}"
        )
        penalty += 0.15

    if (action == "sell" and
            overall_oi in [
                "extremely_bullish", "bullish"
            ]):
        issues.append(
            f"SELL signal but OI shows {overall_oi}"
        )
        penalty += 0.15

    return len(issues) == 0, penalty, issues

# ── Rule 10 — Regime Contradiction ───────────────────
def rule_regime_contradiction(decision,
                               regime=None):
    if not regime:
        return True, 0, []

    action  = decision.get("action", "hold")
    fusion  = regime.get("fusion",   {})
    issues  = []
    penalty = 0

    bias    = fusion.get("primary_bias", "neutral")
    tradeable = fusion.get("tradeable",  True)

    if not tradeable and action != "avoid":
        issues.append(
            "Action is not avoid despite "
            "market being untradeable"
        )
        penalty += 0.30

    if (action == "buy" and
            bias == "bearish"):
        issues.append(
            f"BUY action contradicts "
            f"regime bias={bias}"
        )
        penalty += 0.15

    if (action == "sell" and
            bias == "bullish"):
        issues.append(
            f"SELL action contradicts "
            f"regime bias={bias}"
        )
        penalty += 0.15

    return len(issues) == 0, penalty, issues

# ── Master Validator ──────────────────────────────────
def validate_decision(decision, indicators,
                       regime=None, oi_data=None):
    print(f"\n  🔍 Validating AI decision "
          f"for {decision.get('symbol','?')}...")

    all_issues   = []
    total_penalty = 0.0

    # Run all rules
    rules = [
        ("RSI vs Action",
         rule_rsi_action(decision, indicators)),
        ("VWAP vs Action",
         rule_vwap_action(decision, indicators)),
        ("Trend vs Action",
         rule_trend_action(decision, indicators)),
        ("MACD vs Action",
         rule_macd_action(decision, indicators)),
        ("Bollinger vs Action",
         rule_bollinger_action(
             decision, indicators)),
        ("Confidence vs Risk",
         rule_confidence_risk(
             decision, indicators)),
        ("Volatility vs Strategy",
         rule_volatility_strategy(
             decision, indicators)),
        ("Numeric Consistency",
         rule_numeric_consistency(
             decision, indicators)),
        ("OI Contradiction",
         rule_oi_contradiction(
             decision, oi_data)),
        ("Regime Contradiction",
         rule_regime_contradiction(
             decision, regime))
    ]

    rule_results = []
    for rule_name, (passed, penalty, issues) in rules:
        rule_results.append({
            "rule"   : rule_name,
            "passed" : passed,
            "penalty": penalty,
            "issues" : issues
        })
        if not passed:
            all_issues.extend(issues)
            total_penalty += penalty

    # Original confidence
    original_confidence = float(
        decision.get("confidence", 0.5)
    )

    # Apply penalty
    new_confidence = max(
        0.0,
        round(original_confidence - total_penalty, 3)
    )

    # Determine validation status
    if total_penalty >= 0.5:
        status = "BLOCKED"
        decision["action"] = "avoid"
    elif total_penalty >= 0.3:
        status = "HIGH_RISK"
    elif total_penalty >= 0.15:
        status = "DEGRADED"
    elif total_penalty > 0:
        status = "MINOR_ISSUES"
    else:
        status = "CLEAN"

    # Build validation report
    validation = {
        "symbol"              : decision.get("symbol"),
        "timestamp"           : datetime.now(
                                ).isoformat(),
        "status"              : status,
        "original_confidence" : original_confidence,
        "adjusted_confidence" : new_confidence,
        "total_penalty"       : round(
                                    total_penalty, 3),
        "issues_found"        : len(all_issues),
        "issues"              : all_issues,
        "rule_results"        : rule_results,
        "rules_passed"        : sum(
            1 for r in rule_results if r["passed"]
        ),
        "rules_failed"        : sum(
            1 for r in rule_results if not r["passed"]
        )
    }

    # Update decision
    decision["confidence"]          = new_confidence
    decision["validation_status"]   = status
    decision["validation_penalty"]  = round(
        total_penalty, 3
    )
    decision["hallucination_check"] = validation

    # Append validation warnings
    existing_warnings = decision.get("warnings", [])
    for issue in all_issues:
        if issue not in existing_warnings:
            existing_warnings.append(
                f"[VALIDATION] {issue}"
            )
    decision["warnings"] = existing_warnings

    # Print validation summary
    status_icon = {
        "CLEAN"       : "✅",
        "MINOR_ISSUES": "⚠️",
        "DEGRADED"    : "🟠",
        "HIGH_RISK"   : "🔴",
        "BLOCKED"     : "❌"
    }.get(status, "❓")

    print(f"\n  {'─' * 50}")
    print(f"  {status_icon} Validation Status  : "
          f"{status}")
    print(f"  Rules Passed       : "
          f"{validation['rules_passed']}/10")
    print(f"  Rules Failed       : "
          f"{validation['rules_failed']}/10")
    print(f"  Original Confidence: "
          f"{original_confidence}")
    print(f"  Adjusted Confidence: "
          f"{new_confidence}")
    print(f"  Total Penalty      : "
          f"{total_penalty:.3f}")

    if all_issues:
        print(f"\n  Issues Found:")
        for issue in all_issues:
            print(f"     ⚠ {issue}")

    # Show rule details
    print(f"\n  Rule Breakdown:")
    for r in rule_results:
        icon = "✅" if r["passed"] else "❌"
        print(f"     {icon} {r['rule']:<30} "
              f"penalty={r['penalty']}")

    print(f"  {'─' * 50}")

    return decision, validation

# ── Batch Validate ────────────────────────────────────
def batch_validate(decisions, indicators_map,
                   regimes_map=None,
                   oi_map=None):
    results = []
    for symbol, decision in decisions.items():
        indicators = indicators_map.get(symbol, {})
        regime     = (regimes_map or {}).get(symbol)
        oi_data    = (oi_map     or {}).get(symbol)

        validated, validation = validate_decision(
            decision, indicators, regime, oi_data
        )
        results.append({
            "symbol"    : symbol,
            "decision"  : validated,
            "validation": validation
        })

    return results

# ── Save Validation Report ────────────────────────────
def save_validation_report(results):
    path = os.path.join(
        OUTPUT_DIR,
        f"validation_report_"
        f"{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    )
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  ✅ Validation report → {path}")
    return path

# ── Main Test ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Trading AI — Hallucination Control")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    # Test Case 1 — Clean signal
    print("\n  Test 1 — Clean Signal")
    clean_decision = {
        "symbol"          : "NIFTY",
        "action"          : "buy",
        "market_condition": "uptrend",
        "confidence"      : 0.75,
        "risk_level"      : "medium",
        "strategy"        : "bull_put_spread",
        "entry_zone"      : "23700-23750",
        "stop_loss"       : "23500",
        "target"          : "24000",
        "warnings"        : []
    }
    clean_indicators = {
        "symbol"    : "NIFTY",
        "price"     : 23719.3,
        "trend"     : "uptrend",
        "rsi"       : 58.0,
        "macd"      : "bullish",
        "vwap"      : "above_vwap",
        "bollinger" : "inside_bands",
        "volatility": "normal"
    }
    validated, report = validate_decision(
        clean_decision, clean_indicators
    )
    print(f"  Final Action     : "
          f"{validated['action']}")
    print(f"  Final Confidence : "
          f"{validated['confidence']}")

    # Test Case 2 — Contradictory signal
    print("\n\n  Test 2 — Contradictory Signal")
    bad_decision = {
        "symbol"          : "BANKNIFTY",
        "action"          : "buy",
        "market_condition": "strong_uptrend",
        "confidence"      : 0.85,
        "risk_level"      : "extreme",
        "strategy"        : "naked_call",
        "entry_zone"      : "52000-52100",
        "stop_loss"       : "52500",
        "target"          : "53000",
        "warnings"        : []
    }
    bad_indicators = {
        "symbol"    : "BANKNIFTY",
        "price"     : 51800.0,
        "trend"     : "downtrend",
        "rsi"       : 78.0,
        "macd"      : "bearish",
        "vwap"      : "below_vwap",
        "bollinger" : "above_upper",
        "volatility": "high"
    }
    validated2, report2 = validate_decision(
        bad_decision, bad_indicators
    )
    print(f"  Final Action     : "
          f"{validated2['action']}")
    print(f"  Final Confidence : "
          f"{validated2['confidence']}")
    print(f"  Status           : "
          f"{validated2['validation_status']}")

    # Save reports
    save_validation_report([report, report2])

    print("\n  ✅ Hallucination Control test complete!")
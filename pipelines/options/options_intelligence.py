import os
import sys
import json
import ollama
from datetime        import datetime

ROOT = os.path.dirname(os.path.dirname(
       os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

OPTS = os.path.join(ROOT, "pipelines", "options")
sys.path.insert(0, OPTS)

from options_chain    import fetch_options_chain
from oi_analyzer      import full_oi_analysis
from iv_analyzer      import full_iv_analysis
from greeks_engine    import full_greeks_analysis
from futures_analyzer import full_futures_analysis

OUTPUT_DIR    = os.path.join(ROOT, "outputs")
STRATEGIES_DIR = os.path.join(ROOT, "strategies")
os.makedirs(OUTPUT_DIR,    exist_ok=True)
os.makedirs(STRATEGIES_DIR, exist_ok=True)

SYMBOLS = ["NIFTY", "BANKNIFTY"]

# ── Build Options Intelligence Prompt ─────────────────
def build_options_prompt(symbol, spot, atm,
                          oi_report, iv_report,
                          greeks_report,
                          futures_report):
    # Extract key values safely
    pcr        = oi_report.get("pcr", {})
    walls      = oi_report.get("oi_walls", {})
    oi_change  = oi_report.get("oi_change", {})
    iv_surface = iv_report.get("iv_surface", {})
    max_pain   = iv_report.get("max_pain",   {})
    exp_move   = iv_report.get("exp_move",   {})
    atm_greeks = greeks_report.get(
                     "atm_summary", {}
                 ) if greeks_report else {}
    straddle   = greeks_report.get(
                     "straddle", {}
                 ) if greeks_report else {}
    basis      = (futures_report.get("basis", [{}])
                  or [{}])[0]
    rollover   = futures_report.get(
                     "rollover", {}
                 ) or {}
    fut_oi     = futures_report.get(
                     "oi_analysis", {}
                 ) or {}

    prompt = f"""You are a professional options trading 
intelligence system specializing in Indian index options 
(NIFTY and BANKNIFTY).

Analyze the following comprehensive market data and 
provide a structured options trading recommendation.

═══════════════════════════════════════
SYMBOL        : {symbol}
SPOT PRICE    : ₹{spot}
ATM STRIKE    : {atm}
TIMESTAMP     : {datetime.now().strftime('%Y-%m-%d %H:%M')}
═══════════════════════════════════════

── OI + PCR DATA ──────────────────────
PCR (OI)      : {pcr.get('pcr_oi', 0)}
PCR Signal    : {pcr.get('pcr_oi_signal', 'N/A')}
CE OI         : {pcr.get('total_ce_oi', 0):,}
PE OI         : {pcr.get('total_pe_oi', 0):,}
Resistance    : {walls.get('nearest_resistance', 'N/A')}
Support       : {walls.get('nearest_support', 'N/A')}
OI Activity   : {oi_change.get('dominant_activity', 'N/A')}
OI Signal     : {oi_change.get('activity_signal', 'N/A')}
Overall OI    : {oi_report.get('overall_signal', 'N/A')}

── IV DATA ────────────────────────────
ATM IV        : {iv_surface.get('atm_iv', 0)}%
IV Rank       : {iv_surface.get('iv_rank', 0)}/100
IV Regime     : {iv_surface.get('iv_regime', 'N/A')}
IV Signal     : {iv_surface.get('iv_signal', 'N/A')}
IV Skew       : {iv_surface.get('iv_skew', 0)}
Skew Signal   : {iv_surface.get('skew_signal', 'N/A')}
Max Pain      : {max_pain.get('max_pain', 'N/A') if max_pain else 'N/A'}
Max Pain Sig  : {max_pain.get('signal', 'N/A') if max_pain else 'N/A'}
Expected Move : ±₹{exp_move.get('expected_move', 0) if exp_move else 0}
Upper Range   : ₹{exp_move.get('upper_range', 0) if exp_move else 0}
Lower Range   : ₹{exp_move.get('lower_range', 0) if exp_move else 0}

── GREEKS DATA ────────────────────────
ATM CE Delta  : {atm_greeks.get('atm_ce', {}).get('delta', 0)}
ATM CE Theta  : {atm_greeks.get('atm_ce', {}).get('theta', 0)}
ATM CE Vega   : {atm_greeks.get('atm_ce', {}).get('vega', 0)}
ATM CE IV     : {atm_greeks.get('atm_ce', {}).get('iv', 0)}%
ATM CE LTP    : ₹{atm_greeks.get('atm_ce', {}).get('ltp', 0)}
ATM PE LTP    : ₹{atm_greeks.get('atm_pe', {}).get('ltp', 0)}
Straddle Cost : ₹{straddle.get('net_cost', 0) if straddle else 0}
Straddle BE+  : ₹{straddle.get('breakeven', {}).get('upper', 0) if straddle else 0}
Straddle BE-  : ₹{straddle.get('breakeven', {}).get('lower', 0) if straddle else 0}

── FUTURES DATA ───────────────────────
Futures Price : ₹{basis.get('ltp', spot)}
Basis         : ₹{basis.get('basis', 0)}
Basis Type    : {basis.get('basis_type', 'N/A')}
Futures Signal: {basis.get('signal', 'N/A')}
Rollover %    : {rollover.get('rollover_pct', 0)}%
Rollover Bias : {rollover.get('rollover_bias', 'N/A')}
Carry Signal  : {rollover.get('carry_signal', 'N/A')}
FUT Activity  : {fut_oi.get('activity', 'N/A')}
FUT OI Signal : {fut_oi.get('signal', 'N/A')}

═══════════════════════════════════════

Based on ALL the above data, provide a complete options 
trading recommendation.

Return ONLY a valid JSON object with this exact structure:
{{
  "symbol"           : "{symbol}",
  "spot"             : {spot},
  "atm"              : {atm},
  "market_bias"      : "<bullish/bearish/neutral/volatile>",
  "confidence"       : <0.0 to 1.0>,
  "risk_level"       : "<low/medium/high/extreme>",
  "recommended_strategy": "<strategy name>",
  "strategy_legs"    : [
    {{
      "action"  : "<buy/sell>",
      "type"    : "<CE/PE>",
      "strike"  : <strike price>,
      "reason"  : "<why this leg>"
    }}
  ],
  "entry_condition"  : "<when to enter>",
  "exit_condition"   : "<when to exit>",
  "stop_loss"        : "<stop loss level>",
  "target"           : "<target level>",
  "max_loss"         : "<maximum loss on trade>",
  "max_profit"       : "<maximum profit on trade>",
  "key_levels"       : {{
    "resistance" : <resistance level>,
    "support"    : <support level>,
    "max_pain"   : <max pain level>
  }},
  "iv_edge"          : "<sell_options/buy_options/neutral>",
  "reasoning"        : [
    "<reason 1>",
    "<reason 2>",
    "<reason 3>",
    "<reason 4>"
  ],
  "warnings"         : [
    "<warning 1 if any>"
  ]
}}"""

    return prompt

# ── Query Qwen AI ─────────────────────────────────────
def query_options_ai(prompt, symbol):
    print(f"\n  🧠 Querying Qwen AI for {symbol}...")

    try:
        response = ollama.chat(
            model   = "qwen2.5:1.5b",
            messages= [{
                "role"   : "user",
                "content": prompt
            }],
            options = {
                "temperature": 0.1,
                "num_predict": 1000
            }
        )

        raw = response["message"]["content"].strip()

        # Clean markdown if present
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break

        # Parse JSON
        result = json.loads(raw)
        print(f"  ✅ AI analysis complete")
        return result

    except json.JSONDecodeError as e:
        print(f"  ⚠ JSON parse failed: {e}")
        print(f"  Raw response: {raw[:200]}...")
        return None
    except Exception as e:
        print(f"  ❌ AI query failed: {e}")
        return None

# ── Print Options Signal ──────────────────────────────
def print_options_signal(signal):
    if not signal:
        return

    symbol = signal.get("symbol")
    print(f"\n{'═' * 55}")
    print(f"  🎯 OPTIONS INTELLIGENCE — {symbol}")
    print(f"{'═' * 55}")

    # Bias indicator
    bias = signal.get("market_bias", "neutral")
    bias_icon = {
        "bullish" : "🐂",
        "bearish" : "🐻",
        "neutral" : "😐",
        "volatile": "⚡"
    }.get(bias, "❓")

    print(f"\n  {bias_icon} Market Bias    : "
          f"{bias.upper()}")
    print(f"  📊 Confidence    : "
          f"{signal.get('confidence', 0)}")
    print(f"  ⚠️  Risk Level    : "
          f"{signal.get('risk_level', 'N/A').upper()}")
    print(f"  💡 IV Edge       : "
          f"{signal.get('iv_edge', 'N/A').upper()}")

    print(f"\n  📋 Strategy      : "
          f"{signal.get('recommended_strategy', 'N/A')}")

    # Strategy legs
    legs = signal.get("strategy_legs", [])
    if legs:
        print(f"\n  📌 Strategy Legs:")
        for leg in legs:
            print(
                f"     {leg.get('action','').upper()} "
                f"{leg.get('type','')} "
                f"{leg.get('strike','')} — "
                f"{leg.get('reason','')}"
            )

    print(f"\n  🎯 Key Levels:")
    kl = signal.get("key_levels", {})
    print(f"     Resistance : {kl.get('resistance')}")
    print(f"     Support    : {kl.get('support')}")
    print(f"     Max Pain   : {kl.get('max_pain')}")

    print(f"\n  📈 Trade Plan:")
    print(f"     Entry      : "
          f"{signal.get('entry_condition', 'N/A')}")
    print(f"     Exit       : "
          f"{signal.get('exit_condition', 'N/A')}")
    print(f"     Stop Loss  : "
          f"{signal.get('stop_loss', 'N/A')}")
    print(f"     Target     : "
          f"{signal.get('target', 'N/A')}")
    print(f"     Max Loss   : "
          f"{signal.get('max_loss', 'N/A')}")
    print(f"     Max Profit : "
          f"{signal.get('max_profit', 'N/A')}")

    print(f"\n  💭 Reasoning:")
    for r in signal.get("reasoning", []):
        print(f"     → {r}")

    warnings = signal.get("warnings", [])
    if warnings and warnings[0]:
        print(f"\n  ⚠️  Warnings:")
        for w in warnings:
            print(f"     🔴 {w}")

    print(f"\n{'═' * 55}")

# ── Run Full Options Intelligence ─────────────────────
def run_options_intelligence(symbol):
    print(f"\n{'🔥' * 27}")
    print(f"  OPTIONS INTELLIGENCE — {symbol}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'🔥' * 27}")

    # Step 1 — Fetch all analyses
    print(f"\n  Step 1 — Running OI Analysis...")
    oi_report = full_oi_analysis(symbol)

    print(f"\n  Step 2 — Running IV Analysis...")
    iv_report = full_iv_analysis(symbol)

    print(f"\n  Step 3 — Running Greeks Analysis...")
    greeks_report = full_greeks_analysis(symbol)

    print(f"\n  Step 4 — Running Futures Analysis...")
    futures_report = full_futures_analysis(symbol)

    # Get spot and ATM
    spot = None
    atm  = None

    for report in [oi_report, iv_report,
                   futures_report]:
        if report:
            spot = report.get("spot", spot)
            atm  = report.get("atm",  atm)
            if spot and atm:
                break

    if not spot:
        print(f"  ❌ Could not get spot price")
        return None

    # Step 5 — Build prompt and query AI
    print(f"\n  Step 5 — Building AI Prompt...")
    prompt = build_options_prompt(
        symbol,
        spot,
        atm,
        oi_report      or {},
        iv_report      or {},
        greeks_report  or {},
        futures_report or {}
    )

    # Step 6 — Query AI
    print(f"\n  Step 6 — Querying Options AI...")
    ai_signal = query_options_ai(prompt, symbol)

    if not ai_signal:
        print(f"  ⚠ AI signal failed — building "
              f"rule-based signal...")
        ai_signal = build_rule_based_signal(
            symbol, spot, atm,
            oi_report, iv_report, futures_report
        )

    # Print signal
    print_options_signal(ai_signal)

    # Save complete report
    full_report = {
        "symbol"        : symbol,
        "timestamp"     : datetime.now().isoformat(),
        "spot"          : spot,
        "atm"           : atm,
        "oi_report"     : oi_report,
        "iv_report"     : iv_report,
        "futures_report": futures_report,
        "ai_signal"     : ai_signal
    }

    # Save to strategies
    strat_path = os.path.join(
        STRATEGIES_DIR,
        f"{symbol}_options_strategy_"
        f"{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    )
    with open(strat_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)

    # Save to outputs
    out_path = os.path.join(
        OUTPUT_DIR,
        f"{symbol}_options_signal.json"
    )
    with open(out_path, "w") as f:
        json.dump(ai_signal or {}, f,
                  indent=2, default=str)

    print(f"\n  ✅ Strategy saved → {strat_path}")
    return full_report

# ── Rule Based Fallback ───────────────────────────────
def build_rule_based_signal(symbol, spot, atm,
                             oi_report, iv_report,
                             futures_report):
    pcr       = oi_report.get(
                    "pcr", {}
                ).get("pcr_oi", 1.0) \
                if oi_report else 1.0
    iv_regime = iv_report.get(
                    "iv_surface", {}
                ).get("iv_regime", "normal") \
                if iv_report else "normal"
    walls     = oi_report.get(
                    "oi_walls", {}
                ) if oi_report else {}

    # Determine bias
    if pcr > 1.2:
        bias = "bullish"
    elif pcr < 0.8:
        bias = "bearish"
    else:
        bias = "neutral"

    # Determine strategy
    if iv_regime in ["high", "very_high"]:
        strategy = "short_straddle"
        iv_edge  = "sell_options"
    elif iv_regime in ["low", "very_low"]:
        strategy = "long_straddle"
        iv_edge  = "buy_options"
    else:
        strategy = "bull_put_spread" \
                   if bias == "bullish" \
                   else "bear_call_spread"
        iv_edge  = "neutral"

    return {
        "symbol"              : symbol,
        "spot"                : spot,
        "atm"                 : atm,
        "market_bias"         : bias,
        "confidence"          : 0.55,
        "risk_level"          : "medium",
        "recommended_strategy": strategy,
        "strategy_legs"       : [],
        "entry_condition"     : "Wait for confirmation",
        "exit_condition"      : "At target or stop loss",
        "stop_loss"           : str(
            walls.get("nearest_support", "N/A")
        ),
        "target"              : str(
            walls.get("nearest_resistance", "N/A")
        ),
        "max_loss"            : "Defined by spread width",
        "max_profit"          : "Defined by premium",
        "key_levels"          : {
            "resistance": walls.get(
                              "nearest_resistance"),
            "support"   : walls.get(
                              "nearest_support"),
            "max_pain"  : iv_report.get(
                              "max_pain", {}
                          ).get("max_pain") \
                          if iv_report else None
        },
        "iv_edge"             : iv_edge,
        "reasoning"           : [
            f"PCR at {pcr} indicates {bias} sentiment",
            f"IV regime is {iv_regime}",
            f"Rule based signal — AI unavailable"
        ],
        "warnings"            : [
            "Rule based fallback — verify manually"
        ]
    }

# ── Main Runner ───────────────────────────────────────
def run_all():
    print("\n" + "🔥" * 27)
    print("   TRADING AI — OPTIONS INTELLIGENCE CORE")
    print("🔥" * 27)

    all_signals = []

    for symbol in SYMBOLS:
        report = run_options_intelligence(symbol)
        if report and report.get("ai_signal"):
            all_signals.append(
                report["ai_signal"]
            )

    # Final Summary
    print(f"\n{'═' * 55}")
    print(f"  OPTIONS INTELLIGENCE SUMMARY")
    print(f"{'═' * 55}")
    print(f"  {'SYMBOL':<14} {'BIAS':<12} "
          f"{'STRATEGY':<25} {'CONF'}")
    print(f"  {'─' * 60}")

    for s in all_signals:
        print(
            f"  {s.get('symbol',''):<14} "
            f"{s.get('market_bias',''):<12} "
            f"{s.get('recommended_strategy',''):<25} "
            f"{s.get('confidence', 0)}"
        )

    # Save combined
    combined_path = os.path.join(
        OUTPUT_DIR, "options_intelligence_report.json"
    )
    with open(combined_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "signals"  : all_signals
        }, f, indent=2, default=str)

    print(f"\n  ✅ Combined report → {combined_path}")
    print(f"\n  🎯 Options Intelligence complete!\n")

# ── Entry Point ───────────────────────────────────────
if __name__ == "__main__":
    run_all()
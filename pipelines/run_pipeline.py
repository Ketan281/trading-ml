import json
import os
import sys
from datetime import datetime

# ── Add project root to path ──────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipelines.fetch_data    import fetch_and_save
from pipelines.indicators    import analyze
from pipelines.intelligence  import query_intelligence
from memory.memory_store     import (init_db, save_analysis,
                                     save_outcome, print_summary)
from reflections.reflection_engine import reflect_on_trade

SYMBOLS = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS"]

# ── Pretty Print ──────────────────────────────────────
def header(title):
    print("\n" + "=" * 55)
    print(f"  {title}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

# ── Step 1 — Fetch Market Data ────────────────────────
def step_fetch():
    header("STEP 1 — Fetching Market Data")
    TICKERS = {
        "NIFTY"    : "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "RELIANCE" : "RELIANCE.NS",
        "TCS"      : "TCS.NS"
    }
    results = {}
    for name, ticker in TICKERS.items():
        df = fetch_and_save(name, ticker)
        results[name] = df is not None
    return results

# ── Step 2 — Compute Indicators ───────────────────────
def step_indicators():
    header("STEP 2 — Computing Indicators")
    results = {}
    for symbol in SYMBOLS:
        data = analyze(symbol)
        if data:
            results[symbol] = data
            print(f"  ✅ {symbol}: RSI={data['rsi']} | "
                  f"Trend={data['trend']} | "
                  f"Volatility={data['volatility']}")
    return results

# ── Step 3 — Run Intelligence ─────────────────────────
def step_intelligence(indicator_data):
    header("STEP 3 — Running Trading Intelligence")
    decisions = {}
    for symbol, data in indicator_data.items():
        decision = query_intelligence(data)
        if decision:
            decisions[symbol] = decision
            print(f"\n  ✅ {symbol}:")
            print(f"     Action     → {decision.get('action')}")
            print(f"     Confidence → {decision.get('confidence')}")
            print(f"     Risk       → {decision.get('risk_level')}")
            print(f"     Condition  → {decision.get('market_condition')}")
    return decisions

# ── Step 4 — Save to Memory ───────────────────────────
def step_memory(indicator_data, decisions):
    header("STEP 4 — Saving to Memory")
    analysis_ids = {}
    for symbol in decisions:
        aid = save_analysis(
            symbol,
            indicator_data[symbol],
            decisions[symbol]
        )
        analysis_ids[symbol] = aid
        print(f"  ✅ {symbol} saved → Analysis ID: {aid}")
    return analysis_ids

# ── Step 5 — Save Daily Report ────────────────────────
def step_report(decisions):
    header("STEP 5 — Generating Daily Report")

    report = {
        "date"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary" : [],
        "alerts"  : []
    }

    for symbol, d in decisions.items():
        entry = {
            "symbol"          : symbol,
            "action"          : d.get("action"),
            "confidence"      : d.get("confidence"),
            "risk_level"      : d.get("risk_level"),
            "market_condition": d.get("market_condition"),
            "entry_zone"      : d.get("entry_zone"),
            "stop_loss"       : d.get("stop_loss")
        }
        report["summary"].append(entry)

        # Flag high risk alerts
        if d.get("risk_level") in ["high", "extreme"]:
            report["alerts"].append({
                "symbol" : symbol,
                "alert"  : f"HIGH RISK — {d.get('market_condition')}",
                "action" : d.get("action")
            })

    # Save report
    report_dir  = os.path.join(ROOT, "outputs")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(
        report_dir,
        f"daily_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary table
    print(f"\n  {'SYMBOL':<12} {'ACTION':<10} "
          f"{'CONFIDENCE':<12} {'RISK':<10} {'CONDITION'}")
    print("  " + "-" * 65)

    for entry in report["summary"]:
        print(f"  {entry['symbol']:<12} "
              f"{entry['action']:<10} "
              f"{str(entry['confidence']):<12} "
              f"{entry['risk_level']:<10} "
              f"{entry['market_condition']}")

    if report["alerts"]:
        print(f"\n  ⚠️  ALERTS:")
        for alert in report["alerts"]:
            print(f"     🔴 {alert['symbol']}: {alert['alert']}")

    print(f"\n  ✅ Report saved → {report_path}")
    return report

# ── Step 6 — Optional Reflection ─────────────────────
def step_reflection_demo(indicator_data, analysis_ids):
    header("STEP 6 — Demo Reflection (Simulated Trade)")

    # Use NIFTY as demo
    symbol = "NIFTY"
    if symbol not in indicator_data or symbol not in analysis_ids:
        print("  ⚠ Skipping reflection demo — NIFTY data not available")
        return

    data      = indicator_data[symbol]
    price     = data.get("price", 22000)
    reasoning = [
        f"Trend: {data.get('trend')}",
        f"RSI: {data.get('rsi')}",
        f"Volatility: {data.get('volatility')}"
    ]

    # Simulate a trade outcome
    entry = price
    exit  = round(price * 1.005, 2)   # Simulate 0.5% gain

    reflect_on_trade(
        outcome_id  = analysis_ids[symbol],
        symbol      = symbol,
        action      = "buy",
        entry_price = entry,
        exit_price  = exit,
        market_data = data,
        reasoning   = reasoning
    )

# ── Master Runner ─────────────────────────────────────
def run_full_pipeline(skip_reflection=False):
    print("\n" + "🔥" * 27)
    print("   TRADING AI — FULL PIPELINE STARTING")
    print("🔥" * 27)

    # Initialize memory DB
    init_db()

    # Run all steps
    step_fetch()
    indicator_data = step_indicators()

    if not indicator_data:
        print("\n  ❌ No indicator data. Stopping pipeline.")
        return

    decisions    = step_intelligence(indicator_data)
    analysis_ids = step_memory(indicator_data, decisions)
    report       = step_report(decisions)

    if not skip_reflection:
        step_reflection_demo(indicator_data, analysis_ids)

    # Final summary
    header("PIPELINE COMPLETE")
    print_summary()
    print("\n  🎯 Trading AI pipeline completed successfully!")
    print("  📁 Check /outputs for daily report")
    print("  🗄  Check /memory for stored analyses")
    print("  🪞  Check /reflections for AI reflections\n")

# ── Entry Point ───────────────────────────────────────
if __name__ == "__main__":
    # Pass --skip-reflection to skip the demo reflection
    skip = "--skip-reflection" in sys.argv
    run_full_pipeline(skip_reflection=skip)
import json
import os
import sys
import sqlite3
import pandas as pd
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MEMORY_DB    = os.path.join(ROOT, "memory", "trading_memory.db")
TRAINING_DIR = os.path.dirname(__file__)
os.makedirs(TRAINING_DIR, exist_ok=True)

# ── Load All Memory ───────────────────────────────────
def load_memory():
    if not os.path.exists(MEMORY_DB):
        print("  ⚠ No memory database found.")
        print("  Run run_pipeline.py first to generate data.")
        return [], [], []

    conn = sqlite3.connect(MEMORY_DB)

    analyses    = pd.read_sql("SELECT * FROM analyses",    conn)
    outcomes    = pd.read_sql("SELECT * FROM outcomes",    conn)
    reflections = pd.read_sql("SELECT * FROM reflections", conn)

    conn.close()

    print(f"  ✅ Loaded:")
    print(f"     Analyses    : {len(analyses)}")
    print(f"     Outcomes    : {len(outcomes)}")
    print(f"     Reflections : {len(reflections)}")

    return analyses, outcomes, reflections

# ── Build Raw Dataset ─────────────────────────────────
def build_raw_dataset(analyses, outcomes, reflections):
    records = []

    for _, analysis in analyses.iterrows():
        record = {
            "id"        : int(analysis["id"]),
            "timestamp" : analysis["timestamp"],
            "symbol"    : analysis["symbol"],
            "market_data": json.loads(analysis["market_data"])
                           if analysis["market_data"] else {},
            "ai_decision": json.loads(analysis["ai_decision"])
                           if analysis["ai_decision"] else {},
            "action"        : analysis["action"],
            "confidence"    : analysis["confidence"],
            "risk_level"    : analysis["risk_level"],
            "market_condition": analysis["market_condition"],
            "outcome"       : None,
            "reflection"    : None
        }

        # Match outcome
        match = outcomes[outcomes["analysis_id"] == analysis["id"]]
        if not match.empty:
            o = match.iloc[0]
            record["outcome"] = {
                "result"      : o["result"],
                "pnl"         : o["pnl"],
                "entry_price" : o["entry_price"],
                "exit_price"  : o["exit_price"],
                "action_taken": o["action_taken"]
            }

            # Match reflection
            ref_match = reflections[
                reflections["outcome_id"] == o["id"]
            ]
            if not ref_match.empty:
                r = ref_match.iloc[0]
                try:
                    ref_data = json.loads(r["reflection"])
                except Exception:
                    ref_data = {"raw": r["reflection"]}

                record["reflection"] = {
                    "data"       : ref_data,
                    "improvement": r["improvement"]
                }

        records.append(record)

    return records

# ── Build Fine-Tune Dataset ───────────────────────────
def build_finetune_dataset(records):
    finetune = []

    for r in records:
        market   = r.get("market_data",  {})
        decision = r.get("ai_decision",  {})
        outcome  = r.get("outcome",      None)
        reflect  = r.get("reflection",   None)

        # Skip if no outcome yet
        if not outcome:
            continue

        result    = outcome.get("result",  "unknown")
        pnl       = outcome.get("pnl",     0)
        reasoning = decision.get("reasoning", [])
        ref_data  = reflect.get("data", {}) if reflect else {}

        # Build instruction
        instruction = f"""You are a professional risk-aware trading 
intelligence system analyzing Indian markets.

Analyze the following market data and make a trading decision:

Symbol     : {r['symbol']}
Price      : {market.get('price')}
Trend      : {market.get('trend')}
RSI        : {market.get('rsi')}
MACD       : {market.get('macd')}
Volatility : {market.get('volatility')}
Bollinger  : {market.get('bollinger')}
VWAP       : {market.get('vwap')}
ATR        : {market.get('atr')}

Return a JSON trading decision with:
market_condition, action, confidence,
risk_level, entry_zone, stop_loss, reasoning"""

        # Build ideal response
        ideal_response = json.dumps({
            "symbol"          : r["symbol"],
            "market_condition": r["market_condition"],
            "action"          : r["action"],
            "confidence"      : r["confidence"],
            "risk_level"      : r["risk_level"],
            "entry_zone"      : decision.get("entry_zone"),
            "stop_loss"       : decision.get("stop_loss"),
            "reasoning"       : reasoning
        }, indent=2)

        # Build reflection context
        reflection_context = ""
        if reflect and ref_data:
            reflection_context = f"""
Trade Outcome  : {result.upper()} | PnL: {pnl}
Key Lesson     : {ref_data.get('key_lesson', 'N/A')}
Improvement    : {ref_data.get('improvement', 'N/A')}
Pattern Found  : {ref_data.get('pattern_identified', 'N/A')}"""

        entry = {
            "instruction"       : instruction.strip(),
            "response"          : ideal_response,
            "outcome"           : result,
            "pnl"               : pnl,
            "reflection_context": reflection_context.strip(),
            "symbol"            : r["symbol"],
            "timestamp"         : r["timestamp"],
            "quality_score"     : score_quality(r)
        }

        finetune.append(entry)

    return finetune

# ── Score Entry Quality ───────────────────────────────
def score_quality(record):
    score = 0

    # Has outcome
    if record.get("outcome"):
        score += 30

    # Has reflection
    if record.get("reflection"):
        score += 30

    # Has full market data
    market = record.get("market_data", {})
    if all(k in market for k in
           ["rsi", "trend", "volatility", "macd"]):
        score += 20

    # Has reasoning
    decision = record.get("ai_decision", {})
    if len(decision.get("reasoning", [])) >= 2:
        score += 20

    return score

# ── Build Reflection-Only Dataset ────────────────────
def build_reflection_dataset(records):
    reflection_data = []

    for r in records:
        if not r.get("outcome") or not r.get("reflection"):
            continue

        market   = r.get("market_data", {})
        outcome  = r["outcome"]
        reflect  = r["reflection"]
        ref_data = reflect.get("data", {})

        entry = {
            "symbol"   : r["symbol"],
            "timestamp": r["timestamp"],
            "market_snapshot": {
                "trend"     : market.get("trend"),
                "rsi"       : market.get("rsi"),
                "volatility": market.get("volatility"),
                "macd"      : market.get("macd")
            },
            "trade_result": {
                "action"    : outcome.get("action_taken"),
                "result"    : outcome.get("result"),
                "pnl"       : outcome.get("pnl")
            },
            "lessons": {
                "what_went_right": ref_data.get(
                    "what_went_right", []),
                "what_went_wrong": ref_data.get(
                    "what_went_wrong", []),
                "key_lesson"     : ref_data.get(
                    "key_lesson", ""),
                "improvement"    : ref_data.get(
                    "improvement", ""),
                "pattern"        : ref_data.get(
                    "pattern_identified", "")
            }
        }
        reflection_data.append(entry)

    return reflection_data

# ── Save Datasets ─────────────────────────────────────
def save_datasets(raw, finetune, reflection):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Raw dataset
    raw_path = os.path.join(
        TRAINING_DIR, f"raw_dataset_{timestamp}.json"
    )
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2)

    # Fine-tune dataset
    ft_path = os.path.join(
        TRAINING_DIR, f"finetune_dataset_{timestamp}.json"
    )
    with open(ft_path, "w") as f:
        json.dump(finetune, f, indent=2)

    # Reflection dataset
    ref_path = os.path.join(
        TRAINING_DIR, f"reflection_dataset_{timestamp}.json"
    )
    with open(ref_path, "w") as f:
        json.dump(reflection, f, indent=2)

    # JSONL format for training tools
    jsonl_path = os.path.join(
        TRAINING_DIR, f"finetune_{timestamp}.jsonl"
    )
    with open(jsonl_path, "w") as f:
        for entry in finetune:
            f.write(json.dumps({
                "instruction": entry["instruction"],
                "output"     : entry["response"]
            }) + "\n")

    print(f"\n  ✅ Datasets saved:")
    print(f"     Raw dataset      → {raw_path}")
    print(f"     Fine-tune JSON   → {ft_path}")
    print(f"     Reflection JSON  → {ref_path}")
    print(f"     Fine-tune JSONL  → {jsonl_path}")

    return {
        "raw_path"  : raw_path,
        "ft_path"   : ft_path,
        "ref_path"  : ref_path,
        "jsonl_path": jsonl_path
    }

# ── Print Dataset Stats ───────────────────────────────
def print_stats(raw, finetune, reflection):
    print("\n" + "=" * 55)
    print("  Dataset Statistics")
    print("=" * 55)
    print(f"  Total Records        : {len(raw)}")
    print(f"  Fine-tune Entries    : {len(finetune)}")
    print(f"  Reflection Entries   : {len(reflection)}")

    if finetune:
        avg_quality = sum(
            e["quality_score"] for e in finetune
        ) / len(finetune)
        print(f"  Avg Quality Score    : {avg_quality:.1f}/100")

        profits = [e for e in finetune
                   if e["outcome"] == "profit"]
        losses  = [e for e in finetune
                   if e["outcome"] == "loss"]
        print(f"  Profitable Trades    : {len(profits)}")
        print(f"  Loss Trades          : {len(losses)}")

        if profits:
            avg_profit = sum(
                e["pnl"] for e in profits
            ) / len(profits)
            print(f"  Avg Profit PnL       : {avg_profit:.2f}")
        if losses:
            avg_loss = sum(
                e["pnl"] for e in losses
            ) / len(losses)
            print(f"  Avg Loss PnL         : {avg_loss:.2f}")

    symbols = {}
    for r in raw:
        s = r["symbol"]
        symbols[s] = symbols.get(s, 0) + 1

    print(f"\n  Records by Symbol:")
    for s, count in symbols.items():
        print(f"     {s:<12} : {count}")

    print("=" * 55)

# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Trading AI — Dataset Builder")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    # Load memory
    analyses, outcomes, reflections = load_memory()

    if len(analyses) == 0:
        print("\n  ⚠ No data yet in memory.")
        print("  Run these first:")
        print("    1. python pipelines/run_pipeline.py")
        print("    2. Use the system for a few days")
        print("    3. Come back and run this to build dataset")
    else:
        # Build datasets
        print("\n  Building datasets...")
        raw        = build_raw_dataset(
                         analyses, outcomes, reflections)
        finetune   = build_finetune_dataset(raw)
        reflection = build_reflection_dataset(raw)

        # Print stats
        print_stats(raw, finetune, reflection)

        # Save
        paths = save_datasets(raw, finetune, reflection)

        print(f"\n  🎯 Dataset ready for fine-tuning!")
        print(f"  Use the JSONL file with Unsloth")
        print(f"  when you have 100+ quality entries\n")
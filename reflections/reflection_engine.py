import json
import os
import sqlite3
import ollama
from datetime import datetime

MEMORY_DB = os.path.join(os.path.dirname(__file__), "..", "memory", "trading_memory.db")
REFLECTIONS_DIR = os.path.dirname(__file__)

# ── Build Reflection Prompt ───────────────────────────
def build_reflection_prompt(symbol, action, entry_price,
                             exit_price, pnl, result,
                             market_data, reasoning):
    return f"""You are a professional trading coach and risk analyst.

A trade was executed with the following details:

Symbol      : {symbol}
Action      : {action}
Entry Price : {entry_price}
Exit Price  : {exit_price}
PnL         : {pnl}
Result      : {result.upper()}

Market Conditions at Entry:
- Trend      : {market_data.get('trend')}
- RSI        : {market_data.get('rsi')}
- MACD       : {market_data.get('macd')}
- Volatility : {market_data.get('volatility')}
- Bollinger  : {market_data.get('bollinger')}

Original AI Reasoning:
{chr(10).join(f'- {r}' for r in reasoning)}

Analyze this trade deeply and respond with ONLY a valid JSON object:
{{
  "what_went_right": [
    "<observation 1>",
    "<observation 2>"
  ],
  "what_went_wrong": [
    "<observation 1>",
    "<observation 2>"
  ],
  "key_lesson": "<single most important lesson from this trade>",
  "improvement": "<what should be done differently next time>",
  "pattern_identified": "<market pattern noticed>",
  "confidence_assessment": "<was the confidence level appropriate? why?>",
  "risk_assessment": "<was risk managed properly? why?>"
}}"""

# ── Query AI for Reflection ───────────────────────────
def get_ai_reflection(prompt):
    print("  🤔 Generating AI reflection...")
    response = ollama.chat(
        model="qwen2.5:1.5b",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response["message"]["content"].strip()

    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"  ⚠ Could not parse reflection JSON.\n{raw}")
        return None

# ── Save Reflection to DB ─────────────────────────────
def save_reflection_to_db(outcome_id, symbol, result,
                           reflection_data):
    conn = sqlite3.connect(MEMORY_DB)
    c = conn.cursor()

    c.execute("""
        INSERT INTO reflections (
            outcome_id, timestamp, symbol,
            result, reflection, improvement
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        outcome_id,
        datetime.now().isoformat(),
        symbol,
        result,
        json.dumps(reflection_data),
        reflection_data.get("improvement", "")
    ))

    conn.commit()
    conn.close()
    print(f"  ✅ Reflection saved to database")

# ── Save Reflection to File ───────────────────────────
def save_reflection_to_file(symbol, result, reflection_data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"{symbol}_{result}_{timestamp}.json"
    path      = os.path.join(REFLECTIONS_DIR, filename)

    with open(path, "w") as f:
        json.dump({
            "timestamp"  : datetime.now().isoformat(),
            "symbol"     : symbol,
            "result"     : result,
            "reflection" : reflection_data
        }, f, indent=2)

    print(f"  ✅ Reflection file saved → {path}")
    return path

# ── Run Full Reflection ───────────────────────────────
def reflect_on_trade(outcome_id, symbol, action,
                     entry_price, exit_price,
                     market_data, reasoning):

    pnl    = round(exit_price - entry_price, 2)
    result = "profit" if pnl > 0 else "loss"

    print("\n" + "=" * 55)
    print(f"  Trading AI — Reflection Engine")
    print(f"  Symbol: {symbol} | Result: {result.upper()} | PnL: {pnl}")
    print("=" * 55)

    # Build and send prompt
    prompt     = build_reflection_prompt(
        symbol, action, entry_price,
        exit_price, pnl, result,
        market_data, reasoning
    )
    reflection = get_ai_reflection(prompt)

    if reflection:
        # Print reflection
        print(f"\n  📊 Reflection Results:")
        print(f"\n  ✅ What Went Right:")
        for r in reflection.get("what_went_right", []):
            print(f"     - {r}")

        print(f"\n  ❌ What Went Wrong:")
        for r in reflection.get("what_went_wrong", []):
            print(f"     - {r}")

        print(f"\n  💡 Key Lesson    : {reflection.get('key_lesson')}")
        print(f"  🔧 Improvement   : {reflection.get('improvement')}")
        print(f"  📈 Pattern       : {reflection.get('pattern_identified')}")
        print(f"  🎯 Confidence    : {reflection.get('confidence_assessment')}")
        print(f"  🛡  Risk          : {reflection.get('risk_assessment')}")

        # Save to DB and file
        save_reflection_to_db(outcome_id, symbol, result, reflection)
        save_reflection_to_file(symbol, result, reflection)

    return reflection

# ── Main Test ─────────────────────────────────────────
if __name__ == "__main__":

    # Simulate a completed trade
    test_market_data = {
        "trend"     : "uptrend",
        "rsi"       : 78,
        "macd"      : "bullish",
        "volatility": "high",
        "bollinger" : "above_upper"
    }

    test_reasoning = [
        "Strong uptrend confirmed by EMA alignment",
        "MACD bullish crossover detected",
        "High volatility increases risk"
    ]

    reflect_on_trade(
        outcome_id  = 1,
        symbol      = "NIFTY",
        action      = "buy",
        entry_price = 22500,
        exit_price  = 22250,
        market_data = test_market_data,
        reasoning   = test_reasoning
    )
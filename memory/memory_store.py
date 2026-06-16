import sqlite3
import json
import os
from datetime import datetime

MEMORY_DIR = os.path.dirname(__file__)
DB_PATH    = os.path.join(MEMORY_DIR, "trading_memory.db")

# ── Initialize Database ───────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    # Market Analysis Table
    c.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT,
            symbol          TEXT,
            market_data     TEXT,
            ai_decision     TEXT,
            market_condition TEXT,
            action          TEXT,
            confidence      REAL,
            risk_level      TEXT,
            entry_zone      TEXT,
            stop_loss       TEXT,
            reasoning       TEXT
        )
    """)

    # Trade Outcomes Table
    c.execute("""
        CREATE TABLE IF NOT EXISTS outcomes (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id  INTEGER,
            timestamp    TEXT,
            symbol       TEXT,
            action_taken TEXT,
            entry_price  REAL,
            exit_price   REAL,
            pnl          REAL,
            result       TEXT,
            FOREIGN KEY (analysis_id) REFERENCES analyses(id)
        )
    """)

    # Reflections Table
    c.execute("""
        CREATE TABLE IF NOT EXISTS reflections (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            outcome_id   INTEGER,
            timestamp    TEXT,
            symbol       TEXT,
            result       TEXT,
            reflection   TEXT,
            improvement  TEXT,
            FOREIGN KEY (outcome_id) REFERENCES outcomes(id)
        )
    """)

    conn.commit()
    conn.close()
    print(f"  ✅ Database initialized → {DB_PATH}")

# ── Save Analysis ─────────────────────────────────────
def save_analysis(symbol, market_data, ai_decision):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    c.execute("""
        INSERT INTO analyses (
            timestamp, symbol, market_data, ai_decision,
            market_condition, action, confidence,
            risk_level, entry_zone, stop_loss, reasoning
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        symbol,
        json.dumps(market_data),
        json.dumps(ai_decision),
        ai_decision.get("market_condition"),
        ai_decision.get("action"),
        ai_decision.get("confidence"),
        ai_decision.get("risk_level"),
        ai_decision.get("entry_zone"),
        ai_decision.get("stop_loss"),
        json.dumps(ai_decision.get("reasoning", []))
    ))

    analysis_id = c.lastrowid
    conn.commit()
    conn.close()
    print(f"  ✅ Analysis saved → ID {analysis_id}")
    return analysis_id

# ── Save Trade Outcome ────────────────────────────────
def save_outcome(analysis_id, symbol, action_taken,
                 entry_price, exit_price):
    pnl    = exit_price - entry_price
    result = "profit" if pnl > 0 else "loss"

    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    c.execute("""
        INSERT INTO outcomes (
            analysis_id, timestamp, symbol,
            action_taken, entry_price, exit_price, pnl, result
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        analysis_id,
        datetime.now().isoformat(),
        symbol,
        action_taken,
        entry_price,
        exit_price,
        round(pnl, 2),
        result
    ))

    outcome_id = c.lastrowid
    conn.commit()
    conn.close()
    print(f"  ✅ Outcome saved → ID {outcome_id} | {result.upper()} | PnL: {pnl:.2f}")
    return outcome_id

# ── Save Reflection ───────────────────────────────────
def save_reflection(outcome_id, symbol, result,
                    reflection, improvement):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

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
        reflection,
        improvement
    ))

    conn.commit()
    conn.close()
    print(f"  ✅ Reflection saved for {symbol}")

# ── Query Recent Memory ───────────────────────────────
def get_recent_analyses(symbol=None, limit=10):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    if symbol:
        c.execute("""
            SELECT timestamp, symbol, market_condition,
                   action, confidence, risk_level
            FROM analyses
            WHERE symbol = ?
            ORDER BY timestamp DESC LIMIT ?
        """, (symbol, limit))
    else:
        c.execute("""
            SELECT timestamp, symbol, market_condition,
                   action, confidence, risk_level
            FROM analyses
            ORDER BY timestamp DESC LIMIT ?
        """, (limit,))

    rows = c.fetchall()
    conn.close()
    return rows

# ── Print Memory Summary ──────────────────────────────
def print_summary():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    c.execute("SELECT COUNT(*) FROM analyses")
    analyses = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM outcomes")
    outcomes = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM reflections")
    reflections = c.fetchone()[0]

    c.execute("""
        SELECT result, COUNT(*), ROUND(AVG(pnl),2)
        FROM outcomes GROUP BY result
    """)
    stats = c.fetchall()

    conn.close()

    print("\n" + "=" * 45)
    print("  Trading AI — Memory Summary")
    print("=" * 45)
    print(f"  Total Analyses   : {analyses}")
    print(f"  Total Outcomes   : {outcomes}")
    print(f"  Total Reflections: {reflections}")
    if stats:
        print("\n  Trade Performance:")
        for result, count, avg_pnl in stats:
            print(f"    {result.upper():10} → Count: {count} | Avg PnL: {avg_pnl}")
    print("=" * 45)

# ── Main Test ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 45)
    print("  Trading AI — Memory System Init")
    print("=" * 45)

    # Initialize DB
    init_db()

    # Test with dummy data
    dummy_market = {
        "symbol": "NIFTY", "price": 22350,
        "trend": "uptrend", "rsi": 68,
        "macd": "bullish", "volatility": "medium"
    }

    dummy_decision = {
        "market_condition": "weak_uptrend",
        "action": "hold",
        "confidence": 0.72,
        "risk_level": "medium",
        "entry_zone": "22100-22200",
        "stop_loss": "21800",
        "reasoning": ["RSI elevated", "Momentum slowing"]
    }

    # Save test analysis
    analysis_id = save_analysis("NIFTY", dummy_market, dummy_decision)

    # Save test outcome
    outcome_id = save_outcome(analysis_id, "NIFTY", "hold", 22200, 22450)

    # Save test reflection
    save_reflection(
        outcome_id, "NIFTY", "profit",
        "Hold was correct given slowing momentum",
        "Watch RSI more carefully near 70"
    )

    # Print summary
    print_summary()
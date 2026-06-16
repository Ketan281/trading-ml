import os
import sys
import json
import sqlite3
import numpy as np
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(
       os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MEMORY_DB  = os.path.join(ROOT, "memory",
                           "trading_memory.db")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Feature Vector Builder ────────────────────────────
# Converts market conditions into numeric vector
# for similarity comparison

def build_feature_vector(market_data):
    """
    Convert market snapshot into numeric vector.
    This allows similarity comparison between
    different market conditions.
    """
    vec = []

    # RSI (normalize 0-100 to 0-1)
    rsi = float(market_data.get("rsi", 50))
    vec.append(rsi / 100.0)

    # Trend encoding
    trend_map = {
        "strong_uptrend"   : 1.0,
        "uptrend"          : 0.75,
        "weak_uptrend"     : 0.6,
        "sideways"         : 0.5,
        "weak_downtrend"   : 0.4,
        "downtrend"        : 0.25,
        "strong_downtrend" : 0.0
    }
    trend = market_data.get("trend", "sideways")
    vec.append(trend_map.get(trend, 0.5))

    # MACD encoding
    macd_map = {
        "bullish": 1.0,
        "bearish": 0.0,
        "neutral": 0.5
    }
    macd = market_data.get("macd", "neutral")
    vec.append(macd_map.get(macd, 0.5))

    # Volatility encoding
    vol_map = {
        "very_low": 0.0,
        "low"     : 0.25,
        "medium"  : 0.5,
        "high"    : 0.75,
        "extreme" : 1.0
    }
    vol = market_data.get("volatility", "medium")
    vec.append(vol_map.get(vol, 0.5))

    # Bollinger encoding
    bb_map = {
        "below_lower" : 0.0,
        "inside_bands": 0.5,
        "above_upper" : 1.0
    }
    bb = market_data.get("bollinger", "inside_bands")
    vec.append(bb_map.get(bb, 0.5))

    # VWAP encoding
    vwap_map = {
        "below_vwap": 0.0,
        "above_vwap": 1.0
    }
    vwap = market_data.get("vwap", "above_vwap")
    vec.append(vwap_map.get(vwap, 0.5))

    # ATR normalized (cap at 500)
    atr = float(market_data.get("atr", 100))
    vec.append(min(atr / 500.0, 1.0))

    # Price vs EMA ratio
    price  = float(market_data.get("price",  100))
    ema20  = float(market_data.get("ema_20", 100))
    if ema20 > 0:
        vec.append(min(price / ema20, 2.0) / 2.0)
    else:
        vec.append(0.5)

    return np.array(vec, dtype=float)

# ── Cosine Similarity ─────────────────────────────────
def cosine_similarity(v1, v2):
    if np.linalg.norm(v1) == 0 or \
       np.linalg.norm(v2) == 0:
        return 0.0
    return float(
        np.dot(v1, v2) /
        (np.linalg.norm(v1) * np.linalg.norm(v2))
    )

# ── Euclidean Similarity ──────────────────────────────
def euclidean_similarity(v1, v2):
    dist = np.linalg.norm(v1 - v2)
    return float(1.0 / (1.0 + dist))

# ── Combined Similarity ───────────────────────────────
def combined_similarity(v1, v2):
    cos = cosine_similarity(v1, v2)
    euc = euclidean_similarity(v1, v2)
    return round((cos * 0.6) + (euc * 0.4), 4)

# ── Load All Memory Records ───────────────────────────
def load_all_records(symbol=None):
    if not os.path.exists(MEMORY_DB):
        return []

    conn = sqlite3.connect(MEMORY_DB)
    c    = conn.cursor()

    try:
        if symbol:
            c.execute("""
                SELECT
                    a.id,
                    a.timestamp,
                    a.symbol,
                    a.market_data,
                    a.ai_decision,
                    a.action,
                    a.confidence,
                    a.risk_level,
                    a.market_condition,
                    o.result,
                    o.pnl,
                    o.entry_price,
                    o.exit_price,
                    r.reflection,
                    r.improvement
                FROM analyses a
                LEFT JOIN outcomes  o
                    ON o.analysis_id = a.id
                LEFT JOIN reflections r
                    ON r.outcome_id  = o.id
                WHERE a.symbol = ?
                ORDER BY a.timestamp DESC
            """, (symbol,))
        else:
            c.execute("""
                SELECT
                    a.id,
                    a.timestamp,
                    a.symbol,
                    a.market_data,
                    a.ai_decision,
                    a.action,
                    a.confidence,
                    a.risk_level,
                    a.market_condition,
                    o.result,
                    o.pnl,
                    o.entry_price,
                    o.exit_price,
                    r.reflection,
                    r.improvement
                FROM analyses a
                LEFT JOIN outcomes  o
                    ON o.analysis_id = a.id
                LEFT JOIN reflections r
                    ON r.outcome_id  = o.id
                ORDER BY a.timestamp DESC
            """)

        rows    = c.fetchall()
        records = []

        for row in rows:
            try:
                market_data = json.loads(row[3]) \
                              if row[3] else {}
                ai_decision = json.loads(row[4]) \
                              if row[4] else {}
                reflection  = json.loads(row[13]) \
                              if row[13] else {}
            except Exception:
                market_data = {}
                ai_decision = {}
                reflection  = {}

            records.append({
                "id"              : row[0],
                "timestamp"       : row[1],
                "symbol"          : row[2],
                "market_data"     : market_data,
                "ai_decision"     : ai_decision,
                "action"          : row[5],
                "confidence"      : row[6],
                "risk_level"      : row[7],
                "market_condition": row[8],
                "result"          : row[9],
                "pnl"             : row[10],
                "entry_price"     : row[11],
                "exit_price"      : row[12],
                "reflection"      : reflection,
                "improvement"     : row[14]
            })

        return records

    except Exception as e:
        print(f"  ⚠ Memory load failed: {e}")
        return []
    finally:
        conn.close()

# ── Find Similar Conditions ───────────────────────────
def find_similar_conditions(current_market,
                              symbol=None,
                              top_n=5,
                              min_similarity=0.70):
    """
    Find past trades with similar market conditions
    using vector similarity search.
    """
    records = load_all_records(symbol)

    if not records:
        return []

    # Build current feature vector
    current_vec = build_feature_vector(current_market)

    similarities = []

    for record in records:
        market_data = record.get("market_data", {})
        if not market_data:
            continue

        try:
            past_vec = build_feature_vector(market_data)
            sim      = combined_similarity(
                current_vec, past_vec
            )

            if sim >= min_similarity:
                similarities.append({
                    "record"    : record,
                    "similarity": sim
                })
        except Exception:
            continue

    # Sort by similarity
    similarities.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return similarities[:top_n]

# ── Find Similar Failures ─────────────────────────────
def find_similar_failures(current_market,
                           symbol=None,
                           top_n=3):
    """
    Find past LOSING trades in similar conditions.
    Helps AI avoid repeating mistakes.
    """
    similar = find_similar_conditions(
        current_market, symbol,
        top_n=20, min_similarity=0.65
    )

    failures = [
        s for s in similar
        if s["record"].get("result") == "loss"
    ]

    return failures[:top_n]

# ── Find Best Performing Setups ───────────────────────
def find_best_setups(current_market,
                     symbol=None,
                     top_n=3):
    """
    Find past WINNING trades in similar conditions.
    Helps AI repeat successful patterns.
    """
    similar = find_similar_conditions(
        current_market, symbol,
        top_n=20, min_similarity=0.65
    )

    winners = [
        s for s in similar
        if s["record"].get("result") == "profit"
        and s["record"].get("pnl", 0) > 0
    ]

    # Sort by PnL
    winners.sort(
        key=lambda x: x["record"].get("pnl", 0),
        reverse=True
    )

    return winners[:top_n]

# ── Get Pattern Statistics ────────────────────────────
def get_pattern_stats(current_market,
                       symbol=None):
    """
    Get win/loss statistics for similar conditions.
    """
    similar = find_similar_conditions(
        current_market, symbol,
        top_n=50, min_similarity=0.60
    )

    if not similar:
        return None

    with_outcomes = [
        s for s in similar
        if s["record"].get("result")
    ]

    if not with_outcomes:
        return None

    wins    = [
        s for s in with_outcomes
        if s["record"].get("result") == "profit"
    ]
    losses  = [
        s for s in with_outcomes
        if s["record"].get("result") == "loss"
    ]

    win_pnls  = [
        s["record"].get("pnl", 0) for s in wins
    ]
    loss_pnls = [
        s["record"].get("pnl", 0) for s in losses
    ]

    total     = len(with_outcomes)
    win_rate  = round(
        len(wins) / total * 100, 1
    ) if total > 0 else 0

    # Most common action in similar conditions
    actions   = [
        s["record"].get("action", "hold")
        for s in similar
    ]
    action_counts = {}
    for a in actions:
        action_counts[a] = \
            action_counts.get(a, 0) + 1

    most_common_action = max(
        action_counts,
        key=action_counts.get
    ) if action_counts else "hold"

    return {
        "total_similar"     : len(similar),
        "with_outcomes"     : total,
        "wins"              : len(wins),
        "losses"            : len(losses),
        "win_rate"          : win_rate,
        "avg_win_pnl"       : round(
            sum(win_pnls)  / len(win_pnls),  2
        ) if win_pnls  else 0,
        "avg_loss_pnl"      : round(
            sum(loss_pnls) / len(loss_pnls), 2
        ) if loss_pnls else 0,
        "most_common_action": most_common_action,
        "avg_similarity"    : round(
            sum(s["similarity"]
                for s in similar) / len(similar), 3
        )
    }

# ── Build Memory Context for AI ───────────────────────
def build_memory_context(current_market,
                          symbol=None):
    """
    Build a comprehensive memory context block
    to inject into AI prompts.
    This gives the AI access to relevant history.
    """
    print(f"  🧠 Searching memory for similar "
          f"conditions...")

    similar   = find_similar_conditions(
        current_market, symbol, top_n=5
    )
    failures  = find_similar_failures(
        current_market, symbol, top_n=3
    )
    winners   = find_best_setups(
        current_market, symbol, top_n=3
    )
    stats     = get_pattern_stats(
        current_market, symbol
    )

    if not similar:
        print(f"  ℹ No similar past conditions found")
        return None, {
            "similar_found": 0,
            "stats"        : None
        }

    print(f"  ✅ Found {len(similar)} similar "
          f"past conditions")

    # Build context string for AI prompt
    lines = []
    lines.append(
        "── MEMORY RETRIEVAL CONTEXT ────────────────"
    )

    # Pattern stats
    if stats:
        lines.append(
            f"Similar Conditions Found : "
            f"{stats['total_similar']}"
        )
        lines.append(
            f"Win Rate in Similar      : "
            f"{stats['win_rate']}%"
        )
        lines.append(
            f"Avg Win PnL              : "
            f"₹{stats['avg_win_pnl']}"
        )
        lines.append(
            f"Avg Loss PnL             : "
            f"₹{stats['avg_loss_pnl']}"
        )
        lines.append(
            f"Most Common Action       : "
            f"{stats['most_common_action']}"
        )
        lines.append("")

    # Recent similar trades
    if similar:
        lines.append("Recent Similar Conditions:")
        for i, s in enumerate(similar[:3], 1):
            r   = s["record"]
            sim = s["similarity"]
            lines.append(
                f"  {i}. [{sim:.0%} match] "
                f"{r.get('timestamp','')[:10]} | "
                f"Action={r.get('action')} | "
                f"Result={r.get('result','?')} | "
                f"PnL=₹{r.get('pnl','?')}"
            )
        lines.append("")

    # Past failures
    if failures:
        lines.append("Past Failures in Similar Conditions:")
        for i, f in enumerate(failures[:2], 1):
            r   = f["record"]
            ref = r.get("reflection", {})
            lines.append(
                f"  {i}. {r.get('timestamp','')[:10]}"
                f" | PnL=₹{r.get('pnl','?')}"
            )
            if ref.get("key_lesson"):
                lines.append(
                    f"     Lesson: "
                    f"{ref['key_lesson']}"
                )
            if r.get("improvement"):
                lines.append(
                    f"     Improve: "
                    f"{r['improvement']}"
                )
        lines.append("")

    # Past winners
    if winners:
        lines.append(
            "Successful Setups in Similar Conditions:"
        )
        for i, w in enumerate(winners[:2], 1):
            r = w["record"]
            lines.append(
                f"  {i}. {r.get('timestamp','')[:10]}"
                f" | Action={r.get('action')} | "
                f"PnL=₹{r.get('pnl','?')}"
            )
            ai_dec = r.get("ai_decision", {})
            if ai_dec.get("reasoning"):
                lines.append(
                    f"     Why: "
                    f"{ai_dec['reasoning'][0]}"
                )
        lines.append("")

    lines.append(
        "── END MEMORY CONTEXT ──────────────────────"
    )

    context_str = "\n".join(lines)

    memory_meta = {
        "similar_found"    : len(similar),
        "failures_found"   : len(failures),
        "winners_found"    : len(winners),
        "stats"            : stats,
        "top_similarity"   : similar[0]["similarity"]
                             if similar else 0
    }

    return context_str, memory_meta

# ── Memory Insight Generator ──────────────────────────
def generate_memory_insight(current_market,
                             symbol=None):
    """
    Generate a human-readable insight from memory.
    Tells you what history says about current setup.
    """
    stats = get_pattern_stats(current_market, symbol)

    if not stats or stats["with_outcomes"] == 0:
        return {
            "insight"   : "No historical data available "
                          "for this market condition",
            "confidence": "unknown",
            "suggestion": "proceed_with_caution"
        }

    win_rate = stats["win_rate"]
    avg_win  = stats["avg_win_pnl"]
    avg_loss = stats["avg_loss_pnl"]

    # Generate insight
    if win_rate >= 65:
        insight = (
            f"This setup has historically performed "
            f"well — {win_rate}% win rate in "
            f"{stats['with_outcomes']} similar trades. "
            f"Avg win ₹{avg_win} vs avg loss "
            f"₹{abs(avg_loss)}."
        )
        confidence = "high"
        suggestion = "proceed_with_confidence"

    elif win_rate >= 50:
        insight = (
            f"Mixed historical performance — "
            f"{win_rate}% win rate in "
            f"{stats['with_outcomes']} similar trades. "
            f"Risk carefully."
        )
        confidence = "medium"
        suggestion = "proceed_with_caution"

    else:
        insight = (
            f"This setup has historically failed — "
            f"only {win_rate}% win rate in "
            f"{stats['with_outcomes']} similar trades. "
            f"Consider avoiding."
        )
        confidence = "low"
        suggestion = "avoid_or_reduce_size"

    return {
        "insight"       : insight,
        "confidence"    : confidence,
        "suggestion"    : suggestion,
        "win_rate"      : win_rate,
        "sample_size"   : stats["with_outcomes"],
        "avg_win"       : avg_win,
        "avg_loss"      : avg_loss
    }

# ── Search by Outcome ─────────────────────────────────
def search_by_outcome(result_type="loss",
                       symbol=None,
                       limit=10):
    """
    Get all trades with specific outcome.
    Useful for analyzing what went wrong.
    """
    records = load_all_records(symbol)

    filtered = [
        r for r in records
        if r.get("result") == result_type
    ]

    # Sort by PnL
    if result_type == "loss":
        filtered.sort(
            key=lambda x: x.get("pnl", 0)
        )
    else:
        filtered.sort(
            key=lambda x: x.get("pnl", 0),
            reverse=True
        )

    return filtered[:limit]

# ── Memory Statistics ─────────────────────────────────
def get_memory_statistics(symbol=None):
    records = load_all_records(symbol)

    if not records:
        return {"total": 0}

    with_outcomes = [
        r for r in records
        if r.get("result")
    ]

    wins   = [
        r for r in with_outcomes
        if r.get("result") == "profit"
    ]
    losses = [
        r for r in with_outcomes
        if r.get("result") == "loss"
    ]

    # PnL stats
    win_pnls  = [
        r.get("pnl", 0) for r in wins
    ]
    loss_pnls = [
        r.get("pnl", 0) for r in losses
    ]

    # Action distribution
    action_dist = {}
    for r in records:
        a = r.get("action", "unknown")
        action_dist[a] = action_dist.get(a, 0) + 1

    # Symbol distribution
    symbol_dist = {}
    for r in records:
        s = r.get("symbol", "unknown")
        symbol_dist[s] = symbol_dist.get(s, 0) + 1

    return {
        "total_analyses"  : len(records),
        "with_outcomes"   : len(with_outcomes),
        "total_wins"      : len(wins),
        "total_losses"    : len(losses),
        "win_rate"        : round(
            len(wins) / len(with_outcomes) * 100, 1
        ) if with_outcomes else 0,
        "total_pnl"       : round(
            sum(win_pnls) + sum(loss_pnls), 2
        ),
        "avg_win"         : round(
            sum(win_pnls)  / len(win_pnls),  2
        ) if win_pnls  else 0,
        "avg_loss"        : round(
            sum(loss_pnls) / len(loss_pnls), 2
        ) if loss_pnls else 0,
        "action_dist"     : action_dist,
        "symbol_dist"     : symbol_dist
    }

# ── Main Test ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Trading AI — Memory Retrieval System")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    # Test market condition
    test_market = {
        "symbol"    : "NIFTY",
        "price"     : 23719.3,
        "trend"     : "uptrend",
        "rsi"       : 62.0,
        "macd"      : "bullish",
        "vwap"      : "above_vwap",
        "bollinger" : "inside_bands",
        "volatility": "normal",
        "atr"       : 145.0,
        "ema_20"    : 23500.0
    }

    # Memory statistics
    print("\n  📊 Overall Memory Statistics:")
    stats = get_memory_statistics()
    print(f"  Total Analyses  : {stats.get('total_analyses', 0)}")
    print(f"  With Outcomes   : {stats.get('with_outcomes', 0)}")
    print(f"  Win Rate        : {stats.get('win_rate', 0)}%")
    print(f"  Total PnL       : ₹{stats.get('total_pnl', 0)}")
    print(f"  Action Dist     : {stats.get('action_dist', {})}")

    # Find similar conditions
    print("\n  🔍 Searching similar conditions...")
    context, meta = build_memory_context(
        test_market, "NIFTY"
    )

    if context:
        print(f"\n{context}")
    else:
        print("  No memory context available yet.")
        print("  Run the pipeline daily to build memory.")

    # Memory insight
    print("\n  💡 Memory Insight:")
    insight = generate_memory_insight(
        test_market, "NIFTY"
    )
    print(f"  Insight    : {insight['insight']}")
    print(f"  Confidence : {insight['confidence']}")
    print(f"  Suggestion : {insight['suggestion']}")

    print("\n  ✅ Memory Retrieval test complete!")
    print("  Memory grows as you run the pipeline daily.")
"""
Reflection Engine V2 — self-learning feedback system.

Stores complete trade context in a journal, generates weekly/monthly/learning
reports, discovers patterns in what works and what fails, and produces
calibration data for continuous improvement.
"""

import os
import sys
import json
import sqlite3
import math
from datetime import datetime, timedelta
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DB_PATH = os.path.join(ROOT, "memory", "trading_memory.db")
OUTPUT_DIR = os.path.join(ROOT, "outputs", "reflections")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _conn():
    return sqlite3.connect(DB_PATH)


# ── Journal ────────────────────────────────────────────

def journal_trade(trade, enrichment=None, outcome=None):
    """Store complete trade context into trade_journal."""
    enrichment = enrichment or {}
    outcome = outcome or {}

    trade_id = trade.get("trade_id", f"J_{datetime.now().strftime('%Y%m%d%H%M%S')}")

    conn = _conn()
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO trade_journal
            (trade_id, timestamp, symbol, segment, direction,
             confidence, conviction, grade, trade_quality,
             breadth_score, rs_percentile, sector_phase,
             options_flow, regime, regime_day_type, mtf_alignment,
             psychology_score, entry_price, exit_price, stop_loss,
             target, position_size, capital_pct, pnl, pnl_pct,
             r_multiple, max_favorable, max_adverse,
             hold_duration, exit_reason, enrichment_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        trade_id,
        datetime.now().isoformat(),
        trade.get("symbol", ""),
        trade.get("segment", "equity"),
        trade.get("direction", "long"),
        trade.get("confidence", outcome.get("confidence")),
        trade.get("conviction", outcome.get("conviction")),
        trade.get("grade", outcome.get("grade")),
        trade.get("trade_quality", enrichment.get("trade_quality_score")),
        enrichment.get("breadth", {}).get("composite_score",
                                          enrichment.get("breadth_score")),
        enrichment.get("relative_strength", {}).get("composite_percentile",
                                                     enrichment.get("rs_percentile")),
        enrichment.get("sector_rotation", {}).get("phase",
                                                   enrichment.get("sector_phase")),
        enrichment.get("options_flow", {}).get("sentiment_score",
                                               enrichment.get("options_flow")),
        enrichment.get("regime_v2", {}).get("macro_regime",
                                            enrichment.get("regime")),
        enrichment.get("regime_v2", {}).get("day_type",
                                            enrichment.get("regime_day_type")),
        enrichment.get("multi_timeframe", {}).get("alignment",
                                                   enrichment.get("mtf_alignment")),
        enrichment.get("psychology_score"),
        outcome.get("entry_price", trade.get("entry_price")),
        outcome.get("exit_price", trade.get("exit_price")),
        outcome.get("stop_loss", trade.get("stop_loss")),
        outcome.get("target", trade.get("target")),
        outcome.get("position_size", trade.get("shares")),
        outcome.get("capital_pct"),
        outcome.get("pnl", outcome.get("net_pnl")),
        outcome.get("pnl_pct"),
        outcome.get("r_multiple"),
        outcome.get("max_favorable_excursion", outcome.get("max_favorable")),
        outcome.get("max_adverse_excursion", outcome.get("max_adverse")),
        outcome.get("hold_duration"),
        outcome.get("exit_reason"),
        json.dumps(enrichment, default=str),
    ))
    conn.commit()
    conn.close()
    return trade_id


def get_journal(limit=50, grade=None, symbol=None):
    """Query the trade journal with optional filters."""
    conn = _conn()
    c = conn.cursor()
    query = "SELECT * FROM trade_journal WHERE 1=1"
    params = []
    if grade:
        query += " AND grade = ?"
        params.append(grade)
    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    c.execute(query, params)
    rows = c.fetchall()
    cols = [d[0] for d in c.description]
    conn.close()
    return [dict(zip(cols, r)) for r in rows]


# ── Reports ────────────────────────────────────────────

def _load_trades(start, end):
    conn = _conn()
    c = conn.cursor()
    c.execute("""
        SELECT * FROM trade_journal
        WHERE timestamp >= ? AND timestamp < ?
        ORDER BY timestamp
    """, (start, end))
    rows = c.fetchall()
    cols = [d[0] for d in c.description]
    conn.close()
    return [dict(zip(cols, r)) for r in rows]


def _trade_stats(trades):
    if not trades:
        return {"trades": 0}
    pnls = [t.get("pnl", 0) or 0 for t in trades]
    r_mults = [t.get("r_multiple", 0) or 0 for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    return {
        "trades": len(trades),
        "win_rate": round(len(wins) / len(trades), 3) if trades else 0,
        "avg_r": round(sum(r_mults) / len(r_mults), 3) if r_mults else 0,
        "total_pnl": round(sum(pnls), 2),
        "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if losses and sum(losses) != 0 else 999,
        "expectancy": round(sum(pnls) / len(trades), 2) if trades else 0,
    }


def _group_by(trades, key):
    groups = defaultdict(list)
    for t in trades:
        val = t.get(key, "unknown")
        if val is None:
            val = "unknown"
        groups[str(val)].append(t)
    return {k: _trade_stats(v) for k, v in groups.items()}


def weekly_report(end_date=None):
    """Generate weekly reflection report."""
    if end_date:
        end = datetime.strptime(end_date, "%Y-%m-%d")
    else:
        end = datetime.now()
    start = end - timedelta(days=7)

    trades = _load_trades(start.isoformat(), end.isoformat())
    stats = _trade_stats(trades)

    grade_perf = _group_by(trades, "grade")
    regime_perf = _group_by(trades, "regime")
    sector_perf = _group_by(trades, "sector_phase")
    day_type_perf = _group_by(trades, "regime_day_type")

    sorted_by_r = sorted(trades, key=lambda t: t.get("r_multiple", 0) or 0, reverse=True)
    best_setups = sorted_by_r[:3] if len(sorted_by_r) >= 3 else sorted_by_r
    worst_setups = sorted_by_r[-3:] if len(sorted_by_r) >= 3 else []

    calibration = _confidence_calibration(trades)

    report = {
        "period": f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}",
        "total_trades": stats["trades"],
        "win_rate": stats.get("win_rate", 0),
        "profit_factor": stats.get("profit_factor", 0),
        "avg_r": stats.get("avg_r", 0),
        "total_pnl": stats.get("total_pnl", 0),
        "best_setups": [{
            "symbol": t.get("symbol"), "grade": t.get("grade"),
            "regime": t.get("regime"), "r_multiple": t.get("r_multiple"),
        } for t in best_setups],
        "worst_setups": [{
            "symbol": t.get("symbol"), "grade": t.get("grade"),
            "regime": t.get("regime"), "r_multiple": t.get("r_multiple"),
        } for t in worst_setups],
        "grade_performance": grade_perf,
        "regime_performance": regime_perf,
        "sector_performance": sector_perf,
        "day_type_performance": day_type_perf,
        "confidence_calibration": calibration,
        "reflection_score": compute_reflection_score(trades),
        "edge_score": compute_edge_score(trades),
        "recommendations": _generate_recommendations(trades, grade_perf, regime_perf),
    }

    _save_report("weekly", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), report)
    return report


def monthly_report(month=None):
    """Monthly deep analysis."""
    if month:
        start = datetime.strptime(month + "-01", "%Y-%m-%d")
    else:
        now = datetime.now()
        start = now.replace(day=1)

    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)

    trades = _load_trades(start.isoformat(), end.isoformat())
    base = _trade_stats(trades)

    pattern_discovery = _discover_patterns(trades)
    failure_analysis = _analyze_failures(trades)
    edge_analysis = _analyze_edge(trades)

    report = {
        "period": start.strftime("%Y-%m"),
        **base,
        "pattern_discovery": pattern_discovery,
        "failure_analysis": failure_analysis,
        "edge_analysis": edge_analysis,
        "grade_performance": _group_by(trades, "grade"),
        "regime_performance": _group_by(trades, "regime"),
        "confidence_calibration": _confidence_calibration(trades),
        "reflection_score": compute_reflection_score(trades),
        "edge_score": compute_edge_score(trades),
        "recommendations": _generate_recommendations(
            trades, _group_by(trades, "grade"), _group_by(trades, "regime")),
    }

    _save_report("monthly", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), report)
    return report


def learning_report():
    """Cumulative learning report across all history."""
    conn = _conn()
    c = conn.cursor()
    c.execute("SELECT * FROM trade_journal ORDER BY timestamp")
    rows = c.fetchall()
    cols = [d[0] for d in c.description]
    trades = [dict(zip(cols, r)) for r in rows]
    conn.close()

    if not trades:
        return {"message": "No trades in journal yet"}

    highest_expectancy = _find_best_conditions(trades)
    loss_conditions = _find_worst_conditions(trades)
    regime_edge = _group_by(trades, "regime")
    grade_edge = _group_by(trades, "grade")

    filters = []
    for cond, stats in loss_conditions.items():
        if stats.get("trades", 0) >= 5 and stats.get("win_rate", 1) < 0.35:
            filters.append(f"Avoid {cond}: {stats['win_rate']:.0%} win rate over {stats['trades']} trades")

    return {
        "total_trades": len(trades),
        "highest_expectancy_conditions": highest_expectancy,
        "loss_conditions": loss_conditions,
        "regime_edge_map": regime_edge,
        "grade_edge_map": grade_edge,
        "recommended_filters": filters,
        "confidence_calibration": _confidence_calibration(trades),
        "reflection_score": compute_reflection_score(trades),
        "edge_score": compute_edge_score(trades),
    }


# ── Scoring ────────────────────────────────────────────

def compute_reflection_score(trades):
    """0-100. How well is the system learning over time?"""
    if len(trades) < 10:
        return 50.0
    half = len(trades) // 2
    first_half = _trade_stats(trades[:half])
    second_half = _trade_stats(trades[half:])

    score = 50.0
    if second_half.get("win_rate", 0) > first_half.get("win_rate", 0):
        score += 15
    if second_half.get("avg_r", 0) > first_half.get("avg_r", 0):
        score += 15
    if second_half.get("profit_factor", 0) > first_half.get("profit_factor", 0):
        score += 10
    if second_half.get("expectancy", 0) > first_half.get("expectancy", 0):
        score += 10
    return min(100.0, max(0.0, score))


def compute_edge_score(trades):
    """0-100. Statistical edge strength."""
    if not trades:
        return 0.0
    stats = _trade_stats(trades)
    score = 0.0
    pf = stats.get("profit_factor", 0)
    if pf > 2.0:
        score += 30
    elif pf > 1.5:
        score += 20
    elif pf > 1.0:
        score += 10
    wr = stats.get("win_rate", 0)
    if wr > 0.6:
        score += 25
    elif wr > 0.5:
        score += 15
    elif wr > 0.4:
        score += 5
    avg_r = stats.get("avg_r", 0)
    if avg_r > 1.0:
        score += 25
    elif avg_r > 0.5:
        score += 15
    elif avg_r > 0:
        score += 5
    exp = stats.get("expectancy", 0)
    if exp > 0:
        score += 20
    return min(100.0, max(0.0, score))


# ── Internal Helpers ───────────────────────────────────

def _confidence_calibration(trades, bins=5):
    """Bin trades by predicted confidence, compute actual win rate per bin."""
    if not trades:
        return []
    bin_edges = [i / bins for i in range(bins + 1)]
    result = []
    for i in range(bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = [t for t in trades
                  if lo <= (t.get("confidence") or 0) < hi]
        if in_bin:
            actual_wr = len([t for t in in_bin if (t.get("pnl") or 0) > 0]) / len(in_bin)
            result.append({
                "bin": f"{lo:.0%}-{hi:.0%}",
                "predicted_avg": round(sum(t.get("confidence", 0) or 0 for t in in_bin) / len(in_bin), 3),
                "actual_win_rate": round(actual_wr, 3),
                "trades": len(in_bin),
            })
    return result


def _discover_patterns(trades):
    """Find recurring winning patterns."""
    if len(trades) < 10:
        return {"message": "Need 10+ trades for pattern discovery"}
    combos = defaultdict(list)
    for t in trades:
        key = f"{t.get('grade', '?')}_{t.get('regime', '?')}_{t.get('sector_phase', '?')}"
        combos[key].append(t.get("r_multiple", 0) or 0)
    patterns = {}
    for key, r_mults in combos.items():
        if len(r_mults) >= 3:
            avg_r = sum(r_mults) / len(r_mults)
            wr = len([r for r in r_mults if r > 0]) / len(r_mults)
            patterns[key] = {"avg_r": round(avg_r, 3), "win_rate": round(wr, 3),
                             "trades": len(r_mults)}
    return dict(sorted(patterns.items(), key=lambda x: -x[1]["avg_r"])[:10])


def _analyze_failures(trades):
    """Categorize losing trades."""
    losses = [t for t in trades if (t.get("pnl") or 0) < 0]
    if not losses:
        return {"message": "No losses to analyze"}
    by_reason = defaultdict(list)
    for t in losses:
        by_reason[t.get("exit_reason", "unknown")].append(t)
    return {reason: {"count": len(ts),
                     "avg_loss": round(sum(t.get("pnl", 0) or 0 for t in ts) / len(ts), 2)}
            for reason, ts in by_reason.items()}


def _analyze_edge(trades):
    """Find highest expectancy conditions."""
    if not trades:
        return {}
    by_grade = _group_by(trades, "grade")
    best = max(by_grade.items(), key=lambda x: x[1].get("expectancy", 0), default=(None, {}))
    return {
        "best_grade": best[0],
        "best_grade_stats": best[1],
        "overall_expectancy": _trade_stats(trades).get("expectancy", 0),
    }


def _find_best_conditions(trades):
    combos = defaultdict(list)
    for t in trades:
        key = f"{t.get('grade', '?')}|{t.get('regime', '?')}"
        combos[key].append(t.get("r_multiple", 0) or 0)
    ranked = {}
    for key, rs in combos.items():
        if len(rs) >= 3:
            ranked[key] = {"avg_r": round(sum(rs) / len(rs), 3),
                           "trades": len(rs),
                           "win_rate": round(len([r for r in rs if r > 0]) / len(rs), 3)}
    return dict(sorted(ranked.items(), key=lambda x: -x[1]["avg_r"])[:5])


def _find_worst_conditions(trades):
    combos = defaultdict(list)
    for t in trades:
        key = f"{t.get('grade', '?')}|{t.get('regime', '?')}"
        combos[key].append(t.get("r_multiple", 0) or 0)
    ranked = {}
    for key, rs in combos.items():
        if len(rs) >= 3:
            ranked[key] = _trade_stats([{"pnl": r, "r_multiple": r} for r in rs])
    return dict(sorted(ranked.items(), key=lambda x: x[1].get("avg_r", 0))[:5])


def _generate_recommendations(trades, grade_perf, regime_perf):
    recs = []
    for g, stats in grade_perf.items():
        if stats.get("trades", 0) >= 5 and stats.get("win_rate", 0) < 0.4:
            recs.append(f"Grade {g} underperforming ({stats['win_rate']:.0%} WR) — consider tightening criteria")
    for r, stats in regime_perf.items():
        if stats.get("trades", 0) >= 5 and stats.get("win_rate", 0) < 0.35:
            recs.append(f"Regime '{r}' losing — reduce exposure or avoid trading in this regime")
    if not recs:
        recs.append("No actionable recommendations — system performing within expectations")
    return recs


def _save_report(report_type, start, end, report):
    conn = _conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO reflection_reports
            (report_type, period_start, period_end, generated_at,
             report_json, reflection_score, edge_score, recommendations)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        report_type, start, end, datetime.now().isoformat(),
        json.dumps(report, default=str),
        report.get("reflection_score", 0),
        report.get("edge_score", 0),
        json.dumps(report.get("recommendations", []), default=str),
    ))
    conn.commit()
    conn.close()


# ── CLI ────────────────────────────────────────────────

if __name__ == "__main__":
    from memory.phase2_schema import migrate
    migrate()

    print("=" * 60)
    print("  REFLECTION ENGINE V2")
    print("=" * 60)

    journal = get_journal(limit=5)
    print(f"\n  Journal entries: {len(journal)}")
    for j in journal[:3]:
        print(f"    {j.get('symbol')} {j.get('grade')} R={j.get('r_multiple')} "
              f"PnL={j.get('pnl')}")

    w = weekly_report()
    print(f"\n  Weekly: {w['total_trades']} trades, "
          f"WR={w.get('win_rate', 0):.0%}, PF={w.get('profit_factor', 0):.2f}")
    print(f"  Reflection: {w.get('reflection_score', 0):.0f}, "
          f"Edge: {w.get('edge_score', 0):.0f}")
    for r in w.get("recommendations", []):
        print(f"    → {r}")

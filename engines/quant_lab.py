"""
Quant Research Lab — A/B testing, calibration, and stress testing.

Runs experiments asynchronously and stores results in DB for
continuous system improvement.
"""

import os
import sys
import json
import sqlite3
import uuid
import math
from datetime import datetime
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DB_PATH = os.path.join(ROOT, "memory", "trading_memory.db")


def _conn():
    return sqlite3.connect(DB_PATH)


# ── A/B Testing ────────────────────────────────────────

def create_experiment(name, hypothesis, variant_a, variant_b,
                      metric="r_multiple", min_samples=30):
    """Register a new A/B experiment."""
    exp_id = f"EXP_{uuid.uuid4().hex[:8]}"
    conn = _conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO quant_experiments
            (experiment_id, name, hypothesis, variant_a_config, variant_b_config,
             metric, min_samples, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?)
    """, (
        exp_id, name, hypothesis,
        json.dumps(variant_a, default=str),
        json.dumps(variant_b, default=str),
        metric, min_samples,
        datetime.now().isoformat(),
    ))
    conn.commit()
    conn.close()
    return {"experiment_id": exp_id, "name": name, "status": "active"}


def record_experiment_result(experiment_id, variant, result_value, context=None):
    """Record a single observation for an experiment variant."""
    conn = _conn()
    c = conn.cursor()
    c.execute("SELECT results_json FROM quant_experiments WHERE experiment_id = ?",
              (experiment_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return {"error": "Experiment not found"}

    results = json.loads(row[0]) if row[0] else {"a": [], "b": []}
    key = "a" if variant == "a" else "b"
    results[key].append({
        "value": result_value,
        "context": context or {},
        "timestamp": datetime.now().isoformat(),
    })

    c.execute("""
        UPDATE quant_experiments SET results_json = ?, samples_a = ?, samples_b = ?
        WHERE experiment_id = ?
    """, (json.dumps(results, default=str), len(results["a"]),
          len(results["b"]), experiment_id))
    conn.commit()
    conn.close()
    return {"recorded": True, "variant": variant, "total": len(results[key])}


def analyze_experiment(experiment_id):
    """Statistical analysis of an A/B experiment."""
    conn = _conn()
    c = conn.cursor()
    c.execute("SELECT * FROM quant_experiments WHERE experiment_id = ?",
              (experiment_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return {"error": "Experiment not found"}
    cols = [d[0] for d in c.description]
    exp = dict(zip(cols, row))
    conn.close()

    results = json.loads(exp.get("results_json") or '{"a":[],"b":[]}')
    a_vals = [r["value"] for r in results.get("a", [])]
    b_vals = [r["value"] for r in results.get("b", [])]

    if len(a_vals) < 5 or len(b_vals) < 5:
        return {"experiment_id": experiment_id, "status": "insufficient_data",
                "samples_a": len(a_vals), "samples_b": len(b_vals),
                "min_needed": exp.get("min_samples", 30)}

    mean_a = sum(a_vals) / len(a_vals)
    mean_b = sum(b_vals) / len(b_vals)
    var_a = sum((x - mean_a) ** 2 for x in a_vals) / max(1, len(a_vals) - 1)
    var_b = sum((x - mean_b) ** 2 for x in b_vals) / max(1, len(b_vals) - 1)

    pooled_se = math.sqrt(var_a / len(a_vals) + var_b / len(b_vals)) if (var_a + var_b) > 0 else 1
    t_stat = (mean_b - mean_a) / pooled_se if pooled_se > 0 else 0

    significant = abs(t_stat) > 1.96
    winner = "b" if mean_b > mean_a else "a" if mean_a > mean_b else "tie"
    effect_size = (mean_b - mean_a) / math.sqrt((var_a + var_b) / 2) if (var_a + var_b) > 0 else 0

    min_samples = exp.get("min_samples", 30)
    enough_data = len(a_vals) >= min_samples and len(b_vals) >= min_samples

    analysis = {
        "experiment_id": experiment_id,
        "name": exp.get("name"),
        "status": "conclusive" if significant and enough_data else "inconclusive",
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "std_a": round(math.sqrt(var_a), 4),
        "std_b": round(math.sqrt(var_b), 4),
        "t_statistic": round(t_stat, 4),
        "significant": significant,
        "effect_size": round(effect_size, 4),
        "winner": winner if significant else "undetermined",
        "samples_a": len(a_vals),
        "samples_b": len(b_vals),
        "recommendation": (
            f"Adopt variant {winner}" if significant and enough_data
            else "Continue collecting data"
        ),
    }

    conn = _conn()
    c = conn.cursor()
    if significant and enough_data:
        c.execute("""
            UPDATE quant_experiments SET
                status = 'concluded', winner = ?,
                p_value = ?, effect_size = ?,
                concluded_at = ?
            WHERE experiment_id = ?
        """, (winner, round(1 - min(abs(t_stat) / 4, 0.999), 4),
              round(effect_size, 4),
              datetime.now().isoformat(), experiment_id))
    conn.commit()
    conn.close()

    return analysis


def list_experiments(status=None):
    """List all experiments, optionally filtered by status."""
    conn = _conn()
    c = conn.cursor()
    if status:
        c.execute("SELECT * FROM quant_experiments WHERE status = ? ORDER BY created_at DESC",
                  (status,))
    else:
        c.execute("SELECT * FROM quant_experiments ORDER BY created_at DESC")
    rows = c.fetchall()
    cols = [d[0] for d in c.description]
    conn.close()
    return [dict(zip(cols, r)) for r in rows]


# ── Confidence Calibration ─────────────────────────────

def confidence_calibration(lookback_days=90):
    """Compare predicted confidence to actual win rates."""
    conn = _conn()
    c = conn.cursor()

    if lookback_days:
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        c.execute("""
            SELECT confidence, pnl FROM trade_journal
            WHERE timestamp >= ? AND confidence IS NOT NULL AND pnl IS NOT NULL
        """, (cutoff,))
    else:
        c.execute("""
            SELECT confidence, pnl FROM trade_journal
            WHERE confidence IS NOT NULL AND pnl IS NOT NULL
        """)
    rows = c.fetchall()
    conn.close()

    if not rows:
        return {"bins": [], "message": "No calibration data yet"}

    bins = 10
    bin_edges = [i / bins for i in range(bins + 1)]
    calibration = []

    for i in range(bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = [(conf, pnl) for conf, pnl in rows if lo <= (conf or 0) < hi]
        if in_bin:
            actual_wr = len([x for x in in_bin if x[1] > 0]) / len(in_bin)
            avg_conf = sum(x[0] for x in in_bin) / len(in_bin)
            calibration.append({
                "bin": f"{lo:.0%}-{hi:.0%}",
                "predicted": round(avg_conf, 3),
                "actual": round(actual_wr, 3),
                "gap": round(avg_conf - actual_wr, 3),
                "trades": len(in_bin),
            })

    overconfident = sum(1 for b in calibration if b["gap"] > 0.05)
    underconfident = sum(1 for b in calibration if b["gap"] < -0.05)

    return {
        "bins": calibration,
        "overall_calibration": "overconfident" if overconfident > underconfident else
                               "underconfident" if underconfident > overconfident else "well_calibrated",
        "recommendation": (
            "Reduce confidence outputs — system is overconfident" if overconfident > underconfident
            else "System is well-calibrated" if overconfident == underconfident
            else "System is conservative — could increase position sizes"
        ),
    }


# ── Feature Contribution ──────────────────────────────

def feature_contribution(lookback_days=90):
    """Analyze which enrichment features contribute most to winning trades."""
    conn = _conn()
    c = conn.cursor()

    if lookback_days:
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        c.execute("""
            SELECT enrichment_json, pnl, r_multiple FROM trade_journal
            WHERE timestamp >= ? AND enrichment_json IS NOT NULL
        """, (cutoff,))
    else:
        c.execute("""
            SELECT enrichment_json, pnl, r_multiple FROM trade_journal
            WHERE enrichment_json IS NOT NULL
        """)
    rows = c.fetchall()
    conn.close()

    if not rows:
        return {"features": [], "message": "No data for feature analysis"}

    feature_impact = defaultdict(lambda: {"win_present": 0, "loss_present": 0,
                                          "win_absent": 0, "loss_absent": 0,
                                          "r_with": [], "r_without": []})
    features_to_check = [
        "breadth", "relative_strength", "sector_rotation",
        "options_flow", "multi_timeframe", "intraday_features",
        "portfolio_risk", "regime_v2",
    ]

    for enr_json, pnl, r_mult in rows:
        try:
            enr = json.loads(enr_json) if isinstance(enr_json, str) else {}
        except (json.JSONDecodeError, TypeError):
            continue
        is_win = (pnl or 0) > 0
        r = r_mult or 0

        for feat in features_to_check:
            has_feat = feat in enr and enr[feat]
            if has_feat:
                if is_win:
                    feature_impact[feat]["win_present"] += 1
                else:
                    feature_impact[feat]["loss_present"] += 1
                feature_impact[feat]["r_with"].append(r)
            else:
                if is_win:
                    feature_impact[feat]["win_absent"] += 1
                else:
                    feature_impact[feat]["loss_absent"] += 1
                feature_impact[feat]["r_without"].append(r)

    result = []
    for feat, data in feature_impact.items():
        total_with = data["win_present"] + data["loss_present"]
        total_without = data["win_absent"] + data["loss_absent"]
        wr_with = data["win_present"] / total_with if total_with > 0 else 0
        wr_without = data["win_absent"] / total_without if total_without > 0 else 0
        avg_r_with = sum(data["r_with"]) / len(data["r_with"]) if data["r_with"] else 0
        avg_r_without = sum(data["r_without"]) / len(data["r_without"]) if data["r_without"] else 0

        result.append({
            "feature": feat,
            "win_rate_with": round(wr_with, 3),
            "win_rate_without": round(wr_without, 3),
            "lift": round(wr_with - wr_without, 3),
            "avg_r_with": round(avg_r_with, 3),
            "avg_r_without": round(avg_r_without, 3),
            "trades_with": total_with,
            "trades_without": total_without,
        })

    result.sort(key=lambda x: -x["lift"])
    return {"features": result}


# ── Strategy Comparison ────────────────────────────────

def strategy_comparison(lookback_days=90):
    """Compare performance across different strategy types."""
    conn = _conn()
    c = conn.cursor()
    if lookback_days:
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        c.execute("""
            SELECT * FROM paper_trades_v2
            WHERE status = 'closed' AND exit_time >= ?
        """, (cutoff,))
    else:
        c.execute("SELECT * FROM paper_trades_v2 WHERE status = 'closed'")
    rows = c.fetchall()
    cols = [d[0] for d in c.description]
    trades = [dict(zip(cols, r)) for r in rows]
    conn.close()

    if not trades:
        return {"strategies": [], "message": "No closed trades for comparison"}

    by_segment = defaultdict(list)
    for t in trades:
        seg = t.get("segment", "equity")
        by_segment[seg].append(t)

    strategies = []
    for seg, seg_trades in by_segment.items():
        pnls = [t.get("net_pnl", 0) or 0 for t in seg_trades]
        r_mults = [t.get("r_multiple", 0) or 0 for t in seg_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        strategies.append({
            "strategy": seg,
            "trades": len(seg_trades),
            "win_rate": round(len(wins) / len(seg_trades), 3) if seg_trades else 0,
            "avg_r": round(sum(r_mults) / len(r_mults), 3) if r_mults else 0,
            "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if losses and sum(losses) != 0 else 999,
            "total_pnl": round(sum(pnls), 2),
            "expectancy": round(sum(pnls) / len(seg_trades), 2) if seg_trades else 0,
        })

    strategies.sort(key=lambda x: -x["expectancy"])
    return {"strategies": strategies}


# ── Stress Test ────────────────────────────────────────

def stress_test(scenario="2020_crash"):
    """Simulate portfolio under historical stress scenarios."""
    scenarios = {
        "2020_crash": {
            "name": "COVID-19 Crash (Feb-Mar 2020)",
            "nifty_drawdown": -0.38,
            "vix_spike": 83.6,
            "duration_days": 33,
            "recovery_days": 140,
        },
        "2022_bear": {
            "name": "2022 Bear Market",
            "nifty_drawdown": -0.17,
            "vix_spike": 33.5,
            "duration_days": 210,
            "recovery_days": 180,
        },
        "2018_correction": {
            "name": "2018 Correction (IL&FS + NBFC Crisis)",
            "nifty_drawdown": -0.14,
            "vix_spike": 25.0,
            "duration_days": 90,
            "recovery_days": 120,
        },
        "flash_crash": {
            "name": "Flash Crash Scenario",
            "nifty_drawdown": -0.10,
            "vix_spike": 50.0,
            "duration_days": 1,
            "recovery_days": 5,
        },
    }

    if scenario not in scenarios:
        return {"error": f"Unknown scenario. Available: {list(scenarios.keys())}"}

    sc = scenarios[scenario]

    try:
        from engines.psychology_engine import load_state
        psych = load_state() or {}
    except Exception:
        psych = {}

    from engines.paper_trading_v2 import get_positions
    positions = get_positions("open")
    capital = 1_000_000

    total_exposure = sum(abs(p.get("capital_allocated", 0) or 0) for p in positions)
    gross_pct = total_exposure / capital if capital > 0 else 0

    estimated_loss = total_exposure * abs(sc["nifty_drawdown"]) * 0.8
    loss_pct = estimated_loss / capital * 100

    would_trigger_halt = loss_pct > 8
    would_trigger_restricted = loss_pct > 4

    return {
        "scenario": sc["name"],
        "market_drawdown": f"{sc['nifty_drawdown']:.0%}",
        "vix_spike_to": sc["vix_spike"],
        "duration_days": sc["duration_days"],
        "current_exposure": {
            "positions": len(positions),
            "gross_exposure": round(total_exposure, 2),
            "gross_pct": round(gross_pct, 4),
        },
        "estimated_impact": {
            "estimated_loss": round(estimated_loss, 2),
            "loss_pct": round(loss_pct, 2),
            "would_trigger_halt": would_trigger_halt,
            "would_trigger_restricted": would_trigger_restricted,
        },
        "psychology_response": {
            "current_state": psych.get("risk_state", "unknown"),
            "would_halt": would_trigger_halt,
            "recovery_days": sc["recovery_days"],
        },
        "recommendation": (
            "REDUCE EXPOSURE — current positions would breach loss limits"
            if would_trigger_halt else
            "CAUTION — losses would trigger restricted state"
            if would_trigger_restricted else
            "Position sizing appears defensive enough for this scenario"
        ),
    }


# ── CLI ────────────────────────────────────────────────

if __name__ == "__main__":
    from memory.phase2_schema import migrate
    migrate()

    print("=" * 60)
    print("  QUANT RESEARCH LAB")
    print("=" * 60)

    cal = confidence_calibration()
    print(f"\n  Calibration: {cal.get('overall_calibration', 'no data')}")
    print(f"  → {cal.get('recommendation', 'N/A')}")

    for sc in ["2020_crash", "flash_crash"]:
        result = stress_test(sc)
        print(f"\n  Stress: {result.get('scenario', sc)}")
        impact = result.get("estimated_impact", {})
        print(f"    Est. loss: ₹{impact.get('estimated_loss', 0):,.0f} "
              f"({impact.get('loss_pct', 0):.1f}%)")
        print(f"    → {result.get('recommendation', '')}")

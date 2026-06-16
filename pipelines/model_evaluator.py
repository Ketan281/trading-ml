import os
import sys
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta

ROOT = os.path.dirname(os.path.dirname(
       os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MEMORY_DB  = os.path.join(ROOT, "memory",
                           "trading_memory.db")
OUTPUT_DIR = os.path.join(ROOT, "outputs",
                          "evaluations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Evaluation Config ─────────────────────────────────
EVAL_CONFIG = {
    "rolling_window"      : 20,   # Last 20 trades
    "min_trades"          : 10,   # Min trades to eval
    "drift_threshold"     : 0.15, # 15% perf drop
    "retrain_threshold"   : 0.25, # 25% drop triggers
    "min_win_rate"        : 0.45, # Min 45% win rate
    "min_profit_factor"   : 1.0,  # Min 1.0 PF
    "max_drawdown_limit"  : 0.20, # Max 20% drawdown
    "confidence_accuracy" : 0.10, # Conf calibration
}

# ── Load Trade History ────────────────────────────────
def load_trade_history(symbol=None,
                        days_back=90):
    if not os.path.exists(MEMORY_DB):
        return []

    try:
        conn      = sqlite3.connect(MEMORY_DB)
        c         = conn.cursor()
        cutoff    = str(
            date.today() - timedelta(days=days_back)
        )

        if symbol:
            c.execute("""
                SELECT
                    a.id,
                    a.timestamp,
                    a.symbol,
                    a.action,
                    a.confidence,
                    a.risk_level,
                    a.market_condition,
                    o.result,
                    o.pnl,
                    o.entry_price,
                    o.exit_price
                FROM analyses a
                LEFT JOIN outcomes o
                    ON o.analysis_id = a.id
                WHERE a.symbol = ?
                AND DATE(a.timestamp) >= ?
                AND o.result IS NOT NULL
                ORDER BY a.timestamp ASC
            """, (symbol, cutoff))
        else:
            c.execute("""
                SELECT
                    a.id,
                    a.timestamp,
                    a.symbol,
                    a.action,
                    a.confidence,
                    a.risk_level,
                    a.market_condition,
                    o.result,
                    o.pnl,
                    o.entry_price,
                    o.exit_price
                FROM analyses a
                LEFT JOIN outcomes o
                    ON o.analysis_id = a.id
                WHERE DATE(a.timestamp) >= ?
                AND o.result IS NOT NULL
                ORDER BY a.timestamp ASC
            """, (cutoff,))

        rows    = c.fetchall()
        conn.close()

        trades = []
        for row in rows:
            trades.append({
                "id"              : row[0],
                "timestamp"       : row[1],
                "symbol"          : row[2],
                "action"          : row[3],
                "confidence"      : float(
                    row[4] or 0
                ),
                "risk_level"      : row[5],
                "market_condition": row[6],
                "result"          : row[7],
                "pnl"             : float(
                    row[8] or 0
                ),
                "entry_price"     : float(
                    row[9] or 0
                ),
                "exit_price"      : float(
                    row[10] or 0
                )
            })

        return trades

    except Exception as e:
        print(f"  ⚠ Trade history load failed: {e}")
        return []

# ── Rolling Performance ───────────────────────────────
def calculate_rolling_performance(trades,
                                   window=20):
    if len(trades) < window:
        return []

    results = []

    for i in range(window, len(trades) + 1):
        window_trades = trades[i-window:i]

        wins   = [
            t for t in window_trades
            if t["result"] == "profit"
        ]
        losses = [
            t for t in window_trades
            if t["result"] == "loss"
        ]

        win_rate = round(
            len(wins) / len(window_trades) * 100, 1
        )
        pnls     = [
            t["pnl"] for t in window_trades
        ]
        total_pnl = sum(pnls)

        win_pnls  = [
            t["pnl"] for t in wins
        ]
        loss_pnls = [
            abs(t["pnl"]) for t in losses
        ]

        profit_factor = (
            sum(win_pnls) / sum(loss_pnls)
            if loss_pnls else 99.0
        )

        # Sharpe
        if len(pnls) > 1:
            ret_mean = np.mean(pnls)
            ret_std  = np.std(pnls)
            sharpe   = (
                ret_mean / ret_std * np.sqrt(252)
                if ret_std > 0 else 0
            )
        else:
            sharpe = 0

        results.append({
            "window_end"    : window_trades[-1][
                "timestamp"
            ],
            "window_start"  : window_trades[0][
                "timestamp"
            ],
            "win_rate"      : win_rate,
            "total_pnl"     : round(total_pnl, 2),
            "profit_factor" : round(
                profit_factor, 2
            ),
            "sharpe"        : round(sharpe, 2),
            "trade_count"   : len(window_trades)
        })

    return results

# ── Confidence Calibration ────────────────────────────
def check_confidence_calibration(trades):
    """
    Checks if AI confidence correlates with
    actual win rate.
    High confidence should = higher win rate.
    """
    if len(trades) < 10:
        return None

    # Bucket trades by confidence
    buckets = {
        "low"    : [],  # < 0.5
        "medium" : [],  # 0.5 - 0.7
        "high"   : [],  # 0.7 - 0.85
        "very_high": [] # > 0.85
    }

    for trade in trades:
        conf   = trade["confidence"]
        result = 1 if trade["result"] == "profit" \
                 else 0

        if conf < 0.5:
            buckets["low"].append(result)
        elif conf < 0.7:
            buckets["medium"].append(result)
        elif conf < 0.85:
            buckets["high"].append(result)
        else:
            buckets["very_high"].append(result)

    calibration = {}
    for bucket, results in buckets.items():
        if results:
            actual_wr = round(
                sum(results) / len(results) * 100, 1
            )
            expected  = {
                "low"      : 40,
                "medium"   : 55,
                "high"     : 65,
                "very_high": 75
            }.get(bucket, 50)

            calibration[bucket] = {
                "count"     : len(results),
                "actual_wr" : actual_wr,
                "expected_wr": expected,
                "calibrated": abs(
                    actual_wr - expected
                ) <= 15,
                "gap"       : round(
                    actual_wr - expected, 1
                )
            }

    # Overall calibration score
    gaps = [
        abs(v["gap"])
        for v in calibration.values()
        if v["count"] >= 3
    ]
    cal_score = round(
        100 - min(np.mean(gaps), 50), 1
    ) if gaps else None

    return {
        "buckets"         : calibration,
        "calibration_score": cal_score,
        "is_calibrated"   : cal_score > 70
                            if cal_score else None
    }

# ── Drift Detection ───────────────────────────────────
def detect_performance_drift(rolling_perf,
                               config=None):
    config = config or EVAL_CONFIG

    if len(rolling_perf) < 3:
        return {
            "drift_detected": False,
            "reason"        : "Not enough data"
        }

    # Compare recent vs historical
    recent_count = max(3, len(rolling_perf) // 4)
    recent       = rolling_perf[-recent_count:]
    historical   = rolling_perf[:-recent_count]

    if not historical:
        return {
            "drift_detected": False,
            "reason"        : "Not enough history"
        }

    recent_wr    = np.mean(
        [r["win_rate"] for r in recent]
    )
    hist_wr      = np.mean(
        [r["win_rate"] for r in historical]
    )

    recent_pf    = np.mean(
        [r["profit_factor"] for r in recent]
    )
    hist_pf      = np.mean(
        [r["profit_factor"] for r in historical]
    )

    # Drift magnitude
    wr_drift     = (hist_wr - recent_wr) / hist_wr \
                   if hist_wr > 0 else 0
    pf_drift     = (hist_pf - recent_pf) / hist_pf \
                   if hist_pf > 0 else 0

    drift_score  = max(wr_drift, pf_drift)

    # Win rate trend
    win_rates    = [
        r["win_rate"] for r in rolling_perf
    ]
    trend        = np.polyfit(
        range(len(win_rates)), win_rates, 1
    )[0] if len(win_rates) > 2 else 0

    # Drift classification
    if drift_score >= config["retrain_threshold"]:
        drift_level = "critical"
        action      = "RETRAIN_IMMEDIATELY"
    elif drift_score >= config["drift_threshold"]:
        drift_level = "significant"
        action      = "MONITOR_CLOSELY"
    elif drift_score >= 0.05:
        drift_level = "minor"
        action      = "WATCH"
    else:
        drift_level = "none"
        action      = "CONTINUE"

    return {
        "drift_detected"  : drift_score >= 0.05,
        "drift_level"     : drift_level,
        "drift_score"     : round(drift_score, 3),
        "action"          : action,
        "recent_win_rate" : round(recent_wr,  1),
        "hist_win_rate"   : round(hist_wr,    1),
        "recent_pf"       : round(recent_pf,  2),
        "hist_pf"         : round(hist_pf,    2),
        "wr_drift"        : round(wr_drift,   3),
        "pf_drift"        : round(pf_drift,   3),
        "win_rate_trend"  : round(trend,      3),
        "trend_direction" : (
            "improving"  if trend > 0.2
            else "declining" if trend < -0.2
            else "stable"
        )
    }

# ── Signal Quality Score ──────────────────────────────
def calculate_signal_quality(trades,
                               window=20):
    if len(trades) < 5:
        return None

    recent = trades[-window:] \
             if len(trades) >= window else trades

    wins   = [
        t for t in recent
        if t["result"] == "profit"
    ]
    losses = [
        t for t in recent
        if t["result"] == "loss"
    ]

    if not recent:
        return None

    # Base score from win rate
    win_rate  = len(wins) / len(recent)
    score     = win_rate * 40

    # Profit factor component
    win_pnls  = [t["pnl"] for t in wins]
    loss_pnls = [abs(t["pnl"]) for t in losses]

    if win_pnls and loss_pnls:
        pf     = sum(win_pnls) / sum(loss_pnls)
        score += min(pf / 3 * 30, 30)

    # Consistency component
    pnls      = [t["pnl"] for t in recent]
    if len(pnls) > 1:
        pnl_std = np.std(pnls)
        pnl_mean = np.mean(pnls)
        cv      = abs(pnl_std / pnl_mean) \
                  if pnl_mean != 0 else 99
        cons_score = max(0, 20 - cv * 5)
        score  += min(cons_score, 20)

    # Confidence accuracy
    high_conf  = [
        t for t in recent
        if t["confidence"] >= 0.7
    ]
    if high_conf:
        hc_wins = [
            t for t in high_conf
            if t["result"] == "profit"
        ]
        hc_wr   = len(hc_wins) / len(high_conf)
        score  += hc_wr * 10

    score = round(min(score, 100), 1)

    # Grade
    if score >= 80:
        grade = "A+"
        status = "excellent"
    elif score >= 65:
        grade = "A"
        status = "good"
    elif score >= 50:
        grade = "B"
        status = "average"
    elif score >= 35:
        grade = "C"
        status = "below_average"
    else:
        grade = "D"
        status = "poor"

    return {
        "score"    : score,
        "grade"    : grade,
        "status"   : status,
        "win_rate" : round(win_rate * 100, 1),
        "sample"   : len(recent)
    }

# ── Condition-Based Analysis ──────────────────────────
def analyze_by_condition(trades):
    """
    Breaks down performance by market condition.
    Helps identify which regimes the AI excels in.
    """
    conditions = {}

    for trade in trades:
        cond = trade.get(
            "market_condition", "unknown"
        )
        if cond not in conditions:
            conditions[cond] = []
        conditions[cond].append(trade)

    analysis = {}
    for cond, cond_trades in conditions.items():
        if len(cond_trades) < 3:
            continue

        wins = [
            t for t in cond_trades
            if t["result"] == "profit"
        ]
        pnls = [t["pnl"] for t in cond_trades]

        analysis[cond] = {
            "count"   : len(cond_trades),
            "win_rate": round(
                len(wins) / len(cond_trades) * 100,
                1
            ),
            "avg_pnl" : round(np.mean(pnls), 2),
            "total_pnl": round(sum(pnls), 2)
        }

    # Sort by win rate
    return dict(
        sorted(
            analysis.items(),
            key=lambda x: x[1]["win_rate"],
            reverse=True
        )
    )

# ── Retraining Check ──────────────────────────────────
def check_retraining_needed(drift,
                              quality,
                              calibration,
                              config=None):
    config   = config or EVAL_CONFIG
    reasons  = []
    urgency  = "none"

    if drift.get("action") == "RETRAIN_IMMEDIATELY":
        reasons.append(
            f"Critical drift detected: "
            f"{drift['drift_score']:.1%}"
        )
        urgency = "immediate"

    if quality and quality["score"] < 35:
        reasons.append(
            f"Signal quality very low: "
            f"{quality['score']}/100"
        )
        urgency = "immediate" \
                  if urgency != "immediate" \
                  else urgency

    if quality and \
            quality["win_rate"] < \
            config["min_win_rate"] * 100:
        reasons.append(
            f"Win rate below minimum: "
            f"{quality['win_rate']}%"
        )
        if urgency == "none":
            urgency = "soon"

    if calibration and \
            calibration.get(
                "calibration_score", 100
            ) < 50:
        reasons.append(
            "AI confidence poorly calibrated"
        )
        if urgency == "none":
            urgency = "soon"

    if drift.get("action") == "MONITOR_CLOSELY" \
            and quality \
            and quality["score"] < 50:
        reasons.append(
            "Significant drift + below average quality"
        )
        if urgency == "none":
            urgency = "planned"
    return {
        "retraining_needed": len(reasons) > 0,
        "urgency"          : urgency,
        "reasons"          : reasons,
        "recommendation"   : get_retrain_recommendation(
            urgency
        )
    }

def get_retrain_recommendation(urgency):
    recs = {
        "immediate": {
            "action" : "Retrain model NOW",
            "steps"  : [
                "Run dataset_builder.py",
                "Collect more outcome data",
                "Fine-tune Qwen on new data",
                "Validate on walk-forward test"
            ],
            "icon"   : "🔴"
        },
        "soon": {
            "action" : "Plan retraining this week",
            "steps"  : [
                "Collect 20+ more trade outcomes",
                "Run dataset_builder.py",
                "Review reflection quality",
                "Schedule fine-tuning session"
            ],
            "icon"   : "🟠"
        },
        "planned": {
            "action" : "Monitor for 2 more weeks",
            "steps"  : [
                "Continue collecting data",
                "Review indicator parameters",
                "Check if market regime changed"
            ],
            "icon"   : "🟡"
        },
        "none": {
            "action" : "System performing well",
            "steps"  : [
                "Continue normal operation",
                "Keep collecting trade data",
                "Review monthly"
            ],
            "icon"   : "✅"
        }
    }
    return recs.get(urgency, recs["none"])

# ── Full Evaluation ───────────────────────────────────
def run_full_evaluation(symbol=None,
                         days_back=90):
    print(f"\n{'=' * 60}")
    print(f"  Model Evaluator"
          f"{' — ' + symbol if symbol else ''}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")

    # Load trade history
    trades = load_trade_history(symbol, days_back)

    if not trades:
        print(f"\n  ⚠ No trade history found.")
        print(f"  Run the pipeline daily to build "
              f"trade history.")
        return None

    print(f"\n  📊 Loaded {len(trades)} trades "
          f"from last {days_back} days")

    min_trades = EVAL_CONFIG["min_trades"]
    if len(trades) < min_trades:
        print(f"  ⚠ Need at least {min_trades} "
              f"trades. Have {len(trades)}.")
        print(f"  Continue running pipeline to "
              f"build history.")
        return {
            "status"    : "insufficient_data",
            "trades"    : len(trades),
            "needed"    : min_trades
        }

    # Rolling performance
    print(f"\n  📈 Calculating rolling performance...")
    rolling = calculate_rolling_performance(
        trades, EVAL_CONFIG["rolling_window"]
    )

    # Drift detection
    print(f"  🔍 Detecting performance drift...")
    drift = detect_performance_drift(rolling)

    # Signal quality
    print(f"  🎯 Calculating signal quality...")
    quality = calculate_signal_quality(trades)

    # Confidence calibration
    print(f"  🧮 Checking confidence calibration...")
    calibration = check_confidence_calibration(
        trades
    )

    # Condition analysis
    print(f"  📋 Analyzing by market condition...")
    by_condition = analyze_by_condition(trades)

    # Retraining check
    retrain = check_retraining_needed(
        drift, quality, calibration
    )

    # Overall stats
    wins      = [
        t for t in trades
        if t["result"] == "profit"
    ]
    losses    = [
        t for t in trades
        if t["result"] == "loss"
    ]
    pnls      = [t["pnl"] for t in trades]
    total_pnl = sum(pnls)

    win_rate  = round(
        len(wins) / len(trades) * 100, 1
    )
    avg_win   = round(
        np.mean([t["pnl"] for t in wins]), 2
    ) if wins else 0
    avg_loss  = round(
        np.mean([t["pnl"] for t in losses]), 2
    ) if losses else 0

    pf        = round(
        sum(t["pnl"] for t in wins) /
        abs(sum(t["pnl"] for t in losses)), 2
    ) if losses else 99.0

    # Print results
    print(f"\n  {'─' * 58}")
    print(f"  📊 OVERALL PERFORMANCE")
    print(f"  {'─' * 58}")
    print(f"  Total Trades    : {len(trades)}")
    print(f"  Win Rate        : {win_rate}%")
    print(f"  Total PnL       : ₹{total_pnl:,.2f}")
    print(f"  Avg Win         : ₹{avg_win:,.2f}")
    print(f"  Avg Loss        : ₹{avg_loss:,.2f}")
    print(f"  Profit Factor   : {pf}")

    if quality:
        print(f"\n  {'─' * 58}")
        print(f"  🎯 SIGNAL QUALITY")
        print(f"  {'─' * 58}")
        print(f"  Quality Score   : "
              f"{quality['score']}/100")
        print(f"  Grade           : "
              f"{quality['grade']}")
        print(f"  Status          : "
              f"{quality['status'].upper()}")

    print(f"\n  {'─' * 58}")
    print(f"  🔍 DRIFT ANALYSIS")
    print(f"  {'─' * 58}")
    print(f"  Drift Level     : "
          f"{drift['drift_level'].upper()}")
    print(f"  Drift Score     : "
          f"{drift['drift_score']:.1%}")
    print(f"  Recent WR       : "
          f"{drift.get('recent_win_rate', 0)}%")
    print(f"  Historical WR   : "
          f"{drift.get('hist_win_rate', 0)}%")
    print(f"  Trend           : "
          f"{drift.get('trend_direction', 'N/A')}")
    print(f"  Action          : "
          f"{drift['action']}")

    if calibration:
        cal_score = calibration.get(
            "calibration_score"
        )
        print(f"\n  {'─' * 58}")
        print(f"  🧮 CONFIDENCE CALIBRATION")
        print(f"  {'─' * 58}")
        print(f"  Cal Score       : "
              f"{cal_score}/100")
        print(f"  Calibrated      : "
              f"{'✅ YES' if calibration.get('is_calibrated') else '❌ NO'}")

        for bucket, data in calibration[
            "buckets"
        ].items():
            if data["count"] >= 3:
                gap_icon = (
                    "✅" if abs(data["gap"]) <= 15
                    else "⚠️"
                )
                print(
                    f"     {gap_icon} "
                    f"{bucket:<12} | "
                    f"Actual: {data['actual_wr']}% | "
                    f"Expected: {data['expected_wr']}% | "
                    f"Gap: {data['gap']:+.1f}%"
                )

    if by_condition:
        print(f"\n  {'─' * 58}")
        print(f"  📋 PERFORMANCE BY CONDITION")
        print(f"  {'─' * 58}")
        print(
            f"  {'CONDITION':<25} {'COUNT':<8} "
            f"{'WIN RATE':<12} {'AVG PnL'}"
        )
        print("  " + "─" * 55)
        for cond, stats in list(
            by_condition.items()
        )[:8]:
            icon = (
                "✅" if stats["win_rate"] >= 55
                else "⚠️" if stats["win_rate"] >= 45
                else "❌"
            )
            print(
                f"  {icon} {cond:<23} "
                f"{stats['count']:<8} "
                f"{stats['win_rate']}%{'':<7} "
                f"₹{stats['avg_pnl']:,.2f}"
            )

    # Retraining recommendation
    rec = retrain["recommendation"]
    print(f"\n  {'═' * 58}")
    print(
        f"  {rec['icon']} RETRAINING: "
        f"{retrain['urgency'].upper()}"
    )
    print(f"  Action: {rec['action']}")
    if retrain["reasons"]:
        print(f"  Reasons:")
        for r in retrain["reasons"]:
            print(f"     → {r}")
    print(f"  Next Steps:")
    for step in rec["steps"]:
        print(f"     • {step}")
    print(f"  {'═' * 58}")

    # Build full report
    report = {
        "timestamp"     : datetime.now().isoformat(),
        "symbol"        : symbol,
        "days_analyzed" : days_back,
        "total_trades"  : len(trades),
        "overall"       : {
            "win_rate"     : win_rate,
            "total_pnl"    : round(total_pnl, 2),
            "avg_win"      : avg_win,
            "avg_loss"     : avg_loss,
            "profit_factor": pf
        },
        "quality"       : quality,
        "drift"         : drift,
        "calibration"   : calibration,
        "by_condition"  : by_condition,
        "rolling"       : rolling[-10:]
                          if rolling else [],
        "retraining"    : retrain
    }

    # Save
    path = os.path.join(
        OUTPUT_DIR,
        f"evaluation_"
        f"{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    )
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  ✅ Evaluation saved → {path}")
    return report

# ── Run All Symbols ───────────────────────────────────
def run_all_evaluations():
    symbols = [
        "NIFTY", "BANKNIFTY",
        "RELIANCE", "TCS"
    ]

    print("\n" + "🔥" * 27)
    print("  CONTINUOUS MODEL EVALUATION")
    print("🔥" * 27)

    reports  = []
    alerts   = []

    for symbol in symbols:
        report = run_full_evaluation(symbol)
        if report and isinstance(report, dict):
            reports.append(report)

            # Collect alerts
            retrain = report.get("retraining", {})
            if retrain.get(
                "urgency"
            ) in ["immediate", "soon"]:
                alerts.append({
                    "symbol" : symbol,
                    "urgency": retrain["urgency"],
                    "reasons": retrain.get(
                        "reasons", []
                    )
                })

    # Global summary
    if reports:
        print(f"\n{'=' * 60}")
        print(f"  EVALUATION SUMMARY")
        print(f"{'=' * 60}")

        for r in reports:
            if not isinstance(r, dict):
                continue
            sym      = r.get("symbol", "?")
            overall  = r.get("overall", {})
            quality  = r.get("quality") or {}
            drift    = r.get("drift",   {})
            retrain  = r.get("retraining", {})

            print(
                f"  {sym:<12} | "
                f"WR: {overall.get('win_rate',0)}% | "
                f"Quality: {quality.get('score','N/A')}/100 | "
                f"Drift: {drift.get('drift_level','N/A')} | "
                f"Retrain: {retrain.get('urgency','none')}"
            )

    if alerts:
        print(f"\n  🔴 URGENT ALERTS:")
        for alert in alerts:
            print(f"     {alert['symbol']}: "
                  f"{alert['urgency'].upper()}")
            for reason in alert["reasons"]:
                print(f"       → {reason}")

    return reports

# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Trading AI — Continuous Model Evaluator")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    run_all_evaluations()
    print("\n  ✅ Model Evaluation complete!")
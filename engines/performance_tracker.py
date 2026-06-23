"""
Performance Tracker — log every prediction, measure accuracy over time.

This is how the system gets smarter:
1. Every prediction is logged with timestamp
2. After the horizon passes, actual outcomes are recorded
3. Accuracy, calibration, and decay metrics are computed
4. Retraining is triggered when performance degrades

The model improves because:
- We know WHICH features actually predicted well
- We know WHEN the model started degrading
- We know which market regimes the model works best in
"""

import os
import sys
import json
import sqlite3
import logging
import numpy as np
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

log = logging.getLogger("performance_tracker")
DB_PATH = os.path.join(ROOT, "data", "ml_performance.db")


def _db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = _db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            ml_score REAL,
            ml_rank INTEGER,
            confidence REAL,
            predicted_action TEXT,
            entry_price REAL,
            model_id TEXT,
            horizon TEXT,
            actual_return REAL,
            actual_hit_target INTEGER,
            outcome_recorded INTEGER DEFAULT 0,
            outcome_date TEXT,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT,
            date TEXT,
            period TEXT,
            total_predictions INTEGER,
            accuracy REAL,
            precision_at_10 REAL,
            rank_ic REAL,
            avg_return_top10 REAL,
            avg_return_bottom10 REAL,
            long_short_spread REAL,
            win_rate REAL,
            sharpe REAL,
            computed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS retrain_triggers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trigger_type TEXT,
            reason TEXT,
            metrics TEXT,
            triggered_at TEXT,
            resolved INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_pred_date ON predictions(date);
        CREATE INDEX IF NOT EXISTS idx_pred_sym ON predictions(symbol, date);
        CREATE INDEX IF NOT EXISTS idx_pred_model ON predictions(model_id);
        CREATE INDEX IF NOT EXISTS idx_pred_outcome ON predictions(outcome_recorded);
    """)
    conn.commit()
    conn.close()


def log_predictions(picks, model_id, horizon="21d"):
    """Log a batch of predictions from ml_inference.predict_all()."""
    init_db()
    conn = _db()
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().isoformat()

    for pick in picks:
        conn.execute(
            "INSERT OR REPLACE INTO predictions "
            "(date, symbol, ml_score, ml_rank, confidence, predicted_action, "
            "entry_price, model_id, horizon, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (today, pick["symbol"], pick.get("ml_score"), pick.get("ml_rank", pick.get("rank")),
             pick.get("confidence"), pick.get("action", "buy"),
             pick.get("price", pick.get("entry")), model_id, horizon, now)
        )

    conn.commit()
    conn.close()
    return len(picks)


def record_outcomes():
    """Check past predictions and record actual outcomes.
    Call daily — looks up actual prices from feature store."""
    init_db()
    conn = _db()

    pending = conn.execute(
        "SELECT DISTINCT date, symbol, horizon, entry_price FROM predictions "
        "WHERE outcome_recorded = 0 AND date < date('now', '-5 day')"
    ).fetchall()

    if not pending:
        return 0

    try:
        from engines.feature_store import _db as _feat_db
        feat_conn = _feat_db()
    except Exception:
        return 0

    updated = 0
    for row in pending:
        pred_date = row["date"]
        symbol = row["symbol"]
        horizon = row["horizon"] or "21d"
        entry_price = row["entry_price"]

        horizon_days = int(horizon.replace("d", ""))
        target_date = (datetime.strptime(pred_date, "%Y-%m-%d") +
                       timedelta(days=int(horizon_days * 1.5))).strftime("%Y-%m-%d")

        # Get price at target date
        price_row = feat_conn.execute(
            "SELECT price FROM features WHERE symbol=? AND date >= ? ORDER BY date ASC LIMIT 1",
            (symbol, target_date)
        ).fetchone()

        if price_row and entry_price and entry_price > 0:
            actual_price = price_row[0]
            actual_return = (actual_price / entry_price - 1)
            hit_target = 1 if actual_return > 0 else 0

            conn.execute(
                "UPDATE predictions SET actual_return=?, actual_hit_target=?, "
                "outcome_recorded=1, outcome_date=? "
                "WHERE date=? AND symbol=? AND outcome_recorded=0",
                (round(actual_return, 6), hit_target, datetime.now().strftime("%Y-%m-%d"),
                 pred_date, symbol)
            )
            updated += 1

    conn.commit()
    conn.close()
    try:
        feat_conn.close()
    except Exception:
        pass
    return updated


def compute_performance(model_id=None, lookback_days=30):
    """Compute performance metrics for a model over recent period."""
    init_db()
    conn = _db()

    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    query = ("SELECT * FROM predictions WHERE outcome_recorded = 1 AND date >= ?")
    params = [cutoff]
    if model_id:
        query += " AND model_id = ?"
        params.append(model_id)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    if not rows:
        return {"error": "No completed predictions in this period", "n": 0}

    returns = [r["actual_return"] for r in rows if r["actual_return"] is not None]
    hits = [r["actual_hit_target"] for r in rows if r["actual_hit_target"] is not None]

    # Top 10 vs Bottom 10 per date
    from collections import defaultdict
    by_date = defaultdict(list)
    for r in rows:
        by_date[r["date"]].append(r)

    top10_rets, bot10_rets = [], []
    for date, preds in by_date.items():
        preds.sort(key=lambda x: x["ml_score"] or 0, reverse=True)
        if len(preds) >= 10:
            top10_rets.extend([p["actual_return"] for p in preds[:10] if p["actual_return"] is not None])
            bot10_rets.extend([p["actual_return"] for p in preds[-10:] if p["actual_return"] is not None])

    avg_top10 = float(np.mean(top10_rets)) if top10_rets else 0
    avg_bot10 = float(np.mean(bot10_rets)) if bot10_rets else 0

    metrics = {
        "period_days": lookback_days,
        "total_predictions": len(rows),
        "outcomes_recorded": len(returns),
        "accuracy": round(float(np.mean(hits)) * 100, 1) if hits else 0,
        "win_rate": round(float(np.mean([1 if r > 0 else 0 for r in returns])) * 100, 1) if returns else 0,
        "avg_return": round(float(np.mean(returns)) * 100, 2) if returns else 0,
        "avg_return_top10": round(avg_top10 * 100, 2),
        "avg_return_bottom10": round(avg_bot10 * 100, 2),
        "long_short_spread": round((avg_top10 - avg_bot10) * 100, 2),
        "best_prediction": round(float(max(returns)) * 100, 2) if returns else 0,
        "worst_prediction": round(float(min(returns)) * 100, 2) if returns else 0,
    }

    if returns:
        rets = np.array(returns)
        metrics["sharpe"] = round(float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252)), 3)

    return metrics


def check_retrain_needed(model_id=None):
    """Check if model performance has degraded enough to trigger retraining."""
    recent = compute_performance(model_id, lookback_days=14)
    baseline = compute_performance(model_id, lookback_days=60)

    if recent.get("error") or baseline.get("error"):
        return {"needs_retrain": False, "reason": "insufficient data"}

    triggers = []

    # Accuracy dropped >10% from baseline
    if recent["accuracy"] < baseline["accuracy"] - 10:
        triggers.append(f"Accuracy dropped: {recent['accuracy']:.1f}% vs {baseline['accuracy']:.1f}% baseline")

    # L/S spread turned negative
    if recent["long_short_spread"] < 0:
        triggers.append(f"Long/Short spread negative: {recent['long_short_spread']:.2f}%")

    # Win rate below 45%
    if recent["win_rate"] < 45 and recent["outcomes_recorded"] > 20:
        triggers.append(f"Win rate low: {recent['win_rate']:.1f}%")

    if triggers:
        conn = _db()
        conn.execute(
            "INSERT INTO retrain_triggers (trigger_type, reason, metrics, triggered_at) "
            "VALUES (?, ?, ?, ?)",
            ("performance_decay", "; ".join(triggers),
             json.dumps({"recent": recent, "baseline": baseline}),
             datetime.now().isoformat())
        )
        conn.commit()
        conn.close()

    return {
        "needs_retrain": len(triggers) > 0,
        "triggers": triggers,
        "recent_metrics": recent,
        "baseline_metrics": baseline,
    }


def get_prediction_history(symbol=None, days=30, limit=100):
    """Get prediction history for display in frontend."""
    init_db()
    conn = _db()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    query = "SELECT * FROM predictions WHERE date >= ?"
    params = [cutoff]
    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
    query += " ORDER BY date DESC, ml_rank ASC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_accuracy_trend(days=90):
    """Get daily accuracy over time for charting."""
    init_db()
    conn = _db()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    rows = conn.execute(
        "SELECT date, "
        "COUNT(*) as total, "
        "SUM(CASE WHEN actual_hit_target=1 THEN 1 ELSE 0 END) as correct, "
        "AVG(actual_return) as avg_return "
        "FROM predictions WHERE outcome_recorded=1 AND date >= ? "
        "GROUP BY date ORDER BY date",
        (cutoff,)
    ).fetchall()
    conn.close()

    return [{
        "date": r["date"],
        "total": r["total"],
        "accuracy": round(r["correct"] / r["total"] * 100, 1) if r["total"] > 0 else 0,
        "avg_return": round(r["avg_return"] * 100, 2) if r["avg_return"] else 0,
    } for r in rows]


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    init_db()
    n = record_outcomes()
    print(f"Recorded {n} outcomes")
    metrics = compute_performance()
    print(json.dumps(metrics, indent=2))

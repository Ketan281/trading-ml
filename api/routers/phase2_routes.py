"""
Phase 2 API routes — all endpoints under /phase2/.

Mounted on the main FastAPI app via include_router.
"""

import os
import sys
import json
import time as _time
from typing import Optional

from fastapi import APIRouter, Query

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

router = APIRouter(prefix="/phase2", tags=["Phase 2"])


def _safe(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as e:
        return {"error": str(e)}


_cache = {}

def _cached(key, fn, ttl=300, *a, **kw):
    """Cache expensive computations for ttl seconds."""
    now = _time.time()
    if key in _cache:
        ts, val = _cache[key]
        if now - ts < ttl:
            return val
    val = _safe(fn, *a, **kw)
    _cache[key] = (now, val)
    return val


# ── Psychology ─────────────────────────────────────────

@router.get("/psychology")
def psychology_state():
    """Current psychology state, scores, risk state, and limits usage."""
    from engines.psychology_engine import load_state, get_risk_state
    state = load_state()
    return {
        "state": state,
        "risk_state": get_risk_state(state),
    }


@router.get("/psychology/events")
def psychology_events(limit: int = Query(50, ge=1, le=500)):
    """Recent psychology events (revenge detection, cooldowns, etc)."""
    import sqlite3
    db = os.path.join(ROOT, "memory", "trading_memory.db")
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute("""
        SELECT * FROM psychology_events ORDER BY timestamp DESC LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    cols = [d[0] for d in c.description]
    conn.close()
    return {"events": [dict(zip(cols, r)) for r in rows]}


# ── Regime ─────────────────────────────────────────────

@router.get("/regime")
def regime_status():
    """Current market regime classification + day type."""
    if os.environ.get("AOS_PROFILE") == "micro":
        from engines.regime_v2 import STRATEGY_ADAPTATION
        from pipelines.market_regime import _load_index, regime_at
        macro = "unknown"
        try:
            df = _load_index("NIFTY")
            if df is not None:
                r = regime_at(df)
                macro = r.get("label", "unknown") if isinstance(r, dict) else "unknown"
        except Exception:
            pass
        return {
            "macro_regime": macro,
            "day_type": "range_day",
            "regime_confidence": 0.3,
            "strategy_adaptation": STRATEGY_ADAPTATION["range_day"],
        }
    from engines.regime_v2 import classify_regime_v2
    return _cached("regime", classify_regime_v2, 300)


# ── Conviction Grading ─────────────────────────────────

@router.get("/conviction/{symbol}")
def conviction_grade(symbol: str):
    """Grade a symbol's current signal."""
    from engines.conviction_v2 import grade_signal
    signal = {"symbol": symbol.upper(), "final_action": "buy",
              "fused_confidence": 0.65, "price": 0,
              "volatility": "medium", "agreement": "moderate"}
    try:
        from pipelines.combined_intelligence import run_full_pipeline
        result = run_full_pipeline(symbol.upper())
        if result:
            signal.update(result)
    except Exception:
        pass
    return _safe(grade_signal, signal)


# ── Trade Quality ──────────────────────────────────────

@router.get("/quality/{symbol}")
def trade_quality(symbol: str):
    """Trade quality breakdown for a symbol."""
    from engines.trade_quality import compute_trade_quality
    signal = {"symbol": symbol.upper()}
    enrichment = {}
    try:
        from pipelines.institutional_engine import enrich_signal
        signal = enrich_signal(signal, None)
        enrichment = signal.get("enrichment", {})
    except Exception:
        pass
    return _safe(compute_trade_quality, signal, enrichment)


# ── Explainability ─────────────────────────────────────

@router.get("/explain/{symbol}")
def explain_symbol(symbol: str):
    """Full 'Why This Trade?' explanation."""
    from engines.recommendation_engine import full_pipeline
    signal = {"symbol": symbol.upper(), "final_action": "buy",
              "fused_confidence": 0.65, "price": 0}
    try:
        from pipelines.combined_intelligence import run_full_pipeline
        result = run_full_pipeline(symbol.upper())
        if result:
            signal.update(result)
    except Exception:
        pass
    rec = full_pipeline(signal)
    return {
        "symbol": symbol.upper(),
        "grade": rec.get("grade"),
        "strategy": rec.get("strategy"),
        "explanation": rec.get("explanation"),
        "grade_detail": rec.get("grade_detail"),
        "quality_detail": rec.get("quality_detail"),
    }


# ── Recommendations ────────────────────────────────────

@router.get("/recommendations")
def recommendations(capital: float = Query(1_000_000)):
    """Institutional-grade recommendations — graded, sized, explained."""
    from engines.recommendation_engine import generate_recommendations
    try:
        from pipelines.screener import screen
        screened = screen()
        signals = []
        picks = screened.get("picks") or screened.get("stocks") or []
        for p in picks[:20]:
            sig = p if isinstance(p, dict) else {"symbol": p}
            sig.setdefault("final_action", "buy")
            sig.setdefault("fused_confidence", 0.6)
            signals.append(sig)
    except Exception:
        signals = []

    if not signals:
        return {"recommendations": [], "no_trade_reasons": [],
                "market_summary": "No signals available", "stats": {}}

    return _safe(generate_recommendations, signals, capital)


# ── Paper Trading V2 ──────────────────────────────────

@router.get("/paper/positions")
def paper_positions(status: str = Query("open")):
    """Paper trading positions (open, closed, or all)."""
    from engines.paper_trading_v2 import get_positions
    return {"positions": _safe(get_positions, status)}


@router.get("/paper/metrics")
def paper_metrics(lookback_days: Optional[int] = Query(None)):
    """Paper trading performance metrics."""
    from engines.paper_trading_v2 import compute_metrics
    return _safe(compute_metrics, lookback_days)


@router.get("/paper/equity-curve")
def paper_equity_curve():
    """Paper trading equity curve for charting."""
    from engines.paper_trading_v2 import equity_curve
    return {"curve": _safe(equity_curve)}


@router.get("/paper/readiness")
def paper_readiness():
    """Is the system ready for live trading?"""
    from engines.paper_trading_v2 import readiness_score
    return _safe(readiness_score)


@router.post("/paper/update")
def paper_update():
    """Mark-to-market all open paper positions."""
    from engines.paper_trading_v2 import update_positions
    return _safe(update_positions)


@router.post("/paper/close/{trade_id}")
def paper_close(trade_id: str, price: Optional[float] = Query(None)):
    """Manually close a paper trade."""
    from engines.paper_trading_v2 import close_trade
    return _safe(close_trade, trade_id, price)


# ── Journal & Reflection ──────────────────────────────

@router.get("/journal")
def journal(limit: int = Query(50, ge=1, le=500),
            grade: Optional[str] = Query(None),
            symbol: Optional[str] = Query(None)):
    """Trade journal entries with optional filters."""
    from engines.reflection_v2 import get_journal
    return {"journal": _safe(get_journal, limit, grade,
                             symbol.upper() if symbol else None)}


@router.get("/reflection/weekly")
def reflection_weekly():
    """Latest weekly reflection report."""
    from engines.reflection_v2 import weekly_report
    return _safe(weekly_report)


@router.get("/reflection/monthly")
def reflection_monthly(month: Optional[str] = Query(None)):
    """Monthly reflection report."""
    from engines.reflection_v2 import monthly_report
    return _safe(monthly_report, month)


@router.get("/reflection/learning")
def reflection_learning():
    """Cumulative learning report."""
    from engines.reflection_v2 import learning_report
    return _safe(learning_report)


# ── Quant Lab ──────────────────────────────────────────

@router.get("/lab/experiments")
def lab_experiments(status: Optional[str] = Query(None)):
    """List all experiments."""
    from engines.quant_lab import list_experiments
    return {"experiments": _safe(list_experiments, status)}


@router.get("/lab/experiment/{experiment_id}")
def lab_experiment_detail(experiment_id: str):
    """Analyze a specific experiment."""
    from engines.quant_lab import analyze_experiment
    return _safe(analyze_experiment, experiment_id)


@router.get("/lab/calibration")
def lab_calibration(lookback_days: int = Query(90)):
    """Confidence calibration curve."""
    from engines.quant_lab import confidence_calibration
    return _safe(confidence_calibration, lookback_days)


@router.get("/lab/features")
def lab_features(lookback_days: int = Query(90)):
    """Feature contribution analysis."""
    from engines.quant_lab import feature_contribution
    return _safe(feature_contribution, lookback_days)


@router.get("/lab/strategies")
def lab_strategies(lookback_days: int = Query(90)):
    """Strategy comparison."""
    from engines.quant_lab import strategy_comparison
    return _safe(strategy_comparison, lookback_days)


@router.get("/lab/stress/{scenario}")
def lab_stress(scenario: str):
    """Run a stress test scenario."""
    from engines.quant_lab import stress_test
    return _safe(stress_test, scenario)


# ── Auto Trader (ML Mode) ────────────────────────────

@router.get("/auto/dashboard")
def auto_dashboard():
    """Full auto-trader dashboard: account, open trades, daily P&L."""
    from engines.auto_trader import dashboard
    return _safe(dashboard)


@router.get("/auto/account")
def auto_account():
    """Auto-trader account status."""
    from engines.auto_trader import get_account
    return _safe(get_account)


@router.get("/auto/signals")
def auto_signals():
    """Generate today's signals (wall selling + stock ML picks)."""
    from engines.auto_trader import generate_signals
    return _safe(generate_signals)


@router.post("/auto/trade")
def auto_trade():
    """Place the highest-probability trades for today."""
    from engines.auto_trader import place_best_trades
    return _safe(place_best_trades)


@router.post("/auto/close")
def auto_close():
    """Close all open trades at EOD."""
    from engines.auto_trader import close_trades
    return _safe(close_trades)


@router.post("/auto/reset")
def auto_reset(capital: float = Query(1000000)):
    """Reset paper trading account with given capital."""
    from engines.auto_trader import reset_account
    return _safe(reset_account, capital)

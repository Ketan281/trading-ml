"""Portfolio Management API routes."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional, List

from api import auth
from api.router import _silent
from api.cache import cached

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.get("/brief")
def daily_brief(user: dict = Depends(auth.current_user)):
    """Today's trading plan: signals, positions, risk, and recommended actions."""
    from engines.portfolio_manager import daily_brief
    # Cache per-user — the brief is built from CSV signals + DB reads that don't
    # change intra-minute. Without this, every frontend poll recomputes it and
    # competes with live-tick / NSE work for the single worker, causing 504s.
    return cached(f"brief:{user['id']}",
                  lambda: _silent(daily_brief, user["id"]), ttl=180)


@router.get("/risk")
def risk_dashboard(user: dict = Depends(auth.current_user)):
    """Real-time risk dashboard: exposure, concentration, drawdown, alerts."""
    from engines.portfolio_manager import risk_dashboard
    return cached(f"risk:{user['id']}",
                  lambda: _silent(risk_dashboard, user["id"]), ttl=120)


@router.get("/performance")
def performance(user: dict = Depends(auth.current_user), days: int = 30):
    """Performance attribution: by segment, by symbol, equity curve."""
    from engines.portfolio_manager import performance_report
    return _silent(performance_report, user["id"], days)


@router.get("/equity-curve")
def equity_curve(user: dict = Depends(auth.current_user), days: int = 90):
    """Historical equity curve from daily snapshots."""
    from engines.portfolio_manager import get_equity_curve
    return {"curve": _silent(get_equity_curve, user["id"], days)}


@router.post("/snapshot")
def take_snapshot(user: dict = Depends(auth.current_user)):
    """Record today's equity snapshot (call daily)."""
    from engines.portfolio_manager import take_snapshot
    return _silent(take_snapshot, user["id"])


@router.post("/auto-trade")
def auto_trade(user: dict = Depends(auth.current_user), dry_run: bool = True):
    """Execute the daily plan. dry_run=true shows what would happen."""
    from engines.portfolio_manager import auto_trade
    return _silent(auto_trade, user["id"], dry_run)


@router.get("/alerts")
def alerts(user: dict = Depends(auth.current_user), unread_only: bool = False):
    """Portfolio alerts and notifications."""
    from engines.portfolio_manager import get_alerts
    return {"alerts": _silent(get_alerts, user["id"], 20, unread_only)}


class AlertIds(BaseModel):
    ids: Optional[List[int]] = None

@router.post("/alerts/read")
def mark_read(body: AlertIds, user: dict = Depends(auth.current_user)):
    """Mark alerts as read. Empty ids = mark all read."""
    from engines.portfolio_manager import mark_alerts_read
    return _silent(mark_alerts_read, user["id"], body.ids)


@router.get("/plan")
def get_plan(user: dict = Depends(auth.current_user), date: Optional[str] = None):
    """Get saved trading plan for a date (default: today)."""
    from engines.portfolio_manager import get_plan
    plan = _silent(get_plan, user["id"], date)
    return plan or {"message": "No plan for this date"}


@router.get("/models")
def model_status():
    """Status of all ML models powering the portfolio."""
    from engines.portfolio_manager import model_status
    return _silent(model_status)

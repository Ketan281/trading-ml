from fastapi import APIRouter, Depends, HTTPException

from api import auth
from api.router import _silent
from api.schemas import RoleChange
from aos import user_wallet as uw
from aos import scheduler

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/users")
def admin_users(user: dict = Depends(auth.admin_only)):
    return {"users": auth.list_users()}


@router.post("/users/{uid}/role")
def admin_set_role(uid: int, body: RoleChange, user: dict = Depends(auth.admin_only)):
    if uid == user["id"] and body.role != "admin":
        raise HTTPException(400, "you cannot remove your own admin access")
    try:
        return {"ok": True, "user": auth.set_role(uid, body.role)}
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.get("/overview")
def admin_overview(user: dict = Depends(auth.admin_only)):
    overview = []
    for item in auth.list_users():
        status = _silent(uw.status, item["id"], do_tick=False)
        overview.append({"user": item, "wallet": status.get("wallet"),
                         "live_equity": status.get("live_equity"),
                         "open_trades": status.get("open_trades", [])})
    return {"overview": overview}


@router.get("/learning")
def admin_learning_status(user: dict = Depends(auth.admin_only)):
    from aos import memory as mem
    from aos.meta_learning import load_policy

    policy = _silent(load_policy)
    stats = _silent(mem.stats)
    lessons = _silent(mem.recent_lessons, 5)
    last_postmarket = scheduler.last_run("postmarket")
    last_metalearn = scheduler.last_run("metalearn")
    return {
        "policy": policy or {"status": "not yet trained"},
        "memory_stats": stats or {},
        "recent_lessons": lessons or [],
        "last_postmarket_run": str(last_postmarket) if last_postmarket else None,
        "last_metalearn_run": str(last_metalearn) if last_metalearn else None,
    }


@router.post("/learning/run")
def admin_trigger_learning(user: dict = Depends(auth.admin_only)):
    from aos.meta_learning import learn
    from aos.postmarket import review

    review_result = review()
    policy = learn()
    return {
        "postmarket": {"trades": review_result.get("n_trades", 0),
                       "lessons": len(review_result.get("lessons", []))},
        "meta_learning": policy,
    }


# ── Execution boundary controls ─────────────────────
@router.get("/execution")
def admin_execution_status(user: dict = Depends(auth.admin_only)):
    from broker.executor import get_executor
    gate = get_executor()
    return {
        "mode": gate.mode,
        "broker": gate.adapter.name,
        "is_live": gate.is_live(),
        "kill_switch": gate.kill_switch,
        "daily_loss_limit": gate.daily_loss_limit,
        "max_open_positions": gate.max_open_positions,
        "broker_connected": gate.adapter.is_connected(),
    }


@router.post("/execution/kill-switch")
def admin_kill_switch(user: dict = Depends(auth.admin_only)):
    from broker.executor import get_executor
    gate = get_executor()
    gate.trip_kill_switch(reason=f"admin:{user['id']}")
    return {"kill_switch": True, "message": "all new orders blocked"}


@router.post("/execution/kill-switch/reset")
def admin_kill_switch_reset(user: dict = Depends(auth.admin_only)):
    from broker.executor import get_executor
    gate = get_executor()
    gate.reset_kill_switch()
    return {"kill_switch": False, "message": "orders re-enabled"}

"""
Trading-AI API server — deploy on a server, consume from a frontend.

Run:
    uvicorn api.server:app --host 0.0.0.0 --port 8000
    # or: python api/server.py

Frontend usage:
    POST /query   {"q": "which is the best to enter in banknifty intraday option today"}
        → {"answer": "...", "intent": "...", "data": {...}}

Endpoints
    GET  /health
    POST /query                 natural-language command (the main one)
    GET  /options/{symbol}      full options dashboard
    GET  /book                  constructed portfolio book
    GET  /screen                swing-trade shortlist

NOTE: handlers call the live engines (NSE / model), so a request can take a few
seconds. The frontend should show a loading state. Responses are cached briefly
to keep repeated queries snappy.
"""

import os
import sys
import time
import json
import asyncio
import threading
import logging

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.router import route, _silent
from api import auth
from aos import user_wallet as uw

app = FastAPI(title="Trading-AI", version="1.0",
              description="NSE equity + NIFTY/BANKNIFTY options intelligence")

# CORS: in production set ALLOWED_ORIGINS to your frontend origin(s),
# comma-separated (e.g. "https://app.example.com"). For local dev any
# localhost/127.0.0.1 port is always allowed via the regex below, so serving the
# static frontend on any port (5500, 5601, 8080, …) works without configuration.
# Tokens are sent in the Authorization header (not cookies), so allowing any
# local port carries no credential-leak risk.
_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Tiny TTL cache so repeated identical queries don't re-hit NSE.
_CACHE = {}
_TTL = 60


def _cached(key, fn):
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < _TTL:
        return hit[1]
    val = fn()
    _CACHE[key] = (now, val)
    return val


class Query(BaseModel):
    q: str
    polish: bool = True          # LLM-rephrase the answer (failure-safe)


@app.get("/health")
def health():
    return {"status": "ok", "service": "trading-ai"}


@app.post("/query")
def query(body: Query):
    """The main endpoint: natural-language command → answer + structured data.

    The deterministic answer is instant (served from the pre-computed cache).
    `polish=true` adds an LLM rewrite that is slower on CPU (~seconds) but the
    polished result is cached, so repeats are instant. Set `polish=false` for a
    guaranteed-instant deterministic answer."""
    # On low-RAM hosts set AOS_DISABLE_LLM=1 so the LLM can never be invoked
    # (it would OOM a 1 GB box). Answers are still clean and deterministic.
    want_polish = body.polish and os.getenv("AOS_DISABLE_LLM") != "1"
    key = f"q::{int(want_polish)}::{body.q.lower().strip()}"

    def build():
        res = route(body.q)
        if want_polish and res.get("answer"):
            from api.narrate import polish
            res = {**res, "answer_raw": res["answer"], "answer": polish(res["answer"])}
        return res

    return _cached(key, build)


@app.get("/options/{symbol}")
def options(symbol: str):
    from pipelines.options.options_dashboard import dashboard
    return _cached(f"opt::{symbol.upper()}",
                   lambda: _silent(dashboard, symbol.upper()))


@app.get("/book")
def book():
    from pipelines.portfolio_book import build_book
    return _cached("book", lambda: _silent(build_book))


@app.get("/screen")
def screen_ep():
    from pipelines.screener import screen
    return _cached("screen", lambda: _silent(screen))


# ── Autonomous paper-trading wallet ───────────────────
class Deposit(BaseModel):
    amount: float

@app.get("/wallet")
def wallet_status():
    """Wallet balance, live equity, today's trade + P&L series (for the chart),
    and trade history. Paper money only — no profit guarantee."""
    from aos.sim_wallet import status
    return _silent(status)

@app.post("/wallet/deposit")
def wallet_deposit(body: Deposit):
    from aos.sim_wallet import deposit
    return _silent(deposit, body.amount)

@app.post("/wallet/reset")
def wallet_reset():
    from aos.sim_wallet import reset
    return {"wallet": _silent(reset)}

@app.post("/wallet/trade/start")
def wallet_trade_start():
    """Pick + open today's intraday call (idempotent — once per day). Normally
    the scheduler calls this at the open; exposed for manual trigger/testing."""
    from aos.sim_wallet import start_daily_trade
    return _silent(start_daily_trade)

@app.post("/wallet/tick")
def wallet_tick():
    """Advance the open trade against the live price (records a P&L point,
    exits on stop/target/square-off). Normally the scheduler ticks this."""
    from aos.sim_wallet import tick, status
    _silent(tick)
    return _silent(status)


# ── Auth (multi-user) ─────────────────────────────────
class Credentials(BaseModel):
    email: str
    password: str

class GoogleToken(BaseModel):
    id_token: str

@app.post("/auth/signup")
def auth_signup(body: Credentials):
    try:
        return auth.signup(body.email, body.password)
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.post("/auth/login")
def auth_login(body: Credentials):
    try:
        return auth.login(body.email, body.password)
    except ValueError as e:
        raise HTTPException(401, str(e))

@app.post("/auth/google")
def auth_google(body: GoogleToken):
    """Sign up or log in with a Google account (Gmail). Send the Google ID
    token from the frontend; the server verifies it and returns a JWT."""
    try:
        return auth.google_auth(body.id_token)
    except ValueError as e:
        raise HTTPException(401, str(e))

@app.get("/auth/me")
def auth_me(user: dict = Depends(auth.current_user)):
    return user

class PasswordChange(BaseModel):
    old_password: str
    new_password: str

@app.post("/auth/change-password")
def auth_change_password(body: PasswordChange, user: dict = Depends(auth.current_user)):
    try:
        return auth.change_password(user["id"], body.old_password, body.new_password)
    except ValueError as e:
        raise HTTPException(400, str(e))


# ── Per-user paper trading (auth required) ────────────
class TradeSpec(BaseModel):
    segment: str                       # options | futures | equity | forex
    underlying: str | None = None      # NIFTY | BANKNIFTY (options/futures)
    symbol: str | None = None          # stock symbol (equity)
    pair: str | None = None            # forex pair (EUR/USD, GBP/USD, …)
    leg: str | None = None             # CE | PE (options)
    strike: int | None = None          # options strike (default ATM)
    side: str | None = None            # long | short | buy | sell
    lots: int | None = None
    qty: int | None = None
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    reason: str | None = None

class DepositAmt(BaseModel):
    amount: float

@app.get("/me/wallet")
def me_wallet(user: dict = Depends(auth.current_user)):
    """Wallet, live equity, open positions (ticked on read) and trade history."""
    return _silent(uw.status, user["id"])

@app.post("/me/wallet/deposit")
def me_deposit(body: DepositAmt, user: dict = Depends(auth.current_user)):
    return _silent(uw.deposit, user["id"], body.amount)

@app.post("/me/trade")
def me_trade(body: TradeSpec, user: dict = Depends(auth.current_user)):
    spec = {k: v for k, v in body.model_dump().items() if v is not None}
    return _silent(uw.open_trade, user["id"], spec)

@app.post("/me/trade/{trade_id}/close")
def me_trade_close(trade_id: str, user: dict = Depends(auth.current_user)):
    return _silent(uw.close_trade, user["id"], trade_id)

@app.get("/me/history")
def me_history(user: dict = Depends(auth.current_user)):
    """All trades the user has taken (any status), newest first — for the
    date-grouped history page."""
    return {"trades": _silent(uw.history_full, user["id"])}

@app.post("/me/trade/{trade_id}/explain")
def me_trade_explain(trade_id: str, user: dict = Depends(auth.current_user)):
    """AI explanation of why a trade was taken (cached after first call)."""
    return _silent(uw.explain_trade, user["id"], trade_id)


def _resolve_sse_user(request: Request):
    """Resolve user from query-param token (EventSource can't set headers)."""
    tok = request.query_params.get("token")
    if not tok:
        raise HTTPException(401, "missing token query param")
    payload = auth.decode_token(tok)
    if not payload:
        raise HTTPException(401, "invalid or expired token")
    user = auth.get_user(payload["sub"])
    if not user:
        raise HTTPException(401, "user not found")
    return user


# ── Live P&L streaming (SSE) for real-time charts ────
@app.get("/me/live")
async def me_live(request: Request):
    """Server-Sent Events stream: pushes live P&L for all open trades every
    ~2 seconds. The frontend connects with EventSource and gets a continuous
    feed of price + P&L data for charting.

    Each SSE event is a JSON object:
      {indian_wallet, forex_wallet, trades: [{id, symbol, segment, side,
        entry, current_price, qty, gross_pnl, pnl_pct, ...}],
       indian_equity, forex_equity, timestamp}
    """
    user = _resolve_sse_user(request)
    uid = user["id"]

    async def event_stream():
        while True:
            if await request.is_disconnected():
                break
            try:
                data = _build_live_snapshot(uid)
                yield f"data: {json.dumps(data, default=str)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/me/live/{trade_id}")
async def me_live_trade(trade_id: str, request: Request):
    """SSE stream for a single trade — higher frequency (every 1s) for the
    focused live chart view."""
    user = _resolve_sse_user(request)
    uid = user["id"]

    async def event_stream():
        while True:
            if await request.is_disconnected():
                break
            try:
                data = _build_trade_snapshot(uid, trade_id)
                if data.get("error"):
                    yield f"data: {json.dumps(data)}\n\n"
                    break
                yield f"data: {json.dumps(data, default=str)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _build_live_snapshot(uid):
    """Tick all open trades and return a real-time snapshot for the SSE stream."""
    from datetime import datetime as dt
    try:
        uw.tick_user(uid)
    except Exception:
        pass
    w = uw.get_wallet(uid)
    fw = uw.get_forex_wallet(uid)
    opens = uw._open_trades(uid)
    trades_out = []
    indian_unreal = 0.0
    forex_unreal = 0.0
    for t in opens:
        last_px = t["pnl_series"][-1][1] if t["pnl_series"] else t["entry"]
        last_pnl = t["pnl_series"][-1][2] if t["pnl_series"] else 0.0
        pnl_pct = round((last_pnl / t["cost"]) * 100, 2) if t["cost"] else 0
        is_forex = t.get("segment") == "forex"
        ccy = "$" if is_forex else "₹"
        if is_forex:
            forex_unreal += last_pnl
        else:
            indian_unreal += last_pnl
        trades_out.append({
            "id": t["id"], "symbol": t["symbol"], "segment": t["segment"],
            "kind": t["kind"], "side": t["side"],
            "entry": t["entry"], "current_price": last_px,
            "qty": t["qty"], "lots": t.get("lots"),
            "stop": t["stop"], "target": t["target"],
            "gross_pnl": last_pnl, "pnl_pct": pnl_pct,
            "currency": ccy,
            "pnl_series": t["pnl_series"][-60:],
            "opened_at": t["opened_at"],
        })
    return {
        "indian_wallet": {"balance": w["balance"], "currency": "INR"},
        "forex_wallet": {"balance": fw["balance"], "currency": "USD"},
        "indian_equity": round(w["balance"] + indian_unreal, 2),
        "forex_equity": round(fw["balance"] + forex_unreal, 2),
        "trades": trades_out,
        "timestamp": dt.now().isoformat(),
    }


def _build_trade_snapshot(uid, trade_id):
    """Single-trade live snapshot for focused chart view."""
    from datetime import datetime as dt
    with uw._conn() as c:
        row = c.execute("SELECT * FROM trades WHERE id=? AND user_id=?",
                        (trade_id, uid)).fetchone()
    if not row:
        return {"error": "trade not found"}
    t = uw._row_to_trade(row)
    if t["status"] != "open":
        return {"error": f"trade is {t['status']}", "trade": t}
    px = uw._live_price(t)
    if px is not None:
        gross = uw._signed_gross(t, px)
        t["pnl_series"].append([dt.now().strftime("%H:%M:%S"), round(px, 2), round(gross, 1)])
        with uw._conn() as c:
            c.execute("UPDATE trades SET pnl_series=? WHERE id=?",
                      (json.dumps(t["pnl_series"]), t["id"]))
    last_px = t["pnl_series"][-1][1] if t["pnl_series"] else t["entry"]
    last_pnl = t["pnl_series"][-1][2] if t["pnl_series"] else 0
    is_forex = t.get("segment") == "forex"
    return {
        "id": t["id"], "symbol": t["symbol"], "segment": t["segment"],
        "side": t["side"], "entry": t["entry"], "current_price": last_px,
        "qty": t["qty"], "stop": t["stop"], "target": t["target"],
        "gross_pnl": last_pnl,
        "pnl_pct": round((last_pnl / t["cost"]) * 100, 2) if t["cost"] else 0,
        "currency": "$" if is_forex else "₹",
        "pnl_series": t["pnl_series"],
        "opened_at": t["opened_at"],
        "timestamp": dt.now().isoformat(),
    }


# ── Forex wallet (USD, separate from Indian INR wallet) ─
@app.get("/me/forex-wallet")
def me_forex_wallet(user: dict = Depends(auth.current_user)):
    fw = _silent(uw.get_forex_wallet, user["id"])
    forex_opens = [t for t in _silent(uw.status, user["id"]).get("forex_open_trades", [])
                   if t.get("segment") == "forex"]
    forex_unreal = sum(t["pnl_series"][-1][2] for t in forex_opens if t.get("pnl_series"))
    return {"wallet": fw, "live_equity": round(fw["balance"] + forex_unreal, 2),
            "unrealized": round(forex_unreal, 1), "open_trades": forex_opens}

@app.post("/me/forex-wallet/deposit")
def me_forex_deposit(body: DepositAmt, user: dict = Depends(auth.current_user)):
    return _silent(uw.deposit_forex, user["id"], body.amount)

@app.post("/me/forex-wallet/reset")
def me_forex_reset(user: dict = Depends(auth.current_user)):
    return {"wallet": _silent(uw.reset_forex, user["id"])}


# ── Trade mode (separate toggles for Indian & Forex) ─
class ModeChange(BaseModel):
    mode: str       # "ml" or "custom"
    market: str = "indian"  # "indian" or "forex"


def _trigger_ml_immediately(uid, market):
    """When the user switches to ML mode, immediately open the best trade
    instead of waiting for the background cycle."""
    from datetime import datetime as dt, time as dtime
    from aos.sim_wallet import SQUARE_OFF
    opened = []
    now_time = dt.now().time()
    if market in ("indian", "both"):
        indian_market_open = dtime(9, 15) <= now_time <= SQUARE_OFF
        if indian_market_open:
            try:
                t = uw.auto_open_trade(uid)
                if t:
                    opened.append({"market": "indian", "symbol": t["symbol"], "trade": t})
                else:
                    opened.append({"market": "indian", "symbol": None,
                                   "info": "no actionable Indian trade right now"})
            except Exception as e:
                _ml_log.warning("ML immediate Indian trade for uid %s failed: %s", uid, e)
                opened.append({"market": "indian", "error": str(e)})
        else:
            opened.append({"market": "indian", "symbol": None,
                           "info": "Indian market is closed (09:15–15:15 IST)"})
    if market in ("forex", "both"):
        try:
            t = uw.auto_open_forex_trade(uid)
            if t:
                opened.append({"market": "forex", "symbol": t["symbol"], "trade": t})
            else:
                opened.append({"market": "forex", "symbol": None,
                               "info": "no actionable forex setup meets confluence threshold"})
        except Exception as e:
            _ml_log.warning("ML immediate Forex trade for uid %s failed: %s", uid, e)
            opened.append({"market": "forex", "error": str(e)})
    return opened


@app.get("/me/mode")
def me_mode(user: dict = Depends(auth.current_user)):
    return uw.get_mode(user["id"])

@app.post("/me/mode")
def me_set_mode(body: ModeChange, user: dict = Depends(auth.current_user)):
    uid = user["id"]
    result = _silent(uw.set_mode, uid, body.mode, body.market)
    if body.mode == "ml":
        result["auto_opened"] = _trigger_ml_immediately(uid, body.market)
    result["current_mode"] = uw.get_mode(uid)
    result["status"] = _silent(uw.status, uid, do_tick=False)
    return result

@app.post("/me/mode/indian")
def me_set_indian_mode(body: ModeChange, user: dict = Depends(auth.current_user)):
    uid = user["id"]
    result = _silent(uw.set_mode, uid, body.mode, "indian")
    if body.mode == "ml":
        result["auto_opened"] = _trigger_ml_immediately(uid, "indian")
    result["current_mode"] = uw.get_mode(uid)
    result["status"] = _silent(uw.status, uid, do_tick=False)
    return result

@app.post("/me/mode/forex")
def me_set_forex_mode(body: ModeChange, user: dict = Depends(auth.current_user)):
    uid = user["id"]
    result = _silent(uw.set_mode, uid, body.mode, "forex")
    if body.mode == "ml":
        result["auto_opened"] = _trigger_ml_immediately(uid, "forex")
    result["current_mode"] = uw.get_mode(uid)
    result["status"] = _silent(uw.status, uid, do_tick=False)
    return result


# ── Market data: candles + best recommendation ────────
@app.get("/candles/{symbol}")
def candles_ep(symbol: str, interval: str = "5m", period: str = "1d"):
    from api.market import candles
    return _cached(f"candles::{symbol.upper()}::{interval}::{period}",
                   lambda: _silent(candles, symbol, interval, period))

@app.get("/recommendation")
def recommendation_ep(user: dict = Depends(auth.current_user)):
    from api.market import recommendation
    bal = _silent(uw.get_wallet, user["id"]).get("balance", 10_000)
    return _cached("reco", lambda: _silent(recommendation, bal))


# ── Forex ────────────────────────────────────────────
@app.get("/forex/pairs")
def forex_pairs():
    from pipelines.forex.data import list_pairs
    return {"pairs": list_pairs()}

@app.get("/forex/candles/{pair:path}")
def forex_candles(pair: str, interval: str = "15m", period: str = "5d"):
    from pipelines.forex.data import fetch_candles, candles_to_list
    df = fetch_candles(pair, interval, period)
    return _cached(f"fxc::{pair}::{interval}::{period}",
                   lambda: {"pair": pair, "candles": candles_to_list(df)})

@app.get("/forex/signals/{pair:path}")
def forex_signals(pair: str):
    from pipelines.forex.confluence import score_pair
    return _cached(f"fxsig::{pair}", lambda: _silent(score_pair, pair))

@app.get("/forex/scan")
def forex_scan():
    from pipelines.forex.confluence import scan_all_pairs
    return _cached("fxscan", lambda: {"pairs": _silent(scan_all_pairs)})

@app.get("/forex/recommendation")
def forex_recommendation():
    from pipelines.forex.confluence import best_trade
    r = _cached("fxreco", lambda: _silent(best_trade))
    if not r:
        return {"answer": "No forex pair meets the confluence threshold right now — "
                "the system is waiting for a high-confidence setup.",
                "trade": None}
    return {"answer": f"Best forex setup: {r['direction'].upper()} {r['pair']} "
            f"(confluence {r['score']:.2f}, {r['confidence']} confidence, "
            f"{r['agreeing_timeframes']}/{r['total_timeframes']} TFs agree). "
            f"Entry {r['trade_plan']['entry']}, SL {r['trade_plan']['stop_loss']}, "
            f"TP {r['trade_plan']['take_profit']} "
            f"(R:R {r['trade_plan']['risk_reward']}:1).",
            "trade": r}


# ── Admin (owner only) ────────────────────────────────
@app.get("/admin/users")
def admin_users(user: dict = Depends(auth.admin_only)):
    return {"users": auth.list_users()}

class RoleChange(BaseModel):
    role: str

@app.post("/admin/users/{uid}/role")
def admin_set_role(uid: int, body: RoleChange, user: dict = Depends(auth.admin_only)):
    if uid == user["id"] and body.role != "admin":
        raise HTTPException(400, "you cannot remove your own admin access")
    try:
        return {"ok": True, "user": auth.set_role(uid, body.role)}
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.get("/admin/overview")
def admin_overview(user: dict = Depends(auth.admin_only)):
    """Every user's wallet + open positions — the owner's bird's-eye view."""
    out = []
    for u in auth.list_users():
        st = _silent(uw.status, u["id"], do_tick=False)
        out.append({"user": u, "wallet": st.get("wallet"),
                    "live_equity": st.get("live_equity"),
                    "open_trades": st.get("open_trades", [])})
    return {"overview": out}


# ── Background ML auto-trading loop ──────────────────
_ml_log = logging.getLogger("ml_auto")

def _ml_loop():
    """Every 60 seconds, tick all ML-mode users, auto-open trades in both
    Indian and Forex markets simultaneously, and manage the portfolio.
    Runs in a daemon thread — dies with the process."""
    from datetime import time as dtime
    while True:
        time.sleep(60)
        try:
            results = uw.ml_tick_all()
            if results:
                _ml_log.info("ML auto-traded: %s", results)
        except Exception as e:
            _ml_log.warning("ML loop error: %s", e)

@app.on_event("startup")
def _start_ml_loop():
    t = threading.Thread(target=_ml_loop, daemon=True, name="ml-auto-trader")
    t.start()
    _ml_log.info("ML auto-trading background loop started (60s interval)")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)

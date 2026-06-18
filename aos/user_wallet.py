"""
Per-user paper wallet & trade book.

The single global wallet in aos/sim_wallet.py takes ONE autonomous trade a day.
This module gives every signed-up user their OWN paper wallet and lets them
*manually* open trades across segments (options / futures / intraday equity),
priced by the same live engines. Storage is SQLite (data/users.db — the same
file the auth user table lives in) so a user and their book share one place.

HONEST FRAMING (unchanged): paper money only, no broker, no profit guarantee.
Quant/option engines produce every price; nothing here places a real order.

Reuse: live pricing, lot sizes, stop/target percentages and the won/lost
analysis text are imported from aos.sim_wallet; fees come from
agents.brokerage.charges. We add multi-trade margin accounting and short side
support (futures/equity can go short), which the single-trade global wallet
doesn't need.
"""

import os
import sys
import json
import uuid
import sqlite3
from datetime import datetime, date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from aos.sim_wallet import (
    DEFAULT_BALANCE, SQUARE_OFF, OPT_SL_PCT, OPT_TGT_PCT, LOT,
    _option_ltp, _analysis,
)
from agents.brokerage import charges

# Multi-user paper product gets a higher (fake-money) deposit cap than the
# single autonomous wallet, so index futures — which need lakhs of margin — are
# actually takeable. Still paper money; no real funds involved.
DEPOSIT_CAP = 2_000_000

DB = os.path.join(ROOT, "data", "users.db")
os.makedirs(os.path.dirname(DB), exist_ok=True)

# Futures move far less in % terms than option premiums — tighter bands.
FUT_SL_PCT, FUT_TGT_PCT = 0.004, 0.008
# Index futures are leveraged: only SPAN+exposure margin is blocked (~15% of
# notional), while P&L accrues on the FULL contract value (real leverage).
FUT_MARGIN_PCT = 0.15
# Default intraday-equity bands when the caller doesn't supply levels.
EQ_SL_PCT, EQ_TGT_PCT = 0.01, 0.02
INDICES = ("NIFTY", "BANKNIFTY")


# ── DB ────────────────────────────────────────────────
def _conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS wallets (
                user_id         INTEGER PRIMARY KEY,
                balance         REAL NOT NULL,
                total_deposited REAL NOT NULL,
                realized_pnl    REAL NOT NULL DEFAULT 0,
                created         TEXT NOT NULL
            )""")
        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id          TEXT PRIMARY KEY,
                user_id     INTEGER NOT NULL,
                date        TEXT,
                segment     TEXT,
                kind        TEXT,
                side        TEXT DEFAULT 'long',
                symbol      TEXT,
                chart_symbol TEXT,
                underlying  TEXT,
                strike      INTEGER,
                leg         TEXT,
                qty         INTEGER,
                lots        INTEGER,
                entry       REAL,
                stop        REAL,
                target      REAL,
                cost        REAL,
                status      TEXT,
                opened_at   TEXT,
                exit_price  REAL,
                exit_reason TEXT,
                fees        REAL,
                net_pnl     REAL,
                pnl_series  TEXT,
                reason      TEXT,
                analysis    TEXT,
                ai_why      TEXT
            )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_trades_user ON trades(user_id, status)")
        # Migrations for pre-existing tables.
        cols = [r[1] for r in c.execute("PRAGMA table_info(trades)")]
        if "ai_why" not in cols:
            c.execute("ALTER TABLE trades ADD COLUMN ai_why TEXT")
        wcols = [r[1] for r in c.execute("PRAGMA table_info(wallets)")]
        if "trade_mode" not in wcols:
            c.execute("ALTER TABLE wallets ADD COLUMN trade_mode TEXT DEFAULT 'custom'")


# ── wallet ops ────────────────────────────────────────
def get_wallet(uid):
    init_db()
    with _conn() as c:
        row = c.execute("SELECT * FROM wallets WHERE user_id=?", (uid,)).fetchone()
        if row is None:
            now = datetime.now().isoformat()
            c.execute("INSERT INTO wallets (user_id, balance, total_deposited, "
                      "realized_pnl, created) VALUES (?,?,?,0,?)",
                      (uid, DEFAULT_BALANCE, DEFAULT_BALANCE, now))
            return {"user_id": uid, "balance": float(DEFAULT_BALANCE),
                    "total_deposited": float(DEFAULT_BALANCE), "realized_pnl": 0.0,
                    "created": now}
    return dict(row)


def _save_wallet(w):
    with _conn() as c:
        c.execute("UPDATE wallets SET balance=?, total_deposited=?, realized_pnl=? "
                  "WHERE user_id=?",
                  (w["balance"], w["total_deposited"], w["realized_pnl"], w["user_id"]))


def get_mode(uid):
    get_wallet(uid)  # ensure row exists
    with _conn() as c:
        row = c.execute("SELECT trade_mode FROM wallets WHERE user_id=?", (uid,)).fetchone()
    return (row["trade_mode"] if row and row["trade_mode"] else "custom")


def set_mode(uid, mode):
    if mode not in ("ml", "custom"):
        return {"error": "mode must be 'ml' or 'custom'"}
    get_wallet(uid)
    with _conn() as c:
        c.execute("UPDATE wallets SET trade_mode=? WHERE user_id=?", (mode, uid))
    return {"ok": True, "trade_mode": mode}


def deposit(uid, amount):
    w = get_wallet(uid); amount = float(amount)
    if amount <= 0:
        return {"error": "amount must be positive"}
    if w["total_deposited"] + amount > DEPOSIT_CAP:
        room = DEPOSIT_CAP - w["total_deposited"]
        return {"error": f"deposit cap ₹{DEPOSIT_CAP:,} reached — you can add at most "
                f"₹{room:,.0f} more (profits can still grow the balance past ₹1L)."}
    w["balance"] = round(w["balance"] + amount, 2)
    w["total_deposited"] = round(w["total_deposited"] + amount, 2)
    _save_wallet(w)
    return {"ok": True, "wallet": w}


def reset(uid):
    with _conn() as c:
        c.execute("DELETE FROM trades WHERE user_id=?", (uid,))
        c.execute("DELETE FROM wallets WHERE user_id=?", (uid,))
    return get_wallet(uid)


# ── trade storage helpers ─────────────────────────────
_TRADE_COLS = ["id", "user_id", "date", "segment", "kind", "side", "symbol",
               "chart_symbol", "underlying", "strike", "leg", "qty", "lots",
               "entry", "stop", "target", "cost", "status", "opened_at",
               "exit_price", "exit_reason", "fees", "net_pnl", "pnl_series",
               "reason", "analysis"]


def _insert_trade(t):
    with _conn() as c:
        c.execute(
            f"INSERT INTO trades ({','.join(_TRADE_COLS)}) "
            f"VALUES ({','.join('?' * len(_TRADE_COLS))})",
            [t.get(k) for k in _TRADE_COLS])


def _save_trade(t):
    with _conn() as c:
        c.execute(
            "UPDATE trades SET status=?, exit_price=?, exit_reason=?, fees=?, "
            "net_pnl=?, pnl_series=?, analysis=? WHERE id=?",
            (t["status"], t["exit_price"], t["exit_reason"], t["fees"], t["net_pnl"],
             json.dumps(t["pnl_series"]), t["analysis"], t["id"]))


def _row_to_trade(row):
    t = dict(row)
    t["pnl_series"] = json.loads(t["pnl_series"]) if t.get("pnl_series") else []
    return t


def _open_trades(uid):
    with _conn() as c:
        rows = c.execute("SELECT * FROM trades WHERE user_id=? AND status='open' "
                         "ORDER BY opened_at", (uid,)).fetchall()
    return [_row_to_trade(r) for r in rows]


def _history(uid, limit=60):
    with _conn() as c:
        rows = c.execute("SELECT * FROM trades WHERE user_id=? AND status!='open' "
                         "ORDER BY opened_at DESC LIMIT ?", (uid, limit)).fetchall()
    return [_row_to_trade(r) for r in rows]


def _locked_margin(uid):
    """Capital tied up in currently-open trades (paper margin)."""
    return sum(t["cost"] for t in _open_trades(uid))


# ── live price for a trade ────────────────────────────
def _live_price(t):
    if t["kind"] == "option":
        return _option_ltp(t["underlying"], t["strike"], t["leg"])
    from agents.auto_trader import _stock_price
    return _stock_price(t["chart_symbol"] or t["symbol"])


# ── open a trade ──────────────────────────────────────
def open_trade(uid, spec):
    """spec: {segment, ...}. Returns {status, trade} or {error}.

    Segments:
      options  — underlying NIFTY/BANKNIFTY, leg CE/PE, optional strike (else ATM),
                 optional lots (else max affordable). Always long the option.
      futures  — index NIFTY/BANKNIFTY, side long/short, optional lots.
      equity   — intraday equity, symbol, side long/short, optional qty + levels.
    """
    init_db()
    w = get_wallet(uid)
    available = round(w["balance"] - _locked_margin(uid), 2)
    seg = (spec.get("segment") or "").lower()
    try:
        if seg in ("options", "option"):
            plan = _plan_option(spec, available)
        elif seg in ("futures", "future"):
            plan = _plan_futures(spec, available)
        elif seg in ("equity", "equity_intraday", "intraday"):
            plan = _plan_equity(spec, available)
        else:
            return {"error": f"unknown segment '{seg}'"}
    except _PlanError as e:
        return {"error": str(e)}

    plan.update({
        "id": str(uuid.uuid4())[:8], "user_id": uid, "date": date.today().isoformat(),
        "status": "open", "opened_at": datetime.now().isoformat(), "pnl_series": "[]",
        "exit_price": None, "exit_reason": None, "fees": None, "net_pnl": None,
        "analysis": None,
    })
    _insert_trade(plan)
    plan["pnl_series"] = []
    return {"status": "opened", "trade": plan}


class _PlanError(Exception):
    pass


def _affordable_lots(cost1, available, want):
    if cost1 > available:
        raise _PlanError(f"insufficient free capital: need ₹{cost1:,.0f} for 1 lot, "
                         f"only ₹{available:,.0f} free.")
    max_lots = int(available // cost1)
    lots = int(want) if want else max_lots
    lots = max(1, min(lots, max_lots))
    return lots


def _plan_option(spec, available):
    from pipelines.options.chain_live_intel import fetch_chain
    u = (spec.get("underlying") or spec.get("symbol") or "NIFTY").upper()
    if u not in INDICES:
        raise _PlanError(f"options supported on {INDICES}, not {u}")
    leg = (spec.get("leg") or "CE").upper()
    if leg not in ("CE", "PE"):
        raise _PlanError("leg must be CE or PE")
    ch = fetch_chain(u)
    if not ch:
        raise _PlanError(f"could not read {u} option chain (market closed / NSE down)")
    strike = int(spec.get("strike") or ch["atm"])
    r = ch["df"].iloc[(ch["df"]["strike"] - strike).abs().argmin()]
    strike = int(r["strike"])
    ltp = float(r[f"{leg.lower()}_ltp"])
    if ltp <= 0:
        raise _PlanError(f"{u} {strike} {leg} has no tradable premium right now")
    lot = LOT[u]
    cost1 = ltp * lot
    lots = _affordable_lots(cost1, available, spec.get("lots"))
    qty = lots * lot
    return {"kind": "option", "segment": "options", "side": "long",
            "underlying": u, "chart_symbol": u, "strike": strike, "leg": leg,
            "symbol": f"{u} {strike} {leg}", "qty": qty, "lots": lots,
            "entry": round(ltp, 2), "stop": round(ltp * (1 - OPT_SL_PCT), 2),
            "target": round(ltp * (1 + OPT_TGT_PCT), 2), "cost": round(cost1 * lots, 2),
            "reason": spec.get("reason") or
                      f"Manual {leg} on {u} ATM {strike} @ {round(ltp,2)} × {lots} lot(s)."}


def _plan_futures(spec, available):
    from agents.auto_trader import _stock_price
    u = (spec.get("underlying") or spec.get("symbol") or "NIFTY").upper()
    if u not in INDICES:
        raise _PlanError(f"index futures supported on {INDICES}, not {u}")
    side = (spec.get("side") or "long").lower()
    if side not in ("long", "short"):
        raise _PlanError("side must be long or short")
    px = _stock_price(u)
    if not px:
        raise _PlanError(f"could not read {u} price right now")
    lot = LOT[u]
    margin1 = px * lot * FUT_MARGIN_PCT          # SPAN-style margin per lot
    lots = _affordable_lots(margin1, available, spec.get("lots"))
    qty = lots * lot
    sl = px * (1 - FUT_SL_PCT) if side == "long" else px * (1 + FUT_SL_PCT)
    tgt = px * (1 + FUT_TGT_PCT) if side == "long" else px * (1 - FUT_TGT_PCT)
    return {"kind": "future", "segment": "futures", "side": side,
            "underlying": u, "chart_symbol": u, "strike": None, "leg": None,
            "symbol": f"{u} FUT", "qty": qty, "lots": lots, "entry": round(px, 2),
            "stop": round(sl, 2), "target": round(tgt, 2), "cost": round(margin1 * lots, 2),
            "reason": spec.get("reason") or
                      f"Manual {side} {u} futures @ {round(px,2)} × {lots} lot(s) "
                      f"(~₹{round(margin1*lots):,} margin, {int(FUT_MARGIN_PCT*100)}% SPAN)."}


def _plan_equity(spec, available):
    from agents.auto_trader import _stock_price
    sym = (spec.get("symbol") or "").upper()
    if not sym:
        raise _PlanError("equity trade needs a symbol")
    side = (spec.get("side") or "long").lower()
    px = float(spec.get("entry") or 0) or _stock_price(sym)
    if not px:
        raise _PlanError(f"could not read {sym} price right now")
    qty = int(spec.get("qty") or (available // px))
    if qty < 1:
        raise _PlanError(f"insufficient capital for 1 share of {sym} (₹{px:,.0f})")
    cost = px * qty
    if cost > available:
        raise _PlanError(f"insufficient free capital: need ₹{cost:,.0f}, "
                         f"only ₹{available:,.0f} free.")
    stop = float(spec.get("stop") or 0) or (
        px * (1 - EQ_SL_PCT) if side == "long" else px * (1 + EQ_SL_PCT))
    tgt = float(spec.get("target") or 0) or (
        px * (1 + EQ_TGT_PCT) if side == "long" else px * (1 - EQ_TGT_PCT))
    return {"kind": "equity", "segment": "equity_intraday", "side": side,
            "underlying": None, "chart_symbol": sym, "strike": None, "leg": None,
            "symbol": sym, "qty": qty, "lots": None, "entry": round(px, 2),
            "stop": round(stop, 2), "target": round(tgt, 2), "cost": round(cost, 2),
            "reason": spec.get("reason") or
                      f"Manual {side} intraday {sym} @ {round(px,2)} × {qty} sh."}


# ── tick / close ──────────────────────────────────────
def _signed_gross(t, px):
    sign = 1 if t.get("side", "long") == "long" else -1
    return (px - t["entry"]) * t["qty"] * sign


def _hit_exit(t, px, now):
    long = t.get("side", "long") == "long"
    if long:
        if px <= t["stop"]:
            return "stop_loss"
        if px >= t["target"]:
            return "target"
    else:
        if px >= t["stop"]:
            return "stop_loss"
        if px <= t["target"]:
            return "target"
    if now.time() >= SQUARE_OFF:
        return "square_off"
    return None


def _close(t, w, price, reason):
    qty, entry = t["qty"], t["entry"]
    fee = (charges(t["segment"], "buy", entry, qty)["total"] +
           charges(t["segment"], "sell", price, qty)["total"])
    net = round(_signed_gross(t, price) - fee, 2)
    t.update({"status": "closed", "exit_price": round(price, 2), "exit_reason": reason,
              "fees": round(fee), "net_pnl": net})
    t["analysis"] = _analysis(t)
    w["balance"] = round(w["balance"] + net, 2)
    w["realized_pnl"] = round(w.get("realized_pnl", 0) + net, 2)
    _save_trade(t); _save_wallet(w)
    try:
        from aos import memory as mem
        mem.record_lesson("user_wallet",
                          f"u{t['user_id']} {t['symbol']} {reason} net ₹{net}",
                          {"trade": t["id"]})
    except Exception:
        pass


def tick_user(uid, now=None):
    """Advance every open trade against its live price (records a P&L point,
    auto-exits on stop/target/square-off). Returns the open trades."""
    now = now or datetime.now()
    w = get_wallet(uid)
    for t in _open_trades(uid):
        px = _live_price(t)
        if px is None:
            continue
        gross = _signed_gross(t, px)
        t["pnl_series"].append([now.strftime("%H:%M:%S"), round(px, 2), round(gross, 1)])
        reason = _hit_exit(t, px, now)
        if reason:
            _close(t, w, px, reason)
        else:
            with _conn() as c:
                c.execute("UPDATE trades SET pnl_series=? WHERE id=?",
                          (json.dumps(t["pnl_series"]), t["id"]))
    return _open_trades(uid)


def close_trade(uid, trade_id, price=None):
    """Manual square-off of one open trade at the live (or given) price."""
    with _conn() as c:
        row = c.execute("SELECT * FROM trades WHERE id=? AND user_id=?",
                        (trade_id, uid)).fetchone()
    if not row:
        return {"error": "trade not found"}
    t = _row_to_trade(row)
    if t["status"] != "open":
        return {"error": f"trade already {t['status']}"}
    px = price if price is not None else _live_price(t)
    if px is None:
        return {"error": "could not read a live price to close at"}
    w = get_wallet(uid)
    _close(t, w, px, "manual")
    return {"status": "closed", "trade": t}


# ── history + AI "why this trade" ─────────────────────
def history_full(uid):
    """Every trade the user has taken (any status), newest first — for the
    date-grouped History page."""
    with _conn() as c:
        rows = c.execute("SELECT * FROM trades WHERE user_id=? ORDER BY opened_at DESC",
                         (uid,)).fetchall()
    return [_row_to_trade(r) for r in rows]


def _why_base(t):
    """Deterministic, fact-only explanation of why the trade was taken. Every
    number here is computed from the trade itself — the LLM only rephrases it
    (anti-hallucination rule), never invents figures."""
    side = t.get("side", "long")
    risk = abs(t["entry"] - t["stop"]) * t["qty"]
    reward = abs(t["target"] - t["entry"]) * t["qty"]
    rr = (reward / risk) if risk else 0
    direction = {"long": "a bullish (long)", "short": "a bearish (short)"}.get(side, side)
    seg = {"options": "index-options", "futures": "index-futures",
           "equity_intraday": "intraday-equity"}.get(t["segment"], t["segment"])
    return (f"This was {direction} {seg} trade on {t['symbol']}. "
            f"Entry {t['entry']}, stop-loss {t['stop']}, target {t['target']} — "
            f"risking about ₹{round(risk):,} to make about ₹{round(reward):,} "
            f"(reward-to-risk {rr:.1f}:1) across {t['qty']} units. "
            f"Entry rationale: {t.get('reason', 'discretionary')}.")


def explain_trade(uid, trade_id):
    """Return an AI explanation of why the trade was taken. Cached on the trade
    row after the first call. Failure-safe: if the LLM is unavailable, the
    deterministic fact-based rationale is returned (and cached)."""
    with _conn() as c:
        row = c.execute("SELECT * FROM trades WHERE id=? AND user_id=?",
                        (trade_id, uid)).fetchone()
    if not row:
        return {"error": "trade not found"}
    t = _row_to_trade(row)
    if t.get("ai_why"):
        return {"ai_why": t["ai_why"], "cached": True}
    base = _why_base(t)
    try:
        from api.narrate import polish
        prose = polish(base)
    except Exception:
        prose = base
    with _conn() as c:
        c.execute("UPDATE trades SET ai_why=? WHERE id=?", (prose, t["id"]))
    return {"ai_why": prose, "cached": False}


# ── ML auto-trading ──────────────────────────────────
def ml_users():
    """All user IDs with trade_mode='ml'."""
    init_db()
    with _conn() as c:
        rows = c.execute("SELECT user_id FROM wallets WHERE trade_mode='ml'").fetchall()
    return [r["user_id"] for r in rows]


def auto_open_trade(uid):
    """Pick and open the best trade for an ML-mode user (one at a time).
    Returns the trade dict on success, None if no trade available or already
    has an open position."""
    opens = _open_trades(uid)
    if opens:
        return None  # already has a position — let it ride
    w = get_wallet(uid)
    available = round(w["balance"] - _locked_margin(uid), 2)
    from aos.sim_wallet import pick_trade
    plan = pick_trade(available)
    if not plan:
        return None
    spec = {"segment": plan.get("segment", "options")}
    if plan.get("underlying"):
        spec["underlying"] = plan["underlying"]
    if plan.get("leg"):
        spec["leg"] = plan["leg"]
    if plan.get("strike"):
        spec["strike"] = plan["strike"]
    if plan.get("lots"):
        spec["lots"] = plan["lots"]
    if plan.get("symbol") and plan.get("kind") == "equity":
        spec["symbol"] = plan["symbol"]
    if plan.get("qty") and plan.get("kind") == "equity":
        spec["qty"] = plan["qty"]
    spec["side"] = plan.get("side", "long")
    spec["reason"] = f"[ML auto] {plan.get('reason', 'system pick')}"
    res = open_trade(uid, spec)
    if res.get("error"):
        return None
    return res.get("trade")


def ml_tick_all():
    """Run one cycle for every ML-mode user: tick open trades (auto-closes on
    SL/target/square-off) and open a new trade if flat. Called by the background
    loop in the API server."""
    now = datetime.now()
    from aos.sim_wallet import SQUARE_OFF
    from datetime import time as dtime
    market_open = dtime(9, 15) <= now.time() <= SQUARE_OFF
    results = []
    for uid in ml_users():
        try:
            tick_user(uid, now)
            if market_open:
                t = auto_open_trade(uid)
                if t:
                    results.append({"uid": uid, "opened": t["symbol"]})
        except Exception:
            pass
    return results


# ── view object for the API / frontend ────────────────
def status(uid, do_tick=True):
    if do_tick:
        try:
            tick_user(uid)
        except Exception:
            pass  # never let a price-fetch failure break the dashboard
    w = get_wallet(uid)
    opens = _open_trades(uid)
    unreal = 0.0
    for t in opens:
        if t["pnl_series"]:
            unreal += t["pnl_series"][-1][2]
    return {"wallet": w, "live_equity": round(w["balance"] + unreal, 2),
            "unrealized": round(unreal, 1), "open_trades": opens,
            "history": _history(uid), "trade_mode": get_mode(uid)}


# Ensure tables + migrations are applied whenever this module is imported (e.g.
# by the API server), so direct-query helpers like history_full/explain_trade
# never hit a missing column on a pre-existing DB.
init_db()


if __name__ == "__main__":
    # Deterministic lifecycle test with a throwaway DB and simulated prices.
    import tempfile
    DB = os.path.join(tempfile.gettempdir(), "trading_ai_uw_test.db")
    if os.path.exists(DB):
        os.remove(DB)
    print("=" * 64); print("  USER-WALLET — simulated trade lifecycle"); print("=" * 64)
    UID = 1
    print("  start balance:", get_wallet(UID)["balance"])
    # inject an option trade manually (skip live pricing)
    t = {"id": "t1", "user_id": UID, "date": date.today().isoformat(),
         "segment": "options", "kind": "option", "side": "long",
         "symbol": "BANKNIFTY 54000 CE", "chart_symbol": "BANKNIFTY",
         "underlying": "BANKNIFTY", "strike": 54000, "leg": "CE", "qty": 35,
         "lots": 1, "entry": 200.0, "stop": 130.0, "target": 300.0, "cost": 7000.0,
         "status": "open", "opened_at": datetime.now().isoformat(), "pnl_series": "[]",
         "exit_price": None, "exit_reason": None, "fees": None, "net_pnl": None,
         "reason": "test entry", "analysis": None}
    _insert_trade(t)
    w = get_wallet(UID)
    for px in [210, 240, 280, 305]:                 # rises to target
        for tr in _open_trades(UID):
            tr["pnl_series"].append(["x", px, round(_signed_gross(tr, px), 1)])
            reason = _hit_exit(tr, px, datetime.now())
            if reason:
                _close(tr, w, px, reason)
            else:
                with _conn() as c:
                    c.execute("UPDATE trades SET pnl_series=? WHERE id=?",
                              (json.dumps(tr["pnl_series"]), tr["id"]))
    st = status(UID, do_tick=False)
    h = st["history"][0]
    print(f"  trade {h['symbol']} {h['status']} exit {h['exit_price']} ({h['exit_reason']})")
    print(f"  net P&L ₹{h['net_pnl']} | new balance ₹{st['wallet']['balance']}")
    print(f"  analysis: {h['analysis']}")
    # short-side sanity: a short that the price falls into target
    short = {**t, "id": "t2", "side": "short", "symbol": "NIFTY FUT", "kind": "future",
             "segment": "futures", "underlying": "NIFTY", "chart_symbol": "NIFTY",
             "strike": None, "leg": None, "entry": 100.0, "stop": 102.0,
             "target": 98.0, "qty": 75, "lots": 1, "cost": 7500.0, "status": "open",
             "pnl_series": "[]"}
    _insert_trade(short)
    w = get_wallet(UID)
    tr = [x for x in _open_trades(UID) if x["id"] == "t2"][0]
    reason = _hit_exit(tr, 97.5, datetime.now())
    tr["pnl_series"].append(["x", 97.5, round(_signed_gross(tr, 97.5), 1)])
    _close(tr, w, 97.5, reason)
    h2 = status(UID, do_tick=False)["history"][0]
    print(f"  short {h2['symbol']} {h2['exit_reason']} net ₹{h2['net_pnl']} "
          f"(price fell, short profits)")
    try:
        os.remove(DB)
    except OSError:
        pass

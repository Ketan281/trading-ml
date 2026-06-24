"""
Per-user paper wallet & trade book.

The single global wallet in aos/sim_wallet.py takes ONE autonomous trade a day.
This module gives every signed-up user their OWN paper wallet and lets them
*manually* open trades across segments (options / futures / intraday equity),
priced by the same live engines. Storage is SQLite (data/users.db — the same
file the auth user table lives in) so a user and their book share one place.

All trade opens/closes pass through the broker.ExecutionGate boundary.
TRADING_MODE=paper (default) preserves exact current behavior (instant fills).
TRADING_MODE=live routes orders through the configured broker adapter.

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
DEPOSIT_CAP = 100_000

# Forex wallet: separate USD-denominated wallet for currency trading.
FX_DEFAULT_BALANCE = 10_000.0     # $10,000 starting balance
FX_DEPOSIT_CAP     = 100_000.0    # max $1,00,000 total deposited

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

# Forex: leveraged (50:1 default), pip-based P&L, spread as cost.
FX_LEVERAGE = 50
FX_LOT_SIZES = {"standard": 100_000, "mini": 10_000, "micro": 1_000}
FX_DEFAULT_LOT = "micro"


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
            CREATE TABLE IF NOT EXISTS forex_wallets (
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
                ai_why      TEXT,
                decision_id INTEGER,
                regime      TEXT,
                conviction  REAL,
                signal_source TEXT,
                broker_order_id TEXT
            )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_trades_user ON trades(user_id, status)")
        # Migrations for pre-existing tables.
        cols = [r[1] for r in c.execute("PRAGMA table_info(trades)")]
        if "ai_why" not in cols:
            c.execute("ALTER TABLE trades ADD COLUMN ai_why TEXT")
        for col, typ in [("decision_id", "INTEGER"), ("regime", "TEXT"),
                         ("conviction", "REAL"), ("signal_source", "TEXT"),
                         ("broker_order_id", "TEXT")]:
            if col not in cols:
                c.execute(f"ALTER TABLE trades ADD COLUMN {col} {typ}")
        wcols = [r[1] for r in c.execute("PRAGMA table_info(wallets)")]
        if "trade_mode" not in wcols:
            c.execute("ALTER TABLE wallets ADD COLUMN trade_mode TEXT DEFAULT 'custom'")
        if "indian_trade_mode" not in wcols:
            c.execute("ALTER TABLE wallets ADD COLUMN indian_trade_mode TEXT DEFAULT 'custom'")
        if "forex_trade_mode" not in wcols:
            c.execute("ALTER TABLE wallets ADD COLUMN forex_trade_mode TEXT DEFAULT 'custom'")
        if "trading_mode" not in wcols:
            c.execute("ALTER TABLE wallets ADD COLUMN trading_mode TEXT DEFAULT 'paper'")
        c.execute("""
            CREATE TABLE IF NOT EXISTS broker_config (
                user_id     INTEGER PRIMARY KEY,
                broker      TEXT NOT NULL DEFAULT 'angelone',
                api_key     TEXT,
                client_id   TEXT,
                password    TEXT,
                totp_secret TEXT,
                configured  INTEGER NOT NULL DEFAULT 0,
                updated_at  TEXT
            )""")


# ── wallet ops (Indian INR wallet) ────────────────────
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


# ── Forex USD wallet ─────────────────────────────────
def get_forex_wallet(uid):
    init_db()
    with _conn() as c:
        row = c.execute("SELECT * FROM forex_wallets WHERE user_id=?", (uid,)).fetchone()
        if row is None:
            now = datetime.now().isoformat()
            c.execute("INSERT INTO forex_wallets (user_id, balance, total_deposited, "
                      "realized_pnl, created) VALUES (?,?,?,0,?)",
                      (uid, FX_DEFAULT_BALANCE, FX_DEFAULT_BALANCE, now))
            return {"user_id": uid, "balance": FX_DEFAULT_BALANCE,
                    "total_deposited": FX_DEFAULT_BALANCE, "realized_pnl": 0.0,
                    "created": now, "currency": "USD"}
    d = dict(row)
    d["currency"] = "USD"
    return d


def _save_forex_wallet(w):
    with _conn() as c:
        c.execute("UPDATE forex_wallets SET balance=?, total_deposited=?, realized_pnl=? "
                  "WHERE user_id=?",
                  (w["balance"], w["total_deposited"], w["realized_pnl"], w["user_id"]))


def deposit_forex(uid, amount):
    w = get_forex_wallet(uid); amount = float(amount)
    if amount <= 0:
        return {"error": "amount must be positive"}
    if w["total_deposited"] + amount > FX_DEPOSIT_CAP:
        room = FX_DEPOSIT_CAP - w["total_deposited"]
        return {"error": f"deposit cap ${FX_DEPOSIT_CAP:,.0f} reached — you can add at most "
                f"${room:,.0f} more (profits can still grow the balance)."}
    w["balance"] = round(w["balance"] + amount, 2)
    w["total_deposited"] = round(w["total_deposited"] + amount, 2)
    _save_forex_wallet(w)
    return {"ok": True, "wallet": w}


def reset_forex(uid):
    with _conn() as c:
        c.execute("DELETE FROM trades WHERE user_id=? AND segment='forex'", (uid,))
        c.execute("DELETE FROM forex_wallets WHERE user_id=?", (uid,))
    return get_forex_wallet(uid)


# ── trade mode (separate toggles for Indian & Forex) ─
def get_mode(uid):
    get_wallet(uid)
    with _conn() as c:
        row = c.execute("SELECT trade_mode, indian_trade_mode, forex_trade_mode "
                        "FROM wallets WHERE user_id=?", (uid,)).fetchone()
    return {
        "indian_trade_mode": (row["indian_trade_mode"] if row and row["indian_trade_mode"] else "custom"),
        "forex_trade_mode": (row["forex_trade_mode"] if row and row["forex_trade_mode"] else "custom"),
        "trade_mode": (row["trade_mode"] if row and row["trade_mode"] else "custom"),
    }


def set_mode(uid, mode, market="indian"):
    if mode not in ("ml", "custom"):
        return {"error": "mode must be 'ml' or 'custom'"}
    if market not in ("indian", "forex", "both"):
        return {"error": "market must be 'indian', 'forex', or 'both'"}
    get_wallet(uid)
    if market == "both":
        with _conn() as c:
            c.execute("UPDATE wallets SET indian_trade_mode=?, forex_trade_mode=? "
                      "WHERE user_id=?", (mode, mode, uid))
    else:
        col = "indian_trade_mode" if market == "indian" else "forex_trade_mode"
        with _conn() as c:
            c.execute(f"UPDATE wallets SET {col}=? WHERE user_id=?", (mode, uid))
    return {"ok": True, "market": market, "trade_mode": mode}


def get_trading_mode(uid):
    get_wallet(uid)
    with _conn() as c:
        row = c.execute("SELECT trading_mode FROM wallets WHERE user_id=?", (uid,)).fetchone()
    return (row["trading_mode"] if row and row["trading_mode"] else "paper")


def set_trading_mode(uid, mode):
    if mode not in ("paper", "live"):
        return {"error": "mode must be 'paper' or 'live'"}
    if mode == "live":
        cfg = get_broker_config(uid)
        if not cfg.get("configured"):
            return {"error": "configure your broker account before switching to live"}
    get_wallet(uid)
    with _conn() as c:
        c.execute("UPDATE wallets SET trading_mode=? WHERE user_id=?", (mode, uid))
    return {"ok": True, "trading_mode": mode}


def get_broker_config(uid):
    init_db()
    with _conn() as c:
        row = c.execute("SELECT * FROM broker_config WHERE user_id=?", (uid,)).fetchone()
    if not row:
        return {"user_id": uid, "broker": "angelone", "configured": False,
                "has_api_key": False, "has_client_id": False,
                "has_password": False, "has_totp": False}
    return {
        "user_id": uid, "broker": row["broker"],
        "configured": bool(row["configured"]),
        "has_api_key": bool(row["api_key"]),
        "has_client_id": bool(row["client_id"]),
        "client_id_display": row["client_id"][:2] + "***" if row["client_id"] else None,
        "has_password": bool(row["password"]),
        "has_totp": bool(row["totp_secret"]),
        "updated_at": row["updated_at"],
    }


def save_broker_config(uid, api_key, client_id, password, totp_secret):
    init_db()
    now = datetime.now().isoformat()
    configured = 1 if all([api_key, client_id, password, totp_secret]) else 0
    with _conn() as c:
        c.execute("""INSERT INTO broker_config (user_id, broker, api_key, client_id,
                     password, totp_secret, configured, updated_at)
                     VALUES (?,?,?,?,?,?,?,?)
                     ON CONFLICT(user_id) DO UPDATE SET
                     api_key=excluded.api_key, client_id=excluded.client_id,
                     password=excluded.password, totp_secret=excluded.totp_secret,
                     configured=excluded.configured, updated_at=excluded.updated_at""",
                  (uid, "angelone", api_key, client_id, password, totp_secret,
                   configured, now))
    return {"ok": True, "configured": bool(configured)}


def delete_broker_config(uid):
    init_db()
    with _conn() as c:
        c.execute("DELETE FROM broker_config WHERE user_id=?", (uid,))
        c.execute("UPDATE wallets SET trading_mode='paper' WHERE user_id=?", (uid,))
    return {"ok": True, "trading_mode": "paper"}


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
        c.execute("DELETE FROM trades WHERE user_id=? AND segment!='forex'", (uid,))
        c.execute("DELETE FROM wallets WHERE user_id=?", (uid,))
    return get_wallet(uid)


def reset_all_wallets():
    """Reset every user's wallet to DEFAULT_BALANCE and clear all trades."""
    init_db()
    with _conn() as c:
        c.execute("DELETE FROM trades WHERE segment!='forex'")
        c.execute("DELETE FROM wallets")
    return {"status": "all wallets reset", "default_balance": DEFAULT_BALANCE}


# ── trade storage helpers ─────────────────────────────
_TRADE_COLS = ["id", "user_id", "date", "segment", "kind", "side", "symbol",
               "chart_symbol", "underlying", "strike", "leg", "qty", "lots",
               "entry", "stop", "target", "cost", "status", "opened_at",
               "exit_price", "exit_reason", "fees", "net_pnl", "pnl_series",
               "reason", "analysis", "broker_order_id"]


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
            "net_pnl=?, pnl_series=?, analysis=?, broker_order_id=? WHERE id=?",
            (t["status"], t["exit_price"], t["exit_reason"], t["fees"], t["net_pnl"],
             json.dumps(t["pnl_series"]), t["analysis"],
             t.get("broker_order_id"), t["id"]))


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


def _locked_margin(uid, segment_filter=None):
    """Capital tied up in currently-open trades (paper margin).
    segment_filter: 'forex' for forex-only, 'indian' for non-forex, None for all."""
    trades = _open_trades(uid)
    if segment_filter == "forex":
        trades = [t for t in trades if t.get("segment") == "forex"]
    elif segment_filter == "indian":
        trades = [t for t in trades if t.get("segment") != "forex"]
    return sum(t["cost"] for t in trades)


# ── live price for a trade ────────────────────────────
def _live_price(t):
    if t["kind"] == "option":
        return _option_ltp(t["underlying"], t["strike"], t["leg"])
    if t["kind"] == "forex":
        from pipelines.forex.data import current_price as fx_price
        return fx_price(t["symbol"])
    sym = t["chart_symbol"] or t["symbol"]
    if t["kind"] == "futures" and sym in INDICES:
        from pipelines.options.chain_live_intel import fetch_chain
        ch = fetch_chain(sym)
        return float(ch["spot"]) if ch and "spot" in ch else None
    from agents.auto_trader import _stock_price
    return _stock_price(sym)


# ── open a trade ──────────────────────────────────────
def open_trade(uid, spec):
    """spec: {segment, ...}. Returns {status, trade} or {error}.

    Segments:
      options  — underlying NIFTY/BANKNIFTY, leg CE/PE, optional strike (else ATM),
                 optional lots (else max affordable). Always long the option.
      futures  — index NIFTY/BANKNIFTY, side long/short, optional lots.
      equity   — intraday equity, symbol, side long/short, optional qty + levels.
      forex    — currency pair, side buy/sell. Uses the separate USD forex wallet.
    """
    init_db()
    seg = (spec.get("segment") or "").lower()
    if seg == "forex":
        w = get_forex_wallet(uid)
        available = round(w["balance"] - _locked_margin(uid, segment_filter="forex"), 2)
    else:
        w = get_wallet(uid)
        available = round(w["balance"] - _locked_margin(uid, segment_filter="indian"), 2)
    try:
        if seg in ("options", "option"):
            plan = _plan_option(spec, available)
        elif seg in ("futures", "future"):
            plan = _plan_futures(spec, available)
        elif seg in ("equity", "equity_intraday", "intraday"):
            plan = _plan_equity(spec, available)
        elif seg == "forex":
            plan = _plan_forex(spec, available)
        else:
            return {"error": f"unknown segment '{seg}'"}
    except _PlanError as e:
        return {"error": str(e)}

    regime = _current_regime()
    source = _signal_source(seg, plan.get("reason", ""))
    decision_id, signal_id, conviction = _record_lineage(
        plan.get("symbol", "?"), seg, regime, source, plan.get("reason", ""))

    from broker.executor import get_executor
    gate = get_executor()
    result = gate.open(
        symbol=plan.get("symbol", "?"), segment=seg,
        side=plan.get("side", "long"), qty=plan.get("qty", 0),
        price=plan.get("entry", 0), stop_loss=plan.get("stop", 0),
        target=plan.get("target", 0), uid=uid,
        reason=plan.get("reason", ""))
    if result.error:
        return {"error": f"execution blocked: {result.error}"}
    if result.filled_price and result.filled_price != plan.get("entry"):
        plan["entry"] = result.filled_price

    plan.update({
        "id": str(uuid.uuid4())[:8], "user_id": uid, "date": date.today().isoformat(),
        "status": "open", "opened_at": datetime.now().isoformat(), "pnl_series": "[]",
        "exit_price": None, "exit_reason": None, "fees": None, "net_pnl": None,
        "analysis": None,
        "decision_id": decision_id, "regime": regime,
        "conviction": conviction, "signal_source": source,
        "broker_order_id": result.broker_order_id,
    })
    _insert_trade(plan)
    plan["pnl_series"] = []
    return {"status": "opened", "trade": plan, "decision_id": decision_id,
            "broker_order_id": result.broker_order_id}


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


def _plan_forex(spec, available):
    from pipelines.forex.data import current_price as fx_price, PIP_INFO, SPREADS, list_pairs
    pair = spec.get("pair") or spec.get("symbol") or ""
    if not pair or pair not in list_pairs():
        raise _PlanError(f"unsupported forex pair '{pair}' — supported: {list_pairs()}")
    side = (spec.get("side") or spec.get("direction") or "buy").lower()
    if side not in ("buy", "sell"):
        raise _PlanError("side must be buy or sell")
    px = fx_price(pair)
    if not px:
        raise _PlanError(f"could not read {pair} price right now")
    pip = PIP_INFO.get(pair, {"pip": 0.0001, "pip_value": 10.0})
    pip_size = pip["pip"]
    pip_value = pip["pip_value"]
    lot_type = spec.get("lot_type", FX_DEFAULT_LOT)
    lot_units = FX_LOT_SIZES.get(lot_type, FX_LOT_SIZES["micro"])
    lots = int(spec.get("lots") or 1)
    qty = lots * lot_units
    margin = round(qty * px / FX_LEVERAGE, 2)
    if margin > available:
        max_lots = int(available * FX_LEVERAGE / (lot_units * px))
        if max_lots < 1:
            raise _PlanError(f"insufficient margin: need ${margin:,.0f} for {lots} "
                             f"{lot_type} lot(s), only ${available:,.0f} free.")
        lots = max_lots
        qty = lots * lot_units
        margin = round(qty * px / FX_LEVERAGE, 2)
    sl_pips = float(spec.get("sl_pips") or 30)
    tp_pips = float(spec.get("tp_pips") or 60)
    sign = 1 if side == "buy" else -1
    stop = round(px - sign * sl_pips * pip_size, 5)
    target = round(px + sign * tp_pips * pip_size, 5)
    spread_cost = round(SPREADS.get(pair, 1.5) * pip_value * lots * (lot_units / 100_000), 2)
    return {"kind": "forex", "segment": "forex", "side": side,
            "underlying": None, "chart_symbol": pair, "strike": None, "leg": None,
            "symbol": pair, "qty": qty, "lots": lots, "entry": round(px, 5),
            "stop": stop, "target": target, "cost": margin,
            "reason": spec.get("reason") or
                      f"Forex {side} {pair} @ {px} × {lots} {lot_type} lot(s) "
                      f"(margin ${margin:,.0f}, leverage {FX_LEVERAGE}:1, "
                      f"spread ~${spread_cost})."}


# ── tick / close ──────────────────────────────────────
def _signed_gross(t, px):
    sign = 1 if t.get("side", "long") == "long" else -1
    if t.get("kind") == "forex":
        sign = 1 if t.get("side") == "buy" else -1
    return (px - t["entry"]) * t["qty"] * sign


def _hit_exit(t, px, now):
    is_long = t.get("side", "long") in ("long", "buy")
    if is_long:
        if px <= t["stop"]:
            return "stop_loss"
        if px >= t["target"]:
            return "target"
    else:
        if px >= t["stop"]:
            return "stop_loss"
        if px <= t["target"]:
            return "target"
    # Forex trades run 24/5 — no intraday square-off
    if t.get("kind") != "forex" and now.time() >= SQUARE_OFF:
        return "square_off"
    return None


def _current_regime():
    try:
        from pipelines.market_regime import _load_index, regime_at
        df = _load_index()
        if df is not None:
            return regime_at(df).get("regime", "unknown")
    except Exception:
        pass
    return "unknown"


def _signal_source(segment, reason=""):
    if "forex" in (segment or ""):
        return "forex_confluence"
    if "[ML auto]" in reason:
        return "ml_auto"
    if "Manual" in reason:
        return "manual"
    return "options_engine"


def _record_lineage(symbol, segment, regime, source, reason):
    """Record a signal + decision in Trade Memory and return (decision_id, signal_id, conviction)."""
    try:
        from aos import memory as mem
        conviction = 50.0
        signal_id = mem.record_signal(
            symbol, segment, source, score=None, confidence=None,
            regime=regime, sentiment=None, snapshot={"reason": reason})
        decision_id = mem.record_decision(
            symbol, segment, "BUY", "BUY", conviction, regime,
            vetoed=False, veto_reason=None,
            evidence={"source": source, "signal_id": signal_id, "reason": reason})
        mem.record_trade(decision_id, symbol, segment, "long",
                         0, 0, 0, None, status="open")
        return decision_id, signal_id, conviction
    except Exception:
        return None, None, 50.0


def _close(t, w, price, reason):
    qty, entry = t["qty"], t["entry"]

    from broker.executor import get_executor
    gate = get_executor()
    broker_oid = t.get("broker_order_id", "")
    if broker_oid:
        ex_result = gate.close(broker_oid, qty, price, reason=reason,
                               uid=t.get("user_id", 0), symbol=t.get("symbol", ""))
        if ex_result.error and gate.is_live():
            import logging
            logging.getLogger("user_wallet").warning(
                "broker close failed for %s: %s — closing paper side anyway",
                t["id"], ex_result.error)
        if ex_result.filled_price:
            price = ex_result.filled_price

    fee = (charges(t["segment"], "buy", entry, qty)["total"] +
           charges(t["segment"], "sell", price, qty)["total"])
    net = round(_signed_gross(t, price) - fee, 2)
    t.update({"status": "closed", "exit_price": round(price, 2), "exit_reason": reason,
              "fees": round(fee), "net_pnl": net})
    t["analysis"] = _analysis(t)
    w["balance"] = round(w["balance"] + net, 2)
    w["realized_pnl"] = round(w.get("realized_pnl", 0) + net, 2)
    is_forex = t.get("segment") == "forex"
    _save_trade(t)
    if is_forex:
        _save_forex_wallet(w)
    else:
        _save_wallet(w)
    try:
        from aos import memory as mem
        ccy = "$" if is_forex else "₹"
        did = t.get("decision_id")
        mem.record_lesson("user_wallet",
                          f"u{t['user_id']} {t['symbol']} {reason} net {ccy}{net}",
                          {"trade": t["id"], "decision_id": did})
        if did:
            mem.close_trade(did, net, round(fee), reason)
        sig_source = t.get("signal_source") or _signal_source(t.get("segment", ""), "")
        sigs = mem.query(
            "SELECT id FROM signals WHERE symbol=? AND source=? AND outcome_ret IS NULL "
            "ORDER BY id DESC LIMIT 1", (t["symbol"], sig_source))
        if sigs:
            mem.set_signal_outcome(sigs[0]["id"], outcome_ret=net,
                                   outcome_label=1 if net > 0 else 0)
    except Exception:
        pass


def tick_user(uid, now=None):
    """Advance every open trade against its live price (records a P&L point,
    auto-exits on stop/target/square-off). In live mode, syncs with broker
    position state first. Returns the open trades."""
    now = now or datetime.now()
    w_indian = get_wallet(uid)
    w_forex = get_forex_wallet(uid)

    from broker.executor import get_executor
    gate = get_executor()
    is_live = gate.is_live()

    for t in _open_trades(uid):
        if is_live and t.get("broker_order_id"):
            pos = gate.sync_position(t["broker_order_id"])
            if pos.status == "closed":
                w = w_forex if t.get("segment") == "forex" else w_indian
                exit_px = pos.avg_price if pos.avg_price else _live_price(t) or t["entry"]
                _close(t, w, exit_px, "broker_closed")
                continue

        px = _live_price(t)
        if px is None:
            continue
        gross = _signed_gross(t, px)
        t["pnl_series"].append([now.strftime("%H:%M:%S"), round(px, 2), round(gross, 1)])
        reason = _hit_exit(t, px, now)
        if reason:
            w = w_forex if t.get("segment") == "forex" else w_indian
            _close(t, w, px, reason)
        else:
            with _conn() as c:
                c.execute("UPDATE trades SET pnl_series=? WHERE id=?",
                          (json.dumps(t["pnl_series"]), t["id"]))
    return _open_trades(uid)


def close_trade(uid, trade_id, price=None):
    """Manual square-off of one open trade at the live (or given) price.

    ML Auto mode: after closing, if the user's mode is ML auto for the
    relevant market, the system automatically picks the next best
    recommendation respecting portfolio risk limits.
    """
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
    is_forex = t.get("segment") == "forex"
    w = get_forex_wallet(uid) if is_forex else get_wallet(uid)
    _close(t, w, px, "manual")

    replacement = None
    try:
        modes = get_mode(uid)
        market_key = "forex_trade_mode" if is_forex else "indian_trade_mode"
        if modes.get(market_key) == "ml_auto":
            from aos.sim_wallet import SQUARE_OFF
            from datetime import time as dtime
            now = datetime.now()
            if is_forex:
                rep = auto_open_forex_trade(uid)
                if rep:
                    replacement = rep
            else:
                indian_market_open = dtime(9, 15) <= now.time() <= SQUARE_OFF
                if indian_market_open:
                    rep = auto_open_trade(uid)
                    if rep:
                        replacement = rep
    except Exception:
        pass

    result = {"status": "closed", "trade": t}
    if replacement:
        result["auto_replacement"] = replacement
    return result


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


# ── ML auto-trading (dual market) ────────────────────
def ml_indian_users():
    """All user IDs with indian_trade_mode='ml'."""
    init_db()
    with _conn() as c:
        rows = c.execute("SELECT user_id FROM wallets WHERE indian_trade_mode='ml'").fetchall()
    return [r["user_id"] for r in rows]


def ml_forex_users():
    """All user IDs with forex_trade_mode='ml'."""
    init_db()
    with _conn() as c:
        rows = c.execute("SELECT user_id FROM wallets WHERE forex_trade_mode='ml'").fetchall()
    return [r["user_id"] for r in rows]


def ml_users():
    """Legacy compat: all user IDs with any ML mode enabled."""
    return list(set(ml_indian_users() + ml_forex_users()))


def _pick_best_recommendation(uid, balance):
    """Pick best trade using ML inference engine (instant — reads pre-trained model).
    Falls back to scheduler cache if ML model not available."""
    held_symbols = {t["symbol"] for t in _open_trades(uid)}

    # Try ML inference first (fastest, smartest)
    try:
        from engines.ml_inference import get_top_picks_for_trading
        picks = get_top_picks_for_trading(capital=balance, max_picks=10)
        if picks:
            for pick in picks:
                if pick.get("symbol") in held_symbols:
                    continue
                if pick.get("confidence", 0) < 30:
                    continue
                return pick
    except Exception:
        pass

    # Fallback: scheduler cache
    try:
        from engines.market_scheduler import get_cached_recommendations
        reco = get_cached_recommendations()
        if reco:
            all_picks = []
            for seg in ("equity_intraday", "options", "swing"):
                all_picks.extend(reco.get(seg, []))
            all_picks.sort(key=lambda p: p.get("confidence", 0), reverse=True)
            for pick in all_picks:
                if pick.get("symbol") in held_symbols:
                    continue
                if pick.get("confidence", 0) < 30:
                    continue
                return pick
    except Exception:
        pass
    return None


def _reco_to_spec(pick):
    """Convert a recommendation pick dict to a trade spec for open_trade()."""
    seg = pick.get("segment", "equity_intraday")
    spec = {}
    if seg == "options":
        spec["segment"] = "options"
        spec["underlying"] = pick.get("underlying", "NIFTY")
        leg = pick.get("leg", "CE")
        spec["leg"] = leg
        if pick.get("strike") and pick["strike"] != "—":
            try:
                spec["strike"] = int(pick["strike"])
            except (ValueError, TypeError):
                pass
        if pick.get("lots"):
            spec["lots"] = pick["lots"]
        spec["side"] = "long"
    elif seg == "swing":
        spec["segment"] = "equity_delivery"
        spec["symbol"] = pick["symbol"]
        spec["side"] = "long"
        if pick.get("entry"):
            spec["entry"] = pick["entry"]
        if pick.get("stop"):
            spec["stop"] = pick["stop"]
        if pick.get("target"):
            spec["target"] = pick["target"]
    else:
        spec["segment"] = "equity"
        spec["symbol"] = pick["symbol"]
        spec["side"] = "long"
    return spec


def auto_open_trade(uid):
    """Pick and open the best Indian-market trade for an ML-mode user.

    Uses the full intelligence-backed recommendation engine:
    FII/DII · PCR · intermarket · volume profile · S/R · 30 patterns ·
    20 fundamentals · sector rotation · regime · event calendar.

    Respects portfolio risk: skips if already holding the same symbol,
    applies meta-learning sizing, and enforces minimum confidence threshold.
    """
    indian_opens = [t for t in _open_trades(uid) if t.get("segment") != "forex"]
    if indian_opens:
        return None
    w = get_wallet(uid)
    available = round(w["balance"] - _locked_margin(uid, segment_filter="indian"), 2)
    from aos.meta_learning import sizing_multiplier
    regime = _current_regime()
    size_mult = sizing_multiplier("options_engine", regime)
    adjusted_balance = round(available * size_mult, 2)

    pick = _pick_best_recommendation(uid, adjusted_balance)
    if not pick:
        return None

    if isinstance(pick, dict) and pick.get("segment"):
        spec = _reco_to_spec(pick)
        reason = pick.get("reason", "system pick")
        conf = pick.get("confidence", 0)
        if size_mult != 1.0:
            reason += f" [meta: ×{size_mult} sizing in {regime} regime]"
        spec["reason"] = f"[ML auto] {reason} (confidence {conf:.0f}%)"
    else:
        spec = {"segment": pick.get("segment", "options")}
        if pick.get("underlying"):
            spec["underlying"] = pick["underlying"]
        if pick.get("leg"):
            spec["leg"] = pick["leg"]
        if pick.get("strike"):
            spec["strike"] = pick["strike"]
        if pick.get("lots"):
            spec["lots"] = pick["lots"]
        if pick.get("symbol") and pick.get("kind") == "equity":
            spec["symbol"] = pick["symbol"]
        if pick.get("qty") and pick.get("kind") == "equity":
            spec["qty"] = pick["qty"]
        spec["side"] = pick.get("side", "long")
        reason = pick.get("reason", "system pick")
        if size_mult != 1.0:
            reason += f" [meta: ×{size_mult} sizing in {regime} regime]"
        spec["reason"] = f"[ML auto] {reason}"

    res = open_trade(uid, spec)
    if res.get("error"):
        return None
    return res.get("trade")


def auto_open_forex_trade(uid):
    """Pick and open the best forex trade for an ML-mode user.
    Uses meta-learning sizing multiplier when enough history exists."""
    forex_opens = [t for t in _open_trades(uid) if t.get("segment") == "forex"]
    if forex_opens:
        return None
    w = get_forex_wallet(uid)
    available = round(w["balance"] - _locked_margin(uid, segment_filter="forex"), 2)
    from pipelines.forex.confluence import best_trade
    from aos.meta_learning import sizing_multiplier
    regime = _current_regime()
    size_mult = sizing_multiplier("forex_confluence", regime)
    sig = best_trade()
    if not sig or not sig.get("trade_plan"):
        return None
    tp = sig["trade_plan"]
    lots = max(1, round(1 * size_mult))
    spec = {
        "segment": "forex", "pair": tp["pair"], "side": tp["direction"],
        "sl_pips": tp.get("sl_pips", 30), "tp_pips": tp.get("tp_pips", 60),
        "lot_type": "micro", "lots": lots,
        "reason": f"[ML auto] Confluence {sig['score']:.2f} ({sig['confidence']}), "
                  f"{sig['agreeing_timeframes']}/{sig['total_timeframes']} TFs agree."
                  f"{f' [meta: ×{size_mult} in {regime}]' if size_mult != 1.0 else ''}",
    }
    res = open_trade(uid, spec)
    if res.get("error"):
        return None
    return res.get("trade")


def ml_tick_all():
    """Run one cycle for every ML-mode user: tick open trades (auto-closes on
    SL/target/square-off) and open new trades if flat in either market.
    Indian and Forex markets are independent — both can have open trades
    simultaneously. Called by the 60s background loop in the API server."""
    import logging
    log = logging.getLogger("ml_auto")
    now = datetime.now()
    from aos.sim_wallet import SQUARE_OFF
    from datetime import time as dtime
    indian_market_open = dtime(9, 15) <= now.time() <= SQUARE_OFF
    indian_uids = set(ml_indian_users())
    forex_uids = set(ml_forex_users())
    all_uids = indian_uids | forex_uids
    results = []
    for uid in all_uids:
        try:
            tick_user(uid, now)
        except Exception as e:
            log.warning("tick_user(%s) failed: %s", uid, e)
        if uid in indian_uids and indian_market_open:
            try:
                t = auto_open_trade(uid)
                if t:
                    log.info("ML opened Indian trade for uid %s: %s", uid, t["symbol"])
                    results.append({"uid": uid, "market": "indian", "opened": t["symbol"]})
            except Exception as e:
                log.warning("auto_open_trade(%s) failed: %s", uid, e)
        if uid in forex_uids:
            try:
                t = auto_open_forex_trade(uid)
                if t:
                    log.info("ML opened Forex trade for uid %s: %s", uid, t["symbol"])
                    results.append({"uid": uid, "market": "forex", "opened": t["symbol"]})
            except Exception as e:
                log.warning("auto_open_forex_trade(%s) failed: %s", uid, e)
    return results


# ── view object for the API / frontend ────────────────
def status(uid, do_tick=True):
    if do_tick:
        try:
            tick_user(uid)
        except Exception:
            pass
    w = get_wallet(uid)
    fw = get_forex_wallet(uid)
    opens = _open_trades(uid)
    indian_opens = [t for t in opens if t.get("segment") != "forex"]
    forex_opens = [t for t in opens if t.get("segment") == "forex"]
    indian_unreal = sum(t["pnl_series"][-1][2] for t in indian_opens if t["pnl_series"])
    forex_unreal = sum(t["pnl_series"][-1][2] for t in forex_opens if t["pnl_series"])
    modes = get_mode(uid)
    return {
        "indian_wallet": {**w, "currency": "INR"},
        "forex_wallet": fw,
        "wallet": w,
        "live_equity": round(w["balance"] + indian_unreal, 2),
        "forex_live_equity": round(fw["balance"] + forex_unreal, 2),
        "unrealized": round(indian_unreal, 1),
        "forex_unrealized": round(forex_unreal, 1),
        "open_trades": opens,
        "indian_open_trades": indian_opens,
        "forex_open_trades": forex_opens,
        "history": _history(uid),
        "trade_mode": modes,
        "indian_trade_mode": modes["indian_trade_mode"],
        "forex_trade_mode": modes["forex_trade_mode"],
    }


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

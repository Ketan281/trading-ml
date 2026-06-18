"""
Autonomous paper-trading wallet — one disciplined intraday call per day.

HONEST FRAMING (shown in the UI too): this is a PAPER wallet. It takes the
highest-conviction, risk-managed intraday trade each day and protects capital
with a hard stop — it does NOT guarantee profit. Some days lose. Decisions are
technical + option-chain based (no news feed wired). Paper money only.

Rules
-----
• Default balance ₹10,000.
• Deposits are capped so total *deposited* never exceeds ₹1,00,000 — but the
  BALANCE can grow past ₹1L through profits.
• The full balance carries to the next day and is used for the next call.
• Each trading day: pick ONE intraday trade, size it to the wallet, enter at
  the live price, manage with a stop + target, square off by 15:15 IST.
• Instrument: a BANKNIFTY/NIFTY option lot if the wallet can afford one; else
  (option a) an EQUITY intraday trade; else skip the day (no forced trade).
• A P&L time-series is recorded for the live chart; at close an analysis of why
  it won/lost is produced and stored.
"""

import os
import sys
import json
import uuid
from datetime import datetime, date, time as dtime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

STATE = os.path.join(ROOT, "data", "aos", "sim_wallet.json")
os.makedirs(os.path.dirname(STATE), exist_ok=True)

DEFAULT_BALANCE = 10_000
DEPOSIT_CAP     = 100_000
SQUARE_OFF      = dtime(15, 15)
OPT_SL_PCT      = 0.35      # option stop = 35% premium decay
OPT_TGT_PCT     = 0.50      # option target = +50%
LOT = {"NIFTY": 75, "BANKNIFTY": 35}


# ── state ─────────────────────────────────────────────
def _fresh():
    return {"wallet": {"balance": DEFAULT_BALANCE, "total_deposited": DEFAULT_BALANCE,
                       "realized_pnl": 0.0, "created": datetime.now().isoformat()},
            "active_trade": None, "history": []}

def load_state():
    if os.path.exists(STATE):
        try:
            return json.load(open(STATE))
        except Exception:
            pass
    s = _fresh(); save_state(s); return s

def save_state(s):
    json.dump(s, open(STATE, "w"), indent=2, default=str)


# ── wallet ops ────────────────────────────────────────
def get_wallet():
    return load_state()["wallet"]

def deposit(amount):
    s = load_state(); w = s["wallet"]; amount = float(amount)
    if amount <= 0:
        return {"error": "amount must be positive"}
    if w["total_deposited"] + amount > DEPOSIT_CAP:
        room = DEPOSIT_CAP - w["total_deposited"]
        return {"error": f"deposit cap ₹{DEPOSIT_CAP:,} reached — you can add at most "
                f"₹{room:,.0f} more (profits can still grow the balance past ₹1L)."}
    w["balance"] = round(w["balance"] + amount, 2)
    w["total_deposited"] = round(w["total_deposited"] + amount, 2)
    save_state(s)
    return {"ok": True, "wallet": w}

def reset():
    s = _fresh(); save_state(s); return s["wallet"]


# ── live price helpers ────────────────────────────────
def _option_ltp(underlying, strike, leg):
    from pipelines.options.chain_live_intel import fetch_chain
    ch = fetch_chain(underlying)
    if not ch:
        return None
    r = ch["df"].iloc[(ch["df"]["strike"] - strike).abs().argmin()]
    return float(r[f"{leg.lower()}_ltp"])

def _live_price(t):
    if t["kind"] == "option":
        return _option_ltp(t["underlying"], t["strike"], t["leg"])
    from agents.auto_trader import _stock_price
    return _stock_price(t["symbol"])


# ── the daily pick ────────────────────────────────────
def pick_trade(balance):
    # 1) Index option if a lot is affordable (BANKNIFTY then NIFTY)
    try:
        from pipelines.options.chain_live_intel import fetch_chain
        from pipelines.options_action_engine import chain_prob_up, action_from_probability
        for idx in ("BANKNIFTY", "NIFTY"):
            ch = fetch_chain(idx)
            if not ch:
                continue
            prob = chain_prob_up(ch); act = action_from_probability(prob)
            if act["leg"] not in ("CE", "PE"):
                continue
            leg = act["leg"]; atm = ch["atm"]; lot = LOT[idx]
            r = ch["df"].iloc[(ch["df"]["strike"] - atm).abs().argmin()]
            ltp = float(r[f"{leg.lower()}_ltp"])
            if ltp <= 0:
                continue
            cost1 = ltp * lot
            if cost1 <= balance:
                lots = max(1, int(balance // cost1)); qty = lots * lot
                return {"kind": "option", "segment": "options", "underlying": idx,
                        "strike": int(atm), "leg": leg,
                        "symbol": f"{idx} {atm} {leg}", "qty": qty, "lots": lots,
                        "entry": round(ltp, 2), "stop": round(ltp * (1 - OPT_SL_PCT), 2),
                        "target": round(ltp * (1 + OPT_TGT_PCT), 2),
                        "reason": f"{idx} action {act['action']} (P(up) {prob:.2f}); "
                                  f"bought ATM {atm} {leg} @ {round(ltp,2)} × {lots} lot(s)."}
    except Exception as e:
        pass

    # 2) (option a) Equity intraday fallback — top actionable from the screener
    try:
        from api.precompute import cached_screen
        rep = cached_screen() or {}
        for c in rep.get("actionable", []):
            entry = float(c.get("price") or 0); stop = float(c.get("stop_loss") or 0)
            if entry <= 0 or stop <= 0 or stop >= entry:
                continue
            shares = int(balance // entry)
            if shares < 1:
                continue
            tgt = float(c.get("target") or 0) or round(entry + 2 * (entry - stop), 2)
            return {"kind": "equity", "segment": "equity_intraday", "symbol": c["symbol"],
                    "qty": shares, "entry": round(entry, 2), "stop": round(stop, 2),
                    "target": round(tgt, 2),
                    "reason": f"No affordable index-option lot for ₹{balance:,.0f}; "
                              f"equity intraday on {c['symbol']} (grade {c.get('grade')}) "
                              f"@ {round(entry,2)} × {shares} sh."}
    except Exception:
        pass
    return None


def start_daily_trade(force=False):
    s = load_state(); today = date.today().isoformat()
    act = s.get("active_trade")
    if act and act.get("date") == today and act.get("status") in ("open", "closed") and not force:
        return {"status": "already_traded_today", "trade": act}
    plan = pick_trade(s["wallet"]["balance"])
    if not plan:
        s["active_trade"] = {"date": today, "status": "no_trade", "pnl_series": [],
                             "reason": "No affordable index-option lot and no clean equity "
                                       "setup today — skipped (capital preserved)."}
        save_state(s); return {"status": "no_trade", "trade": s["active_trade"]}
    plan.update({"id": str(uuid.uuid4())[:8], "date": today, "status": "open",
                 "opened_at": datetime.now().isoformat(), "pnl_series": [],
                 "exit_price": None, "exit_reason": None, "net_pnl": None})
    s["active_trade"] = plan
    save_state(s)
    return {"status": "opened", "trade": plan}


# ── tick (called during market hours) ─────────────────
def tick(price=None, now=None):
    s = load_state(); t = s.get("active_trade")
    if not t or t.get("status") != "open":
        return s
    now = now or datetime.now()
    px = price if price is not None else _live_price(t)
    if px is None:
        return s
    gross = (px - t["entry"]) * t["qty"]
    t["pnl_series"].append([now.strftime("%H:%M:%S"), round(px, 2), round(gross, 1)])

    reason = ("stop_loss" if px <= t["stop"] else
              "target" if px >= t["target"] else
              "square_off" if now.time() >= SQUARE_OFF else None)
    if reason:
        _close(s, t, px, reason)
    save_state(s)
    return s


def _close(s, t, price, reason):
    from agents.brokerage import charges
    qty, entry = t["qty"], t["entry"]
    fee = charges(t["segment"], "buy", entry, qty)["total"] + \
          charges(t["segment"], "sell", price, qty)["total"]
    gross = (price - entry) * qty
    net = round(gross - fee, 2)
    t.update({"status": "closed", "exit_price": round(price, 2), "exit_reason": reason,
              "fees": round(fee), "net_pnl": net})
    t["analysis"] = _analysis(t)
    w = s["wallet"]
    w["balance"] = round(w["balance"] + net, 2)
    w["realized_pnl"] = round(w.get("realized_pnl", 0) + net, 2)
    s.setdefault("history", []).insert(0, {
        "date": t["date"], "symbol": t["symbol"], "kind": t["kind"], "qty": t["qty"],
        "entry": t["entry"], "exit": t["exit_price"], "reason": reason, "net_pnl": net})
    s["history"] = s["history"][:60]
    # learn from it
    try:
        from aos import memory as mem
        mem.record_lesson("paper_wallet",
                          f"{t['symbol']} {reason} net ₹{net}", {"trade": t["id"]})
    except Exception:
        pass


def _analysis(t):
    net = t["net_pnl"]; verdict = "PROFIT ✅" if net > 0 else "LOSS ❌" if net < 0 else "FLAT"
    moved = "up" if t["exit_price"] > t["entry"] else "down"
    why = {"stop_loss": "the move went against the position and the stop-loss protected the wallet",
           "target": "the move went in favour and the target was booked",
           "square_off": "the position was squared off at the session close"}.get(t["exit_reason"], "")
    return (f"{verdict}: {t['symbol']} went from {t['entry']} to {t['exit_price']} ({moved}); "
            f"{why}. Net P&L ₹{net:,} after ₹{t.get('fees',0):,} brokerage. "
            f"Entry rationale — {t['reason']}")


# ── view object for the API / frontend ────────────────
def status():
    s = load_state(); t = s.get("active_trade"); w = s["wallet"]
    live_equity = w["balance"]; unreal = 0.0
    if t and t.get("status") == "open" and t.get("pnl_series"):
        unreal = t["pnl_series"][-1][2]
        live_equity = round(w["balance"] + unreal, 2)
    return {"wallet": w, "live_equity": live_equity, "unrealized": round(unreal, 1),
            "active_trade": t, "history": s.get("history", [])}


if __name__ == "__main__":
    # Deterministic lifecycle test with simulated prices (no NSE needed).
    print("=" * 64); print("  SIM-WALLET — simulated trade lifecycle"); print("=" * 64)
    s = reset(); print("  start balance:", s["balance"])
    # inject a trade manually (skip live picker)
    st = load_state()
    st["active_trade"] = {"id": "test", "date": date.today().isoformat(), "kind": "option",
        "segment": "options", "underlying": "BANKNIFTY", "strike": 54000, "leg": "CE",
        "symbol": "BANKNIFTY 54000 CE", "qty": 35, "lots": 1, "entry": 200.0,
        "stop": 130.0, "target": 300.0, "status": "open", "opened_at": "x",
        "pnl_series": [], "reason": "test entry"}
    save_state(st)
    for px in [210, 240, 280, 305]:                 # rises to target
        tick(price=px, now=datetime.now())
    st = status()
    t = st["active_trade"]
    print(f"  trade {t['symbol']} {t['status']} exit {t['exit_price']} ({t['exit_reason']})")
    print(f"  net P&L ₹{t['net_pnl']} | new balance ₹{st['wallet']['balance']}")
    print(f"  analysis: {t['analysis']}")
    print(f"  pnl points: {len(t['pnl_series'])}")
    reset()  # clean up

"""
Auto Trader — unified ML-driven paper trading engine.

Combines:
  1. OI Wall Selling (scored, tiered) on NIFTY + BANKNIFTY
  2. Intraday ML stock picks (top ranked by direction model)

Runs autonomously on backend. Places only the highest-probability trades.
Paper account: Rs.10,00,000 (10 lakhs).
"""

import os
import sys
import json
import sqlite3
import time
from datetime import datetime, date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DB_PATH = os.path.join(ROOT, "memory", "trading_memory.db")

INITIAL_CAPITAL = 1000000
SLIPPAGE_BPS = 5
BROKERAGE_BPS = 20
INDEX_LOT = {"NIFTY": 75, "BANKNIFTY": 30}
MARGIN_PER_LOT = 120000


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def _init_schema():
    conn = _conn()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS auto_trader_account (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            capital         REAL NOT NULL,
            deployed        REAL DEFAULT 0,
            total_pnl       REAL DEFAULT 0,
            total_trades    INTEGER DEFAULT 0,
            wins            INTEGER DEFAULT 0,
            losses          INTEGER DEFAULT 0,
            max_capital     REAL,
            max_drawdown    REAL DEFAULT 0,
            updated_at      TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS auto_trader_trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id        TEXT UNIQUE NOT NULL,
            date            TEXT NOT NULL,
            segment         TEXT NOT NULL,
            symbol          TEXT NOT NULL,
            action          TEXT NOT NULL,
            strike          INTEGER,
            entry_price     REAL NOT NULL,
            exit_price      REAL,
            qty             INTEGER NOT NULL,
            lots            INTEGER DEFAULT 0,
            score           REAL DEFAULT 0,
            tier            INTEGER DEFAULT 0,
            pnl             REAL,
            pnl_pct         REAL,
            status          TEXT DEFAULT 'open',
            entry_time      TEXT,
            exit_time       TEXT,
            exit_reason     TEXT,
            signal_data     TEXT,
            created_at      TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS auto_trader_daily (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT UNIQUE NOT NULL,
            trades          INTEGER DEFAULT 0,
            wins            INTEGER DEFAULT 0,
            pnl             REAL DEFAULT 0,
            capital_start   REAL,
            capital_end     REAL,
            drawdown_pct    REAL DEFAULT 0,
            signals_json    TEXT
        )
    """)

    # Initialize account if empty
    row = c.execute("SELECT COUNT(*) FROM auto_trader_account").fetchone()
    if row[0] == 0:
        c.execute("""
            INSERT INTO auto_trader_account (capital, deployed, total_pnl,
                total_trades, wins, losses, max_capital, max_drawdown, updated_at)
            VALUES (?, 0, 0, 0, 0, 0, ?, 0, ?)
        """, (INITIAL_CAPITAL, INITIAL_CAPITAL, datetime.now().isoformat()))

    conn.commit()
    conn.close()


def get_account():
    _init_schema()
    conn = _conn()
    c = conn.cursor()
    c.execute("SELECT * FROM auto_trader_account ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    cols = [d[0] for d in c.description]
    conn.close()
    return dict(zip(cols, row)) if row else None


def _score_wall_trade(dist_pct, entry_prem, dow, wall_type, oi_building):
    score = 0
    if dist_pct >= 3.0: score += 30
    elif dist_pct >= 2.5: score += 25
    elif dist_pct >= 2.0: score += 20
    elif dist_pct >= 1.5: score += 15
    elif dist_pct >= 1.0: score += 10

    if entry_prem < 5: score += 25
    elif entry_prem < 10: score += 20
    elif entry_prem < 25: score += 15
    elif entry_prem < 50: score += 5

    if dow == 2: score += 20
    elif dow == 1: score += 10
    elif dow == 0: score += 5
    elif dow == 3: score -= 15

    if wall_type == "put": score += 10
    else: score += 5

    if oi_building: score += 5
    return score


def _wall_ev(sig):
    """Expected value of an OI wall sell in rupees:
    EV = P(win)·max_profit − (1−P(win))·max_loss.
    P(win) is the ML-calibrated prob the wall holds (win_pct)."""
    p = (sig.get("win_pct", 0) or 0) / 100.0
    max_profit = sig.get("max_profit", 0) or 0
    max_loss = sig.get("max_loss", 0) or 0
    return round(p * max_profit - (1 - p) * max_loss, 1)


def rank_wall_trades(capital=1_000_000, skip_underlyings=None):
    """All tradeable OI wall sells across NIFTY + BANKNIFTY, ranked by EXPECTED
    VALUE (desc). EV captures 'highest probability of *profit*' honestly — it
    weights the ML win-probability by what's actually won vs. risked, so a rich
    82%-win wall outranks a thin 90%-win one. Ties break by win% then score.

    Each returned signal is enriched with `score`, `ev`, `underlying`, `wall_type`.
    """
    from pipelines.options_action_engine import simple_signal
    from datetime import datetime
    skip = set(skip_underlyings or ())
    dow = datetime.now().weekday()
    out = []
    for sym in ("NIFTY", "BANKNIFTY"):
        if sym in skip:
            continue
        try:
            raw = simple_signal(sym, capital)
        except Exception:
            continue
        if not isinstance(raw, list):
            continue
        for s in raw:
            sig = s.get("signal", "")
            if "STRANGLE" in sig or " at " not in sig:
                continue
            wtype = "put" if "PE" in sig else "call"
            score = _score_wall_trade(s.get("dist_pct", 0), s.get("premium", 0),
                                      dow, wtype, s.get("oi_building", False))
            if score < 20:                        # below tradeable threshold
                continue
            out.append({**s, "underlying": sym, "wall_type": wtype,
                        "score": score, "ev": _wall_ev(s)})
    out.sort(key=lambda c: (c["ev"], c.get("win_pct", 0), c["score"]), reverse=True)
    return out


def best_trade_today(capital=1_000_000):
    """The single highest expected-value OI wall sell right now, or None.
    This is the system's canonical 'daily best trade' — used by the autonomous
    loop and the Home suggestion so they always agree."""
    ranked = rank_wall_trades(capital)
    return ranked[0] if ranked else None


def _get_wall_signals():
    """Get OI wall selling signals from live chain data."""
    try:
        from pipelines.options_action_engine import simple_signal
        signals = []
        account = get_account()
        capital = account["capital"] if account else INITIAL_CAPITAL

        for sym in ["NIFTY", "BANKNIFTY"]:
            result = simple_signal(sym, capital)
            if not result or isinstance(result, dict):
                continue
            for s in result:
                dist = s.get("dist_pct", 0)
                prem = s.get("premium", 0)
                dow = datetime.now().weekday()
                wtype = "put" if "PE" in s.get("signal", "") else "call"
                oi_build = s.get("oi_building", False)

                sc = _score_wall_trade(dist, prem, dow, wtype, oi_build)

                if sc >= 55:
                    tier = 1
                elif sc >= 35:
                    tier = 2
                elif sc >= 20:
                    tier = 3
                else:
                    continue

                lot = INDEX_LOT[sym]
                tier_mult = {1: 1.0, 2: 0.5, 3: 0.25}[tier]
                max_lots = max(1, int(capital / MARGIN_PER_LOT * tier_mult))
                max_lots = min(max_lots, s.get("lots", 1))

                signals.append({
                    "segment": "index_options",
                    "symbol": sym,
                    "action": s.get("signal", ""),
                    "strike": s.get("strike", 0),
                    "entry_price": prem,
                    "target": s.get("target", 0),
                    "stoploss": s.get("stoploss", 0),
                    "qty": max_lots * lot,
                    "lots": max_lots,
                    "score": sc,
                    "tier": tier,
                    "win_pct": s.get("win_pct", 0),
                    "dist_pct": dist,
                    "funds_required": s.get("funds_required", 0),
                    "oi_building": oi_build,
                })
        return signals
    except Exception as e:
        print(f"  [auto_trader] Wall signals error: {e}")
        return []


def _get_stock_signals():
    """Get top intraday stock picks from ML ranker."""
    try:
        from pipelines.screener import screen
        rep = screen()
        if not rep:
            return []

        signals = []
        account = get_account()
        capital = account["capital"] if account else INITIAL_CAPITAL
        per_stock = capital * 0.05  # 5% per stock max

        actionable = rep.get("actionable", [])
        for pick in actionable[:5]:
            sym = pick.get("symbol", "")
            price = pick.get("price")
            stop = pick.get("stop_loss")
            target = pick.get("target")
            conv = pick.get("conviction", 0)
            grade = pick.get("grade", "C")

            if not price or not stop:
                continue
            try:
                price = float(str(price).replace(",", "").split("-")[-1])
                stop = float(str(stop).replace(",", ""))
                target = float(str(target).replace(",", "")) if target else price * 1.02
            except (ValueError, TypeError):
                continue

            if stop >= price or conv < 50:
                continue

            risk_pct = (price - stop) / price * 100
            if risk_pct > 3:
                continue

            qty = max(1, int(per_stock / price))
            rr = (target - price) / (price - stop) if (price - stop) > 0 else 0

            score = conv
            if grade in ("A+", "A"):
                score += 10
            if rr >= 2:
                score += 5

            signals.append({
                "segment": "equity_intraday",
                "symbol": sym,
                "action": "BUY",
                "strike": 0,
                "entry_price": round(price, 2),
                "target": round(target, 2),
                "stoploss": round(stop, 2),
                "qty": qty,
                "lots": 0,
                "score": round(score, 1),
                "tier": 1 if grade in ("A+", "A") else 2,
                "win_pct": round(conv, 1),
                "dist_pct": 0,
                "funds_required": round(qty * price, 0),
                "grade": grade,
                "rr": round(rr, 2),
            })

        signals.sort(key=lambda x: x["score"], reverse=True)
        return signals[:3]
    except Exception as e:
        print(f"  [auto_trader] Stock signals error: {e}")
        return []


def generate_signals():
    """Generate all signals, pick only the best."""
    _init_schema()

    today = date.today().isoformat()
    conn = _conn()
    c = conn.cursor()
    existing = c.execute(
        "SELECT COUNT(*) FROM auto_trader_trades WHERE date=? AND status='open'",
        (today,)).fetchone()[0]
    conn.close()

    if existing > 0:
        return {"status": "already_traded", "date": today,
                "message": f"{existing} trades already open today"}

    wall_signals = _get_wall_signals()
    stock_signals = _get_stock_signals()
    all_signals = wall_signals + stock_signals

    if not all_signals:
        return {"status": "no_signals", "date": today}

    all_signals.sort(key=lambda x: x["score"], reverse=True)

    return {
        "status": "signals_ready",
        "date": today,
        "wall_signals": wall_signals,
        "stock_signals": stock_signals,
        "best": all_signals[0],
        "all_ranked": all_signals,
    }


def place_best_trades():
    """Place the highest-probability trades for today."""
    _init_schema()

    today = date.today().isoformat()
    account = get_account()
    capital = account["capital"]

    conn = _conn()
    c = conn.cursor()
    existing = c.execute(
        "SELECT COUNT(*) FROM auto_trader_trades WHERE date=?",
        (today,)).fetchone()[0]
    conn.close()

    if existing > 0:
        return {"status": "already_traded", "date": today}

    sig_data = generate_signals()
    if sig_data["status"] != "signals_ready":
        return sig_data

    placed = []
    total_deployed = 0

    # Place best wall selling trades (max 2: 1 NIFTY + 1 BANKNIFTY)
    wall_placed = set()
    for sig in sig_data.get("wall_signals", []):
        sym = sig["symbol"]
        if sym in wall_placed:
            continue
        if sig["score"] < 20:
            continue
        if total_deployed + sig.get("funds_required", MARGIN_PER_LOT) > capital * 0.8:
            continue

        trade_id = f"AT_{sym}_{today}_{sig['action'][:8]}_{int(time.time())}"
        conn = _conn()
        c = conn.cursor()
        c.execute("""
            INSERT INTO auto_trader_trades
                (trade_id, date, segment, symbol, action, strike, entry_price,
                 qty, lots, score, tier, status, entry_time, signal_data, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)
        """, (
            trade_id, today, sig["segment"], sym, sig["action"],
            sig.get("strike", 0), sig["entry_price"],
            sig["qty"], sig["lots"], sig["score"], sig["tier"],
            datetime.now().isoformat(),
            json.dumps(sig, default=str),
            datetime.now().isoformat(),
        ))
        conn.commit()
        conn.close()

        total_deployed += sig.get("funds_required", MARGIN_PER_LOT)
        wall_placed.add(sym)
        placed.append({
            "trade_id": trade_id,
            "segment": "index_options",
            "symbol": sym,
            "action": sig["action"],
            "premium": sig["entry_price"],
            "target": sig.get("target", 0),
            "stoploss": sig.get("stoploss", 0),
            "qty": sig["qty"],
            "score": sig["score"],
            "tier": sig["tier"],
            "win_pct": sig.get("win_pct", 0),
        })

    # Place best stock trade (only 1 — highest score)
    for sig in sig_data.get("stock_signals", [])[:1]:
        if sig["score"] < 50:
            continue
        if total_deployed + sig.get("funds_required", 50000) > capital * 0.9:
            continue

        trade_id = f"AT_{sig['symbol']}_{today}_{int(time.time())}"
        conn = _conn()
        c = conn.cursor()
        c.execute("""
            INSERT INTO auto_trader_trades
                (trade_id, date, segment, symbol, action, strike, entry_price,
                 qty, lots, score, tier, status, entry_time, signal_data, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)
        """, (
            trade_id, today, sig["segment"], sig["symbol"], sig["action"],
            0, sig["entry_price"],
            sig["qty"], 0, sig["score"], sig["tier"],
            datetime.now().isoformat(),
            json.dumps(sig, default=str),
            datetime.now().isoformat(),
        ))
        conn.commit()
        conn.close()

        total_deployed += sig.get("funds_required", 0)
        placed.append({
            "trade_id": trade_id,
            "segment": "equity_intraday",
            "symbol": sig["symbol"],
            "action": "BUY",
            "entry": sig["entry_price"],
            "target": sig.get("target", 0),
            "stoploss": sig.get("stoploss", 0),
            "qty": sig["qty"],
            "score": sig["score"],
            "grade": sig.get("grade", ""),
        })

    # Update account
    conn = _conn()
    conn.execute(
        "UPDATE auto_trader_account SET deployed=?, updated_at=?",
        (total_deployed, datetime.now().isoformat()))
    conn.commit()
    conn.close()

    # Save daily record
    conn = _conn()
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO auto_trader_daily (date, trades, capital_start, signals_json)
        VALUES (?, ?, ?, ?)
    """, (today, len(placed), capital, json.dumps(placed, default=str)))
    conn.commit()
    conn.close()

    return {
        "status": "trades_placed",
        "date": today,
        "trades": placed,
        "total_deployed": round(total_deployed, 0),
        "capital": round(capital, 0),
    }


def close_trades(current_prices=None):
    """Close all open trades at EOD (intraday exit)."""
    conn = _conn()
    c = conn.cursor()
    c.execute("SELECT * FROM auto_trader_trades WHERE status='open'")
    rows = c.fetchall()
    cols = [d[0] for d in c.description]
    trades = [dict(zip(cols, r)) for r in rows]
    conn.close()

    if not trades:
        return {"status": "no_open_trades"}

    if current_prices is None:
        current_prices = _fetch_current_prices(trades)

    closed = []
    total_pnl = 0

    for trade in trades:
        sym = trade["symbol"]
        segment = trade["segment"]
        entry = trade["entry_price"]
        qty = trade["qty"]

        if segment == "index_options":
            sig_data = json.loads(trade.get("signal_data", "{}"))
            # For wall selling, we need current option premium
            # Use target/stoploss logic
            exit_price = current_prices.get(f"{sym}_option_{trade['strike']}", entry)
            pnl = (entry - exit_price) * qty  # selling: profit = entry - exit
            sl = sig_data.get("stoploss", entry * 1.5)
            tgt = sig_data.get("target", entry * 0.4)
            if exit_price >= sl:
                pnl = (entry - sl) * qty
                reason = "stoploss"
            elif exit_price <= tgt:
                pnl = (entry - tgt) * qty
                reason = "target"
            else:
                reason = "eod_exit"
        else:
            exit_price = current_prices.get(sym, entry)
            slippage = exit_price * SLIPPAGE_BPS / 10000
            exit_price -= slippage
            brokerage = (entry + exit_price) * qty * BROKERAGE_BPS / 10000
            pnl = (exit_price - entry) * qty - brokerage
            reason = "eod_exit"

        pnl_pct = pnl / (entry * qty) * 100 if entry * qty > 0 else 0
        is_win = pnl > 0

        conn = _conn()
        conn.execute("""
            UPDATE auto_trader_trades
            SET exit_price=?, pnl=?, pnl_pct=?, status='closed',
                exit_time=?, exit_reason=?
            WHERE trade_id=?
        """, (round(exit_price, 2), round(pnl, 2), round(pnl_pct, 2),
              datetime.now().isoformat(), reason, trade["trade_id"]))
        conn.commit()
        conn.close()

        total_pnl += pnl
        closed.append({
            "trade_id": trade["trade_id"],
            "symbol": sym,
            "segment": segment,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "win": is_win,
            "reason": reason,
        })

    # Update account
    account = get_account()
    new_capital = account["capital"] + total_pnl
    new_total_pnl = account["total_pnl"] + total_pnl
    new_trades = account["total_trades"] + len(closed)
    new_wins = account["wins"] + sum(1 for c in closed if c["win"])
    new_losses = account["losses"] + sum(1 for c in closed if not c["win"])
    max_cap = max(account.get("max_capital", INITIAL_CAPITAL), new_capital)
    dd = (max_cap - new_capital) / max_cap * 100 if max_cap > 0 else 0
    max_dd = max(account.get("max_drawdown", 0), dd)

    conn = _conn()
    conn.execute("""
        UPDATE auto_trader_account
        SET capital=?, deployed=0, total_pnl=?, total_trades=?,
            wins=?, losses=?, max_capital=?, max_drawdown=?, updated_at=?
    """, (round(new_capital, 2), round(new_total_pnl, 2), new_trades,
          new_wins, new_losses, round(max_cap, 2), round(max_dd, 2),
          datetime.now().isoformat()))
    conn.commit()
    conn.close()

    # Update daily record
    today = date.today().isoformat()
    conn = _conn()
    conn.execute("""
        UPDATE auto_trader_daily
        SET wins=?, pnl=?, capital_end=?
        WHERE date=?
    """, (sum(1 for c in closed if c["win"]), round(total_pnl, 2),
          round(new_capital, 2), today))
    conn.commit()
    conn.close()

    return {
        "status": "trades_closed",
        "closed": closed,
        "total_pnl": round(total_pnl, 2),
        "capital": round(new_capital, 2),
        "win_rate": round(new_wins / new_trades * 100, 1) if new_trades > 0 else 0,
        "total_return": round(new_total_pnl / INITIAL_CAPITAL * 100, 2),
    }


def _fetch_current_prices(trades):
    """Fetch live prices for open positions."""
    prices = {}
    try:
        from models.cross_sectional import load_prices
        symbols = [t["symbol"] for t in trades if t["segment"] == "equity_intraday"]
        if symbols:
            data = load_prices(universe=set(symbols))
            for s, df in data.items():
                prices[s] = float(df["Close"].iloc[-1])
    except Exception:
        pass

    try:
        from pipelines.options_action_engine import simple_signal
        for sym in ["NIFTY", "BANKNIFTY"]:
            result = simple_signal(sym, INITIAL_CAPITAL)
            if result and not isinstance(result, dict):
                for s in result:
                    key = f"{sym}_option_{s.get('strike', 0)}"
                    prices[key] = s.get("premium", 0)
    except Exception:
        pass

    return prices


def dashboard():
    """Full account dashboard."""
    _init_schema()
    account = get_account()
    if not account:
        return {"error": "No account found"}

    conn = _conn()
    c = conn.cursor()

    # Recent trades
    c.execute("""
        SELECT * FROM auto_trader_trades
        ORDER BY created_at DESC LIMIT 20
    """)
    rows = c.fetchall()
    cols = [d[0] for d in c.description]
    recent = [dict(zip(cols, r)) for r in rows]

    # Open trades
    c.execute("SELECT * FROM auto_trader_trades WHERE status='open'")
    rows = c.fetchall()
    open_trades = [dict(zip(cols, r)) for r in rows]

    # Daily history
    c.execute("SELECT * FROM auto_trader_daily ORDER BY date DESC LIMIT 30")
    rows = c.fetchall()
    day_cols = [d[0] for d in c.description]
    daily = [dict(zip(day_cols, r)) for r in rows]

    conn.close()

    total_trades = account["total_trades"]
    win_rate = account["wins"] / total_trades * 100 if total_trades > 0 else 0
    pf = 0
    if total_trades > 0:
        conn = _conn()
        c = conn.cursor()
        c.execute("SELECT pnl FROM auto_trader_trades WHERE status='closed'")
        pnls = [r[0] for r in c.fetchall() if r[0] is not None]
        conn.close()
        gw = sum(p for p in pnls if p > 0)
        gl = abs(sum(p for p in pnls if p <= 0))
        pf = gw / gl if gl > 0 else 0

    return {
        "account": {
            "capital": round(account["capital"], 0),
            "initial": INITIAL_CAPITAL,
            "total_pnl": round(account["total_pnl"], 0),
            "return_pct": round(account["total_pnl"] / INITIAL_CAPITAL * 100, 2),
            "total_trades": total_trades,
            "wins": account["wins"],
            "losses": account["losses"],
            "win_rate": round(win_rate, 1),
            "profit_factor": round(pf, 2),
            "max_drawdown": round(account["max_drawdown"], 2),
            "deployed": round(account.get("deployed", 0), 0),
        },
        "open_trades": [{
            "trade_id": t["trade_id"],
            "symbol": t["symbol"],
            "segment": t["segment"],
            "action": t["action"],
            "entry": t["entry_price"],
            "qty": t["qty"],
            "score": t["score"],
            "tier": t["tier"],
        } for t in open_trades],
        "recent_trades": [{
            "date": t["date"],
            "symbol": t["symbol"],
            "segment": t["segment"],
            "action": t["action"],
            "entry": t["entry_price"],
            "exit": t.get("exit_price"),
            "pnl": t.get("pnl"),
            "status": t["status"],
            "score": t["score"],
        } for t in recent],
        "daily_pnl": [{
            "date": d["date"],
            "trades": d["trades"],
            "wins": d.get("wins", 0),
            "pnl": d.get("pnl", 0),
        } for d in daily],
    }


def reset_account(capital=INITIAL_CAPITAL):
    """Reset paper trading account."""
    _init_schema()
    conn = _conn()
    conn.execute("DELETE FROM auto_trader_trades")
    conn.execute("DELETE FROM auto_trader_daily")
    conn.execute("DELETE FROM auto_trader_account")
    conn.execute("""
        INSERT INTO auto_trader_account
            (capital, deployed, total_pnl, total_trades, wins, losses,
             max_capital, max_drawdown, updated_at)
        VALUES (?, 0, 0, 0, 0, 0, ?, 0, ?)
    """, (capital, capital, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    return {"status": "reset", "capital": capital}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["signals", "trade", "close", "dashboard", "reset"])
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL)
    args = parser.parse_args()

    if args.cmd == "signals":
        result = generate_signals()
        print(json.dumps(result, indent=2, default=str))
    elif args.cmd == "trade":
        result = place_best_trades()
        print(json.dumps(result, indent=2, default=str))
    elif args.cmd == "close":
        result = close_trades()
        print(json.dumps(result, indent=2, default=str))
    elif args.cmd == "dashboard":
        result = dashboard()
        print(json.dumps(result, indent=2, default=str))
    elif args.cmd == "reset":
        result = reset_account(args.capital)
        print(json.dumps(result, indent=2, default=str))

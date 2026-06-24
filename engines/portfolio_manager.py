"""
Intelligent Portfolio Management Service.

Ties together ALL ML models, signals, and risk engines into a single
portfolio management layer that:

1. DAILY BRIEF: What to trade today (index options V2 + stock picks)
2. POSITION SIZING: Kelly-aware, conviction-weighted, drawdown-adjusted
3. RISK DASHBOARD: Real-time portfolio risk, exposure, concentration
4. AUTO-REBALANCE: Exit stale trades, trim winners, cut losers
5. PERFORMANCE: Track, attribute, and report returns
6. ALERTS: Notify on signal changes, stop triggers, opportunity

Works with the existing user_wallet (aos/user_wallet.py) for execution.
"""

import os
import sys
import json
import logging
import sqlite3
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

log = logging.getLogger("portfolio_manager")

DB_PATH = os.path.join(ROOT, "data", "portfolio_mgmt.db")


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


def _migrate():
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            date TEXT NOT NULL,
            total_equity REAL,
            cash REAL,
            positions_value REAL,
            day_pnl REAL,
            day_pnl_pct REAL,
            cumulative_pnl REAL,
            n_open_trades INTEGER,
            max_drawdown REAL,
            sharpe_30d REAL,
            win_rate_30d REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, date)
        );

        CREATE TABLE IF NOT EXISTS portfolio_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            severity TEXT DEFAULT 'info',
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            data TEXT,
            read INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS portfolio_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            date TEXT NOT NULL,
            plan TEXT NOT NULL,
            executed INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, date)
        );
        """)


_migrate()


# ── 1. DAILY BRIEF ────────────────────────────────────

def daily_brief(user_id, capital=100000):
    """Generate today's trading plan combining all signal sources."""
    try:
        from aos import user_wallet as uw
        wallet = uw.get_wallet(user_id)
        balance = wallet.get("balance", capital)
        open_trades = uw._open_trades(user_id)
    except Exception:
        balance = capital
        open_trades = []

    brief = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "market_status": _market_status(),
        "wallet": {"balance": round(balance, 2), "currency": "INR"},
        "open_positions": len(open_trades),
        "signals": {},
        "plan": [],
        "risk_summary": {},
    }

    # Index Options V2 signals
    try:
        from engines.index_options_v2 import predict_index_options_v2
        v2 = predict_index_options_v2()
        brief["signals"]["index_options"] = []
        for t in v2:
            if t.get("tier") == "NO_TRADE":
                brief["signals"]["index_options"].append({
                    "symbol": t["symbol"], "action": "NO_TRADE",
                    "reason": t["reason"],
                })
            else:
                entry = {
                    "symbol": t["symbol"],
                    "action": "BUY",
                    "direction": t["direction"],
                    "contract": t.get("contract"),
                    "tier": t["tier"],
                    "win_rate_est": t["win_rate_est"],
                    "ltp": t.get("ltp", 0),
                    "target": t.get("target_premium", 0),
                    "stoploss": t.get("sl_premium", 0),
                    "position_pct": t.get("position_pct", 0),
                    "signals": t.get("signals", {}),
                    "reason": t["reason"],
                }
                brief["signals"]["index_options"].append(entry)

                lots = _compute_lots(
                    t["symbol"], balance, t.get("ltp", 0),
                    t.get("win_rate_est", 50), t.get("position_pct", 50)
                )
                brief["plan"].append({
                    "priority": 1,
                    "segment": "index_options",
                    "action": f"BUY {t.get('contract', '')}",
                    "lots": lots,
                    "entry": t.get("ltp", 0),
                    "target": t.get("target_premium", 0),
                    "stoploss": t.get("sl_premium", 0),
                    "win_rate": t["win_rate_est"],
                    "risk_amount": round(lots * _lot_size(t["symbol"]) * abs(t.get("ltp", 0) - t.get("sl_premium", 0)), 0),
                })
    except Exception as e:
        brief["signals"]["index_options"] = [{"error": str(e)}]

    # Stock equity picks (skip on micro/Oracle — yfinance hangs for NIFTY.NS)
    if not os.environ.get("AOS_PROFILE") == "micro":
        try:
            from engines.intraday_inference import get_intraday_equity_trades
            equity = get_intraday_equity_trades(capital=balance, max_picks=5)
            if isinstance(equity, dict):
                brief["signals"]["equity"] = equity.get("trades", [])
            else:
                brief["signals"]["equity"] = []
        except Exception as e:
            brief["signals"]["equity"] = [{"error": str(e)}]
    else:
        brief["signals"]["equity"] = []

    # Stock options picks (skip on micro — same yfinance issue)
    if not os.environ.get("AOS_PROFILE") == "micro":
        try:
            from engines.intraday_inference import get_intraday_options_trades
            opts = get_intraday_options_trades(capital=balance, max_picks=5)
            if isinstance(opts, dict):
                brief["signals"]["stock_options"] = opts.get("trades", [])
            else:
                brief["signals"]["stock_options"] = []
        except Exception as e:
            brief["signals"]["stock_options"] = [{"error": str(e)}]
    else:
        brief["signals"]["stock_options"] = []

    # Risk summary
    brief["risk_summary"] = _compute_risk_summary(user_id, balance, open_trades)

    # Check for stale positions to exit
    exits = _check_exit_signals(open_trades)
    for ex in exits:
        brief["plan"].append({
            "priority": 0,
            "segment": "exit",
            "action": f"CLOSE {ex['symbol']}",
            "reason": ex["reason"],
            "trade_id": ex["trade_id"],
        })

    brief["plan"].sort(key=lambda x: x["priority"])

    # Save plan
    _save_plan(user_id, brief)

    return brief


def _market_status():
    """Quick market regime check."""
    now = datetime.now()
    hour = now.hour
    minute = now.minute
    dow = now.weekday()

    if dow >= 5:
        return {"status": "closed", "reason": "Weekend"}
    if hour < 9 or (hour == 9 and minute < 15):
        mins_left = (9 * 60 + 15) - (hour * 60 + minute)
        return {"status": "pre_market", "opens_in": f"{mins_left // 60}h {mins_left % 60}m"}
    if hour > 15 or (hour == 15 and minute > 30):
        return {"status": "closed", "reason": "After hours"}
    return {"status": "open", "session": "regular"}


def _lot_size(symbol):
    sizes = {"NIFTY": 75, "BANKNIFTY": 30, "FINNIFTY": 40}
    return sizes.get(symbol, 1)


def _compute_lots(symbol, capital, premium, win_rate, position_pct):
    lot = _lot_size(symbol)
    if premium <= 0:
        return 1
    cost_per_lot = lot * premium
    max_by_capital = max(1, int(capital * (position_pct / 100) / cost_per_lot))
    kelly = max(0, (win_rate / 100 * 2 - 1))
    kelly_lots = max(1, int(max_by_capital * kelly * 0.25))
    return min(kelly_lots, max_by_capital, 10)


# ── 2. RISK DASHBOARD ─────────────────────────────────

def risk_dashboard(user_id):
    """Real-time portfolio risk metrics."""
    try:
        from aos import user_wallet as uw
        wallet = uw.get_wallet(user_id)
        balance = wallet.get("balance", 0)
        open_trades = uw._open_trades(user_id)
        history = uw.history_full(user_id)
    except Exception:
        balance = 0
        open_trades = []
        history = []

    total_exposure = 0
    segment_exposure = {}
    unrealized_pnl = 0

    for t in open_trades:
        cost = abs(t.get("cost", 0))
        seg = t.get("segment", "unknown")
        total_exposure += cost
        segment_exposure[seg] = segment_exposure.get(seg, 0) + cost
        if t.get("pnl_series"):
            unrealized_pnl += t["pnl_series"][-1][2]

    live_equity = balance + unrealized_pnl

    # Historical metrics
    closed = [t for t in history if t.get("status") == "closed"]
    recent = [t for t in closed if _is_recent(t.get("closed_at", ""), 30)]

    wins_30d = sum(1 for t in recent if (t.get("net_pnl") or 0) > 0)
    total_30d = len(recent)
    win_rate_30d = round(wins_30d / total_30d * 100, 1) if total_30d > 0 else 0

    total_pnl = sum(t.get("net_pnl", 0) or 0 for t in closed)
    pnl_30d = sum(t.get("net_pnl", 0) or 0 for t in recent)

    # Drawdown
    peak = balance + total_pnl
    dd = round((live_equity - peak) / peak * 100, 2) if peak > 0 else 0

    # Concentration
    concentration = {}
    for seg, exp in segment_exposure.items():
        concentration[seg] = round(exp / max(live_equity, 1) * 100, 1)

    return {
        "timestamp": datetime.now().isoformat(),
        "equity": {
            "total": round(live_equity, 2),
            "cash": round(balance, 2),
            "invested": round(total_exposure, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "cash_pct": round(balance / max(live_equity, 1) * 100, 1),
        },
        "exposure": {
            "total": round(total_exposure, 2),
            "by_segment": segment_exposure,
            "concentration": concentration,
            "leverage": round(total_exposure / max(balance, 1), 2),
        },
        "performance": {
            "total_pnl": round(total_pnl, 2),
            "pnl_30d": round(pnl_30d, 2),
            "win_rate_30d": win_rate_30d,
            "total_trades": len(closed),
            "trades_30d": total_30d,
            "drawdown_pct": dd,
        },
        "open_positions": [{
            "id": t["id"],
            "symbol": t["symbol"],
            "segment": t.get("segment"),
            "side": t.get("side"),
            "entry": t.get("entry"),
            "current": t["pnl_series"][-1][1] if t.get("pnl_series") else t.get("entry"),
            "pnl": t["pnl_series"][-1][2] if t.get("pnl_series") else 0,
            "pnl_pct": round((t["pnl_series"][-1][2] / t["cost"]) * 100, 1)
                        if t.get("pnl_series") and t.get("cost") else 0,
            "stop": t.get("stop"),
            "target": t.get("target"),
            "opened_at": t.get("opened_at"),
        } for t in open_trades],
        "risk_limits": {
            "max_daily_loss_pct": 2.0,
            "max_positions": 8,
            "max_single_position_pct": 20,
            "max_segment_concentration_pct": 50,
            "positions_used": len(open_trades),
        },
        "alerts": _get_risk_alerts(user_id, live_equity, balance, open_trades, dd),
    }


def _get_risk_alerts(user_id, equity, cash, trades, drawdown):
    alerts = []
    if drawdown < -5:
        alerts.append({"severity": "critical", "message": f"Drawdown at {drawdown:.1f}% - consider reducing exposure"})
    if len(trades) > 6:
        alerts.append({"severity": "warning", "message": f"{len(trades)} open positions - approaching max limit"})
    if cash / max(equity, 1) < 0.2:
        alerts.append({"severity": "warning", "message": f"Cash only {cash/max(equity,1)*100:.0f}% of equity - low liquidity"})

    losing = [t for t in trades if t.get("pnl_series") and t["pnl_series"][-1][2] < 0]
    if len(losing) > len(trades) * 0.7 and len(trades) > 2:
        alerts.append({"severity": "critical", "message": f"{len(losing)}/{len(trades)} positions are underwater"})

    return alerts


def _is_recent(date_str, days):
    if not date_str:
        return False
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00").replace("+00:00", ""))
        return (datetime.now() - dt).days <= days
    except Exception:
        return False


# ── 3. AUTO-TRADE (ML-driven execution) ───────────────

def auto_trade(user_id, dry_run=False):
    """Execute the daily plan automatically.

    1. Close any positions flagged for exit
    2. Open new positions from the daily brief
    3. Respect risk limits
    """
    from aos import user_wallet as uw

    brief = daily_brief(user_id)
    wallet = uw.get_wallet(user_id)
    balance = wallet.get("balance", 0)
    open_trades = uw._open_trades(user_id)
    executed = []

    # Risk gate
    if len(open_trades) >= 8:
        return {"status": "blocked", "reason": "Max 8 positions reached", "plan": brief["plan"]}

    for action in brief["plan"]:
        if action["segment"] == "exit" and not dry_run:
            try:
                result = uw.close_trade(user_id, action["trade_id"])
                executed.append({"action": "CLOSED", "trade_id": action["trade_id"], "result": result})
                _create_alert(user_id, "trade_closed", "info",
                              f"Closed {action['action']}", action.get("reason", ""))
            except Exception as e:
                executed.append({"action": "CLOSE_FAILED", "trade_id": action["trade_id"], "error": str(e)})
            continue

        if action["segment"] == "index_options":
            risk_amt = action.get("risk_amount", 0)
            if risk_amt > balance * 0.05:
                executed.append({"action": "SKIPPED", "reason": f"Risk Rs.{risk_amt} > 5% of balance"})
                continue

            if not dry_run:
                # Parse contract: "NIFTY 23800 CE"
                parts = action["action"].replace("BUY ", "").split()
                if len(parts) >= 3:
                    spec = {
                        "segment": "options",
                        "underlying": parts[0],
                        "strike": int(parts[1]),
                        "leg": parts[2],
                        "side": "long",
                        "lots": action.get("lots", 1),
                        "entry": action.get("entry"),
                        "stop": action.get("stoploss"),
                        "target": action.get("target"),
                        "reason": f"V2 auto-trade: {action.get('win_rate', 0)}% win rate",
                    }
                    try:
                        result = uw.open_trade(user_id, spec)
                        executed.append({"action": "OPENED", "spec": spec, "result": result})
                        _create_alert(user_id, "trade_opened", "info",
                                      f"Opened {action['action']}",
                                      f"Win rate: {action.get('win_rate', 0)}%")
                    except Exception as e:
                        executed.append({"action": "OPEN_FAILED", "spec": spec, "error": str(e)})
            else:
                executed.append({"action": "DRY_RUN", "would_execute": action})

    return {
        "status": "executed" if not dry_run else "dry_run",
        "timestamp": datetime.now().isoformat(),
        "executed": executed,
        "plan": brief["plan"],
        "signals": brief["signals"],
    }


# ── 4. EXIT SIGNALS ───────────────────────────────────

def _check_exit_signals(open_trades):
    exits = []
    now = datetime.now()

    for t in open_trades:
        reasons = []

        # Time-based: close intraday positions before market close
        if now.hour >= 15 and now.minute >= 15:
            if t.get("segment") in ("options", "futures"):
                reasons.append("Market closing - intraday square-off")

        # Stop hit
        if t.get("pnl_series") and t.get("stop"):
            last_px = t["pnl_series"][-1][1]
            side = t.get("side", "long")
            if side == "long" and last_px <= t["stop"]:
                reasons.append(f"Stop loss hit: {last_px} <= {t['stop']}")
            elif side == "short" and last_px >= t["stop"]:
                reasons.append(f"Stop loss hit: {last_px} >= {t['stop']}")

        # Target hit
        if t.get("pnl_series") and t.get("target"):
            last_px = t["pnl_series"][-1][1]
            side = t.get("side", "long")
            if side == "long" and last_px >= t["target"]:
                reasons.append(f"Target hit: {last_px} >= {t['target']}")
            elif side == "short" and last_px <= t["target"]:
                reasons.append(f"Target hit: {last_px} <= {t['target']}")

        # Big loss
        if t.get("pnl_series") and t.get("cost"):
            pnl = t["pnl_series"][-1][2]
            if t["cost"] > 0 and pnl / t["cost"] < -0.3:
                reasons.append(f"Loss exceeds 30% of cost")

        if reasons:
            exits.append({
                "trade_id": t["id"],
                "symbol": t["symbol"],
                "reason": "; ".join(reasons),
            })

    return exits


# ── 5. RISK SUMMARY ───────────────────────────────────

def _compute_risk_summary(user_id, balance, open_trades):
    total_risk = 0
    for t in open_trades:
        if t.get("stop") and t.get("entry") and t.get("qty"):
            risk = abs(t["entry"] - t["stop"]) * t["qty"]
            total_risk += risk

    return {
        "total_risk_amount": round(total_risk, 0),
        "risk_pct_of_capital": round(total_risk / max(balance, 1) * 100, 1),
        "open_positions": len(open_trades),
        "max_allowed": 8,
        "can_open_more": len(open_trades) < 8,
    }


# ── 6. PERFORMANCE REPORT ─────────────────────────────

def performance_report(user_id, days=30):
    """Detailed performance attribution."""
    try:
        from aos import user_wallet as uw
        history = uw.history_full(user_id)
    except Exception:
        history = []
    closed = [t for t in history if t.get("status") == "closed"]

    if not closed:
        return {"message": "No closed trades yet", "trades": 0}

    recent = [t for t in closed if _is_recent(t.get("closed_at", ""), days)]

    # By segment
    by_segment = {}
    for t in recent:
        seg = t.get("segment", "unknown")
        if seg not in by_segment:
            by_segment[seg] = {"trades": 0, "wins": 0, "pnl": 0, "gross_win": 0, "gross_loss": 0}
        by_segment[seg]["trades"] += 1
        pnl = t.get("net_pnl") or t.get("gross_pnl") or 0
        by_segment[seg]["pnl"] += pnl
        if pnl > 0:
            by_segment[seg]["wins"] += 1
            by_segment[seg]["gross_win"] += pnl
        else:
            by_segment[seg]["gross_loss"] += abs(pnl)

    for seg in by_segment:
        s = by_segment[seg]
        s["win_rate"] = round(s["wins"] / max(s["trades"], 1) * 100, 1)
        s["profit_factor"] = round(s["gross_win"] / max(s["gross_loss"], 1), 2)
        s["avg_pnl"] = round(s["pnl"] / max(s["trades"], 1), 0)

    # By symbol (top winners/losers)
    by_symbol = {}
    for t in recent:
        sym = t.get("symbol", "?")
        pnl = t.get("net_pnl") or t.get("gross_pnl") or 0
        if sym not in by_symbol:
            by_symbol[sym] = {"trades": 0, "pnl": 0}
        by_symbol[sym]["trades"] += 1
        by_symbol[sym]["pnl"] += pnl

    sorted_symbols = sorted(by_symbol.items(), key=lambda x: x[1]["pnl"], reverse=True)
    top_winners = [{"symbol": s, **d} for s, d in sorted_symbols[:5] if d["pnl"] > 0]
    top_losers = [{"symbol": s, **d} for s, d in sorted_symbols[-5:] if d["pnl"] < 0]

    total_pnl = sum(t.get("net_pnl") or t.get("gross_pnl") or 0 for t in recent)
    wins = sum(1 for t in recent if (t.get("net_pnl") or 0) > 0)

    # Daily P&L curve
    daily_pnl = {}
    for t in recent:
        d = (t.get("closed_at") or "")[:10]
        if d:
            daily_pnl[d] = daily_pnl.get(d, 0) + (t.get("net_pnl") or t.get("gross_pnl") or 0)

    equity_curve = []
    cumulative = 0
    for d in sorted(daily_pnl.keys()):
        cumulative += daily_pnl[d]
        equity_curve.append({"date": d, "daily_pnl": round(daily_pnl[d], 0), "cumulative": round(cumulative, 0)})

    return {
        "period_days": days,
        "total_trades": len(recent),
        "total_pnl": round(total_pnl, 0),
        "win_rate": round(wins / max(len(recent), 1) * 100, 1),
        "by_segment": by_segment,
        "top_winners": top_winners,
        "top_losers": top_losers,
        "equity_curve": equity_curve,
        "all_time": {
            "total_trades": len(closed),
            "total_pnl": round(sum(t.get("net_pnl") or t.get("gross_pnl") or 0 for t in closed), 0),
        },
    }


# ── 7. ALERTS ─────────────────────────────────────────

def _create_alert(user_id, alert_type, severity, title, message, data=None):
    with _conn() as c:
        c.execute(
            "INSERT INTO portfolio_alerts (user_id, alert_type, severity, title, message, data) VALUES (?,?,?,?,?,?)",
            (user_id, alert_type, severity, title, message, json.dumps(data) if data else None)
        )


def get_alerts(user_id, limit=20, unread_only=False):
    with _conn() as c:
        if unread_only:
            rows = c.execute(
                "SELECT * FROM portfolio_alerts WHERE user_id=? AND read=0 ORDER BY created_at DESC LIMIT ?",
                (user_id, limit)
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM portfolio_alerts WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit)
            ).fetchall()
    return [dict(r) for r in rows]


def mark_alerts_read(user_id, alert_ids=None):
    with _conn() as c:
        if alert_ids:
            c.execute(
                f"UPDATE portfolio_alerts SET read=1 WHERE user_id=? AND id IN ({','.join('?' * len(alert_ids))})",
                [user_id] + alert_ids
            )
        else:
            c.execute("UPDATE portfolio_alerts SET read=1 WHERE user_id=?", (user_id,))
    return {"ok": True}


# ── 8. SNAPSHOT (for equity curve tracking) ────────────

def take_snapshot(user_id):
    """Daily equity snapshot for long-term tracking."""
    try:
        from aos import user_wallet as uw
        wallet = uw.get_wallet(user_id)
        balance = wallet.get("balance", 0)
        opens = uw._open_trades(user_id)
    except Exception:
        balance = 0
        opens = []

    unrealized = 0
    for t in opens:
        if t.get("pnl_series"):
            unrealized += t["pnl_series"][-1][2]

    equity = balance + unrealized
    today = datetime.now().strftime("%Y-%m-%d")

    # Get yesterday's equity for day P&L
    with _conn() as c:
        prev = c.execute(
            "SELECT total_equity FROM portfolio_snapshots WHERE user_id=? AND date<? ORDER BY date DESC LIMIT 1",
            (user_id, today)
        ).fetchone()

    prev_equity = prev["total_equity"] if prev else equity
    day_pnl = equity - prev_equity
    day_pnl_pct = round(day_pnl / max(prev_equity, 1) * 100, 2)

    # Cumulative P&L from first snapshot
    with _conn() as c:
        first = c.execute(
            "SELECT total_equity FROM portfolio_snapshots WHERE user_id=? ORDER BY date ASC LIMIT 1",
            (user_id,)
        ).fetchone()
    initial = first["total_equity"] if first else equity
    cumulative = equity - initial

    # Max drawdown
    with _conn() as c:
        all_eq = c.execute(
            "SELECT total_equity FROM portfolio_snapshots WHERE user_id=? ORDER BY date",
            (user_id,)
        ).fetchall()
    peak = equity
    max_dd = 0
    for row in all_eq:
        peak = max(peak, row["total_equity"])
        dd = (row["total_equity"] - peak) / peak * 100
        max_dd = min(max_dd, dd)

    with _conn() as c:
        c.execute("""
            INSERT OR REPLACE INTO portfolio_snapshots
            (user_id, date, total_equity, cash, positions_value, day_pnl, day_pnl_pct,
             cumulative_pnl, n_open_trades, max_drawdown)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (user_id, today, round(equity, 2), round(balance, 2),
              round(unrealized, 2), round(day_pnl, 2), day_pnl_pct,
              round(cumulative, 2), len(opens), round(max_dd, 2)))

    return {
        "date": today,
        "equity": round(equity, 2),
        "day_pnl": round(day_pnl, 2),
        "day_pnl_pct": day_pnl_pct,
        "cumulative_pnl": round(cumulative, 2),
        "max_drawdown": round(max_dd, 2),
    }


def get_equity_curve(user_id, days=90):
    with _conn() as c:
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        rows = c.execute(
            "SELECT date, total_equity, day_pnl, day_pnl_pct, cumulative_pnl, max_drawdown, n_open_trades "
            "FROM portfolio_snapshots WHERE user_id=? AND date>=? ORDER BY date",
            (user_id, cutoff)
        ).fetchall()
    return [dict(r) for r in rows]


# ── 9. PLAN MANAGEMENT ────────────────────────────────

def _save_plan(user_id, brief):
    today = datetime.now().strftime("%Y-%m-%d")
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO portfolio_plans (user_id, date, plan) VALUES (?,?,?)",
            (user_id, today, json.dumps(brief, default=str))
        )


def get_plan(user_id, date=None):
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM portfolio_plans WHERE user_id=? AND date=?",
            (user_id, date)
        ).fetchone()
    if row:
        plan = dict(row)
        plan["plan"] = json.loads(plan["plan"])
        return plan
    return None


# ── 10. MODEL STATUS ──────────────────────────────────

def model_status():
    """Status of all ML models powering the portfolio."""
    status = {"models": {}, "data": {}}

    # Index Options V2
    try:
        from engines.index_options_v2 import predict_index_options_v2
        v2 = predict_index_options_v2()
        status["models"]["index_options_v2"] = {
            "status": "active",
            "backtest_win_rate": {"NIFTY": 89.0, "BANKNIFTY": 87.8},
            "signals_today": len([t for t in v2 if t.get("tier") != "NO_TRADE"]),
            "approach": "OI-flow + PCR + multi-signal ensemble",
        }
    except Exception as e:
        status["models"]["index_options_v2"] = {"status": "error", "error": str(e)}

    # Stock direction
    meta_path = os.path.join(ROOT, "models", "intraday", "latest_direction_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        status["models"]["stock_direction"] = {
            "status": "active",
            "rank_ic": meta.get("validation", {}).get("mean_rank_ic"),
            "win_rate": meta.get("backtest", {}).get("win_rate"),
            "n_stocks": meta.get("n_symbols"),
            "edge": meta.get("edge"),
        }

    # Strike ranker
    sr_path = os.path.join(ROOT, "models", "intraday", "latest_strike_ranker_meta.json")
    if os.path.exists(sr_path):
        with open(sr_path) as f:
            meta = json.load(f)
        status["models"]["strike_ranker"] = {
            "status": "active",
            "n_features": meta.get("n_features"),
            "n_contracts": meta.get("n_contracts"),
            "n_days": meta.get("n_days"),
        }

    # Data status
    raw_dir = os.path.join(ROOT, "data", "option_chain", "raw")
    for sym in ["NIFTY", "BANKNIFTY"]:
        sym_dir = os.path.join(raw_dir, sym)
        if os.path.exists(sym_dir):
            files = [f for f in os.listdir(sym_dir) if f.endswith(".csv")]
            status["data"][sym] = {
                "days": len(files),
                "first": sorted(files)[0].replace(".csv", "") if files else None,
                "last": sorted(files)[-1].replace(".csv", "") if files else None,
            }

    return status

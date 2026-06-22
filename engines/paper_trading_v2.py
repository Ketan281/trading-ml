"""
Paper Trading V2 — institutional-grade forward-test simulation.

Full lifecycle tracking: open → partial exit → close, with realistic
slippage, brokerage, and comprehensive performance metrics.

Target: 1000+ paper trades before any live deployment.
"""

import os
import sys
import json
import sqlite3
import uuid
import math
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DB_PATH = os.path.join(ROOT, "memory", "trading_memory.db")
COST_PER_SIDE_BPS = 20
SLIPPAGE_BPS = 5
READINESS_THRESHOLD = 1000
MIN_SHARPE = 0.5
MIN_PROFIT_FACTOR = 1.5
MAX_DRAWDOWN_PCT = 15.0


def _conn():
    return sqlite3.connect(DB_PATH)


def _dict_row(cursor, row):
    return dict(zip([d[0] for d in cursor.description], row))


# ── Open Trade ─────────────────────────────────────────

def open_trade(signal, sizing, grade_result=None, enrichment=None):
    """Open a paper position with realistic costs."""
    grade_result = grade_result or {}
    enrichment = enrichment or {}

    symbol = signal.get("symbol", "")
    price = signal.get("price")
    if isinstance(price, str):
        try:
            price = float(price.split("-")[0])
        except (ValueError, IndexError):
            return {"error": "No valid price"}
    if not price:
        return {"error": "No valid price"}

    slippage = price * SLIPPAGE_BPS / 10000
    entry_price = price + slippage
    shares = sizing.get("shares", 0)
    if shares <= 0:
        return {"error": "Zero position size"}

    brokerage = entry_price * shares * COST_PER_SIDE_BPS / 10000
    trade_id = f"PT_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    stop = signal.get("stop_loss")
    if isinstance(stop, str):
        try:
            stop = float(stop)
        except ValueError:
            stop = None

    target = signal.get("target")
    if isinstance(target, str):
        try:
            target = float(target)
        except ValueError:
            target = None

    risk = abs(entry_price - stop) if stop else entry_price * 0.02
    t1 = entry_price + risk * 1.0 if signal.get("final_action") == "buy" else entry_price - risk * 1.0
    t2 = entry_price + risk * 2.0 if signal.get("final_action") == "buy" else entry_price - risk * 2.0
    t3 = target or (entry_price + risk * 3.0 if signal.get("final_action") == "buy" else entry_price - risk * 3.0)

    direction = "long" if signal.get("final_action", "buy") == "buy" else "short"

    conn = _conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO paper_trades_v2
            (trade_id, symbol, segment, direction, entry_price, entry_time,
             entry_signal, shares, lots, capital_allocated,
             initial_stop, current_stop, target_1, target_2, target_3,
             grade, conviction, trade_quality, regime_at_entry,
             psychology_at_entry, status, slippage_entry, brokerage,
             total_costs, last_price, last_update)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open',
                ?, ?, ?, ?, ?)
    """, (
        trade_id, symbol,
        signal.get("segment", "equity"), direction,
        round(entry_price, 2), datetime.now().isoformat(),
        json.dumps(signal, default=str),
        shares, sizing.get("lots", shares),
        round(sizing.get("capital_allocated", shares * entry_price), 2),
        stop, stop,
        round(t1, 2), round(t2, 2), round(t3, 2),
        grade_result.get("grade", "?"),
        grade_result.get("conviction", 0),
        enrichment.get("trade_quality_score",
                       enrichment.get("trade_quality", 0)),
        enrichment.get("regime_v2", {}).get("day_type", "unknown"),
        enrichment.get("psychology_score", 100),
        round(slippage, 2), round(brokerage, 2),
        round(slippage * shares + brokerage, 2),
        round(entry_price, 2), datetime.now().isoformat(),
    ))
    conn.commit()
    conn.close()

    return {"trade_id": trade_id, "symbol": symbol, "direction": direction,
            "entry_price": round(entry_price, 2), "shares": shares,
            "stop": stop, "targets": [round(t1, 2), round(t2, 2), round(t3, 2)]}


# ── Update Positions ───────────────────────────────────

def update_positions(current_prices=None):
    """Mark-to-market all open positions. Check stops and targets."""
    conn = _conn()
    c = conn.cursor()
    c.execute("SELECT * FROM paper_trades_v2 WHERE status IN ('open', 'partial')")
    rows = c.fetchall()
    cols = [d[0] for d in c.description]
    positions = [dict(zip(cols, r)) for r in rows]
    conn.close()

    if not positions:
        return {"positions": [], "closed_today": [], "equity": 0}

    if current_prices is None:
        try:
            from models.cross_sectional import load_prices
            symbols = list({p["symbol"] for p in positions})
            prices = load_prices(universe=set(symbols))
            current_prices = {s: float(df["Close"].iloc[-1])
                              for s, df in prices.items()}
        except Exception:
            current_prices = {}

    closed_today = []
    for pos in positions:
        symbol = pos["symbol"]
        current = current_prices.get(symbol)
        if current is None:
            continue

        entry = pos["entry_price"]
        stop = pos["current_stop"]
        direction = pos["direction"]
        is_long = direction == "long"

        mfe = pos.get("max_favorable_excursion") or 0
        mae = pos.get("max_adverse_excursion") or 0
        if is_long:
            mfe = max(mfe, current - entry)
            mae = min(mae, current - entry)
        else:
            mfe = max(mfe, entry - current)
            mae = min(mae, entry - current)

        hit_stop = (is_long and stop and current <= stop) or \
                   (not is_long and stop and current >= stop)
        hit_t3 = False
        t3 = pos.get("target_3")
        if t3:
            hit_t3 = (is_long and current >= t3) or (not is_long and current <= t3)

        conn = _conn()
        c = conn.cursor()

        if hit_stop or hit_t3:
            exit_reason = "stop_loss" if hit_stop else "target_3"
            exit_price = stop if hit_stop else t3
            slippage_exit = exit_price * SLIPPAGE_BPS / 10000
            if is_long:
                exit_price -= slippage_exit
            else:
                exit_price += slippage_exit
            brokerage_exit = exit_price * pos["shares"] * COST_PER_SIDE_BPS / 10000
            total_costs = (pos.get("total_costs", 0) or 0) + slippage_exit * pos["shares"] + brokerage_exit

            if is_long:
                gross_pnl = (exit_price - entry) * pos["shares"]
            else:
                gross_pnl = (entry - exit_price) * pos["shares"]
            net_pnl = gross_pnl - total_costs
            pnl_pct = net_pnl / (entry * pos["shares"]) * 100 if entry > 0 else 0

            risk = abs(entry - (pos.get("initial_stop") or entry * 0.98))
            r_multiple = net_pnl / (risk * pos["shares"]) if risk > 0 and pos["shares"] > 0 else 0

            c.execute("""
                UPDATE paper_trades_v2 SET
                    status = 'closed', exit_price = ?, exit_time = ?,
                    exit_reason = ?, slippage_exit = ?, brokerage = ?,
                    total_costs = ?, gross_pnl = ?, net_pnl = ?,
                    pnl_pct = ?, r_multiple = ?,
                    max_favorable_excursion = ?, max_adverse_excursion = ?,
                    last_price = ?, last_update = ?
                WHERE trade_id = ?
            """, (
                round(exit_price, 2), datetime.now().isoformat(),
                exit_reason, round(slippage_exit, 2),
                round((pos.get("brokerage", 0) or 0) + brokerage_exit, 2),
                round(total_costs, 2), round(gross_pnl, 2),
                round(net_pnl, 2), round(pnl_pct, 2), round(r_multiple, 3),
                round(mfe, 2), round(mae, 2),
                round(current, 2), datetime.now().isoformat(),
                pos["trade_id"],
            ))
            closed_today.append({
                "trade_id": pos["trade_id"], "symbol": symbol,
                "pnl": round(net_pnl, 2), "r_multiple": round(r_multiple, 3),
                "exit_reason": exit_reason,
            })
        else:
            c.execute("""
                UPDATE paper_trades_v2 SET
                    last_price = ?, max_favorable_excursion = ?,
                    max_adverse_excursion = ?, last_update = ?
                WHERE trade_id = ?
            """, (round(current, 2), round(mfe, 2), round(mae, 2),
                  datetime.now().isoformat(), pos["trade_id"]))

        conn.commit()
        conn.close()

    return {
        "positions": [p for p in positions
                      if p["trade_id"] not in {c["trade_id"] for c in closed_today}],
        "closed_today": closed_today,
    }


# ── Close Trade ────────────────────────────────────────

def close_trade(trade_id, price=None, reason="manual"):
    """Manually close a specific paper position."""
    conn = _conn()
    c = conn.cursor()
    c.execute("SELECT * FROM paper_trades_v2 WHERE trade_id = ?", (trade_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return {"error": "Trade not found"}
    cols = [d[0] for d in c.description]
    pos = dict(zip(cols, row))
    conn.close()

    if pos["status"] == "closed":
        return {"error": "Trade already closed"}

    if price is None:
        try:
            from models.cross_sectional import load_prices
            prices = load_prices(universe={pos["symbol"]})
            price = float(prices[pos["symbol"]]["Close"].iloc[-1])
        except Exception:
            return {"error": "Cannot determine exit price"}

    entry = pos["entry_price"]
    is_long = pos["direction"] == "long"
    slippage = price * SLIPPAGE_BPS / 10000
    exit_price = price - slippage if is_long else price + slippage
    brokerage_exit = exit_price * pos["shares"] * COST_PER_SIDE_BPS / 10000
    total_costs = (pos.get("total_costs", 0) or 0) + slippage * pos["shares"] + brokerage_exit

    gross_pnl = (exit_price - entry) * pos["shares"] if is_long else (entry - exit_price) * pos["shares"]
    net_pnl = gross_pnl - total_costs
    pnl_pct = net_pnl / (entry * pos["shares"]) * 100 if entry > 0 else 0
    risk = abs(entry - (pos.get("initial_stop") or entry * 0.98))
    r_multiple = net_pnl / (risk * pos["shares"]) if risk > 0 and pos["shares"] > 0 else 0

    conn = _conn()
    c = conn.cursor()
    c.execute("""
        UPDATE paper_trades_v2 SET
            status = 'closed', exit_price = ?, exit_time = ?,
            exit_reason = ?, slippage_exit = ?,
            brokerage = ?, total_costs = ?,
            gross_pnl = ?, net_pnl = ?, pnl_pct = ?,
            r_multiple = ?, last_price = ?, last_update = ?
        WHERE trade_id = ?
    """, (
        round(exit_price, 2), datetime.now().isoformat(), reason,
        round(slippage, 2),
        round((pos.get("brokerage", 0) or 0) + brokerage_exit, 2),
        round(total_costs, 2), round(gross_pnl, 2), round(net_pnl, 2),
        round(pnl_pct, 2), round(r_multiple, 3),
        round(exit_price, 2), datetime.now().isoformat(), trade_id,
    ))
    conn.commit()
    conn.close()

    return {"trade_id": trade_id, "net_pnl": round(net_pnl, 2),
            "r_multiple": round(r_multiple, 3), "exit_reason": reason}


# ── Metrics ────────────────────────────────────────────

def compute_metrics(lookback_days=None):
    """Compute comprehensive performance metrics from closed trades."""
    conn = _conn()
    c = conn.cursor()
    if lookback_days:
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        c.execute("""
            SELECT * FROM paper_trades_v2
            WHERE status = 'closed' AND exit_time >= ?
            ORDER BY exit_time
        """, (cutoff,))
    else:
        c.execute("SELECT * FROM paper_trades_v2 WHERE status = 'closed' ORDER BY exit_time")
    rows = c.fetchall()
    cols = [d[0] for d in c.description]
    trades = [dict(zip(cols, r)) for r in rows]
    conn.close()

    if not trades:
        return {"total_trades": 0, "message": "No closed trades yet"}

    pnls = [t.get("net_pnl", 0) or 0 for t in trades]
    r_mults = [t.get("r_multiple", 0) or 0 for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total = len(trades)
    win_rate = len(wins) / total if total > 0 else 0
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0
    profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
    expectancy = sum(pnls) / total if total > 0 else 0
    avg_r = sum(r_mults) / len(r_mults) if r_mults else 0
    avg_win_r = sum(r for r in r_mults if r > 0) / max(1, len([r for r in r_mults if r > 0]))
    avg_loss_r = sum(r for r in r_mults if r <= 0) / max(1, len([r for r in r_mults if r <= 0]))

    cumulative = []
    running = 0
    peak = 0
    max_dd = 0
    for p in pnls:
        running += p
        cumulative.append(running)
        if running > peak:
            peak = running
        dd = (running - peak) / max(peak, 1) * 100 if peak > 0 else 0
        if dd < max_dd:
            max_dd = dd

    daily_returns = [p / 1_000_000 for p in pnls]
    avg_ret = sum(daily_returns) / len(daily_returns) if daily_returns else 0
    std_ret = (sum((r - avg_ret) ** 2 for r in daily_returns) / max(1, len(daily_returns) - 1)) ** 0.5
    sharpe = (avg_ret / std_ret * math.sqrt(252)) if std_ret > 0 else 0

    neg_returns = [r for r in daily_returns if r < 0]
    neg_std = (sum((r - avg_ret) ** 2 for r in neg_returns) / max(1, len(neg_returns) - 1)) ** 0.5 if neg_returns else 0
    sortino = (avg_ret / neg_std * math.sqrt(252)) if neg_std > 0 else 0

    total_return = sum(pnls) / 1_000_000
    days = total
    cagr = ((1 + total_return) ** (252 / max(days, 1)) - 1) * 100 if total_return > -1 else 0

    hold_days = []
    for t in trades:
        try:
            entry_t = datetime.fromisoformat(t["entry_time"])
            exit_t = datetime.fromisoformat(t["exit_time"])
            hold_days.append((exit_t - entry_t).days)
        except (ValueError, TypeError):
            pass

    metrics = {
        "total_trades": total,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(min(profit_factor, 999), 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown": round(abs(max_dd), 2),
        "cagr": round(cagr, 2),
        "expectancy": round(expectancy, 2),
        "avg_r_multiple": round(avg_r, 3),
        "avg_win_r": round(avg_win_r, 3),
        "avg_loss_r": round(avg_loss_r, 3),
        "best_trade_r": round(max(r_mults), 3) if r_mults else 0,
        "worst_trade_r": round(min(r_mults), 3) if r_mults else 0,
        "avg_hold_days": round(sum(hold_days) / len(hold_days), 1) if hold_days else 0,
        "total_pnl": round(sum(pnls), 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
    }

    conn = _conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO paper_metrics
            (computed_at, total_trades, win_rate, profit_factor, sharpe,
             sortino, max_drawdown, cagr, expectancy, avg_r_multiple,
             avg_win_r, avg_loss_r, best_trade_r, worst_trade_r,
             avg_hold_days, metrics_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        metrics["total_trades"], metrics["win_rate"],
        metrics["profit_factor"], metrics["sharpe"], metrics["sortino"],
        metrics["max_drawdown"], metrics["cagr"], metrics["expectancy"],
        metrics["avg_r_multiple"], metrics["avg_win_r"], metrics["avg_loss_r"],
        metrics["best_trade_r"], metrics["worst_trade_r"],
        metrics["avg_hold_days"], json.dumps(metrics),
    ))
    conn.commit()
    conn.close()

    return metrics


def equity_curve():
    """Return equity curve data for charting."""
    conn = _conn()
    c = conn.cursor()
    c.execute("SELECT * FROM paper_equity_curve ORDER BY date")
    rows = c.fetchall()
    cols = [d[0] for d in c.description]
    conn.close()
    return [dict(zip(cols, r)) for r in rows]


def get_positions(status="open"):
    """Get current paper positions."""
    conn = _conn()
    c = conn.cursor()
    if status == "all":
        c.execute("SELECT * FROM paper_trades_v2 ORDER BY entry_time DESC")
    else:
        c.execute("SELECT * FROM paper_trades_v2 WHERE status = ? ORDER BY entry_time DESC",
                  (status,))
    rows = c.fetchall()
    cols = [d[0] for d in c.description]
    conn.close()
    return [dict(zip(cols, r)) for r in rows]


def trade_count():
    """How many paper trades have been completed."""
    conn = _conn()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM paper_trades_v2 WHERE status = 'closed'")
    count = c.fetchone()[0]
    conn.close()
    return count


def readiness_score():
    """Is the system ready for live trading?"""
    count = trade_count()
    metrics = compute_metrics() if count > 0 else {}

    checks = {
        "trades_completed": count >= READINESS_THRESHOLD,
        "min_sharpe_met": metrics.get("sharpe", 0) >= MIN_SHARPE,
        "min_profit_factor_met": metrics.get("profit_factor", 0) >= MIN_PROFIT_FACTOR,
        "max_drawdown_acceptable": metrics.get("max_drawdown", 100) <= MAX_DRAWDOWN_PCT,
    }

    reasons = []
    if not checks["trades_completed"]:
        reasons.append(f"Need {READINESS_THRESHOLD - count} more trades "
                       f"({count}/{READINESS_THRESHOLD})")
    if not checks["min_sharpe_met"]:
        reasons.append(f"Sharpe {metrics.get('sharpe', 0):.2f} < {MIN_SHARPE}")
    if not checks["min_profit_factor_met"]:
        reasons.append(f"Profit factor {metrics.get('profit_factor', 0):.2f} < {MIN_PROFIT_FACTOR}")
    if not checks["max_drawdown_acceptable"]:
        reasons.append(f"Max DD {metrics.get('max_drawdown', 0):.1f}% > {MAX_DRAWDOWN_PCT}%")

    return {
        "ready": all(checks.values()),
        "trades_completed": count,
        "trades_needed": READINESS_THRESHOLD,
        "progress_pct": round(min(count / READINESS_THRESHOLD * 100, 100), 1),
        "checks": checks,
        "reasons": reasons,
        "metrics": metrics,
    }


# ── CLI ────────────────────────────────────────────────

if __name__ == "__main__":
    from memory.phase2_schema import migrate
    migrate()

    print("=" * 60)
    print("  PAPER TRADING V2")
    print("=" * 60)

    readiness = readiness_score()
    print(f"\n  Readiness: {'READY' if readiness['ready'] else 'NOT READY'}")
    print(f"  Progress: {readiness['progress_pct']:.0f}% "
          f"({readiness['trades_completed']}/{readiness['trades_needed']})")
    for r in readiness["reasons"]:
        print(f"    - {r}")

    positions = get_positions()
    print(f"\n  Open positions: {len(positions)}")
    for p in positions[:5]:
        print(f"    {p['symbol']} {p['direction']} @ {p['entry_price']} "
              f"(last: {p.get('last_price', '?')})")

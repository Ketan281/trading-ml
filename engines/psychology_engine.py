"""
Psychology Engine — THE GATE.

Every trade must pass through can_trade(). This module models the mental
discipline of an experienced trader who knows when NOT to trade.

States: normal → caution → restricted → halt

The gate returns a boolean. No interpretation, no overrides.
"""

import os
import sys
import json
import sqlite3
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DB_PATH = os.path.join(ROOT, "memory", "trading_memory.db")

# ── Configurable Thresholds ────────────────────────────
DAILY_LOSS_LIMIT_PCT = 0.02
WEEKLY_LOSS_LIMIT_PCT = 0.04
MONTHLY_LOSS_LIMIT_PCT = 0.08
MAX_DAILY_TRADES = 5
MAX_CONSECUTIVE_LOSSES = 3
COOLDOWN_MINUTES = 30
MIN_CONFIDENCE_THRESHOLD = 0.60
MIN_CONVICTION_THRESHOLD = 50
REVENGE_TRADE_WINDOW_SEC = 300


# ── Database Operations ───────────────────────────────

def _conn():
    return sqlite3.connect(DB_PATH)


def load_state(date=None):
    """Load or create today's psychology state."""
    date = date or datetime.now().strftime("%Y-%m-%d")
    conn = _conn()
    c = conn.cursor()
    c.execute("SELECT * FROM psychology_state WHERE date = ?", (date,))
    row = c.fetchone()
    if row:
        cols = [d[0] for d in c.description]
        state = dict(zip(cols, row))
        conn.close()
        return state

    now = datetime.now().isoformat()
    c.execute("""
        INSERT INTO psychology_state
            (date, daily_loss, daily_trades, weekly_loss, monthly_loss,
             consecutive_losses, risk_state, psychology_score,
             discipline_score, updated_at)
        VALUES (?, 0, 0, 0, 0, 0, 'normal', 100, 100, ?)
    """, (date, now))
    conn.commit()

    c.execute("SELECT * FROM psychology_state WHERE date = ?", (date,))
    row = c.fetchone()
    cols = [d[0] for d in c.description]
    state = dict(zip(cols, row))
    conn.close()

    _carry_forward(state, date)
    return state


def _carry_forward(state, today_str):
    """Carry forward weekly/monthly losses and consecutive losses from
    prior days so limits accumulate correctly across sessions."""
    conn = _conn()
    c = conn.cursor()
    today = datetime.strptime(today_str, "%Y-%m-%d")

    week_start = (today - timedelta(days=today.weekday())).strftime("%Y-%m-%d")
    c.execute("""
        SELECT COALESCE(SUM(daily_loss), 0) FROM psychology_state
        WHERE date >= ? AND date < ?
    """, (week_start, today_str))
    weekly = c.fetchone()[0]

    month_start = today.replace(day=1).strftime("%Y-%m-%d")
    c.execute("""
        SELECT COALESCE(SUM(daily_loss), 0) FROM psychology_state
        WHERE date >= ? AND date < ?
    """, (month_start, today_str))
    monthly = c.fetchone()[0]

    c.execute("""
        SELECT consecutive_losses FROM psychology_state
        WHERE date < ? ORDER BY date DESC LIMIT 1
    """, (today_str,))
    row = c.fetchone()
    consec = row[0] if row else 0

    c.execute("""
        UPDATE psychology_state
        SET weekly_loss = ?, monthly_loss = ?, consecutive_losses = ?,
            updated_at = ?
        WHERE date = ?
    """, (weekly, monthly, consec, datetime.now().isoformat(), today_str))
    conn.commit()
    conn.close()

    state["weekly_loss"] = weekly
    state["monthly_loss"] = monthly
    state["consecutive_losses"] = consec


def save_state(state):
    """Persist state to DB (update by date)."""
    conn = _conn()
    c = conn.cursor()
    now = datetime.now().isoformat()
    risk_state = get_risk_state(state)
    psych = compute_psychology_score(state)
    disc = compute_discipline_score(state)

    c.execute("""
        UPDATE psychology_state SET
            daily_loss = ?, daily_trades = ?, weekly_loss = ?,
            monthly_loss = ?, consecutive_losses = ?,
            last_trade_ts = ?, cooldown_until = ?,
            risk_state = ?, psychology_score = ?,
            discipline_score = ?, notes = ?, updated_at = ?
        WHERE date = ?
    """, (
        state.get("daily_loss", 0),
        state.get("daily_trades", 0),
        state.get("weekly_loss", 0),
        state.get("monthly_loss", 0),
        state.get("consecutive_losses", 0),
        state.get("last_trade_ts"),
        state.get("cooldown_until"),
        risk_state,
        psych,
        disc,
        state.get("notes"),
        now,
        state["date"],
    ))
    conn.commit()
    conn.close()

    state["risk_state"] = risk_state
    state["psychology_score"] = psych
    state["discipline_score"] = disc


def _log_event(event_type, details, state_before, state_after):
    """Write to psychology_events table."""
    conn = _conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO psychology_events
            (timestamp, event_type, details, risk_state_before, risk_state_after)
        VALUES (?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        event_type,
        json.dumps(details) if isinstance(details, dict) else str(details),
        state_before,
        state_after,
    ))
    conn.commit()
    conn.close()


# ── Core Gate ──────────────────────────────────────────

def can_trade(capital, signal, state=None):
    """THE GATE. Returns {allowed, reason, risk_state, psychology_score,
    discipline_score, restrictions}.

    Checks in order — first failure stops:
      1. Trading halt
      2. Monthly loss limit
      3. Weekly loss limit
      4. Daily loss limit
      5. Cooldown active
      6. Max daily trades
      7. Consecutive loss protection
      8. Minimum confidence
      9. Minimum conviction
     10. Revenge trading detection
    """
    if state is None:
        state = load_state()

    restrictions = []
    risk_state = get_risk_state(state)
    psych = compute_psychology_score(state)
    disc = compute_discipline_score(state)

    base = {
        "risk_state": risk_state,
        "psychology_score": round(psych, 1),
        "discipline_score": round(disc, 1),
    }

    # 1. Halt
    if risk_state == "halt":
        return {**base, "allowed": False,
                "reason": "Trading HALTED — circuit breaker active",
                "restrictions": ["halt"]}

    # 2. Monthly loss limit
    monthly_limit = capital * MONTHLY_LOSS_LIMIT_PCT
    monthly_used = abs(state.get("monthly_loss", 0))
    if monthly_used >= monthly_limit:
        _log_event("monthly_limit_hit",
                   {"used": monthly_used, "limit": monthly_limit},
                   risk_state, "halt")
        return {**base, "allowed": False,
                "reason": f"Monthly loss limit hit: ₹{monthly_used:,.0f} / ₹{monthly_limit:,.0f}",
                "restrictions": ["monthly_limit"]}

    # 3. Weekly loss limit
    weekly_limit = capital * WEEKLY_LOSS_LIMIT_PCT
    weekly_used = abs(state.get("weekly_loss", 0))
    if weekly_used >= weekly_limit:
        _log_event("weekly_limit_hit",
                   {"used": weekly_used, "limit": weekly_limit},
                   risk_state, "halt")
        return {**base, "allowed": False,
                "reason": f"Weekly loss limit hit: ₹{weekly_used:,.0f} / ₹{weekly_limit:,.0f}",
                "restrictions": ["weekly_limit"]}

    # 4. Daily loss limit
    daily_limit = capital * DAILY_LOSS_LIMIT_PCT
    daily_used = abs(state.get("daily_loss", 0))
    if daily_used >= daily_limit:
        _log_event("daily_limit_hit",
                   {"used": daily_used, "limit": daily_limit},
                   risk_state, "halt")
        return {**base, "allowed": False,
                "reason": f"Daily loss limit hit: ₹{daily_used:,.0f} / ₹{daily_limit:,.0f}",
                "restrictions": ["daily_limit"]}

    # 5. Cooldown
    cooldown = state.get("cooldown_until")
    if cooldown:
        try:
            cd_time = datetime.fromisoformat(cooldown)
            if datetime.now() < cd_time:
                remaining = (cd_time - datetime.now()).seconds // 60
                restrictions.append("cooldown")
                return {**base, "allowed": False,
                        "reason": f"Cooldown active — {remaining} min remaining",
                        "restrictions": restrictions}
        except (ValueError, TypeError):
            pass

    # 6. Max daily trades
    daily_trades = state.get("daily_trades", 0)
    if daily_trades >= MAX_DAILY_TRADES:
        restrictions.append("max_trades")
        return {**base, "allowed": False,
                "reason": f"Max daily trades reached: {daily_trades}/{MAX_DAILY_TRADES}",
                "restrictions": restrictions}

    # 7. Consecutive losses
    consec = state.get("consecutive_losses", 0)
    if consec >= MAX_CONSECUTIVE_LOSSES:
        restrictions.append("consecutive_losses")
        return {**base, "allowed": False,
                "reason": f"Consecutive losses: {consec} — take a break",
                "restrictions": restrictions}

    # 8. Min confidence
    confidence = signal.get("fused_confidence", signal.get("confidence", 0))
    if confidence < MIN_CONFIDENCE_THRESHOLD:
        restrictions.append("low_confidence")
        return {**base, "allowed": False,
                "reason": f"Confidence {confidence:.1%} below minimum {MIN_CONFIDENCE_THRESHOLD:.0%}",
                "restrictions": restrictions}

    # 9. Min conviction
    conviction = signal.get("conviction", signal.get("conviction_score", 100))
    if conviction < MIN_CONVICTION_THRESHOLD:
        restrictions.append("low_conviction")
        return {**base, "allowed": False,
                "reason": f"Conviction {conviction:.0f} below minimum {MIN_CONVICTION_THRESHOLD}",
                "restrictions": restrictions}

    # 10. Revenge trading detection
    last_ts = state.get("last_trade_ts")
    last_was_loss = consec > 0
    if last_ts and last_was_loss:
        try:
            last_time = datetime.fromisoformat(last_ts)
            elapsed = (datetime.now() - last_time).total_seconds()
            if elapsed < REVENGE_TRADE_WINDOW_SEC:
                _log_event("revenge_detected",
                           {"elapsed_sec": elapsed, "window": REVENGE_TRADE_WINDOW_SEC},
                           risk_state, risk_state)
                restrictions.append("revenge_trade")
                return {**base, "allowed": False,
                        "reason": f"Revenge trade detected — only {elapsed:.0f}s since last loss. Wait {REVENGE_TRADE_WINDOW_SEC}s.",
                        "restrictions": restrictions}
        except (ValueError, TypeError):
            pass

    # Caution-level warnings (allowed but noted)
    if risk_state == "caution":
        restrictions.append("caution_state")
    if risk_state == "restricted":
        restrictions.append("restricted_state")

    return {**base, "allowed": True,
            "reason": "All checks passed",
            "restrictions": restrictions}


# ── Trade Result Recording ─────────────────────────────

def record_trade_result(pnl, capital, trade_meta=None):
    """Update psychology state after a trade closes.

    Returns updated state dict.
    """
    state = load_state()
    old_risk = get_risk_state(state)
    trade_meta = trade_meta or {}

    state["daily_trades"] = state.get("daily_trades", 0) + 1
    state["last_trade_ts"] = datetime.now().isoformat()

    if pnl < 0:
        loss = abs(pnl)
        state["daily_loss"] = state.get("daily_loss", 0) + loss
        state["weekly_loss"] = state.get("weekly_loss", 0) + loss
        state["monthly_loss"] = state.get("monthly_loss", 0) + loss
        state["consecutive_losses"] = state.get("consecutive_losses", 0) + 1

        if state["consecutive_losses"] >= MAX_CONSECUTIVE_LOSSES:
            cooldown_end = datetime.now() + timedelta(minutes=COOLDOWN_MINUTES)
            state["cooldown_until"] = cooldown_end.isoformat()
            _log_event("cooldown_start",
                       {"consecutive_losses": state["consecutive_losses"],
                        "cooldown_minutes": COOLDOWN_MINUTES,
                        "pnl": pnl},
                       old_risk, "restricted")
    else:
        state["consecutive_losses"] = 0

    save_state(state)
    new_risk = get_risk_state(state)

    if old_risk != new_risk:
        _log_event("risk_state_change",
                   {"pnl": pnl, "from": old_risk, "to": new_risk},
                   old_risk, new_risk)

    return state


# ── Scoring ────────────────────────────────────────────

def compute_psychology_score(state):
    """0-100. Fresh/disciplined = 100. Degrades with losses and overtrading."""
    score = 100.0

    consec = state.get("consecutive_losses", 0)
    score -= consec * 15

    daily_trades = state.get("daily_trades", 0)
    trade_usage = daily_trades / MAX_DAILY_TRADES if MAX_DAILY_TRADES > 0 else 0
    if trade_usage > 0.8:
        score -= 10
    elif trade_usage > 0.5:
        score -= 5

    daily_loss = abs(state.get("daily_loss", 0))
    if daily_loss > 0:
        score -= 10
    weekly_loss = abs(state.get("weekly_loss", 0))
    if weekly_loss > 0:
        score -= 5

    if state.get("cooldown_until"):
        try:
            cd = datetime.fromisoformat(state["cooldown_until"])
            if datetime.now() < cd:
                score -= 15
        except (ValueError, TypeError):
            pass

    return max(0.0, min(100.0, score))


def compute_discipline_score(state, recent_trades=None):
    """0-100. Measures adherence to trading rules."""
    score = 100.0

    if state.get("daily_trades", 0) > MAX_DAILY_TRADES:
        score -= 25

    consec = state.get("consecutive_losses", 0)
    if consec > 0:
        score -= min(consec * 10, 30)

    conn = _conn()
    c = conn.cursor()
    today = state.get("date", datetime.now().strftime("%Y-%m-%d"))
    c.execute("""
        SELECT COUNT(*) FROM psychology_events
        WHERE event_type = 'revenge_detected' AND timestamp LIKE ?
    """, (today + "%",))
    revenge_count = c.fetchone()[0]
    conn.close()

    score -= revenge_count * 20

    return max(0.0, min(100.0, score))


def get_risk_state(state):
    """Derive risk state from current psychology state.

    Returns: 'normal' | 'caution' | 'restricted' | 'halt'
    """
    consec = state.get("consecutive_losses", 0)
    daily_loss = abs(state.get("daily_loss", 0))
    daily_trades = state.get("daily_trades", 0)

    if consec >= MAX_CONSECUTIVE_LOSSES:
        return "halt"
    if daily_trades >= MAX_DAILY_TRADES:
        return "halt"

    if consec == 2:
        return "restricted"

    if consec == 1:
        return "caution"
    if daily_loss > 0:
        return "caution"

    return "normal"


def dynamic_risk_reduction(base_risk_pct, state=None):
    """Scale per-trade risk based on psychology state.

    normal=1.0x, caution=0.75x, restricted=0.5x, halt=0x
    """
    if state is None:
        state = load_state()
    risk_state = get_risk_state(state)
    multipliers = {
        "normal": 1.0,
        "caution": 0.75,
        "restricted": 0.5,
        "halt": 0.0,
    }
    mult = multipliers.get(risk_state, 1.0)
    return round(base_risk_pct * mult, 6)


def get_events(limit=20, event_type=None):
    """Fetch recent psychology events."""
    conn = _conn()
    c = conn.cursor()
    if event_type:
        c.execute("""
            SELECT * FROM psychology_events
            WHERE event_type = ?
            ORDER BY timestamp DESC LIMIT ?
        """, (event_type, limit))
    else:
        c.execute("""
            SELECT * FROM psychology_events
            ORDER BY timestamp DESC LIMIT ?
        """, (limit,))
    rows = c.fetchall()
    cols = [d[0] for d in c.description]
    conn.close()
    return [dict(zip(cols, r)) for r in rows]


# ── CLI ────────────────────────────────────────────────

if __name__ == "__main__":
    from memory.phase2_schema import migrate
    migrate()

    print("=" * 60)
    print("  PSYCHOLOGY ENGINE — THE GATE")
    print("=" * 60)

    capital = 1_000_000
    state = load_state()
    print(f"\n  Date           : {state['date']}")
    print(f"  Risk State     : {get_risk_state(state)}")
    print(f"  Psychology     : {compute_psychology_score(state):.0f}")
    print(f"  Discipline     : {compute_discipline_score(state):.0f}")
    print(f"  Daily Trades   : {state['daily_trades']}/{MAX_DAILY_TRADES}")
    print(f"  Consec Losses  : {state['consecutive_losses']}/{MAX_CONSECUTIVE_LOSSES}")

    good_signal = {"fused_confidence": 0.75, "conviction": 70}
    result = can_trade(capital, good_signal, state)
    print(f"\n  Test 1 (good signal): allowed={result['allowed']} — {result['reason']}")

    weak_signal = {"fused_confidence": 0.45, "conviction": 30}
    result = can_trade(capital, weak_signal, state)
    print(f"  Test 2 (weak signal): allowed={result['allowed']} — {result['reason']}")

    print(f"\n  Simulating 3 consecutive losses...")
    for i in range(3):
        record_trade_result(-5000, capital, {"test": True})
    state = load_state()
    result = can_trade(capital, good_signal, state)
    print(f"  After 3 losses: allowed={result['allowed']} — {result['reason']}")
    print(f"  Risk State: {get_risk_state(state)}")
    print(f"  Psychology: {compute_psychology_score(state):.0f}")

    risk = dynamic_risk_reduction(0.01, state)
    print(f"\n  Risk reduction: 1.0% base → {risk*100:.2f}% effective")

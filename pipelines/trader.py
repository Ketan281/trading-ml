"""
The "20-year trader" playbook (goal #5).

The ranker, fundamentals and pattern layers each produce a signal. A novice
treats every signal as a trade. A seasoned trader does three more things,
and that judgment is what this module encodes:

  1. CONFLUENCE  — grade how many independent edges line up (rank + quality
     + price-action + trend + momentum + reward:risk + liquidity). One green
     light is noise; five aligned is an A+ setup.
  2. DISCIPLINE  — a checklist of red flags that make a pro PASS or trim:
     chasing an overbought move, fighting the trend, thin liquidity, weak
     business, poor reward:risk, stretched far from value. Some are vetoes.
  3. MONEY MANAGEMENT — size by RISK, not by gut. Risk a fixed % of capital
     per trade; the stop distance (not a round lot) sets the share count.

The output is a written trade plan a human can act on or audit: grade,
conviction, position size in shares + rupees, entry / stop / scale-out
targets, reward:risk, the invalidation level, and a plain-English rationale.
This module makes NO new prediction — it synthesises what the other layers
already decided into how an experienced discretionary trader would act.
"""

import math

# ── Account / risk defaults (a real desk fixes these in advance) ──
CAPITAL          = 1_000_000   # ₹10 lakh notional book
RISK_PCT         = 0.01        # risk 1% of capital per trade
MAX_POSITION_PCT = 0.20        # never more than 20% of book in one name
MIN_RR           = 1.5         # below this reward:risk, a pro passes
GOOD_TURNOVER_CR = 5.0         # comfortable liquidity (₹ crore/day)


# ── helpers ───────────────────────────────────────────
def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _reward_risk(action, price, stop, target):
    if None in (price, stop, target) or price <= 0:
        return None
    if action == "buy":
        risk, reward = price - stop, target - price
    else:
        risk, reward = stop - price, price - target
    if risk <= 0:
        return None
    return reward / risk


# ── confluence scoring (the judgment) ─────────────────
def _conviction(c, rr):
    """0–100 conviction from how many independent edges align.
    Each component contributes its weight × a 0..1 quality."""
    rsi   = _num(c.get("rsi")) or 50.0
    qual  = _num(c.get("quality"))
    trend = c.get("trend", "sideways")
    trig  = c.get("entry_signal") == "trigger"
    ctx   = c.get("pattern_context", "mid")
    turn  = _num(c.get("turnover_cr")) or 0.0
    score = _num(c.get("rank_score")) or 0.5     # ranker prob ~0.5..0.6

    parts = {}
    # Relative-strength rank (model prob is compressed; stretch it).
    parts["rank"]    = (min(max((score - 0.50) / 0.12, 0), 1), 0.22)
    # Fundamental quality (0..100 → 0..1); neutral 0.5 if unknown.
    parts["quality"] = ((qual / 100.0) if qual is not None else 0.5, 0.18)
    # Fresh, confirmed entry trigger.
    parts["trigger"] = (1.0 if trig else 0.0, 0.18)
    # Trend alignment with the trade direction.
    aligned = ((c.get("action") == "buy"  and trend == "uptrend") or
               (c.get("action") == "sell" and trend == "downtrend"))
    against = ((c.get("action") == "buy"  and trend == "downtrend") or
               (c.get("action") == "sell" and trend == "uptrend"))
    parts["trend"]   = (1.0 if aligned else (0.0 if against else 0.5), 0.15)
    # Momentum health: for longs, 45–65 is the sweet spot; >70 overbought.
    if c.get("action") == "buy":
        rsi_q = 1.0 if 45 <= rsi <= 65 else (0.4 if rsi < 45 else 0.2)
    elif c.get("action") == "sell":
        rsi_q = 1.0 if 35 <= rsi <= 55 else (0.4 if rsi > 55 else 0.2)
    else:
        rsi_q = 0.5
    parts["momentum"] = (rsi_q, 0.10)
    # Reward:risk.
    rr_q = 0.0 if rr is None else min(max((rr - 1.0) / 2.0, 0), 1)
    parts["rr"]       = (rr_q, 0.10)
    # Liquidity comfort.
    parts["liquidity"] = (min(turn / GOOD_TURNOVER_CR, 1.0), 0.07)

    total = sum(q * w for q, w in parts.values())
    return round(total * 100, 1), {k: round(q, 2) for k, (q, w) in parts.items()}


# ── discipline: red flags a pro respects ──────────────
def _red_flags(c, rr):
    rsi  = _num(c.get("rsi")) or 50.0
    qual = _num(c.get("quality"))
    turn = _num(c.get("turnover_cr")) or 0.0
    trend = c.get("trend", "sideways")
    act   = c.get("action")
    ctx   = c.get("pattern_context", "mid")

    flags, vetoes = [], []
    if act == "buy" and trend == "downtrend":
        vetoes.append("counter-trend: buying a downtrend")
    if act == "sell" and trend == "uptrend":
        vetoes.append("counter-trend: shorting an uptrend")
    if act == "buy" and rsi >= 72:
        flags.append(f"chasing — RSI {rsi:.0f} overbought")
    if act == "buy" and ctx == "extended":
        flags.append("stretched far above value; wait for a pullback")
    if c.get("entry_signal") != "trigger":
        flags.append("no fresh entry trigger yet")
    if rr is not None and rr < MIN_RR:
        flags.append(f"poor reward:risk ({rr:.1f} < {MIN_RR})")
    if qual is not None and qual < 40:
        flags.append(f"weak business quality (Q{qual:.0f})")
    if turn < 2.0:
        flags.append(f"thin liquidity (₹{turn:.1f} cr/day)")
    return flags, vetoes


def _grade(conviction, vetoes, flags):
    if vetoes:
        return "AVOID"
    if conviction >= 75 and len(flags) == 0:
        return "A+"
    if conviction >= 62:
        return "A"
    if conviction >= 48:
        return "B"
    if conviction >= 35:
        return "C"
    return "AVOID"


# ── money management: size by risk ────────────────────
def _position(action, price, stop, capital, risk_pct):
    if None in (price, stop) or price <= 0:
        return {"shares": 0, "rupees": 0, "pct_capital": 0.0, "risk_rupees": 0.0}
    stop_dist = abs(price - stop)
    if stop_dist <= 0:
        return {"shares": 0, "rupees": 0, "pct_capital": 0.0, "risk_rupees": 0.0}
    risk_budget = capital * risk_pct
    shares      = math.floor(risk_budget / stop_dist)
    # Cap exposure so one name can't dominate the book.
    max_shares  = math.floor((capital * MAX_POSITION_PCT) / price)
    capped      = shares > max_shares
    shares      = min(shares, max_shares)
    rupees      = shares * price
    return {
        "shares":       int(shares),
        "rupees":       round(rupees, 0),
        "pct_capital":  round(100 * rupees / capital, 1),
        "risk_rupees":  round(shares * stop_dist, 0),
        "capped":       capped,
    }


# ── scale-out targets (pros bank partials) ────────────
def _targets(action, price, stop, final_target):
    """Two-stage exit: T1 at 1R (take half, move stop to breakeven),
    T2 at the engine's 2R target (trail the rest)."""
    if None in (price, stop):
        return {}
    r = abs(price - stop)
    if action == "buy":
        return {"t1": round(price + r, 2), "t2": round(final_target or price + 2 * r, 2),
                "breakeven_after": "T1"}
    return {"t1": round(price - r, 2), "t2": round(final_target or price - 2 * r, 2),
            "breakeven_after": "T1"}


# ── main: build the playbook for one candidate ────────
def build_playbook(c, capital=CAPITAL, risk_pct=RISK_PCT):
    """`c` is a screener candidate dict. Returns the trade plan a seasoned
    trader would write: grade, conviction, sizing, scale-out, invalidation,
    red flags and a plain-English rationale."""
    action = c.get("action", "hold")
    price  = _num(c.get("price"))
    stop   = _num(c.get("stop_loss"))
    target = _num(c.get("target"))

    rr               = _reward_risk(action, price, stop, target)
    conviction, comp = _conviction(c, rr)
    flags, vetoes    = _red_flags(c, rr)
    grade            = _grade(conviction, vetoes, flags)
    pos              = _position(action, price, stop, capital, risk_pct)
    tgts             = _targets(action, price, stop, target)
    rationale        = _rationale(c, grade, conviction, rr, comp, flags, vetoes)

    return {
        "symbol":      c.get("symbol"),
        "grade":       grade,
        "conviction":  conviction,
        "components":  comp,
        "action":      action,
        "reward_risk": round(rr, 2) if rr is not None else None,
        "entry_zone":  c.get("entry_zone"),
        "stop_loss":   stop,
        "targets":     tgts,
        "position":    pos,
        "red_flags":   flags,
        "vetoes":      vetoes,
        "invalidation": (f"close beyond {stop} (initial stop) "
                         f"or failure of the {c.get('pattern','setup')} pattern"),
        "rationale":   rationale,
    }


def _rationale(c, grade, conviction, rr, comp, flags, vetoes):
    sym   = c.get("symbol", "this name")
    act   = c.get("action", "hold")
    qual  = _num(c.get("quality"))
    trend = c.get("trend", "sideways")
    pat   = c.get("pattern", "-")
    rsi   = _num(c.get("rsi")) or 50.0

    bits = []
    if vetoes:
        bits.append(f"PASS — {vetoes[0]}. A 20-year trader doesn't fight tape.")
        return " ".join(bits)

    lead = {"A+": "Textbook setup", "A": "Strong setup",
            "B": "Tradeable but B-grade", "C": "Marginal — starter size only",
            "AVOID": "Skip"}.get(grade, "Setup")
    bits.append(f"{lead} on {sym}.")
    # confluence narrative
    edges = []
    if comp.get("rank", 0) >= 0.6:   edges.append("top relative-strength")
    if qual is not None and qual >= 60: edges.append(f"quality business (Q{qual:.0f})")
    if comp.get("trigger", 0) >= 1:  edges.append(f"fresh {pat} trigger")
    if comp.get("trend", 0) >= 1:    edges.append(f"with the {trend}")
    if rr is not None and rr >= MIN_RR: edges.append(f"{rr:.1f}:1 reward:risk")
    if edges:
        bits.append("Edges aligned: " + ", ".join(edges) + ".")
    bits.append(f"Conviction {conviction}/100.")
    if flags:
        bits.append("Watch: " + "; ".join(flags) + ".")
    # the experienced-trader move
    if grade in ("A+", "A"):
        bits.append("Take it on the trigger, bank half at T1, trail the rest.")
    elif grade == "B":
        bits.append("Half size; wait for the trigger if it hasn't fired.")
    elif grade == "C":
        bits.append("Quarter size at most, or leave it on the watchlist.")
    return " ".join(bits)


# ── pretty text for a single plan ─────────────────────
def playbook_text(p):
    lines = []
    lines.append(f"  ── {p['symbol']}  [{p['grade']}]  conviction {p['conviction']}/100 ──")
    lines.append(f"     {p['rationale']}")
    pos = p["position"]
    lines.append(f"     Action   : {p['action'].upper()}   "
                 f"R:R {p['reward_risk']}")
    lines.append(f"     Entry    : {p['entry_zone']}   Stop: {p['stop_loss']}")
    if p["targets"]:
        lines.append(f"     Targets  : T1 {p['targets'].get('t1')} (bank half, "
                     f"stop→breakeven) → T2 {p['targets'].get('t2')} (trail)")
    lines.append(f"     Size     : {pos['shares']} sh ≈ ₹{pos['rupees']:,.0f} "
                 f"({pos['pct_capital']}% of book) | risk ₹{pos['risk_rupees']:,.0f}"
                 + ("  [capped]" if pos.get("capped") else ""))
    lines.append(f"     Invalid  : {p['invalidation']}")
    if p["red_flags"]:
        lines.append(f"     Flags    : {', '.join(p['red_flags'])}")
    return "\n".join(lines)


if __name__ == "__main__":
    demo = {
        "symbol": "NAM-INDIA", "price": 1100.2, "rank_score": 0.546,
        "quality": 92.0, "action": "buy", "entry_signal": "trigger",
        "pattern": "Morning Star", "pattern_context": "pullback_to_support",
        "trend": "uptrend", "rsi": 58.0, "turnover_cr": 8.0,
        "entry_zone": "1089.49-1100.2", "stop_loss": "1046.63", "target": "1207.33",
    }
    print(playbook_text(build_playbook(demo)))

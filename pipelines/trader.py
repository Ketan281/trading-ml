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


# ── market intelligence overlay (lazy, cached per call) ──
_intel_cache = {}

def _fetch_intel(symbol):
    """Fetch market + stock intelligence. Cached for the process lifetime
    of one screening run so we don't hit APIs per-candidate."""
    if symbol in _intel_cache:
        return _intel_cache[symbol]
    try:
        from pipelines.market_intel import market_context, stock_context
        mkt = market_context("NIFTY")
        stk = stock_context(symbol)
    except Exception:
        mkt = {"conviction_multiplier": 1.0, "signals": {}, "composite_score": 0}
        stk = {"conviction_multiplier": 1.0, "signals": {}}
    _intel_cache[symbol] = (mkt, stk)
    return mkt, stk


def _intel_score(signals, key, default=0.5):
    """Extract a normalised 0..1 score from an intel signals dict.
    Raw intel scores are in [-1, +1]; we map to [0, 1]."""
    sig = signals.get(key, {})
    raw = sig.get("score", 0) if isinstance(sig, dict) else 0
    return max(0, min(1, (raw + 1) / 2)) if raw else default


# ── confluence scoring (the judgment) ─────────────────
def _conviction(c, rr):
    """0–100 conviction from how many independent edges align.

    Factors (weights sum to 1.0):
      Technical (40%): rank 12%, trigger 10%, trend 8%, momentum 5%, patterns 5%
      Fundamental (15%): quality score from 20 parameters
      Market intel (25%): FII/DII 8%, PCR 6%, intermarket 5%, volume/delivery 6%
      Risk/execution (15%): R:R 7%, liquidity 4%, S/R proximity 4%
      Sentiment (5%): news/event awareness
    """
    rsi   = _num(c.get("rsi")) or 50.0
    qual  = _num(c.get("quality"))
    trend = c.get("trend", "sideways")
    trig  = c.get("entry_signal") == "trigger"
    ctx   = c.get("pattern_context", "mid")
    turn  = _num(c.get("turnover_cr")) or 0.0
    score = _num(c.get("rank_score")) or 0.5
    sym   = c.get("symbol", "NIFTY")

    mkt, stk = _fetch_intel(sym)
    mkt_signals = mkt.get("signals", {})
    stk_signals = stk.get("signals", {})

    parts = {}

    # ── Technical analysis (40%) ──
    parts["rank"]     = (min(max((score - 0.50) / 0.12, 0), 1), 0.12)
    parts["trigger"]  = (1.0 if trig else 0.0, 0.10)
    aligned = ((c.get("action") == "buy"  and trend == "uptrend") or
               (c.get("action") == "sell" and trend == "downtrend"))
    against = ((c.get("action") == "buy"  and trend == "downtrend") or
               (c.get("action") == "sell" and trend == "uptrend"))
    parts["trend"]    = (1.0 if aligned else (0.0 if against else 0.5), 0.08)
    if c.get("action") == "buy":
        rsi_q = 1.0 if 45 <= rsi <= 65 else (0.4 if rsi < 45 else 0.2)
    elif c.get("action") == "sell":
        rsi_q = 1.0 if 35 <= rsi <= 55 else (0.4 if rsi > 55 else 0.2)
    else:
        rsi_q = 0.5
    parts["momentum"] = (rsi_q, 0.05)
    pat_score = _num(c.get("pattern_score")) or 0
    pat_q = max(0, min(1, (pat_score + 1) / 2))
    parts["patterns"] = (pat_q, 0.05)

    # ── Fundamental quality (15%) — 20 parameters via models/fundamentals.py ──
    parts["quality"]  = ((qual / 100.0) if qual is not None else 0.5, 0.15)

    # ── Market intelligence (25%) — FII/DII, PCR, intermarket, volume/delivery ──
    parts["fii_dii"]      = (_intel_score(mkt_signals, "fii_dii"), 0.08)
    parts["pcr"]          = (_intel_score(mkt_signals, "pcr"), 0.06)
    parts["intermarket"]  = (_intel_score(mkt_signals, "intermarket"), 0.05)
    delivery_q = _intel_score(stk_signals, "delivery", 0.5)
    volume_q   = _intel_score(stk_signals, "volume_profile", 0.5)
    parts["volume_delivery"] = ((delivery_q * 0.5 + volume_q * 0.5), 0.06)

    # ── Risk / execution (15%) — R:R, liquidity, S/R proximity ──
    rr_q = 0.0 if rr is None else min(max((rr - 1.0) / 2.0, 0), 1)
    parts["rr"]        = (rr_q, 0.07)
    parts["liquidity"] = (min(turn / GOOD_TURNOVER_CR, 1.0), 0.04)
    sr_q = _intel_score(stk_signals, "support_resistance", 0.5)
    parts["sr_proximity"] = (sr_q, 0.04)

    # ── News / sentiment (5%) ──
    news_q = _intel_score(mkt_signals, "sentiment", 0.5)
    parts["sentiment"] = (news_q, 0.05)

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
    sym   = c.get("symbol", "NIFTY")

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

    # Intelligence-layer flags
    try:
        mkt, stk = _fetch_intel(sym)
        mkt_signals = mkt.get("signals", {})
        stk_signals = stk.get("signals", {})
        fii = mkt_signals.get("fii_dii", {})
        if isinstance(fii, dict) and fii.get("score", 0) < -0.6 and act == "buy":
            flags.append("heavy FII selling — institutional headwind")
        pcr = mkt_signals.get("pcr", {})
        if isinstance(pcr, dict) and pcr.get("score", 0) < -0.5 and act == "buy":
            flags.append(f"low PCR ({pcr.get('read', 'bearish')}) — options sentiment negative")
        delivery = stk_signals.get("delivery", {})
        if isinstance(delivery, dict) and delivery.get("score", 0) < -0.5:
            flags.append("distribution detected — smart money likely selling")
    except Exception:
        pass

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
    if comp.get("fii_dii", 0) >= 0.7: edges.append("FII flows supportive")
    if comp.get("pcr", 0) >= 0.7:     edges.append("PCR favourable")
    if comp.get("volume_delivery", 0) >= 0.7: edges.append("accumulation detected")
    if comp.get("sr_proximity", 0) >= 0.7: edges.append("near support")
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

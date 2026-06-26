"""
Multi-segment recommendation engine.

Returns per-segment best picks (intraday equity, options, swing) with
confidence scores, plus a capital allocator that distributes budget
across segments proportional to confidence.
"""

import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _safe(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception:
        traceback.print_exc()
        return None


def _get_market_context():
    """Fetch macro/institutional market context overlay."""
    from pipelines.market_intel import market_context
    return _safe(market_context, "NIFTY") or {"conviction_multiplier": 1.0, "overall": "unknown"}


def _get_stock_context(symbol):
    """Fetch per-stock institutional signal overlay."""
    from pipelines.market_intel import stock_context
    return _safe(stock_context, symbol) or {"conviction_multiplier": 1.0, "read": "unavailable"}


def _equity_intraday_picks(balance=100_000):
    """Top intraday equity picks from the screener, sorted by conviction."""
    from pipelines.screener import screen
    mkt = _get_market_context()
    rep = _safe(screen)
    if not rep:
        return []
    picks = []
    for c in (rep.get("actionable", []) + rep.get("watchlist", []))[:20]:
        entry = c.get("price")
        stop = c.get("stop_loss")
        target = c.get("target")
        conv = c.get("conviction", 0)
        if not entry or not stop:
            continue
        try:
            entry = float(str(entry).replace(",", "").split("-")[-1])
            stop = float(str(stop).replace(",", ""))
            target = float(str(target).replace(",", "")) if target else entry * 1.03
        except (ValueError, TypeError):
            continue
        if stop >= entry:
            continue
        risk = entry - stop
        rr = round((target - entry) / risk, 2) if risk > 0 else 0
        grade = c.get("grade", "—")
        stk_ctx = _get_stock_context(c["symbol"])
        adj_conv = conv * mkt.get("conviction_multiplier", 1.0) * stk_ctx.get("conviction_multiplier", 1.0)
        adj_conv = min(99, max(0, adj_conv))
        picks.append({
            "segment": "equity_intraday",
            "symbol": c["symbol"],
            "action": "BUY",
            "entry": round(entry, 2),
            "stop": round(stop, 2),
            "target": round(target, 2),
            "confidence": round(adj_conv, 1),
            "raw_conviction": round(conv, 1),
            "grade": grade,
            "reward_risk": rr,
            "trend": c.get("trend", "—"),
            "rsi": round(c.get("rsi", 0), 1),
            "pattern": c.get("pattern", "—"),
            "entry_signal": c.get("entry_signal", "wait"),
            "market_context": mkt.get("overall", "—"),
            "stock_signal": stk_ctx.get("read", "—"),
            "reason": f"{c['symbol']} — grade {grade}, conviction {adj_conv:.0f}% "
                      f"(raw {conv:.0f}% × market {mkt.get('overall', '?')} × "
                      f"{stk_ctx.get('read', '?')}), R:R {rr}:1, "
                      f"{c.get('pattern', 'no pattern')}",
        })
    picks.sort(key=lambda x: x["confidence"], reverse=True)
    return picks


def _options_picks(capital=100_000):
    """OI wall selling signals for NIFTY and BANKNIFTY.
    Walk-forward tested: 90%+ win rate at 80%+ participation."""
    from pipelines.options_action_engine import simple_signal
    picks = []
    for sym in ("NIFTY", "BANKNIFTY"):
        signals = _safe(simple_signal, sym, capital)
        if not signals:
            continue
        if isinstance(signals, dict):
            continue  # no trade or error
        for s in signals:
            picks.append({
                "segment": "options",
                "symbol": s["signal"],
                "underlying": sym,
                "action": s["signal"],
                "entry": s.get("premium", 0),
                "stop": s.get("stoploss", 0),
                "target": s.get("target", 0),
                "confidence": s.get("win_pct", 0),
                "conviction": s.get("conviction", "none"),
                "lots": s.get("lots", 0),
                "qty": s.get("qty", 0),
                "capital_deployed": s.get("funds_required", 0),
                "max_loss": s.get("max_loss", 0),
                "max_profit": s.get("max_profit", 0),
                "win_pct": s.get("win_pct", 0),
                "funds_required": s.get("funds_required", 0),
                "wall_oi": s.get("wall_oi", 0),
                "oi_building": s.get("oi_building", False),
                "dist_pct": s.get("dist_pct", 0),
                "hold": s.get("hold", "1 day"),
                "reason": f"{s['signal']} - {s.get('win_pct',0)}% win rate, "
                          f"wall {s.get('dist_pct',0)}% from spot, "
                          f"OI {'building' if s.get('oi_building') else 'unwinding'}",
            })
    picks.sort(key=lambda x: x.get("win_pct", 0), reverse=True)
    return picks


def _swing_picks(balance=100_000):
    """Swing / positional delivery picks from the screener — for multi-day holds."""
    from pipelines.screener import screen
    mkt = _get_market_context()
    rep = _safe(screen)
    if not rep:
        return []
    picks = []
    for c in rep.get("actionable", [])[:15]:
        entry = c.get("price")
        stop = c.get("stop_loss")
        target = c.get("target")
        conv = c.get("conviction", 0)
        quality = c.get("quality", 0)
        if not entry or not stop:
            continue
        try:
            entry = float(str(entry).replace(",", "").split("-")[-1])
            stop = float(str(stop).replace(",", ""))
            target = float(str(target).replace(",", "")) if target else entry * 1.06
        except (ValueError, TypeError):
            continue
        if stop >= entry:
            continue
        risk = entry - stop
        rr = round((target - entry) / risk, 2) if risk > 0 else 0
        swing_conv = conv * 0.6 + quality * 0.4 if quality else conv
        grade = c.get("grade", "—")
        if grade in ("C", "AVOID") or conv < 40:
            continue
        stk_ctx = _get_stock_context(c["symbol"])
        adj_conv = swing_conv * mkt.get("conviction_multiplier", 1.0) * stk_ctx.get("conviction_multiplier", 1.0)
        adj_conv = min(99, max(0, adj_conv))
        picks.append({
            "segment": "swing",
            "symbol": c["symbol"],
            "action": "BUY",
            "entry": round(entry, 2),
            "stop": round(stop, 2),
            "target": round(target, 2),
            "confidence": round(adj_conv, 1),
            "raw_conviction": round(swing_conv, 1),
            "grade": grade,
            "quality_score": round(quality, 1) if quality else None,
            "reward_risk": rr,
            "trend": c.get("trend", "—"),
            "pattern": c.get("pattern", "—"),
            "holding_period": "2–10 days",
            "market_context": mkt.get("overall", "—"),
            "stock_signal": stk_ctx.get("read", "—"),
            "reason": f"{c['symbol']} — grade {grade}, quality {quality:.0f}, "
                      f"conviction {adj_conv:.0f}% (raw {swing_conv:.0f}% × "
                      f"market {mkt.get('overall', '?')} × {stk_ctx.get('read', '?')}), "
                      f"R:R {rr}:1 (swing/delivery)",
        })
    picks.sort(key=lambda x: x["confidence"], reverse=True)
    return picks


def segment_recommendations(balance=100_000):
    """Fetch best picks across all segments. Returns per-segment sorted lists."""
    mkt = _get_market_context()
    options = _safe(_options_picks, balance) or []
    # On the 1GB server, the equity/swing paths run the screener (twice) plus a
    # live stock_context per pick (~35 calls) — that blows past the nginx timeout
    # and 504s. Skip them on micro; options (OI wall selling) is the strategy.
    if os.environ.get("AOS_PROFILE") == "micro":
        equity, swing = [], []
    else:
        equity = _safe(_equity_intraday_picks, balance) or []
        swing = _safe(_swing_picks, balance) or []
    return {
        "equity_intraday": equity,
        "options": options,
        "swing": swing,
        "market_context": {
            "overall": mkt.get("overall", "unknown"),
            "composite_score": mkt.get("composite_score", 0),
            "conviction_multiplier": mkt.get("conviction_multiplier", 1.0),
            "signals": {
                name: {"score": sig.get("score", 0), "read": sig.get("read", "")}
                for name, sig in mkt.get("signals", {}).items()
            },
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_per_segment": {
            "equity_intraday": equity[0] if equity else None,
            "options": options[0] if options else None,
            "swing": swing[0] if swing else None,
        },
    }


def allocate_capital(balance, segments_data):
    """
    Divide capital across segments proportional to confidence of the best
    pick in each. Zero-confidence segments get nothing.

    Returns: {segment: {allocation_pct, amount, pick}}
    """
    best = segments_data.get("best_per_segment", {})
    scores = {}
    for seg, pick in best.items():
        if pick and pick.get("confidence", 0) > 0:
            scores[seg] = pick["confidence"]
    total_conf = sum(scores.values()) or 1
    allocation = {}
    for seg, conf in scores.items():
        pct = round(conf / total_conf * 100, 1)
        amt = round(balance * conf / total_conf, 2)
        allocation[seg] = {
            "allocation_pct": pct,
            "amount": amt,
            "confidence": conf,
            "pick": best[seg],
        }
    for seg in ("equity_intraday", "options", "swing"):
        if seg not in allocation:
            allocation[seg] = {
                "allocation_pct": 0,
                "amount": 0,
                "confidence": 0,
                "pick": None,
            }
    return allocation

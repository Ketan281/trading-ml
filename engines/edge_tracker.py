"""
Edge tracker — a reality check on each trading engine.

The backtest says wall-selling wins ~82% and the intraday-equity model wins
~87%. Those are *promises*. This module measures what each engine actually does
in LIVE paper trading (from the user wallet's closed autonomous trades) and
turns the comparison into a per-engine TRUST multiplier in [0, 1].

The autonomous loop multiplies position size by this trust, so an engine whose
live results hold up keeps trading at full size, while one that diverges below
its backtest is automatically down-weighted — and skipped entirely if it falls
apart (loses money live). This is what stops a backtest that was too good to be
true (e.g. the 3,583% CAGR equity model) from quietly draining the account.

Backend-only. No UI — the frontend just sees the managed result.
"""

import os
import sys
import time
import sqlite3
import logging
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

log = logging.getLogger("edge_tracker")

# Backtest baselines the engines were validated at (the numbers to live up to).
BACKTEST = {
    "wall_selling":    {"win_rate": 82.0, "profit_factor": 6.0},
    "intraday_equity": {"win_rate": 86.9, "profit_factor": 3.0},
}

# Trust before we have enough live data. Wall selling is a structural micro-
# structure edge → trusted at full size. The intraday-equity model has an
# implausibly good backtest → start at HALF size until live results earn trust.
UNPROVEN_DEFAULT = {"wall_selling": 1.0, "intraday_equity": 0.5}

MIN_TRADES = 12      # need this many live closes before judging divergence
FLOOR      = 0.2     # never down-weight a sampled engine below this (unless losing)
LOOKBACK   = 45      # days of live history to score

_cache = {}
_TTL = 300


def engine_of(trade):
    """Classify a closed trade into an engine bucket."""
    seg = (trade.get("segment") or "").lower()
    reason = (trade.get("reason") or "").lower()
    side = (trade.get("side") or "").lower()
    if seg in ("options", "option") and (side == "short" or "wall" in reason):
        return "wall_selling"
    if seg in ("equity", "equity_intraday", "intraday"):
        return "intraday_equity"
    return "other"


def _closed_auto_trades(lookback_days=LOOKBACK):
    """Closed AUTONOMOUS trades (reason tagged '[ML auto]') within lookback."""
    from aos import user_wallet as uw
    cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
    try:
        conn = sqlite3.connect(uw.DB)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM trades WHERE status='closed'").fetchall()
        conn.close()
    except Exception:
        return []
    out = []
    for r in rows:
        d = dict(r)
        if d.get("net_pnl") is None:
            continue
        if "[ml auto]" not in (d.get("reason") or "").lower():
            continue
        if (d.get("opened_at") or "") and d["opened_at"] < cutoff:
            continue
        out.append(d)
    return out


def live_stats(lookback_days=LOOKBACK):
    """Per-engine live paper performance from closed autonomous trades."""
    buckets = {}
    for r in _closed_auto_trades(lookback_days):
        eng = engine_of(r)
        if eng == "other":
            continue
        b = buckets.setdefault(eng, {"n": 0, "wins": 0, "pnl": 0.0,
                                     "gross_win": 0.0, "gross_loss": 0.0})
        pnl = r.get("net_pnl") or 0
        b["n"] += 1
        b["pnl"] += pnl
        if pnl > 0:
            b["wins"] += 1
            b["gross_win"] += pnl
        else:
            b["gross_loss"] += abs(pnl)
    out = {}
    for eng, b in buckets.items():
        n = b["n"]
        pf = (b["gross_win"] / b["gross_loss"]) if b["gross_loss"] > 0 \
            else (99.0 if b["gross_win"] > 0 else 0.0)
        out[eng] = {
            "n": n,
            "win_rate": round(b["wins"] / n * 100, 1) if n else 0,
            "avg_pnl": round(b["pnl"] / n, 1) if n else 0,
            "total_pnl": round(b["pnl"], 1),
            "profit_factor": round(pf, 2),
        }
    return out


def trust(engine, lookback_days=LOOKBACK):
    """Trust multiplier in [0,1] for an engine, from live-vs-backtest divergence.

      • not enough live trades  → UNPROVEN_DEFAULT (walls 1.0, equity 0.5)
      • losing money live        → 0.0 (engine benched until it ages out)
      • else                     → min(win%, profit-factor) ratio vs backtest,
                                   clamped to [FLOOR, 1.0]
    """
    key = ("trust", engine)
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < _TTL:
        return hit[1]

    stats = live_stats(lookback_days).get(engine)
    bt = BACKTEST.get(engine)
    if not bt:
        val = 1.0
    elif not stats or stats["n"] < MIN_TRADES:
        val = UNPROVEN_DEFAULT.get(engine, 1.0)
    elif stats["total_pnl"] <= 0:
        val = 0.0       # losing money live → bench the engine
    else:
        wr_ratio = stats["win_rate"] / max(bt["win_rate"], 1.0)
        pf_ratio = stats["profit_factor"] / max(bt["profit_factor"], 0.1)
        val = round(max(FLOOR, min(1.0, min(wr_ratio, pf_ratio))), 3)

    _cache[key] = (now, val)
    return val


def report(lookback_days=LOOKBACK):
    """Full live-vs-backtest comparison + trust per engine (for logging/admin)."""
    stats = live_stats(lookback_days)
    out = {}
    for eng, bt in BACKTEST.items():
        out[eng] = {
            "live": stats.get(eng, {"n": 0}),
            "backtest": bt,
            "trust": trust(eng, lookback_days),
        }
    return out


def invalidate():
    _cache.clear()


if __name__ == "__main__":
    import json
    print(json.dumps(report(), indent=2, default=str))

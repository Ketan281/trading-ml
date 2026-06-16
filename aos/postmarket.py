"""
Post-market self-review — how the system learns from its own mistakes.

After the close, this job reads the day's trades and decisions from Trade
Memory, classifies what went wrong into three buckets, writes LESSONS back to
memory, and updates each signal's realised OUTCOME (growing the meta-learning
dataset):

  • PREDICTION errors   high-conviction losers; low win-rate → the signal was
                        wrong, especially in a given regime
  • RISK-MGMT errors    losses taken against an adverse regime; oversized risk
  • EXECUTION errors    fees eating the edge (overtrading small moves)

Lessons are heuristic pattern-detection (honest: real ML on them is the
Meta-Learning Layer, once enough history accrues). Nothing is fabricated —
every lesson cites the trades that triggered it.
"""

import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from aos import memory as mem

HIGH_CONVICTION = 65.0
LOW_WINRATE = 0.40


def _closed_trades():
    return mem.query(
        "SELECT t.id, t.symbol, t.segment, t.net_pnl, t.fees, t.exit_reason, "
        "d.conviction, d.regime, d.vetoed "
        "FROM trades t LEFT JOIN decisions d ON t.decision_id = d.id "
        "WHERE t.status='closed'")


def _sync_signal_outcomes(trades):
    """Mark the realised outcome on each traded signal (for meta-learning)."""
    for t in trades:
        pnl = t.get("net_pnl") or 0
        rows = mem.query("SELECT id FROM signals WHERE symbol=? AND outcome_ret IS NULL "
                         "ORDER BY id DESC LIMIT 1", (t["symbol"],))
        if rows:
            mem.set_signal_outcome(rows[0]["id"], outcome_ret=round(pnl, 2),
                                   outcome_label=1 if pnl > 0 else 0)


def review():
    trades = _closed_trades()
    if not trades:
        return {"note": "no closed trades to review today", "lessons": []}

    _sync_signal_outcomes(trades)
    wins = [t for t in trades if (t["net_pnl"] or 0) > 0]
    total_pnl = round(sum(t["net_pnl"] or 0 for t in trades), 2)
    total_fees = round(sum(t["fees"] or 0 for t in trades), 2)
    win_rate = len(wins) / len(trades)
    gross_wins = sum(t["net_pnl"] for t in wins) or 0

    lessons = []
    def add(cat, text, ev):
        lessons.append({"category": cat, "text": text, "evidence": ev})

    # PREDICTION — high-conviction losers
    for t in trades:
        if (t["net_pnl"] or 0) < 0 and (t["conviction"] or 0) >= HIGH_CONVICTION:
            add("prediction",
                f"High-conviction loss on {t['symbol']} (conviction {t['conviction']}) "
                f"in {t['regime']} regime — model was over-confident here.",
                {"symbol": t["symbol"], "pnl": t["net_pnl"], "conviction": t["conviction"]})

    # PREDICTION — low overall win-rate
    if len(trades) >= 3 and win_rate < LOW_WINRATE:
        add("prediction",
            f"Win rate {win_rate:.0%} over {len(trades)} trades — signal selection "
            f"needs review; tighten the entry gate.",
            {"win_rate": round(win_rate, 2), "n": len(trades)})

    # RISK-MGMT — losses taken against an adverse regime
    for t in trades:
        if (t["net_pnl"] or 0) < 0 and t["regime"] in ("bear", "volatile"):
            add("risk_management",
                f"{t['symbol']} traded in a {t['regime']} regime and lost "
                f"₹{abs(t['net_pnl']):.0f} — the regime gate should have been tighter.",
                {"symbol": t["symbol"], "pnl": t["net_pnl"], "regime": t["regime"]})

    # EXECUTION — fees eating the edge
    if gross_wins > 0 and total_fees > 0.20 * gross_wins:
        add("execution",
            f"Fees ₹{total_fees:.0f} are {total_fees/gross_wins:.0%} of gross winning "
            f"P&L — avoid overtrading small-edge setups.",
            {"fees": total_fees, "gross_wins": round(gross_wins)})

    for ln in lessons:
        mem.record_lesson(ln["category"], ln["text"], ln["evidence"])

    by_exit = {}
    for t in trades:
        by_exit[t["exit_reason"] or "?"] = by_exit.get(t["exit_reason"] or "?", 0) + 1

    return {"date": datetime.now().strftime("%Y-%m-%d"),
            "n_trades": len(trades), "win_rate_pct": round(win_rate * 100, 1),
            "net_pnl": total_pnl, "fees": total_fees, "by_exit": by_exit,
            "lessons": lessons}


def run_postmarket():
    r = review()
    today = r.get("date", datetime.now().strftime("%Y-%m-%d"))
    print("=" * 68)
    print(f"  POST-MARKET SELF-REVIEW — {today}")
    print("=" * 68)
    if r.get("note"):
        print(f"  {r['note']}"); return r
    print(f"  Trades {r['n_trades']} | win-rate {r['win_rate_pct']}% | "
          f"net P&L ₹{r['net_pnl']:,} | fees ₹{r['fees']:,}")
    print(f"  Exits  : {r['by_exit']}")
    print(f"\n  LESSONS LEARNED ({len(r['lessons'])}):")
    for ln in r["lessons"]:
        print(f"   • [{ln['category']}] {ln['text']}")
    if not r["lessons"]:
        print("   • (clean day — no error patterns detected)")
    print(f"\n  Stored to Trade Memory. Total lessons: "
          f"{mem.stats()['lessons']}")
    return r


if __name__ == "__main__":
    run_postmarket()

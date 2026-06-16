"""
Performance attribution — WHERE did the P&L come from?

Total return tells you nothing about why. Attribution decomposes realised P&L
so you can tell skill from luck and double down on what works:

  • by SECTOR    — which sectors made/lost money (rotation skill)
  • by SIGNAL    — winners vs losers split by what the ensemble said drove them
  • by POSITION  — best/worst trades, win rate, profit factor, avg win/loss

Runs off the paper-trading ledger (realised closed trades + open marks), so it
grows more meaningful as the forward test accumulates trades.
"""

import os
import sys
import json
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LEDGER = os.path.join(ROOT, "data", "paper", "ledger.json")


def _signal_of(why):
    """Extract the dominant ensemble signal from the explanation text."""
    if not why:
        return "unknown"
    for key, tag in (("price momentum", "momentum"),
                     ("fundamental quality", "quality"),
                     ("sector strength", "sector_rs"),
                     ("intra-sector", "intra_rs")):
        if key in why:
            return tag
    return "other"


def attribute():
    if not os.path.exists(LEDGER):
        return None
    s = json.load(open(LEDGER))
    closed = s.get("closed", [])
    if not closed:
        return {"note": "no closed trades yet — attribution grows as the "
                "paper book realises trades"}

    by_sector, by_signal = defaultdict(float), defaultdict(float)
    by_sector_n, by_signal_n = defaultdict(int), defaultdict(int)
    wins, losses = [], []
    for c in closed:
        pnl = c.get("pnl", 0)
        by_sector[c.get("sector") or "?"] += pnl
        by_sector_n[c.get("sector") or "?"] += 1
        sig = _signal_of(c.get("why"))
        by_signal[sig] += pnl; by_signal_n[sig] += 1
        (wins if pnl > 0 else losses).append(pnl)

    gross_win = sum(wins); gross_loss = -sum(losses)
    total = gross_win - gross_loss
    return {
        "total_realized_pnl": round(total),
        "n_trades": len(closed),
        "win_rate_pct": round(len(wins) / len(closed) * 100, 1),
        "avg_win": round(gross_win / len(wins)) if wins else 0,
        "avg_loss": round(-gross_loss / len(losses)) if losses else 0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else None,
        "by_sector": {k: round(v) for k, v in sorted(by_sector.items(), key=lambda x: -x[1])},
        "by_signal": {k: round(v) for k, v in sorted(by_signal.items(), key=lambda x: -x[1])},
        "by_signal_trades": dict(by_signal_n),
        "best_trade": max(closed, key=lambda c: c.get("pnl", 0)),
        "worst_trade": min(closed, key=lambda c: c.get("pnl", 0)),
    }


def report():
    a = attribute()
    print("=" * 64)
    print("  PERFORMANCE ATTRIBUTION")
    print("=" * 64)
    if a is None:
        print("  No paper ledger yet — run pipelines/paper_trading.py enter")
        return
    if a.get("note"):
        print(f"  {a['note']}"); return
    print(f"  Realized P&L : ₹{a['total_realized_pnl']:,} over {a['n_trades']} trades")
    print(f"  Win rate     : {a['win_rate_pct']}% | profit factor "
          f"{a['profit_factor']} | avg win ₹{a['avg_win']:,} / avg loss ₹{a['avg_loss']:,}")
    print("\n  By sector:")
    for k, v in a["by_sector"].items():
        print(f"     {k:<30} ₹{v:>10,}")
    print("\n  By signal (which edge drove the P&L):")
    for k, v in a["by_signal"].items():
        print(f"     {k:<12} ₹{v:>10,}  ({a['by_signal_trades'].get(k,0)} trades)")
    bt, wt = a["best_trade"], a["worst_trade"]
    print(f"\n  Best : {bt['symbol']} ₹{bt['pnl']:,} ({bt.get('return_pct')}%)")
    print(f"  Worst: {wt['symbol']} ₹{wt['pnl']:,} ({wt.get('return_pct')}%)")


if __name__ == "__main__":
    report()

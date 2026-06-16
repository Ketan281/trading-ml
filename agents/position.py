"""
Position agent — one autonomous agent per open trade.

Each open trade gets its own agent that watches price and decides, on every
tick, what to do: hold, book a scale-out target, trail the stop, or exit on the
stop. This is the auto stop-loss / target / profit-booking brain.

Behaviour (works for LONG equity/option-buy and SHORT option-sell):
  • STOP        — price hits the stop → exit the remaining quantity
  • TARGETS     — scale-out plan, e.g. book 50% at T1 (=1R), 50% at T2 (=2R)
  • BREAKEVEN   — after the first target books, stop jumps to entry (free trade)
  • TRAIL       — after the first target, a trailing stop locks in more profit

The agent only DECIDES; the manager executes (and charges fees on each fill).
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Position:
    id: str
    symbol: str
    segment: str                 # equity_delivery | equity_intraday | options
    side: str                    # 'long' (buy) or 'short' (option sell)
    entry: float
    qty: int
    stop: float
    targets: List[dict]          # [{price, frac, booked}]
    trail_pct: float = 0.0       # trailing stop distance (fraction), 0 = off
    entry_cost: float = 0.0
    remaining: int = 0
    realized: float = 0.0
    status: str = "open"
    high_water: float = 0.0
    moved_breakeven: bool = False
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.remaining == 0:
            self.remaining = self.qty
        self.high_water = self.entry


class PositionAgent:
    def __init__(self, pos: Position):
        self.p = pos

    @property
    def is_long(self):
        return self.p.side in ("long", "buy")

    def evaluate(self, price):
        """Return a list of actions for the manager to execute this tick."""
        p = self.p
        actions = []
        if p.status != "open" or p.remaining <= 0:
            return actions
        long = self.is_long

        # 1) Stop check (highest priority).
        if (long and price <= p.stop) or (not long and price >= p.stop):
            actions.append({"type": "exit", "reason": "stop_loss",
                            "qty": p.remaining, "price": price})
            return actions

        # 2) Scale-out targets.
        for t in p.targets:
            if t.get("booked"):
                continue
            hit = (long and price >= t["price"]) or (not long and price <= t["price"])
            if hit:
                bq = min(p.remaining, max(1, round(p.qty * t["frac"])))
                actions.append({"type": "book", "reason": "target",
                                "qty": bq, "price": price, "target": t["price"]})
                t["booked"] = True
                # First target → move stop to breakeven, arm trailing.
                if not p.moved_breakeven:
                    p.stop = p.entry
                    p.moved_breakeven = True
                    actions.append({"type": "stop_update", "stop": round(p.entry, 2),
                                    "reason": "breakeven_after_T1"})

        # 3) Trailing stop (only after first target, if configured).
        if p.moved_breakeven and p.trail_pct > 0:
            if long:
                p.high_water = max(p.high_water, price)
                new_stop = p.high_water * (1 - p.trail_pct)
                if new_stop > p.stop:
                    p.stop = new_stop
                    actions.append({"type": "stop_update", "stop": round(new_stop, 2),
                                    "reason": "trail"})
            else:
                p.high_water = min(p.high_water, price)
                new_stop = p.high_water * (1 + p.trail_pct)
                if new_stop < p.stop:
                    p.stop = new_stop
                    actions.append({"type": "stop_update", "stop": round(new_stop, 2),
                                    "reason": "trail"})
        return actions

    def unrealized(self, price):
        p = self.p
        d = (price - p.entry) if self.is_long else (p.entry - price)
        return d * p.remaining

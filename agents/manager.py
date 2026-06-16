"""
Trade manager — the orchestrator that coordinates the agents.

Wires together the Wallet agent (capital), the Brokerage agent (costs) and one
Position agent per open trade. It:

  • OPENS a trade only if the wallet can fund cost-basis + entry charges
  • on every price TICK, asks each position agent what to do and EXECUTES it —
    booking scale-out targets, trailing stops, exiting on stop — charging real
    brokerage on every fill
  • books NET P&L (gross − entry-fee share − exit fee) back to the wallet
  • PERSISTS everything to data/agents/state.json so it survives restarts

Works for swing equity (equity_delivery) and option BUYING (options). Option
SELLING / credit structures need broker margin modelling — flagged, not faked.
"""

import os
import sys
import json
import uuid
from datetime import datetime
from dataclasses import asdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.brokerage import charges
from agents.wallet import WalletAgent
from agents.position import Position, PositionAgent

STATE_PATH = os.path.join(ROOT, "data", "agents", "state.json")
os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)


class TradeManager:
    def __init__(self, starting_cash=100_000, load=True):
        self.wallet = WalletAgent(starting_cash)
        self.agents = []          # list[PositionAgent]
        self.log = []
        if load:
            self._load()

    # ── persistence ──────────────────────────────────────
    def _save(self):
        json.dump({
            "wallet": {"start": self.wallet.start, "cash": self.wallet.cash,
                       "deployed": self.wallet.deployed,
                       "realized_pnl": self.wallet.realized_pnl,
                       "fees_paid": self.wallet.fees_paid},
            "positions": [asdict(a.p) for a in self.agents],
            "log": self.log[-200:],
        }, open(STATE_PATH, "w"), indent=2, default=str)

    def _load(self):
        if not os.path.exists(STATE_PATH):
            return
        try:
            s = json.load(open(STATE_PATH))
        except Exception:
            return
        w = s.get("wallet", {})
        self.wallet.start = w.get("start", self.wallet.start)
        self.wallet.cash = w.get("cash", self.wallet.cash)
        self.wallet.deployed = w.get("deployed", 0.0)
        self.wallet.realized_pnl = w.get("realized_pnl", 0.0)
        self.wallet.fees_paid = w.get("fees_paid", 0.0)
        self.agents = [PositionAgent(Position(**p)) for p in s.get("positions", [])]
        self.log = s.get("log", [])

    def _event(self, **kw):
        kw["t"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log.append(kw)
        return kw

    # ── open ─────────────────────────────────────────────
    def open_trade(self, symbol, segment, entry, qty, stop, targets,
                   side="long", trail_pct=0.0, meta=None):
        """targets: list of (price, fraction). Charges entry fees, reserves capital."""
        entry_side = "buy" if side in ("long", "buy") else "sell"
        fee = charges(segment, entry_side, entry, qty)["total"]
        cost_basis = entry * qty
        if not self.wallet.open_cost(cost_basis, fee):
            return self._event(action="rejected", symbol=symbol,
                               reason="insufficient_funds",
                               need=round(cost_basis + fee, 2),
                               have=round(self.wallet.cash, 2))
        pos = Position(
            id=str(uuid.uuid4())[:8], symbol=symbol, segment=segment, side=side,
            entry=float(entry), qty=int(qty), stop=float(stop),
            targets=[{"price": float(p), "frac": float(f), "booked": False}
                     for p, f in targets],
            trail_pct=trail_pct, entry_cost=fee, meta=meta or {})
        self.agents.append(PositionAgent(pos))
        self._save()
        return self._event(action="opened", id=pos.id, symbol=symbol,
                           segment=segment, side=side, entry=entry, qty=qty,
                           stop=stop, targets=pos.targets, entry_fee=fee)

    # ── tick ─────────────────────────────────────────────
    def on_prices(self, price_map):
        events = []
        for ag in self.agents:
            p = ag.p
            if p.status != "open":
                continue
            price = price_map.get(p.symbol)
            if price is None:
                continue
            for a in ag.evaluate(price):
                if a["type"] in ("book", "exit"):
                    events.append(self._execute_fill(ag, a))
                elif a["type"] == "stop_update":
                    events.append(self._event(action="stop_update", id=p.id,
                                  symbol=p.symbol, stop=a["stop"], reason=a["reason"]))
        if events:
            self._save()
        return events

    def _execute_fill(self, ag, a):
        p = ag.p; qty = a["qty"]; price = a["price"]
        long = ag.is_long
        exit_side = "sell" if long else "buy"
        fee = charges(p.segment, exit_side, price, qty)["total"]
        gross = (price - p.entry) * qty if long else (p.entry - price) * qty
        entry_fee_share = p.entry_cost * qty / p.qty
        realized = gross - entry_fee_share - fee
        cost_basis_released = p.entry * qty
        self.wallet.close_proceeds(cost_basis_released, price * qty, fee, realized)
        p.remaining -= qty
        p.realized += realized
        if p.remaining <= 0:
            p.status = "closed"
        return self._event(action="booked" if a["type"] == "book" else "exited",
                           id=p.id, symbol=p.symbol, reason=a["reason"],
                           qty=qty, price=price, gross=round(gross, 2),
                           exit_fee=round(fee, 2), net_pnl=round(realized, 2),
                           remaining=p.remaining, status=p.status)

    # ── reporting ────────────────────────────────────────
    def open_mtm(self, price_map):
        return sum(ag.unrealized(price_map.get(ag.p.symbol, ag.p.entry))
                   for ag in self.agents if ag.p.status == "open")

    def status(self, price_map=None):
        price_map = price_map or {}
        mtm_value = sum(price_map.get(ag.p.symbol, ag.p.entry) * ag.p.remaining
                        for ag in self.agents if ag.p.status == "open")
        snap = self.wallet.snapshot(mtm_value)
        opens = [{"id": ag.p.id, "symbol": ag.p.symbol, "side": ag.p.side,
                  "entry": ag.p.entry, "remaining": ag.p.remaining,
                  "stop": round(ag.p.stop, 2),
                  "now": price_map.get(ag.p.symbol),
                  "unreal_pnl": round(ag.unrealized(price_map.get(ag.p.symbol, ag.p.entry)), 2)}
                 for ag in self.agents if ag.p.status == "open"]
        return {"wallet": snap, "open_positions": opens,
                "closed": sum(1 for ag in self.agents if ag.p.status == "closed")}


# ── Demo simulation ───────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  TRADE-MANAGER AGENTS — simulated lifecycle (real brokerage)")
    print("=" * 70)
    tm = TradeManager(starting_cash=300_000, load=False)

    # Swing equity long: book 50% at T1, 50% at T2, trail 2% after T1.
    e = tm.open_trade("BSE", "equity_delivery", entry=4146, qty=30, stop=4053,
                      targets=[(4300, 0.5), (4450, 0.5)], trail_pct=0.02)
    print(f"  OPEN  {e['symbol']} swing: {e['qty']}@{e['entry']} stop {e['stop']} "
          f"(entry fee ₹{e['entry_fee']})")
    # Option buy (BANKNIFTY CE, lot 35): T1 +20%, T2 +40%.
    e = tm.open_trade("BANKNIFTY24CE", "options", entry=300, qty=35, stop=210,
                      targets=[(360, 0.5), (420, 0.5)], trail_pct=0.0)
    print(f"  OPEN  option: {e['qty']}@{e['entry']} stop {e['stop']} "
          f"(entry fee ₹{e['entry_fee']})")

    print("\n  Price path → agent actions:")
    path = [
        {"BSE": 4200, "BANKNIFTY24CE": 330},
        {"BSE": 4305, "BANKNIFTY24CE": 365},     # both hit T1 → book 50% + breakeven
        {"BSE": 4380, "BANKNIFTY24CE": 300},     # BSE trails up; option falls back
        {"BSE": 4455, "BANKNIFTY24CE": 300},     # BSE hits T2
        {"BSE": 4455, "BANKNIFTY24CE": 300},     # option now at breakeven stop (300)
    ]
    for i, pm in enumerate(path, 1):
        for ev in tm.on_prices(pm):
            if ev["action"] in ("booked", "exited"):
                print(f"   t{i}: {ev['action'].upper()} {ev['symbol']} {ev['qty']}@{ev['price']} "
                      f"({ev['reason']}) net ₹{ev['net_pnl']} | rem {ev['remaining']}")
            elif ev["action"] == "stop_update":
                print(f"   t{i}: {ev['symbol']} stop → {ev['stop']} ({ev['reason']})")

    print("\n  FINAL")
    st = tm.status(path[-1])
    w = st["wallet"]
    print(f"   Equity ₹{w['equity']:,} ({w['total_return_pct']:+}%) | "
          f"realized P&L ₹{w['realized_pnl']:,} | fees ₹{w['fees_paid']:,} | "
          f"free cash ₹{w['free_cash']:,}")
    print(f"   Open {len(st['open_positions'])} | closed {st['closed']}")

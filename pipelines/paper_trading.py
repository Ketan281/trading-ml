"""
Paper-trading environment — forward-test the system before risking money.

A backtest is in-sample history; paper trading is the honest forward test:
take real signals, open positions on a persistent ledger, mark-to-market each
day against live prices, honour stops, and track the equity curve — all with
transaction costs. If the paper book doesn't make money going forward, the
live book won't either.

State persists in data/paper/ so it accumulates across days/runs.

CLI:
    python pipelines/paper_trading.py enter     # open the latest portfolio_book
    python pipelines/paper_trading.py update    # mark-to-market + honour stops
    python pipelines/paper_trading.py status    # positions, equity, P&L
"""

import os
import sys
import json
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.cross_sectional import load_prices

PAPER_DIR = os.path.join(ROOT, "data", "paper")
os.makedirs(PAPER_DIR, exist_ok=True)
STATE_PATH = os.path.join(PAPER_DIR, "ledger.json")

START_CASH = 1_000_000
COST_PER_SIDE = 0.0020      # 20 bps brokerage+impact+charges


def _load():
    if os.path.exists(STATE_PATH):
        return json.load(open(STATE_PATH))
    return {"cash": START_CASH, "start_equity": START_CASH, "positions": {},
            "closed": [], "equity_curve": [], "created": datetime.now().isoformat()}


def _save(s):
    json.dump(s, open(STATE_PATH, "w"), indent=2, default=str)


def _latest_prices(symbols):
    prices = load_prices(universe=set(symbols))
    return {s: float(df["Close"].iloc[-1]) for s, df in prices.items()}, \
           {s: str(df.index[-1].date()) for s, df in prices.items()}


def enter_book(book_path=None):
    s = _load()
    book_path = book_path or os.path.join(ROOT, "outputs", "portfolio_book.json")
    if not os.path.exists(book_path):
        print("  ⚠ No portfolio_book.json — run pipelines/portfolio_book.py first.")
        return
    book = json.load(open(book_path))
    px, _ = _latest_prices([h["symbol"] for h in book["holdings"]])
    opened = 0
    for h in book["holdings"]:
        sym = h["symbol"]
        if sym in s["positions"]:
            continue
        price = px.get(sym) or h.get("price")
        shares = h["shares"]
        if not price or not shares:
            continue
        cost = price * shares * COST_PER_SIDE
        cash_needed = price * shares + cost
        if cash_needed > s["cash"]:
            continue
        s["cash"] -= cash_needed
        s["positions"][sym] = {
            "shares": shares, "entry": price, "stop": h.get("stop"),
            "sector": h.get("sector"), "entry_date": datetime.now().strftime("%Y-%m-%d"),
            "entry_cost": round(cost), "why": h.get("why")}
        opened += 1
    _save(s)
    print(f"  📝 Opened {opened} paper position(s). Cash left ₹{s['cash']:,.0f}")


def update():
    s = _load()
    if not s["positions"]:
        print("  No open positions."); return
    px, asof = _latest_prices(list(s["positions"]))
    closed = []
    for sym, p in list(s["positions"].items()):
        price = px.get(sym)
        if price is None:
            continue
        if p["stop"] and price <= p["stop"]:           # stop hit → close
            closed.append((sym, price, "stop_hit"))
    for sym, price, reason in closed:
        _close(s, sym, price, reason)
    # Mark-to-market equity
    equity = s["cash"] + sum(px.get(sym, p["entry"]) * p["shares"]
                             for sym, p in s["positions"].items())
    asof_date = max(asof.values()) if asof else datetime.now().strftime("%Y-%m-%d")
    s["equity_curve"].append({"date": asof_date, "equity": round(equity)})
    _save(s)
    ret = equity / s["start_equity"] - 1
    print(f"  📈 MTM {asof_date}: equity ₹{equity:,.0f} ({ret:+.2%}) | "
          f"{len(s['positions'])} open | {len(closed)} stopped out")


def _close(s, sym, price, reason):
    p = s["positions"].pop(sym)
    proceeds = price * p["shares"]
    cost = proceeds * COST_PER_SIDE
    s["cash"] += proceeds - cost
    pnl = (price - p["entry"]) * p["shares"] - p.get("entry_cost", 0) - cost
    s["closed"].append({**p, "symbol": sym, "exit": price, "reason": reason,
                        "exit_date": datetime.now().strftime("%Y-%m-%d"),
                        "pnl": round(pnl),
                        "return_pct": round((price / p["entry"] - 1) * 100, 2)})


def close_all():
    s = _load()
    px, _ = _latest_prices(list(s["positions"]))
    for sym in list(s["positions"]):
        if sym in px:
            _close(s, sym, px[sym], "manual_close")
    _save(s); print("  Closed all positions.")


def status():
    s = _load()
    px, _ = _latest_prices(list(s["positions"])) if s["positions"] else ({}, {})
    mtm = sum(px.get(sym, p["entry"]) * p["shares"] for sym, p in s["positions"].items())
    equity = s["cash"] + mtm
    realized = sum(c["pnl"] for c in s["closed"])
    print("=" * 70)
    print("  PAPER-TRADING LEDGER")
    print("=" * 70)
    print(f"  Equity ₹{equity:,.0f}  ({equity/s['start_equity']-1:+.2%})  | "
          f"cash ₹{s['cash']:,.0f} | realized P&L ₹{realized:,.0f}")
    if s["positions"]:
        print(f"\n  OPEN ({len(s['positions'])}):")
        print(f"  {'SYMBOL':<12}{'SH':>6}{'ENTRY':>9}{'NOW':>9}{'STOP':>9}{'P&L%':>8}  SECTOR")
        for sym, p in s["positions"].items():
            now = px.get(sym, p["entry"])
            print(f"  {sym:<12}{p['shares']:>6}{p['entry']:>9.1f}{now:>9.1f}"
                  f"{(p['stop'] or 0):>9.1f}{(now/p['entry']-1)*100:>8.1f}  {p.get('sector','-')}")
    if s["closed"]:
        wins = [c for c in s["closed"] if c["pnl"] > 0]
        print(f"\n  CLOSED: {len(s['closed'])} | win rate "
              f"{len(wins)/len(s['closed'])*100:.0f}% | realized ₹{realized:,.0f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"enter": enter_book, "update": update, "status": status,
     "close_all": close_all}.get(cmd, status)()

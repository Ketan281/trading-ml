"""
Auto-trader — wire the signals to the agents and tick them on live prices.

Closes the loop:
  1. AUTO-OPEN  — pull the screener's swing buys and the options action call,
     size them by wallet risk, and open them through the TradeManager (which
     gates on capital + charges entry fees).
  2. LIVE PRICES— fetch the current price for every open position (stocks via
     yfinance, option legs via the live NSE chain LTP).
  3. TICK       — feed prices to the agents; they auto-book targets, trail, and
     exit on stops, charging exit fees and booking net P&L.

`monitor_once()` runs one cycle (schedule it every minute) and `run()` loops
during market hours. Paper execution — no broker orders are placed.
"""

import os
import sys
import time
from datetime import datetime, time as dtime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd

from agents.manager import TradeManager

RISK_PCT     = 0.01
MAX_SWING    = 3
MAX_OPTIONS  = 1
SL_OPT_PCT   = 0.35       # option stop = 35% premium decay
MARKET_OPEN  = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)
LOT = {"NIFTY": 75, "BANKNIFTY": 35}


def _num(x, default=None):
    try:
        if isinstance(x, str):
            x = x.replace(",", "").split("-")[-1].strip()   # "1089-1100" → 1100
        return float(x)
    except (TypeError, ValueError):
        return default


def _held(tm):
    return {ag.p.symbol for ag in tm.agents if ag.p.status == "open"}


# ── Auto-open from signals ────────────────────────────
def auto_open_swing(tm, max_n=MAX_SWING, risk_pct=RISK_PCT):
    from api.precompute import cached_screen
    rep = cached_screen()
    if not rep or rep.get("_error"):
        return []
    held = _held(tm); events = []
    for c in rep.get("actionable", [])[:max_n * 2]:
        if len(events) >= max_n or c["symbol"] in held:
            continue
        entry = _num(c.get("price")); stop = _num(c.get("stop_loss"))
        tgt = _num(c.get("target"))
        if not entry or not stop or stop >= entry:
            continue
        risk = entry - stop
        budget = tm.wallet.cash * risk_pct
        qty = max(1, int(budget / risk))
        qty = min(qty, int(tm.wallet.cash * 0.2 / entry))     # ≤20% of cash/name
        if qty < 1:
            continue
        t1 = entry + risk                                     # 1R
        t2 = tgt if (tgt and tgt > t1) else entry + 2 * risk  # plan target or 2R
        ev = tm.open_trade(c["symbol"], "equity_delivery", entry, qty, stop,
                           targets=[(t1, 0.5), (t2, 0.5)], trail_pct=0.03,
                           meta={"type": "swing", "grade": c.get("grade")})
        events.append(ev)
    return events


def auto_open_option(tm, symbol="BANKNIFTY", risk_pct=RISK_PCT):
    from pipelines.options.chain_live_intel import fetch_chain
    from pipelines.options_action_engine import chain_prob_up, action_from_probability
    chain = fetch_chain(symbol)
    if not chain:
        return []
    act = action_from_probability(chain_prob_up(chain))
    if act["leg"] is None:                       # NO_TRADE
        return []
    leg = act["leg"].lower(); atm = chain["atm"]
    row = chain["df"].iloc[(chain["df"]["strike"] - atm).abs().argmin()]
    ltp = float(row[f"{leg}_ltp"])
    if ltp <= 0:
        return []
    tag = f"{symbol}{atm}{act['leg']}"
    if tag in _held(tm):
        return []
    lot = LOT[symbol]
    risk_per_lot = ltp * SL_OPT_PCT * lot
    budget = tm.wallet.cash * risk_pct * act["size_factor"]
    lots = max(1, int(budget / risk_per_lot))
    qty = lots * lot
    stop = round(ltp * (1 - SL_OPT_PCT), 2)
    ev = tm.open_trade(tag, "options", ltp, qty, stop,
                       targets=[(round(ltp * 1.25, 2), 0.5), (round(ltp * 1.5, 2), 0.5)],
                       trail_pct=0.0,
                       meta={"type": "option", "underlying": symbol,
                             "strike": int(atm), "leg": act["leg"]})
    return [ev]


# ── Live prices for open positions ────────────────────
def live_prices(tm):
    import yfinance as yf
    from pipelines.options.chain_live_intel import fetch_chain
    from pipelines.intraday import INDEX_YF
    pm = {}; chains = {}
    for ag in tm.agents:
        if ag.p.status != "open":
            continue
        p = ag.p
        if p.meta.get("type") == "option":
            u = p.meta["underlying"]
            if u not in chains:
                chains[u] = fetch_chain(u)
            ch = chains[u]
            if ch is not None:
                r = ch["df"].iloc[(ch["df"]["strike"] - p.meta["strike"]).abs().argmin()]
                pm[p.symbol] = float(r[f"{p.meta['leg'].lower()}_ltp"])
        else:
            px = _stock_price(p.symbol)
            if px:
                pm[p.symbol] = px
    return pm


_price_cache = {}   # symbol -> (timestamp, price_or_None)
_CACHE_TTL = 60     # seconds — avoid hammering yfinance
_FAIL_TTL = 300     # seconds — back off longer on failures

def _stock_price(symbol):
    """Robust last price with caching to avoid yfinance spam."""
    import time as _time
    now = _time.time()
    if symbol in _price_cache:
        ts, px = _price_cache[symbol]
        ttl = _CACHE_TTL if px is not None else _FAIL_TTL
        if now - ts < ttl:
            return px

    import yfinance as yf
    from pipelines.intraday import INDEX_YF
    yf_sym = INDEX_YF.get(symbol, f"{symbol}.NS")
    try:
        t = yf.Ticker(yf_sym)
        fi = t.fast_info
        for k in ("last_price", "lastPrice"):
            v = fi.get(k) if hasattr(fi, "get") else getattr(fi, k, None)
            if v:
                px = float(v)
                _price_cache[symbol] = (now, px)
                return px
    except Exception:
        pass
    try:
        h = yf.Ticker(yf_sym).history(period="1d")
        if len(h):
            px = float(h["Close"].iloc[-1])
            _price_cache[symbol] = (now, px)
            return px
    except Exception:
        pass
    _price_cache[symbol] = (now, None)
    return None


def market_open(now=None):
    now = now or datetime.now()
    return now.weekday() < 5 and MARKET_OPEN <= now.time() <= MARKET_CLOSE


# ── One monitoring cycle ──────────────────────────────
def monitor_once(tm, do_open=True, option_symbol="BANKNIFTY"):
    opened = []
    if do_open:
        opened += auto_open_swing(tm)
        opened += auto_open_option(tm, option_symbol)
    pm = live_prices(tm)
    fills = tm.on_prices(pm) if pm else []
    return {"opened": opened, "prices": pm, "fills": fills,
            "status": tm.status(pm)}


def run(interval=60, option_symbol="BANKNIFTY", starting_cash=300_000):
    tm = TradeManager(starting_cash=starting_cash)
    print(f"  Auto-trader live. Interval {interval}s. Paper mode (no broker orders).")
    while True:
        if market_open():
            r = monitor_once(tm, option_symbol=option_symbol)
            w = r["status"]["wallet"]
            for e in r["opened"]:
                if e.get("action") == "opened":
                    print(f"  OPEN {e['symbol']} {e['qty']}@{e['entry']} stop {e['stop']}")
            for f in r["fills"]:
                if f["action"] in ("booked", "exited"):
                    print(f"  {f['action'].upper()} {f['symbol']} {f['qty']}@{f['price']} "
                          f"net ₹{f['net_pnl']} ({f['reason']})")
            print(f"  [{datetime.now():%H:%M:%S}] equity ₹{w['equity']:,} "
                  f"({w['total_return_pct']:+}%) | open {len(r['status']['open_positions'])}")
        else:
            print(f"  [{datetime.now():%H:%M:%S}] market closed — idle")
        time.sleep(interval)


if __name__ == "__main__":
    tm = TradeManager(starting_cash=300_000)
    r = monitor_once(tm)
    print("=" * 64)
    print("  AUTO-TRADER — one cycle")
    print("=" * 64)
    print(f"  Opened : {[e.get('symbol') for e in r['opened'] if e.get('action')=='opened']}")
    print(f"  Prices : {r['prices']}")
    print(f"  Fills  : {[(f['symbol'], f['action'], f.get('net_pnl')) for f in r['fills']]}")
    w = r["status"]["wallet"]
    print(f"  Equity ₹{w['equity']:,} ({w['total_return_pct']:+}%) | "
          f"open {len(r['status']['open_positions'])} | cash ₹{w['free_cash']:,}")
    for op in r["status"]["open_positions"]:
        print(f"     {op['symbol']:<16} {op['remaining']}@{op['entry']} "
              f"stop {op['stop']} now {op['now']} unreal ₹{op['unreal_pnl']}")

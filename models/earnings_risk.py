"""
Earnings-event risk model — don't get gapped by a results print.

Quarterly results are a scheduled BINARY event: a stock can gap 10-15% overnight
on numbers, and no price-action stop protects you from a gap. A disciplined book
reduces or avoids fresh positions into results. This flags that risk:

  • fetch each name's NEXT earnings date (yfinance calendar), cached + resumable
  • days-to-earnings → action: AVOID (≤3d), REDUCE (≤7d), CLEAR (otherwise)

It feeds the portfolio book's gates so names reporting this week are dropped or
down-weighted, and the screener can warn on any actionable name.

HONEST LIMITS: yfinance earnings dates are estimates and sometimes stale/missing
— treat a flag as "check before trading", not gospel. Only the NEXT date is
available (no deep history).
"""

import os
import sys
import json
import time
from datetime import date, datetime

import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "historical", "earnings_dates.json")

AVOID_DAYS  = 3        # results imminent → no fresh position
REDUCE_DAYS = 7        # results this week → half size
CACHE_TTL_H = 24 * 3   # refetch a symbol's date at most every ~3 days


def _load():
    if os.path.exists(CACHE):
        try:
            return json.load(open(CACHE, encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save(d):
    json.dump(d, open(CACHE, "w", encoding="utf-8"), indent=2, default=str)


def _fetch_one(sym):
    try:
        cal = yf.Ticker(f"{sym}.NS").calendar
        ed = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if isinstance(ed, (list, tuple)) and ed:
            ed = ed[0]
        return str(ed) if ed else None
    except Exception:
        return None


def fetch_earnings(symbols, refresh=False, sleep=0.3):
    cache = _load()
    now = time.time()
    todo = [s for s in symbols
            if refresh or s not in cache
            or (now - cache.get(s, {}).get("ts", 0)) > CACHE_TTL_H * 3600]
    for i, s in enumerate(todo, 1):
        cache[s] = {"date": _fetch_one(s), "ts": now}
        if i % 10 == 0:
            _save(cache)
        time.sleep(sleep)
    _save(cache)
    return cache


def next_earnings(symbol):
    return _load().get(symbol, {}).get("date")


def earnings_risk(symbol, as_of=None):
    as_of = as_of or date.today()
    ed = next_earnings(symbol)
    if not ed:
        return {"symbol": symbol, "earnings_date": None, "days": None,
                "risk": "unknown", "action": "none"}
    try:
        d = datetime.fromisoformat(str(ed)[:10]).date()
    except Exception:
        return {"symbol": symbol, "earnings_date": ed, "days": None,
                "risk": "unknown", "action": "none"}
    days = (d - as_of).days
    if days < 0:
        risk, action = "post_results", "clear"          # already reported
    elif days <= AVOID_DAYS:
        risk, action = "imminent", "avoid"
    elif days <= REDUCE_DAYS:
        risk, action = "this_week", "reduce"
    else:
        risk, action = "clear", "none"
    return {"symbol": symbol, "earnings_date": str(d), "days": days,
            "risk": risk, "action": action}


def assess(symbols, fetch=True):
    if fetch:
        fetch_earnings(symbols)
    return {s: earnings_risk(s) for s in symbols}


if __name__ == "__main__":
    syms = sys.argv[1:]
    if not syms:
        bp = os.path.join(ROOT, "outputs", "portfolio_book.json")
        if os.path.exists(bp):
            syms = [h["symbol"] for h in json.load(open(bp))["holdings"]]
        else:
            syms = ["RELIANCE", "TCS", "INFY", "HDFCBANK"]
    print("=" * 60)
    print("  EARNINGS-EVENT RISK")
    print("=" * 60)
    res = assess(syms)
    for s in syms:
        r = res[s]
        icon = "❓" if r["risk"] == "unknown" else \
               {"avoid": "⛔", "reduce": "⚠"}.get(r["action"], "✅")
        d = f"{r['days']}d → {r['earnings_date']}" if r["days"] is not None else "no date"
        print(f"  {icon} {s:<12} {r['risk']:<13} {r['action']:<7} ({d})")

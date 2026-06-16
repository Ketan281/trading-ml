"""
Intraday signal engine for stocks — rule-based, multi-timeframe (5m + 15m).

HONEST SCOPE
------------
This is NOT the validated swing ranker. Intraday history is ~60 days max, far
too short to train/walk-forward an ML edge, so this engine is deliberately
RULE-BASED and transparent: proven classic setups, computed live on intraday
bars, with hard stops and time guards. It tells you WHERE a textbook intraday
setup exists right now — it does not claim a statistically proven edge, and
intraday is where trading costs/slippage hurt most. Trade it small, with
discipline, and verify on your own broker feed.

Design
------
• Signals are generated on the 5-minute chart and CONFIRMED by the 15-minute
  trend (only take longs in a 15m uptrend, shorts in a 15m downtrend).
• Three setups, each a well-known intraday pattern:
    1. ORB   — opening-range breakout (break of first 15 min range on volume)
    2. VWAP  — VWAP reclaim (long) / rejection (short)
    3. MOM   — trend pullback continuation (pull back to EMA20/VWAP, resume)
• Every call carries entry, ATR/structure stop, T1=1R / T2=2R targets, a
  confluence-based grade, and risk-based size (1% risk, stop sets quantity).
• Time guards: skip the first 5 min, no NEW entries after 14:45 IST, flat by
  15:15 IST — intraday positions are not held overnight.
"""

import os
import sys
from datetime import time as dtime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yfinance as yf

# ── Config ────────────────────────────────────────────
CAPITAL          = 1_000_000
RISK_PCT         = 0.01          # 1% of capital risked per trade
MAX_POSITION_PCT = 0.20          # never more than 20% of book in one name
OR_BARS          = 3             # opening range = first 3 × 5m bars (15 min)
ATR_PERIOD       = 14
ATR_STOP_MULT    = 1.2           # stop = ATR_STOP_MULT × 5m ATR from entry
MIN_RVOL         = 1.2           # require above-average volume on the trigger
NO_NEW_ENTRY     = dtime(14, 45) # IST — no fresh entries after this
SQUARE_OFF       = dtime(15, 15) # IST — be flat by here
SKIP_OPEN_UNTIL  = dtime(9, 20)  # skip the first 5 min auction noise

# A focused, liquid default watchlist (intraday needs tight spreads).
DEFAULT_WATCHLIST = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "SBIN", "AXISBANK",
    "KOTAKBANK", "LT", "BHARTIARTL", "ITC", "TATAMOTORS", "TATASTEEL",
    "MARUTI", "BAJFINANCE", "HINDUNILVR", "ADANIENT", "ADANIPORTS",
    "SUNPHARMA", "WIPRO",
]


# Yahoo tickers for indices differ from the ".NS" stock convention.
INDEX_YF = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK",
            "FINNIFTY": "^CNXFIN", "MIDCPNIFTY": "^NSEMDCP50", "SENSEX": "^BSESN"}


# ── Data ──────────────────────────────────────────────
def fetch_intraday(symbol, interval, period="5d"):
    yf_sym = INDEX_YF.get(symbol, f"{symbol}.NS")
    try:
        df = yf.Ticker(yf_sym).history(period=period, interval=interval)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df = df.rename(columns=str.title)
    # Localise index to IST for the time-of-day guards.
    try:
        df.index = df.index.tz_convert("Asia/Kolkata")
    except Exception:
        pass
    return df


# ── Indicators ────────────────────────────────────────
def _vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    day = df.index.date
    pv = (tp * df["Volume"]).groupby(day).cumsum()
    vv = df["Volume"].groupby(day).cumsum().replace(0, np.nan)
    return pv / vv


def _atr(df, period=ATR_PERIOD):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _rsi(close, period=14):
    d = close.diff()
    g = d.clip(lower=0).rolling(period).mean()
    los = (-d.clip(upper=0)).rolling(period).mean()
    return 100 - 100 / (1 + g / los)


def _ema(s, span):
    return s.ewm(span=span).mean()


def _opening_range(df):
    """High/low of the first OR_BARS bars of the LATEST trading day."""
    last_day = df.index[-1].date()
    today = df[df.index.map(lambda x: x.date() == last_day)]
    if len(today) < OR_BARS:
        return None
    o = today.iloc[:OR_BARS]
    return {"high": float(o["High"].max()), "low": float(o["Low"].min()),
            "session": today}


# ── 15m trend filter ──────────────────────────────────
def trend_15m(df15):
    if df15 is None or len(df15) < 25:
        return "unknown"
    c = df15["Close"]
    e20 = _ema(c, 20)
    price = float(c.iloc[-1])
    slope = float(e20.iloc[-1] - e20.iloc[-4])
    if price > e20.iloc[-1] and slope > 0:
        return "up"
    if price < e20.iloc[-1] and slope < 0:
        return "down"
    return "flat"


# ── Setup detectors (operate on the latest 5m bar) ────
def _detect(symbol, df5, df15):
    if df5 is None or len(df5) < 30:
        return None
    df5 = df5.copy()
    df5["vwap"] = _vwap(df5)
    df5["atr"]  = _atr(df5)
    df5["ema9"] = _ema(df5["Close"], 9)
    df5["ema20"]= _ema(df5["Close"], 20)
    df5["rsi"]  = _rsi(df5["Close"])
    df5["rvol"] = df5["Volume"] / df5["Volume"].rolling(20).mean()

    bar  = df5.iloc[-1]
    prev = df5.iloc[-2]
    px   = float(bar["Close"])
    atr  = float(bar["atr"])
    if not np.isfinite(atr) or atr <= 0:
        return None

    t = df5.index[-1].time()
    trend = trend_15m(df15)
    orng  = _opening_range(df5)
    rvol  = float(bar["rvol"]) if np.isfinite(bar["rvol"]) else 0.0
    vwap  = float(bar["vwap"])

    sig = None   # (setup, side, reason)

    # 1) Opening-range breakout, aligned with the 15m trend.
    if orng:
        if (px > orng["high"] and prev["Close"] <= orng["high"]
                and trend == "up" and rvol >= MIN_RVOL):
            sig = ("ORB", "long", "5m close breaks opening-range high on volume")
        elif (px < orng["low"] and prev["Close"] >= orng["low"]
                and trend == "down" and rvol >= MIN_RVOL):
            sig = ("ORB", "short", "5m close breaks opening-range low on volume")

    # 2) VWAP reclaim / rejection.
    if sig is None and np.isfinite(vwap):
        if (prev["Close"] < prev["vwap"] and px > vwap
                and trend in ("up", "flat") and rvol >= 1.0):
            sig = ("VWAP", "long", "reclaims VWAP from below")
        elif (prev["Close"] > prev["vwap"] and px < vwap
                and trend in ("down", "flat") and rvol >= 1.0):
            sig = ("VWAP", "short", "rejects VWAP from above")

    # 3) Trend pullback continuation (EMA9/EMA20).
    if sig is None:
        pulled_long = (px > bar["ema20"] and prev["Low"] <= prev["ema20"]
                       and bar["ema9"] > bar["ema20"])
        pulled_short = (px < bar["ema20"] and prev["High"] >= prev["ema20"]
                        and bar["ema9"] < bar["ema20"])
        if pulled_long and trend == "up":
            sig = ("MOM", "long", "pullback to EMA20 resumes in uptrend")
        elif pulled_short and trend == "down":
            sig = ("MOM", "short", "pullback to EMA20 resumes in downtrend")

    if sig is None:
        return None

    setup, side, reason = sig

    # ── Levels & sizing ──
    if side == "long":
        stop   = px - ATR_STOP_MULT * atr
        t1, t2 = px + (px - stop), px + 2 * (px - stop)
    else:
        stop   = px + ATR_STOP_MULT * atr
        t1, t2 = px - (stop - px), px - 2 * (stop - px)

    risk_per_share = abs(px - stop)
    if risk_per_share <= 0:
        return None
    qty = int((CAPITAL * RISK_PCT) / risk_per_share)
    cap_qty = int((CAPITAL * MAX_POSITION_PCT) / px)
    capped = qty > cap_qty
    qty = min(qty, cap_qty)

    # ── Confluence grade ──
    score = 0
    score += 30 if (setup == "ORB") else 20 if setup == "VWAP" else 15
    score += 25 if ((side == "long" and trend == "up") or
                    (side == "short" and trend == "down")) else 0
    score += min(20, int((rvol - 1) * 20))            # volume conviction
    rsi = float(bar["rsi"]) if np.isfinite(bar["rsi"]) else 50
    score += 10 if (side == "long" and 50 <= rsi <= 70) or \
                   (side == "short" and 30 <= rsi <= 50) else 0
    score += 15 if (side == "long" and px > vwap) or \
                   (side == "short" and px < vwap) else 0
    score = max(0, min(100, score))
    grade = "A" if score >= 75 else "B" if score >= 60 else "C"

    # ── Time guards ──
    tradeable = True
    note = ""
    if t < SKIP_OPEN_UNTIL:
        tradeable, note = False, "wait — first 5 min auction"
    elif t >= NO_NEW_ENTRY:
        tradeable, note = False, "too late — no new intraday entry after 14:45"

    return {
        "symbol": symbol, "setup": setup, "side": side, "grade": grade,
        "score": score, "price": round(px, 2),
        "entry": round(px, 2), "stop": round(stop, 2),
        "t1": round(t1, 2), "t2": round(t2, 2),
        "rr": 2.0, "qty": qty, "capital_at_risk": round(qty * risk_per_share),
        "rvol": round(rvol, 2), "rsi": round(rsi, 1),
        "trend_15m": trend, "vwap": round(vwap, 2),
        "bar_time": df5.index[-1].strftime("%Y-%m-%d %H:%M"),
        "tradeable": tradeable, "note": note, "reason": reason,
        "capped": capped,
    }


# ── Scanner ───────────────────────────────────────────
def scan(symbols=None):
    symbols = symbols or DEFAULT_WATCHLIST
    print("=" * 72)
    print("  INTRADAY SIGNAL ENGINE  (5m setups · 15m trend filter)")
    print("  ⚠ Rule-based, NOT a validated-edge model. Trade small, use stops.")
    print("=" * 72)

    calls = []
    for sym in symbols:
        df5  = fetch_intraday(sym, "5m",  period="5d")
        df15 = fetch_intraday(sym, "15m", period="10d")
        s = _detect(sym, df5, df15)
        if s:
            calls.append(s)

    longs  = sorted([c for c in calls if c["side"] == "long"],
                    key=lambda c: c["score"], reverse=True)
    shorts = sorted([c for c in calls if c["side"] == "short"],
                    key=lambda c: c["score"], reverse=True)

    _print(" 🟢 LONG setups", longs)
    _print(" 🔴 SHORT setups", shorts)

    if calls:
        bt = calls[0]["bar_time"]
        print(f"\n  Signals as of last bar: {bt} IST  "
              f"(run live during market hours for fresh calls)")
    else:
        print("\n  No textbook intraday setup on the watchlist right now.")
    return calls


def _liquid_universe(max_symbols=120, min_turnover_cr=5.0, days=30):
    """Pick the most liquid names from the daily history (tight intraday
    spreads need real turnover). Returns symbols sorted by turnover desc."""
    from models.cross_sectional import load_prices
    prices = load_prices()
    rows = []
    for sym, df in prices.items():
        recent = df.tail(days)
        if len(recent) < days:
            continue
        turn_cr = float((recent["Close"] * recent["Volume"]).median()) / 1e7
        if turn_cr >= min_turnover_cr:
            rows.append((sym, turn_cr))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in rows[:max_symbols]]


def scan_buckets(symbols=None, max_symbols=120):
    """Scan a liquid universe and group the intraday calls into price bands:
    ₹100–500, ₹500–1000, ₹1000+."""
    symbols = symbols or _liquid_universe(max_symbols=max_symbols)
    print("=" * 72)
    print("  INTRADAY CALLS BY PRICE BAND  (5m setups · 15m trend filter)")
    print(f"  Scanning {len(symbols)} liquid names")
    print("  ⚠ Rule-based, NOT a validated-edge model. Trade small, use stops.")
    print("=" * 72)

    calls = []
    for sym in symbols:
        df5  = fetch_intraday(sym, "5m",  period="5d")
        df15 = fetch_intraday(sym, "15m", period="10d")
        s = _detect(sym, df5, df15)
        if s:
            calls.append(s)

    bands = [("₹100–500 stocks", 100, 500),
             ("₹500–1000 stocks", 500, 1000),
             ("Above ₹1000 stocks", 1000, float("inf"))]
    for title, lo, hi in bands:
        sel = [c for c in calls if lo <= c["price"] < hi]
        sel.sort(key=lambda c: c["score"], reverse=True)
        print(f"\n  {'═' * 68}")
        print(f"  📊 {title}   ({len(sel)} setup{'s' if len(sel)!=1 else ''})")
        print(f"  {'═' * 68}")
        _print(" 🟢 LONG", [c for c in sel if c["side"] == "long"])
        _print(" 🔴 SHORT", [c for c in sel if c["side"] == "short"])

    if calls:
        print(f"\n  Signals as of last bar: {calls[0]['bar_time']} IST  "
              f"(run live during market hours for fresh calls)")
    else:
        print("\n  No textbook intraday setup across the universe right now.")
    return calls


def _print(title, rows):
    print(f"\n  {title}")
    print("  " + "─" * 68)
    if not rows:
        print("     (none)")
        return
    print(f"  {'GRD':<4}{'SCR':>4} {'SYMBOL':<12}{'SETUP':>6}{'ENTRY':>9}"
          f"{'STOP':>9}{'T1':>9}{'T2':>9}{'QTY':>6} {'15mTR':>6} {'NOTE'}")
    for c in rows:
        flag = "" if c["tradeable"] else f"  ⏸ {c['note']}"
        print(f"  {c['grade']:<4}{c['score']:>4} {c['symbol']:<12}"
              f"{c['setup']:>6}{c['entry']:>9.1f}{c['stop']:>9.1f}"
              f"{c['t1']:>9.1f}{c['t2']:>9.1f}{c['qty']:>6} "
              f"{c['trend_15m']:>6}{flag}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--buckets":
        scan_buckets()
    elif args:
        scan(args)
    else:
        scan_buckets()

"""
Price-action pattern detector (goal #3) — sharper ENTRY TIMING.

The ranker says WHICH stock; the fundamentals say it's a QUALITY business;
this module answers WHEN to pull the trigger by reading the candles the way
a discretionary trader does. It recognises the 15 classic price-action
patterns and, crucially, reads them IN CONTEXT (a hammer only counts after a
pulldown; an engulfing only counts at the right location), then decides
whether a clean entry trigger has actually fired on the most recent bars.

The 28 patterns
---------------
Single  : 1 Hammer  2 Inverted Hammer  3 Shooting Star  4 Hanging Man
          5 Doji  6 Dragonfly Doji  7 Gravestone Doji  8 Long-legged Doji
          9 Marubozu  10 Spinning Top
          11 Bullish Belt Hold  12 Bearish Belt Hold
Double  : 13 Bullish Engulfing  14 Bearish Engulfing  15 Bullish Harami
          16 Bearish Harami  17 Piercing Line  18 Dark Cloud Cover
          19 Tweezer Top  20 Tweezer Bottom
          21 Bullish Kicker  22 Bearish Kicker
Triple  : 23 Morning Star  24 Evening Star
          25 Three White Soldiers  26 Three Black Crows
          27 Three Inside Up  28 Three Inside Down
          29 Bullish Abandoned Baby  30 Bearish Abandoned Baby

Output (detect_patterns):
    {
      "patterns":      [ {name, direction, strength, bars_ago}, ... ],
      "pattern_score": float in [-1, +1],   # net, recency-weighted
      "primary":       "Bullish Engulfing" | None,
      "context":       "pullback_to_support" | "breakout" | "extended" | "mid",
      "entry_trigger": "long" | "short" | None,  # a clean trigger fired NOW
    }
"""

import numpy as np
import pandas as pd

# How many of the most-recent bars can still count as a fresh signal.
RECENCY_BARS = 2
DOJI_BODY    = 0.10   # body <= 10% of range  → doji
MARUBOZU     = 0.90   # body >= 90% of range  → marubozu
LONG_WICK    = 2.0    # wick >= 2x body        → hammer/star family


# ── candle geometry ───────────────────────────────────
def _parts(o, h, l, c):
    body  = abs(c - o)
    rng   = (h - l) or 1e-9
    upper = h - max(o, c)
    lower = min(o, c) - l
    return body, rng, upper, lower


def _bull(o, c):   # green candle
    return c > o


# ── trend context BEFORE the signal candle ────────────
def _prior_trend(closes):
    """Direction of the 5 bars leading INTO the signal (not incl. last)."""
    seg = closes[-6:-1]
    if len(seg) < 3:
        return "flat"
    slope = (seg[-1] - seg[0]) / (abs(seg[0]) + 1e-9)
    if slope < -0.01:
        return "down"
    if slope > 0.01:
        return "up"
    return "flat"


# ── individual pattern tests (evaluated at index i = last bar) ──
def _single(o, h, l, c, prior):
    body, rng, up, lo = _parts(o, h, l, c)
    out = []
    # Doji — indecision; bias by context
    if body <= DOJI_BODY * rng:
        d = 1 if prior == "down" else (-1 if prior == "up" else 0)
        # Dragonfly Doji — long lower shadow, no upper shadow
        if lo >= 2 * rng * DOJI_BODY and up <= rng * DOJI_BODY:
            out.append(("Dragonfly Doji", 1 if prior == "down" else 0, 0.65))
        # Gravestone Doji — long upper shadow, no lower shadow
        elif up >= 2 * rng * DOJI_BODY and lo <= rng * DOJI_BODY:
            out.append(("Gravestone Doji", -1 if prior == "up" else 0, 0.65))
        # Long-legged Doji — long shadows both sides
        elif up >= rng * 0.3 and lo >= rng * 0.3:
            out.append(("Long-legged Doji", d, 0.4))
        else:
            out.append(("Doji", d, 0.3))
        return out
    # Marubozu — strong continuation
    if body >= MARUBOZU * rng:
        out.append(("Marubozu", 1 if _bull(o, c) else -1, 0.7))
        return out
    # Spinning Top — small body, shadows on both sides
    if body <= 0.3 * rng and up >= 0.3 * rng and lo >= 0.3 * rng:
        out.append(("Spinning Top", 0, 0.2))
    # Long-lower-wick family
    if lo >= LONG_WICK * body and up <= body:
        if prior == "down":
            out.append(("Hammer", 1, 0.7))           # bullish reversal
        elif prior == "up":
            out.append(("Hanging Man", -1, 0.6))      # bearish reversal
    # Long-upper-wick family
    if up >= LONG_WICK * body and lo <= body:
        if prior == "up":
            out.append(("Shooting Star", -1, 0.7))    # bearish reversal
        elif prior == "down":
            out.append(("Inverted Hammer", 1, 0.6))   # bullish reversal
    # Belt Hold — opens at extreme, closes near opposite extreme
    if _bull(o, c) and lo <= 0.05 * rng and body >= 0.6 * rng and prior == "down":
        out.append(("Bullish Belt Hold", 1, 0.55))
    elif not _bull(o, c) and up <= 0.05 * rng and body >= 0.6 * rng and prior == "up":
        out.append(("Bearish Belt Hold", -1, 0.55))
    return out


def _double(po, ph, pl, pc, o, h, l, c, prior):
    out = []
    pbody = abs(pc - po)
    body  = abs(c - o)
    # Engulfing
    if (not _bull(po, pc)) and _bull(o, c) and o <= pc and c >= po and body > pbody:
        out.append(("Bullish Engulfing", 1, 0.85))
    if _bull(po, pc) and (not _bull(o, c)) and o >= pc and c <= po and body > pbody:
        out.append(("Bearish Engulfing", -1, 0.85))
    # Harami (small body inside the prior big body)
    if (not _bull(po, pc)) and _bull(o, c) and o >= pc and c <= po and body < pbody:
        out.append(("Bullish Harami", 1, 0.55))
    if _bull(po, pc) and (not _bull(o, c)) and o <= pc and c >= po and body < pbody:
        out.append(("Bearish Harami", -1, 0.55))
    # Piercing Line / Dark Cloud Cover (need real prior body)
    if pbody > 0:
        pmid = (po + pc) / 2
        if (not _bull(po, pc)) and _bull(o, c) and o < pl and pc < c < po and c > pmid:
            out.append(("Piercing Line", 1, 0.7))
        if _bull(po, pc) and (not _bull(o, c)) and o > ph and po < c < pc and c < pmid:
            out.append(("Dark Cloud Cover", -1, 0.7))
    # Tweezer Top / Bottom — same high or same low
    prng = (ph - pl) or 1e-9
    rng = (h - l) or 1e-9
    if abs(pl - l) <= 0.002 * l and prior == "down":
        out.append(("Tweezer Bottom", 1, 0.6))
    if abs(ph - h) <= 0.002 * h and prior == "up":
        out.append(("Tweezer Top", -1, 0.6))
    # Kicker — gap + opposite direction (very strong)
    if (not _bull(po, pc)) and _bull(o, c) and o > po:
        out.append(("Bullish Kicker", 1, 0.9))
    if _bull(po, pc) and (not _bull(o, c)) and o < po:
        out.append(("Bearish Kicker", -1, 0.9))
    return out


def _triple(c2o, c2h, c2l, c2c, c1o, c1h, c1l, c1c, o, h, l, c):
    out = []
    b2 = abs(c2c - c2o); b1 = abs(c1c - c1o); b0 = abs(c - o)
    rng2 = (c2h - c2l) or 1e-9
    # Morning / Evening star: big body, small body, big opposite body
    small1 = b1 <= 0.5 * b2
    if (not _bull(c2o, c2c)) and small1 and _bull(o, c) and c > (c2o + c2c) / 2:
        out.append(("Morning Star", 1, 0.9))
    if _bull(c2o, c2c) and small1 and (not _bull(o, c)) and c < (c2o + c2c) / 2:
        out.append(("Evening Star", -1, 0.9))
    # Three soldiers / crows
    ups   = _bull(c2o, c2c) and _bull(c1o, c1c) and _bull(o, c)
    downs = (not _bull(c2o, c2c)) and (not _bull(c1o, c1c)) and (not _bull(o, c))
    if ups and c1c > c2c and c > c1c:
        out.append(("Three White Soldiers", 1, 0.85))
    if downs and c1c < c2c and c < c1c:
        out.append(("Three Black Crows", -1, 0.85))
    # Three Inside Up / Down (harami + confirmation)
    if (not _bull(c2o, c2c)) and _bull(c1o, c1c) and c1o >= c2c and c1c <= c2o:
        if _bull(o, c) and c > c2o:
            out.append(("Three Inside Up", 1, 0.8))
    if _bull(c2o, c2c) and (not _bull(c1o, c1c)) and c1o <= c2c and c1c >= c2o:
        if (not _bull(o, c)) and c < c2o:
            out.append(("Three Inside Down", -1, 0.8))
    # Abandoned Baby (gap + doji + gap)
    if (not _bull(c2o, c2c)) and b1 <= 0.1 * (c1h - c1l + 1e-9):
        if c1h < c2l and o > c1h and _bull(o, c):
            out.append(("Bullish Abandoned Baby", 1, 0.9))
    if _bull(c2o, c2c) and b1 <= 0.1 * (c1h - c1l + 1e-9):
        if c1l > c2h and o < c1l and (not _bull(o, c)):
            out.append(("Bearish Abandoned Baby", -1, 0.9))
    return out


# ── structural context: where is price? ───────────────
def _context(df):
    close = df["Close"]
    price = float(close.iloc[-1])
    ema20 = float(close.ewm(span=20).mean().iloc[-1])
    atr   = _atr(df)
    hi20  = float(df["High"].iloc[-21:-1].max()) if len(df) > 21 else price
    # near EMA20 from above = healthy pullback-to-support
    if atr > 0 and 0 <= (price - ema20) <= 0.8 * atr:
        return "pullback_to_support"
    if price >= hi20:
        return "breakout"
    if atr > 0 and (price - ema20) > 3 * atr:
        return "extended"
    return "mid"


def _atr(df, period=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    v = tr.rolling(period).mean().iloc[-1]
    return float(v) if v == v else 0.0


# ── main entry point ──────────────────────────────────
def detect_patterns(df):
    """Scan the most recent RECENCY_BARS candles for the 15 patterns,
    in trend/location context, and decide if a clean entry trigger fired."""
    if df is None or len(df) < 12:
        return {"patterns": [], "pattern_score": 0.0, "primary": None,
                "context": "mid", "entry_trigger": None}

    o = df["Open"].values; h = df["High"].values
    l = df["Low"].values;  c = df["Close"].values
    closes = c.copy()
    n = len(df)

    found = []   # (name, direction, strength, bars_ago)
    for bars_ago in range(RECENCY_BARS):
        i = n - 1 - bars_ago
        if i < 4:
            break
        prior = _prior_trend(closes[:i + 1])
        hits  = []
        hits += _single(o[i], h[i], l[i], c[i], prior)
        hits += _double(o[i-1], h[i-1], l[i-1], c[i-1],
                        o[i], h[i], l[i], c[i], prior)
        hits += _triple(o[i-2], h[i-2], l[i-2], c[i-2],
                        o[i-1], h[i-1], l[i-1], c[i-1],
                        o[i], h[i], l[i], c[i])
        for name, direction, strength in hits:
            found.append({"name": name, "direction": int(direction),
                          "strength": float(strength), "bars_ago": bars_ago})

    # Recency-weighted net score (newer bars count more).
    score = 0.0
    for p in found:
        recency = 1.0 if p["bars_ago"] == 0 else 0.6
        score += p["direction"] * p["strength"] * recency
    score = float(np.clip(score, -1.0, 1.0))

    # Strongest, most-recent directional pattern is the headline.
    directional = [p for p in found if p["direction"] != 0]
    primary = None
    if directional:
        directional.sort(key=lambda p: (p["bars_ago"], -p["strength"]))
        primary = directional[0]["name"]

    context = _context(df)

    # A clean entry trigger needs BOTH a fresh strong pattern AND a sensible
    # location — that is the whole point of "sharper entry timing".
    trigger = None
    fresh = [p for p in found if p["bars_ago"] == 0 and p["strength"] >= 0.6]
    if fresh:
        net = sum(p["direction"] * p["strength"] for p in fresh)
        if net > 0 and context in ("pullback_to_support", "breakout", "mid"):
            trigger = "long"
        elif net < 0 and context in ("extended", "mid", "breakout"):
            trigger = "short"

    return {"patterns": found, "pattern_score": round(score, 3),
            "primary": primary, "context": context, "entry_trigger": trigger}


if __name__ == "__main__":
    import os, glob
    HIST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "historical")
    files = sorted(glob.glob(os.path.join(HIST, "*.csv")))[:25]
    print(f"  Scanning {len(files)} symbols for fresh price-action triggers...\n")
    for path in files:
        sym = os.path.basename(path).replace(".csv", "")
        try:
            df = pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()
        except Exception:
            continue
        r = detect_patterns(df)
        if r["primary"]:
            trig = r["entry_trigger"] or "-"
            print(f"  {sym:<14} score={r['pattern_score']:+.2f}  "
                  f"primary={r['primary']:<22} ctx={r['context']:<20} "
                  f"trigger={trig}")

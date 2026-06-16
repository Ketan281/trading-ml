"""
4-class market-regime classifier — bull / bear / sideways / volatile.

The earlier market_regime.py gave a coarse risk-on/off exposure number. This
classifies the market into the four regimes a discretionary trader actually
names, by combining NIFTY trend + momentum + drawdown + realised-vol percentile
with the breadth read (participation confirms the regime):

  • VOLATILE : realised vol in its top percentile — size down, widen stops,
               regardless of direction (vol dominates the playbook)
  • BULL     : above the 200-DMA, positive trend, broad participation
  • BEAR     : below the 200-DMA, negative trend / deep drawdown, weak breadth
  • SIDEWAYS : no decisive trend — range / chop

Output: the label, a confidence, the components, and a playbook hint. Designed
to feed position sizing, the portfolio risk manager, and the ensemble.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipelines.breadth import breadth_read

TRADING_DAYS = 252
VOL_HIGH_PCTL = 80.0

HIST_DIR = os.path.join(ROOT, "data", "historical")
DATA_DIR = os.path.join(ROOT, "data")


def _load_index(symbol):
    """Merge data/historical/<SYM>.csv and data/<SYM>_daily.csv into the
    longest up-to-date daily series (the historical file alone ends in 2024)."""
    frames = []
    for path in (os.path.join(HIST_DIR, f"{symbol}.csv"),
                 os.path.join(DATA_DIR, f"{symbol}_daily.csv")):
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df.columns = [c.title() for c in df.columns]
                dcol = "Date" if "Date" in df.columns else df.columns[0]
                df[dcol] = pd.to_datetime(df[dcol])
                frames.append(df.rename(columns={dcol: "Date"}).set_index("Date"))
            except Exception:
                pass
    if not frames:
        return None
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.dropna(subset=["Close"]) if "Close" in out.columns else None


def _components(df):
    c = df["Close"]
    price = float(c.iloc[-1])
    ma50  = float(c.rolling(50).mean().iloc[-1])
    ma200 = float(c.rolling(200).mean().iloc[-1])
    slope50 = float(c.rolling(50).mean().iloc[-1] - c.rolling(50).mean().iloc[-11])
    ret_20 = float(c.iloc[-1] / c.iloc[-21] - 1)
    ret_60 = float(c.iloc[-1] / c.iloc[-61] - 1)
    hi_1y  = float(c.iloc[-252:].max())
    dd     = price / hi_1y - 1

    rets = c.pct_change()
    vol20 = rets.rolling(20).std() * np.sqrt(TRADING_DAYS)
    v_now = float(vol20.iloc[-1])
    v_1y  = vol20.iloc[-252:].dropna()
    v_pct = float((v_1y < v_now).mean() * 100) if len(v_1y) else 50.0

    return {"price": price, "ma50": ma50, "ma200": ma200, "slope50": slope50,
            "ret_20": ret_20, "ret_60": ret_60, "drawdown": dd,
            "vol_now": round(v_now * 100, 1), "vol_pctile": round(v_pct, 1)}


def classify(symbol="NIFTY", use_breadth=True):
    df = _load_index(symbol)
    if df is None or len(df) < 260:
        return {"regime": "unknown", "reason": "insufficient index history"}
    k = _components(df)

    breadth = breadth_read() if use_breadth else None
    b_a200 = breadth["pct_above_200dma"] if breadth else None

    above_trend = k["price"] > k["ma200"]
    up = k["price"] > k["ma50"] and k["slope50"] > 0 and k["ret_60"] > 0
    down = k["price"] < k["ma50"] and k["slope50"] < 0 and k["ret_60"] < 0
    volatile = k["vol_pctile"] >= VOL_HIGH_PCTL

    # ── Decide ───────────────────────────────────────────
    if volatile:
        direction = "up" if above_trend else "down"
        regime = "volatile"
        hint = ("high-volatility regime — cut size, widen stops, favour "
                f"defined-risk; underlying tilt {direction}")
    elif above_trend and up and (b_a200 is None or b_a200 >= 45):
        regime = "bull"
        hint = "trend-following longs, buy dips, let winners run"
    elif (not above_trend) and (down or k["drawdown"] < -0.12) and \
         (b_a200 is None or b_a200 <= 55):
        regime = "bear"
        hint = "capital preservation — reduce gross, hedge, only A+ setups"
    else:
        regime = "sideways"
        hint = "range/chop — fade extremes, smaller size, mean-reversion"

    # Confidence from how decisively the components agree.
    conf = 0.4
    conf += 0.2 if (up or down) else 0
    conf += 0.2 if abs(k["ret_60"]) > 0.05 else 0
    conf += 0.2 if volatile or abs(k["drawdown"]) > 0.10 else 0.1
    conf = round(min(0.95, conf), 2)

    return {
        "regime": regime, "confidence": conf, "hint": hint,
        "symbol": symbol, "as_of": str(df.index[-1].date()),
        "price_vs_200dma_pct": round((k["price"] / k["ma200"] - 1) * 100, 2),
        "ret_60d_pct": round(k["ret_60"] * 100, 2),
        "drawdown_pct": round(k["drawdown"] * 100, 2),
        "vol_pctile": k["vol_pctile"],
        "breadth_pct_above_200dma": b_a200,
    }


if __name__ == "__main__":
    for sym in (sys.argv[1:] or ["NIFTY"]):
        r = classify(sym)
        print("=" * 60)
        print(f"  MARKET REGIME — {sym}  ({r.get('as_of','?')})")
        print("=" * 60)
        print(f"  Regime        : {r['regime'].upper()}  (conf {r.get('confidence')})")
        print(f"  Playbook      : {r.get('hint','')}")
        if "ret_60d_pct" in r:
            print(f"  vs 200-DMA    : {r['price_vs_200dma_pct']:+}%  | "
                  f"60d {r['ret_60d_pct']:+}%  | dd {r['drawdown_pct']}%")
            print(f"  Vol percentile: {r['vol_pctile']}  | "
                  f"breadth>200DMA {r['breadth_pct_above_200dma']}%")

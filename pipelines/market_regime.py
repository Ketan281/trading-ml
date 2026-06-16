"""
Market-regime overlay for the cross-sectional book — roadmap #5 (part A).

models/regime_detector.py already reads a rich, LIVE per-instrument regime for
the OPTIONS engine (trend/vol/expiry/breadth, needs NSELive). This module is
different and complementary: a lightweight, OFFLINE read of the BROAD market
(the NIFTY index) whose only job is to answer one question for the equity
book —

        "How risk-on should the long book be RIGHT NOW?"

A long-only relative-strength book still carries full market beta: in the
2018/2020/2022 drawdowns the ranker's picks fell with everything else. A
regime filter that cuts gross exposure when the index is below trend and
volatility is spiking is the single cheapest way to tame that −41% drawdown.

Signal (all point-in-time, computable from index history alone):
  • trend      : index vs its 200-day MA (above = risk-on)
  • drawdown   : distance from the trailing 1-year high
  • volatility : 20-day realised vol percentile over the last year
Combined into a regime label and an EXPOSURE MULTIPLIER in [0, 1].

This is a SIGNAL provider. It does not silently resize anything; the screener
/ backtest decide whether to apply the multiplier (and the backtest is where
we prove it actually helps before trusting it live).
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST_DIR = os.path.join(ROOT, "data", "historical")
DATA_DIR = os.path.join(ROOT, "data")

# Exposure ladder. Deliberately coarse — regime timing is a blunt tool and
# over-tuning these invites curve-fitting.
VOL_HIGH_PCTL = 80.0      # realised-vol percentile above which we de-risk
DD_RISK_OFF   = -0.10     # >10% below the 1-yr high = caution


def _load_index(symbol="NIFTY"):
    for path in (os.path.join(HIST_DIR, f"{symbol}.csv"),
                 os.path.join(DATA_DIR, f"{symbol}_daily.csv")):
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, index_col="Date", parse_dates=True)
                if "Close" in df.columns and len(df) > 260:
                    return df.sort_index()
            except Exception:
                pass
    return None


def regime_at(df, as_of=None):
    """Point-in-time regime read from index history up to `as_of`
    (inclusive). Returns label, exposure multiplier, and the components."""
    hist = df.loc[:as_of] if as_of is not None else df
    if len(hist) < 260:
        return {"regime": "unknown", "exposure": 1.0,
                "reason": "insufficient history"}

    close = hist["Close"]
    price = float(close.iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1])
    hi_1y = float(close.iloc[-252:].max())
    dd    = price / hi_1y - 1.0

    rets  = close.pct_change()
    vol20 = rets.rolling(20).std() * np.sqrt(252)
    v_now = float(vol20.iloc[-1])
    v_1y  = vol20.iloc[-252:].dropna()
    v_pct = float((v_1y < v_now).mean() * 100) if len(v_1y) else 50.0

    above_trend = price > ma200

    # ── Classify + exposure ladder ───────────────────────
    if above_trend and v_pct < VOL_HIGH_PCTL and dd > DD_RISK_OFF:
        regime, exposure = "risk_on", 1.0
    elif above_trend and (v_pct >= VOL_HIGH_PCTL or dd <= DD_RISK_OFF):
        regime, exposure = "cautious_uptrend", 0.6
    elif not above_trend and dd > -0.20:
        regime, exposure = "risk_off", 0.3
    else:
        regime, exposure = "deep_risk_off", 0.0

    return {
        "regime":        regime,
        "exposure":      exposure,
        "above_trend":   bool(above_trend),
        "price_vs_ma200": round((price / ma200 - 1) * 100, 2),
        "drawdown_pct":  round(dd * 100, 2),
        "vol_pctile":    round(v_pct, 1),
        "reason":        f"{'above' if above_trend else 'below'} 200dMA, "
                         f"vol {v_pct:.0f}pct, dd {dd*100:.0f}%",
    }


def current_regime(symbol="NIFTY"):
    df = _load_index(symbol)
    if df is None:
        return {"regime": "unknown", "exposure": 1.0,
                "reason": f"no index history for {symbol}"}
    return regime_at(df)


if __name__ == "__main__":
    r = current_regime("NIFTY")
    print("=" * 56)
    print("  MARKET REGIME (NIFTY) — equity-book exposure overlay")
    print("=" * 56)
    print(f"  Regime          : {r['regime'].upper()}")
    print(f"  Exposure mult.  : {r['exposure']:.2f}  (of full gross)")
    if "price_vs_ma200" in r:
        print(f"  Price vs 200dMA : {r['price_vs_ma200']:+.1f}%")
        print(f"  Drawdown (1yr)  : {r['drawdown_pct']:+.1f}%")
        print(f"  Vol percentile  : {r['vol_pctile']:.0f}")
    print(f"  Reason          : {r['reason']}")

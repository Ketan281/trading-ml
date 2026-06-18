"""
Market data helpers for the frontend: candlesticks + today's best recommendation.

Both reuse existing engines:
  • candles()        → pipelines.intraday.fetch_intraday (yfinance OHLCV)
  • recommendation() → aos.sim_wallet.pick_trade (the same disciplined daily
    picker the autonomous wallet uses), reshaped into a ready-to-submit trade
    spec so the UI can offer one-click "Take this trade".
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# yfinance intraday history only reliably covers the last ~60 days; map a few
# friendly ranges to (yf interval, yf period).
_RANGES = {
    "1d":  ("5m", "1d"),
    "5d":  ("15m", "5d"),
    "1mo": ("1h", "1mo"),
}


def candles(symbol, interval=None, period=None):
    """Return OHLCV candles for lightweight-charts: a list of
    {time(epoch s, UTC), open, high, low, close, volume}. `interval`/`period`
    override the default 5m/1d intraday view."""
    from pipelines.intraday import fetch_intraday
    interval = interval or "5m"
    period = period or "1d"
    df = fetch_intraday(symbol.upper(), interval, period)
    if df is None or len(df) == 0:
        return {"symbol": symbol.upper(), "interval": interval, "candles": [],
                "error": "no data (market closed or symbol unavailable)"}
    out = []
    for ts, row in df.iterrows():
        try:
            t = int(ts.timestamp())
        except Exception:
            continue
        out.append({"time": t, "open": round(float(row["Open"]), 2),
                    "high": round(float(row["High"]), 2),
                    "low": round(float(row["Low"]), 2),
                    "close": round(float(row["Close"]), 2),
                    "volume": int(row["Volume"]) if row["Volume"] == row["Volume"] else 0})
    return {"symbol": symbol.upper(), "interval": interval, "period": period,
            "candles": out}


def recommendation(balance=10_000):
    """Today's single best disciplined trade, as text + a submit-ready spec."""
    from aos.sim_wallet import pick_trade
    plan = pick_trade(balance)
    if not plan:
        return {"answer": "No clean trade today — no affordable index-option lot and "
                "no high-quality equity setup. Capital preserved.",
                "spec": None, "plan": None}
    if plan["kind"] == "option":
        spec = {"segment": "options", "underlying": plan["underlying"],
                "strike": plan["strike"], "leg": plan["leg"]}
        chart = plan["underlying"]
    else:
        spec = {"segment": "equity", "symbol": plan["symbol"],
                "entry": plan["entry"], "stop": plan["stop"], "target": plan["target"]}
        chart = plan["symbol"]
    answer = (f"Best idea today: {plan['symbol']} — entry ≈ ₹{plan['entry']}, "
              f"stop ₹{plan['stop']}, target ₹{plan['target']}. {plan['reason']} "
              f"⚠ Paper-trading, rule-based read — no profit guarantee.")
    return {"answer": answer, "spec": spec, "chart_symbol": chart, "plan": plan}


if __name__ == "__main__":
    import json
    c = candles("NIFTY")
    print("candles:", len(c.get("candles", [])), "| first:",
          c["candles"][0] if c.get("candles") else c.get("error"))
    print("recommendation:", json.dumps(recommendation(), default=str)[:300])

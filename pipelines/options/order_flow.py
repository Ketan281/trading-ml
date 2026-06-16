"""
Order-flow imbalance — PROXY (we have no tick/L2 data).

True order-flow imbalance needs level-2 / tick data (bid-vs-ask aggressor
volume) which we do NOT have. This is an honest PROXY built from the two flow
signals we DO see:

  1. Option-chain flow : at each strike, signed by the option's price move —
     volume on rising CE = call buying (bullish), on rising PE = put buying
     (bearish). Net = a directional pressure read.
  2. Index tape flow    : on the collected 5-min bars, up-bar volume minus
     down-bar volume = crude buying/selling pressure.

Clearly labelled a proxy — it captures DIRECTION of pressure, not the precise
microstructure imbalance a real OFI feed would give. Don't trade it as if it
were L2.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from pipelines.options.chain_live_intel import fetch_chain
from pipelines.intraday import fetch_intraday


def chain_flow(symbol, chain=None):
    """Directional pressure from option volume signed by each option's move."""
    if chain is None:
        chain = fetch_chain(symbol)
    if not chain:
        return None
    df = chain["df"]
    # call buying (CE vol where CE price up) vs put buying (PE vol where PE price up)
    call_buy = float((df["ce_vol"] * (df["ce_chg"] > 0)).sum())
    call_sell = float((df["ce_vol"] * (df["ce_chg"] < 0)).sum())
    put_buy = float((df["pe_vol"] * (df["pe_chg"] > 0)).sum())
    put_sell = float((df["pe_vol"] * (df["pe_chg"] < 0)).sum())
    # bullish: call buying + put selling ; bearish: put buying + call selling
    bull = call_buy + put_sell
    bear = put_buy + call_sell
    tot = bull + bear + 1
    imbalance = (bull - bear) / tot          # -1..1
    return {"spot": chain["spot"], "call_buy": int(call_buy), "put_buy": int(put_buy),
            "imbalance": round(imbalance, 3),
            "read": ("bullish flow" if imbalance > 0.1 else
                     "bearish flow" if imbalance < -0.1 else "balanced flow")}


def tape_flow(symbol, bars=24):
    """Up-volume minus down-volume on recent 5-min bars (pressure proxy)."""
    df = fetch_intraday(symbol, "5m", period="2d")
    if df is None or len(df) < bars:
        return None
    d = df.tail(bars).copy()
    if float(d["Volume"].sum()) < 1:        # indices carry no intraday volume here
        return {"bars": bars, "imbalance": None,
                "read": "unavailable — this feed has no index intraday volume"}
    up = d["Close"] >= d["Open"]
    up_vol = float(d.loc[up, "Volume"].sum()); dn_vol = float(d.loc[~up, "Volume"].sum())
    tot = up_vol + dn_vol + 1
    imb = (up_vol - dn_vol) / tot
    return {"bars": bars, "up_vol": int(up_vol), "down_vol": int(dn_vol),
            "imbalance": round(imb, 3),
            "read": ("buying pressure" if imb > 0.1 else
                     "selling pressure" if imb < -0.1 else "balanced")}


def report(symbol):
    print("=" * 60)
    print(f"  ORDER-FLOW IMBALANCE (PROXY) — {symbol}")
    print("  ⚠ Proxy from chain+tape; not true L2/tick order flow")
    print("=" * 60)
    cf = chain_flow(symbol)
    if cf:
        print(f"  Chain flow : imbalance {cf['imbalance']:+}  → {cf['read']}")
        print(f"               call-buy {cf['call_buy']:,} vs put-buy {cf['put_buy']:,}")
    tf = tape_flow(symbol)
    if tf:
        print(f"  Tape flow  : imbalance {tf['imbalance']:+}  → {tf['read']}  "
              f"(last {tf['bars']} 5m bars)")
    return {"chain": cf, "tape": tf}


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["NIFTY", "BANKNIFTY"]):
        report(s); print()

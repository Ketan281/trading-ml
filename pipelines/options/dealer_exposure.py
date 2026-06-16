"""
Dealer gamma / vanna / charm exposure — the second-order positioning map.

Extends the gamma-regime read with the two greeks that drive index behaviour
near expiry and around vol shocks:

  • GAMMA  (∂Δ/∂S)   — long-gamma dealers suppress vol (range); short-gamma
                       chase moves (trend). [the regime read]
  • VANNA  (∂Δ/∂σ)   — how dealer hedging shifts when IV moves: positive vanna
                       exposure → a vol spike forces dealers to BUY (supportive),
                       negative → a vol spike forces selling (accelerant).
  • CHARM  (∂Δ/∂t)   — how dealer delta bleeds as time passes: large charm into
                       expiry creates the mechanical drift / pin toward big-OI
                       strikes on expiry day.

HONEST ASSUMPTION: dealer books are unobservable. We use the standard
positioning heuristic (long gamma where call OI dominates, short where put OI
dominates) and Black-Scholes greeks (the free feed has none). This is a regime
/ flow heuristic, NOT measured dealer inventory.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from pipelines.options.chain_live_intel import fetch_chain

RISK_FREE = 0.065
LOT = {"NIFTY": 75, "BANKNIFTY": 35}


def _greeks(S, K, dte, iv):
    """Return gamma, vanna, charm (per-share, leg-agnostic where symmetric)."""
    T = max(dte, 0.5) / 365.0
    sigma = (iv or 0) / 100.0
    if sigma <= 0 or S <= 0 or K <= 0:
        return 0.0, 0.0, 0.0
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (RISK_FREE + sigma ** 2 / 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    pdf = norm.pdf(d1)
    gamma = pdf / (S * sigma * sqrtT)
    vanna = -pdf * d2 / sigma                       # ∂Δ/∂σ
    charm = -pdf * (2 * RISK_FREE * T - d2 * sigma * sqrtT) / (2 * T * sigma * sqrtT)
    return gamma, vanna, charm


def dealer_exposure(chain):
    df, spot, sym = chain["df"], chain["spot"], chain["symbol"]
    dte, lot = chain["dte"], LOT[chain["symbol"]]
    scale = lot * spot ** 2 * 0.01
    g_tot = v_tot = c_tot = 0.0
    by = []
    for r in df.itertuples():
        gc, vc, cc = _greeks(spot, float(r.strike), dte, r.ce_iv)
        gp, vp, cp = _greeks(spot, float(r.strike), dte, r.pe_iv)
        # Dealer long gamma where calls dominate, short where puts dominate.
        g = (gc * r.ce_oi - gp * r.pe_oi) * scale
        v = (vc * r.ce_oi - vp * r.pe_oi) * scale
        ch = (cc * r.ce_oi - cp * r.pe_oi) * scale
        g_tot += g; v_tot += v; c_tot += ch
        by.append((int(r.strike), g, v, ch))
    bdf = pd.DataFrame(by, columns=["strike", "gamma", "vanna", "charm"])

    # Raw products mix units across greeks, so report a unit-free INTENSITY:
    # net / Σ|contributions| ∈ [-1, 1] = how one-sided the dealer book is.
    def intensity(col):
        denom = bdf[col].abs().sum()
        return round(float(bdf[col].sum() / denom), 2) if denom else 0.0

    g_i, v_i, c_i = intensity("gamma"), intensity("vanna"), intensity("charm")
    return {
        "symbol": sym, "spot": round(spot, 1), "dte": dte,
        "gamma_intensity": g_i, "vanna_intensity": v_i, "charm_intensity": c_i,
        "gamma_regime": ("long-gamma → RANGE (vol suppressed)" if g_i > 0
                         else "short-gamma → TREND (vol amplified)"),
        "vanna_read": ("positive — a vol spike forces dealer BUYING (supportive)"
                       if v_i > 0 else
                       "negative — a vol spike forces dealer SELLING (accelerant)"),
        "charm_read": (("strong charm — delta bleed pins/drifts price into expiry"
                        if abs(c_i) > 0.3 else "moderate charm into expiry")
                       if dte <= 2 else "charm minor (far from expiry)"),
        "peak_gamma_strike": int(bdf.loc[bdf["gamma"].abs().idxmax(), "strike"]),
        "peak_vanna_strike": int(bdf.loc[bdf["vanna"].abs().idxmax(), "strike"]),
    }


def report(symbol):
    chain = fetch_chain(symbol)
    if not chain:
        print(f"  {symbol}: chain fetch failed"); return None
    e = dealer_exposure(chain)
    print("=" * 66)
    print(f"  DEALER EXPOSURE — {symbol}  (spot {e['spot']}, {e['dte']}d)")
    print("=" * 66)
    print(f"  GAMMA : intensity {e['gamma_intensity']:+}  → {e['gamma_regime']}")
    print(f"          peak gamma strike {e['peak_gamma_strike']}")
    print(f"  VANNA : intensity {e['vanna_intensity']:+}  → {e['vanna_read']}")
    print(f"  CHARM : intensity {e['charm_intensity']:+}  → {e['charm_read']}")
    print("  (intensity ∈ [-1,1] = how one-sided the dealer book is per greek)")
    return e


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["NIFTY", "BANKNIFTY"]):
        report(s); print()

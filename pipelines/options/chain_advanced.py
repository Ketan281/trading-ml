"""
Advanced options reads — gamma regime, IV skew, OI/IV velocity, pin risk.

These are the higher-order reads a seasoned index-options trader watches, all
computable from data we already have (a live chain + the snapshots the
collector is accumulating):

  • gamma_exposure  → trend-vs-range regime + gamma-flip / call & put walls
  • iv_skew         → 25-delta put-vs-call skew = the market's fear gauge
  • oi_iv_velocity  → how fast PCR / OI / IV are MOVING (sentiment momentum),
                      read from the collected intraday snapshot time series
  • pin_risk        → near expiry, pull of price toward max pain

HONEST ASSUMPTION (gamma): we do not see dealer books. Gamma-exposure uses the
standard positioning HEURISTIC (dealers net short the calls and puts written
to the public → long-gamma where call OI dominates, etc.). It is a regime
heuristic for trend-vs-range, NOT a measured dealer position.
"""

import os
import sys
import glob

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from pipelines.options.chain_live_intel import fetch_chain, bs_greeks, STEP

LOT = {"NIFTY": 75, "BANKNIFTY": 35}


# ── Gamma exposure → trend vs range regime ────────────
def gamma_exposure(chain):
    df, spot, sym = chain["df"], chain["spot"], chain["symbol"]
    dte, lot = chain["dte"], LOT[chain["symbol"]]
    gex_by_strike = []
    for r in df.itertuples():
        g_ce = bs_greeks(spot, float(r.strike), dte, r.ce_iv, "ce")["gamma"]
        g_pe = bs_greeks(spot, float(r.strike), dte, r.pe_iv, "pe")["gamma"]
        # Dealer-long-gamma where calls dominate, short where puts dominate.
        gex = (g_ce * r.ce_oi - g_pe * r.pe_oi) * lot * spot ** 2 * 0.01
        gex_by_strike.append((int(r.strike), gex))
    gdf = pd.DataFrame(gex_by_strike, columns=["strike", "gex"])
    total = float(gdf["gex"].sum())

    # Gamma flip = strike where cumulative GEX crosses zero.
    gdf = gdf.sort_values("strike")
    gdf["cum"] = gdf["gex"].cumsum()
    flip = None
    sign = np.sign(gdf["cum"].values)
    for i in range(1, len(sign)):
        if sign[i] != sign[i - 1] and sign[i] != 0:
            flip = int(gdf["strike"].iloc[i]); break

    call_wall = int(gdf.loc[gdf["gex"].idxmax(), "strike"])      # strongest +gamma
    put_wall  = int(gdf.loc[gdf["gex"].idxmin(), "strike"])      # strongest -gamma
    regime = ("positive-gamma → RANGE / mean-revert (vol suppressed)"
              if total > 0 else
              "negative-gamma → TREND / momentum (vol amplified)")
    return {"total_gex": round(total, 1), "regime": regime,
            "gamma_flip": flip, "call_wall": call_wall, "put_wall": put_wall,
            "bias_for_trading": "favor range/credit spreads" if total > 0
                                else "favor directional/long-premium"}


# ── IV skew = fear gauge ──────────────────────────────
def iv_skew(chain):
    df, spot, dte = chain["df"], chain["spot"], chain["dte"]
    # Find ~25-delta OTM put and OTM call by computed delta.
    best_call, best_put = None, None
    for r in df.itertuples():
        cd = bs_greeks(spot, float(r.strike), dte, r.ce_iv, "ce")["delta"]
        pd_ = bs_greeks(spot, float(r.strike), dte, r.pe_iv, "pe")["delta"]
        if r.ce_iv and best_call is None or (r.ce_iv and abs(cd - 0.25) <
                abs(best_call[1] - 0.25)):
            best_call = (int(r.strike), cd, float(r.ce_iv))
        if r.pe_iv and best_put is None or (r.pe_iv and abs(pd_ + 0.25) <
                abs(best_put[1] + 0.25)):
            best_put = (int(r.strike), pd_, float(r.pe_iv))
    if not best_call or not best_put:
        return {"skew": None, "note": "insufficient IV data"}
    put_iv, call_iv = best_put[2], best_call[2]
    skew = round(put_iv - call_iv, 2)
    if skew > 3:
        interp = "steep put skew — market paying up for downside protection (fear)"
    elif skew > 1:
        interp = "normal put skew (typical downside premium)"
    elif skew < -1:
        interp = "call skew — upside chase / squeeze risk"
    else:
        interp = "flat skew — complacency"
    return {"put_25d_strike": best_put[0], "put_25d_iv": put_iv,
            "call_25d_strike": best_call[0], "call_25d_iv": call_iv,
            "skew": skew, "interpretation": interp}


# ── OI / IV velocity from collected snapshots ─────────
def oi_iv_velocity(symbol, lookback=6):
    path = os.path.join(ROOT, "data", "option_chain", "agg", f"{symbol}.csv")
    if not os.path.exists(path):
        return {"note": "no collected snapshots yet — run the chain collector"}
    df = pd.read_csv(path).tail(lookback)
    if len(df) < 2:
        return {"snapshots": int(len(df)),
                "note": "need ≥2 snapshots for velocity (collector accumulating)"}
    first, last = df.iloc[0], df.iloc[-1]
    span = max(1, len(df) - 1)

    def rate(col):
        return round((last[col] - first[col]) / span, 3)

    pcr_v = rate("pcr_oi")
    iv_v  = rate("atm_iv")
    ce_oi_v = rate("tot_ce_oi"); pe_oi_v = rate("tot_pe_oi")
    sentiment = ("put-writing accelerating (bullish)" if pcr_v > 0.02 else
                 "call-writing accelerating (bearish)" if pcr_v < -0.02 else
                 "OI balance steady")
    iv_note = ("IV expanding — premiums rising" if iv_v > 0.05 else
               "IV contracting — premiums bleeding" if iv_v < -0.05 else
               "IV stable")
    return {"snapshots": int(len(df)),
            "pcr_velocity": pcr_v, "iv_velocity": iv_v,
            "ce_oi_velocity": ce_oi_v, "pe_oi_velocity": pe_oi_v,
            "sentiment": sentiment, "iv_note": iv_note}


# ── Pin risk near expiry ──────────────────────────────
def pin_risk(chain):
    df, spot, dte, step = chain["df"], chain["spot"], chain["dte"], chain["step"]
    strikes = df["strike"].values
    pains = [((k - df["strike"]).clip(lower=0) * df["ce_oi"]).sum() +
             ((df["strike"] - k).clip(lower=0) * df["pe_oi"]).sum() for k in strikes]
    mp = int(strikes[int(pd.Series(pains).idxmin())])
    dist = abs(spot - mp)
    if dte <= 1 and dist <= step:
        risk = "HIGH — expiry + price within a strike of max pain; expect pinning"
    elif dte <= 2 and dist <= 2 * step:
        risk = "elevated — pull toward max pain likely into expiry"
    else:
        risk = "low — far from expiry / max pain"
    return {"max_pain": mp, "distance_pts": round(float(dist), 1),
            "dte": dte, "pin_risk": risk}


# ── One-shot advanced read ────────────────────────────
def advanced_read(symbol):
    chain = fetch_chain(symbol)
    if not chain:
        return None
    gex = gamma_exposure(chain); sk = iv_skew(chain)
    vel = oi_iv_velocity(symbol); pin = pin_risk(chain)
    print("=" * 68)
    print(f"  ADVANCED OPTIONS READ — {symbol}  (spot {chain['spot']:.1f}, "
          f"{chain['dte']}d)")
    print("=" * 68)
    print(f"  Gamma regime : {gex['regime']}")
    print(f"     flip {gex['gamma_flip']} | call-wall {gex['call_wall']} | "
          f"put-wall {gex['put_wall']} | total GEX {gex['total_gex']:,}")
    print(f"     → {gex['bias_for_trading']}")
    if sk.get("skew") is not None:
        print(f"  IV skew      : {sk['skew']} (put25d {sk['put_25d_iv']} vs "
              f"call25d {sk['call_25d_iv']})")
        print(f"     {sk['interpretation']}")
    print(f"  Velocity     : {vel.get('sentiment', vel.get('note',''))}")
    if "iv_velocity" in vel:
        print(f"     PCRΔ {vel['pcr_velocity']}/snap | IVΔ {vel['iv_velocity']}/snap "
              f"({vel['iv_note']}) | {vel['snapshots']} snaps")
    print(f"  Pin risk     : {pin['pin_risk']} (max pain {pin['max_pain']}, "
          f"{pin['distance_pts']} pts away)")
    return {"symbol": symbol, "gamma": gex, "skew": sk,
            "velocity": vel, "pin": pin}


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["NIFTY", "BANKNIFTY"]):
        advanced_read(s); print()

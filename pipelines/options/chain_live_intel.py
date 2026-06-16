"""
Options live-intelligence — NIFTY / BANKNIFTY (works on TODAY's chain).

Everything here is computable from a single LIVE option-chain snapshot, so it
delivers value now — no waiting on the months of history the ML chain models
need. It is the professional "read the chain" layer:

  • OI walls         → support / resistance from where writers sit
  • OI build-up      → long / short buildup, short covering, long unwinding
  • Smart strikes    → ATM / ITM / OTM pick by delta + liquidity, + spreads
  • Greeks summary   → net delta/theta/vega of a chosen leg
  • Liquidity check  → bid-ask spread, OI, volume → slippage estimate
  • Expected range   → IV-based 1σ move + straddle-implied move

HONEST SCOPE: this is rule-based market structure, not a trained predictor.
It tells you what the chain IS saying right now and which strike is cleanest
to trade — it does not forecast premium moves (that's the ML model, gated on
collected history).
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from jugaad_data.nse import NSELive

nse = NSELive()
STEP = {"NIFTY": 50, "BANKNIFTY": 100}
RISK_FREE = 0.065      # ~India 10y; greeks are not sensitive to small changes


# ── Black-Scholes greeks (the free NSE feed returns empty greeks) ──
def bs_greeks(S, K, dte, iv, leg):
    """Compute delta/theta/gamma/vega from spot, strike, days-to-expiry and
    IV%, because NSELive does not populate greeks. leg in {'ce','pe'}."""
    T = max(dte, 0.5) / 365.0
    sigma = (iv or 0) / 100.0
    if sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": 0.0, "theta": 0.0, "gamma": 0.0, "vega": 0.0}
    d1 = (np.log(S / K) + (RISK_FREE + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100
    if leg == "ce":
        delta = norm.cdf(d1)
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 - RISK_FREE * K * np.exp(-RISK_FREE * T) * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 + RISK_FREE * K * np.exp(-RISK_FREE * T) * norm.cdf(-d2)) / 365
    return {"delta": round(float(delta), 4), "theta": round(float(theta), 2),
            "gamma": round(float(gamma), 6), "vega": round(float(vega), 2)}


# ── Fetch + parse a rich strike-level chain ───────────
def fetch_chain(symbol):
    raw = nse.index_option_chain(symbol)
    recs = raw["records"]
    spot = recs.get("underlyingValue")
    expiries = recs.get("expiryDates", [])
    if not spot or not expiries:
        return None
    nearest = expiries[0]
    rows = []
    for item in recs["data"]:
        if nearest not in item.get("expiryDates", []):
            continue
        k = item.get("strikePrice", 0)
        if not k:
            continue
        ce, pe = item.get("CE", {}), item.get("PE", {})
        rows.append({
            "strike": k,
            "ce_oi": ce.get("openInterest", 0), "ce_chg_oi": ce.get("changeinOpenInterest", 0),
            "ce_vol": ce.get("totalTradedVolume", 0), "ce_iv": ce.get("impliedVolatility", 0),
            "ce_ltp": ce.get("lastPrice", 0), "ce_chg": ce.get("change", 0),
            "ce_bid": ce.get("bidprice", 0), "ce_ask": ce.get("askPrice", 0),
            "ce_delta": ce.get("delta", 0), "ce_theta": ce.get("theta", 0),
            "ce_gamma": ce.get("gamma", 0), "ce_vega": ce.get("vega", 0),
            "pe_oi": pe.get("openInterest", 0), "pe_chg_oi": pe.get("changeinOpenInterest", 0),
            "pe_vol": pe.get("totalTradedVolume", 0), "pe_iv": pe.get("impliedVolatility", 0),
            "pe_ltp": pe.get("lastPrice", 0), "pe_chg": pe.get("change", 0),
            "pe_bid": pe.get("bidprice", 0), "pe_ask": pe.get("askPrice", 0),
            "pe_delta": pe.get("delta", 0), "pe_theta": pe.get("theta", 0),
            "pe_gamma": pe.get("gamma", 0), "pe_vega": pe.get("vega", 0),
        })
    df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
    step = STEP[symbol]
    atm = int(round(spot / step) * step)
    df["moneyness"] = np.where(df["strike"] == atm, "ATM",
                       np.where(df["strike"] > atm, "OTM_call_ITM_put", "ITM_call_OTM_put"))

    # days to expiry
    dte = max(0, (pd.Timestamp(nearest) - pd.Timestamp.now().normalize()).days)
    prev_close = _prev_close(symbol)
    return {"symbol": symbol, "spot": float(spot), "prev_close": prev_close,
            "expiry": nearest, "dte": dte, "atm": atm, "step": step, "df": df}


def _prev_close(symbol):
    try:
        from models.index_options_model import load_index
        d = load_index(symbol)
        return float(d["Close"].iloc[-1]) if d is not None and len(d) else None
    except Exception:
        return None


# ── OI walls = support / resistance ───────────────────
def oi_walls(chain, top=3):
    df, atm = chain["df"], chain["atm"]
    calls = df.nlargest(top, "ce_oi")[["strike", "ce_oi", "ce_chg_oi"]]
    puts  = df.nlargest(top, "pe_oi")[["strike", "pe_oi", "pe_chg_oi"]]
    return {
        "resistance": [{"strike": int(r.strike), "ce_oi": int(r.ce_oi),
                        "added": int(r.ce_chg_oi)} for r in calls.itertuples()],
        "support":    [{"strike": int(r.strike), "pe_oi": int(r.pe_oi),
                        "added": int(r.pe_chg_oi)} for r in puts.itertuples()],
    }


# ── OI build-up quadrant (per strike) ─────────────────
def _quad(price_chg, oi_chg):
    if oi_chg > 0:
        return "Long Buildup" if price_chg > 0 else "Short Buildup"
    if oi_chg < 0:
        return "Short Covering" if price_chg > 0 else "Long Unwinding"
    return "Neutral"


def buildup(chain, near=6):
    """Per-strike buildup near ATM + an aggregate writer-sentiment read."""
    df, atm, step = chain["df"], chain["atm"], chain["step"]
    win = df[(df["strike"] >= atm - near * step) & (df["strike"] <= atm + near * step)]
    notable = []
    for r in win.itertuples():
        notable.append({"strike": int(r.strike),
                        "CE": _quad(r.ce_chg, r.ce_chg_oi),
                        "PE": _quad(r.pe_chg, r.pe_chg_oi)})
    # Aggregate: who is writing more — calls (bearish) or puts (bullish)?
    ce_added, pe_added = df["ce_chg_oi"].sum(), df["pe_chg_oi"].sum()
    if pe_added > ce_added * 1.15:
        bias = "Bullish (put writers dominant — support building)"
    elif ce_added > pe_added * 1.15:
        bias = "Bearish (call writers dominant — resistance building)"
    else:
        bias = "Balanced / range-bound"
    return {"aggregate_bias": bias, "ce_oi_added": int(ce_added),
            "pe_oi_added": int(pe_added), "per_strike": notable}


# ── Liquidity / slippage ──────────────────────────────
def liquidity(row, leg):
    bid = row[f"{leg}_bid"]; ask = row[f"{leg}_ask"]; oi = row[f"{leg}_oi"]; vol = row[f"{leg}_vol"]
    mid = (bid + ask) / 2 if (bid and ask) else row[f"{leg}_ltp"]
    spread_pct = round((ask - bid) / mid * 100, 2) if mid and ask >= bid > 0 else None

    if spread_pct is not None:                       # bid/ask available
        if spread_pct <= 1 and oi > 1000:
            rating = "excellent"
        elif spread_pct <= 3 and oi > 500:
            rating = "ok"
        else:
            rating = "poor (wide / thin)"
        est_slip = round((ask - bid) / 2, 2)
    else:                                            # feed has no bid/ask → use OI+vol
        if oi > 5000 and vol > 1000:
            rating = "excellent (by OI/vol)"
        elif oi > 1000 and vol > 200:
            rating = "ok (by OI/vol)"
        else:
            rating = "thin (low OI/vol)"
        est_slip = None
    return {"spread_pct": spread_pct, "oi": int(oi), "volume": int(vol),
            "rating": rating, "est_slippage_per_share": est_slip}


# ── Smart strike selection ────────────────────────────
def smart_strikes(chain, view):
    """view in {bullish, bearish, neutral}. Returns concrete strikes for a
    directional buy (ATM/ITM/OTM by delta+liquidity) and a defined-risk spread."""
    df, atm, step, spot = chain["df"], chain["atm"], chain["step"], chain["spot"]
    leg = "ce" if view == "bullish" else "pe" if view == "bearish" else None

    def row_at(k):
        return df.iloc[(df["strike"] - k).abs().argmin()]

    out = {"view": view}
    if leg:
        atm_r = row_at(atm)
        # ITM ≈ one step in-the-money for the chosen leg; OTM ≈ one step out.
        itm_k = atm - step if leg == "ce" else atm + step
        otm_k = atm + step if leg == "ce" else atm - step
        picks = {}
        for label, k in (("ATM", atm), ("ITM", itm_k), ("OTM", otm_k)):
            r = row_at(k)
            iv = r[f"{leg}_iv"]
            g = bs_greeks(spot, float(r.strike), chain["dte"], iv, leg)
            picks[label] = {"strike": int(r.strike), "ltp": float(r[f"{leg}_ltp"]),
                            "delta": g["delta"], "theta": g["theta"],
                            "liquidity": liquidity(r, leg)}
        out["directional"] = {"leg": leg.upper(), "picks": picks,
            "guide": "ATM = balanced; ITM = higher delta, costlier, less theta; "
                     "OTM = cheap, high leverage, fastest theta decay"}
        # Defined-risk debit spread: buy ATM, sell 2-steps OTM.
        long_k = atm
        short_k = atm + 2 * step if leg == "ce" else atm - 2 * step
        lr, sr = row_at(long_k), row_at(short_k)
        net = round(float(lr[f"{leg}_ltp"]) - float(sr[f"{leg}_ltp"]), 2)
        width = abs(int(sr.strike) - int(lr.strike))
        out["spread"] = {
            "type": f"{'Bull Call' if leg=='ce' else 'Bear Put'} debit spread",
            "buy": int(lr.strike), "sell": int(sr.strike), "width": width,
            "net_debit": net,
            "max_profit": round(width - net, 2) if 0 < net < width else None}
    else:
        # Neutral → sell the ATM straddle / iron condor wings.
        atm_r = row_at(atm)
        straddle = round(float(atm_r["ce_ltp"]) + float(atm_r["pe_ltp"]), 2)
        out["neutral"] = {
            "short_straddle_strike": atm, "straddle_credit": straddle,
            "iron_condor": {"sell_call": atm + 2 * step, "sell_put": atm - 2 * step,
                            "buy_call": atm + 4 * step, "buy_put": atm - 4 * step},
            "note": "Sell premium only if you expect range + vol contraction"}
    return out


# ── Expected range ────────────────────────────────────
def expected_range(chain):
    df, atm, spot, dte = chain["df"], chain["atm"], chain["spot"], chain["dte"]
    atm_r = df.iloc[(df["strike"] - atm).abs().argmin()]
    atm_iv = float((atm_r["ce_iv"] + atm_r["pe_iv"]) / 2)
    # IV-based 1σ moves
    daily_sigma = spot * (atm_iv / 100) / np.sqrt(252) if atm_iv else None
    exp_sigma = spot * (atm_iv / 100) * np.sqrt(max(dte, 1) / 365) if atm_iv else None
    # Straddle-implied move to expiry
    straddle = float(atm_r["ce_ltp"]) + float(atm_r["pe_ltp"])
    return {
        "atm_iv": round(atm_iv, 2),
        "daily_1sigma_pts": round(float(daily_sigma), 1) if daily_sigma else None,
        "daily_range": [round(float(spot - daily_sigma), 1),
                        round(float(spot + daily_sigma), 1)] if daily_sigma else None,
        "to_expiry_1sigma_pts": round(float(exp_sigma), 1) if exp_sigma else None,
        "straddle_implied_move_pts": round(float(straddle), 1),
        "straddle_range": [round(float(spot - straddle), 1),
                           round(float(spot + straddle), 1)],
    }


# ── Greeks summary for a leg/quantity ─────────────────
def greeks_for(chain, strike, leg, lots=1):
    df = chain["df"]; lot = 75 if chain["symbol"] == "NIFTY" else 35
    r = df.iloc[(df["strike"] - strike).abs().argmin()]
    q = lots * lot
    g = bs_greeks(chain["spot"], float(r.strike), chain["dte"], r[f"{leg}_iv"], leg)
    return {"strike": int(r.strike), "leg": leg.upper(), "qty": q,
            "position_delta": round(g["delta"] * q, 1),
            "position_theta": round(g["theta"] * q, 1),
            "position_gamma": round(g["gamma"] * q, 4),
            "position_vega":  round(g["vega"] * q, 1),
            "per_lot": g}


# ── One-shot analysis + pretty print ──────────────────
def analyze(symbol, view=None):
    chain = fetch_chain(symbol)
    if not chain:
        print(f"  {symbol}: chain fetch failed."); return None
    spot, prev = chain["spot"], chain["prev_close"]
    chg = f"{(spot/prev-1)*100:+.2f}%" if prev else "n/a"
    if view is None:
        view = ("bullish" if prev and spot > prev * 1.002 else
                "bearish" if prev and spot < prev * 0.998 else "neutral")

    walls = oi_walls(chain); bu = buildup(chain)
    er = expected_range(chain); ss = smart_strikes(chain, view)

    print("=" * 66)
    print(f"  OPTIONS LIVE INTELLIGENCE — {symbol}  "
          f"(spot {spot:.1f}, {chg}, expiry {chain['expiry']}, {chain['dte']}d)")
    print("=" * 66)
    print(f"  Chain bias    : {bu['aggregate_bias']}")
    print(f"  Resistance    : " + " | ".join(
        f"{w['strike']} (OI {w['ce_oi']:,})" for w in walls["resistance"]))
    print(f"  Support       : " + " | ".join(
        f"{w['strike']} (OI {w['pe_oi']:,})" for w in walls["support"]))
    print(f"  Expected range: ATM IV {er['atm_iv']}% | day ±{er['daily_1sigma_pts']} "
          f"→ {er['daily_range']} | straddle move ±{er['straddle_implied_move_pts']}")
    print(f"  View taken    : {view.upper()}")
    if "directional" in ss:
        d = ss["directional"]
        print(f"  Strike picks ({d['leg']}):")
        for lab, p in d["picks"].items():
            print(f"     {lab:<4} {p['strike']} @ {p['ltp']:<8} "
                  f"Δ{p['delta']:<7} θ{p['theta']:<7} liq:{p['liquidity']['rating']}")
        sp = ss["spread"]
        print(f"  Spread        : {sp['type']} {sp['buy']}/{sp['sell']} "
              f"(width {sp['width']}) net ₹{sp['net_debit']} max-profit ₹{sp['max_profit']}")
    else:
        n = ss["neutral"]
        print(f"  Neutral play  : short straddle @ {n['short_straddle_strike']} "
              f"credit ₹{n['straddle_credit']} | IC "
              f"{n['iron_condor']['sell_put']}/{n['iron_condor']['sell_call']}")
    print("\n  Notable build-up near ATM:")
    for s in bu["per_strike"]:
        if s["CE"] != "Neutral" or s["PE"] != "Neutral":
            print(f"     {s['strike']:<7} CE:{s['CE']:<16} PE:{s['PE']}")
    return {"chain_meta": {k: chain[k] for k in ("symbol","spot","expiry","dte","atm")},
            "bias": bu, "walls": walls, "expected_range": er, "strikes": ss}


if __name__ == "__main__":
    syms = sys.argv[1:] or ["NIFTY", "BANKNIFTY"]
    for s in syms:
        analyze(s)
        print()

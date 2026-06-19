"""
Options action engine + risk manager — NIFTY / BANKNIFTY.

This is the layer your blueprint correctly flags as mattering MORE than the
model: it turns a probability-of-up into a concrete, risk-managed trade.

    P(up)        Action
    > 0.75       Buy CE (full)
    0.60–0.75    Small CE
    0.40–0.60    No Trade
    0.25–0.40    Small PE
    < 0.25       Buy PE

…then attaches a stop-loss, a risk-based position size, and a target. It is
deterministic (no hallucination) and model-agnostic: feed it P(up) from the
walk-forward chain model once that has data. UNTIL THEN it can run on a clearly
labelled INTERIM rule-based bias derived from the live chain (PCR / OI / Max
Pain) — useful for plumbing and paper-trading, NOT a proven edge.
"""

import os
import sys
import glob

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Live-intelligence layer (strike selection, walls, greeks, expected range).
from pipelines.options.chain_live_intel import (
    fetch_chain, buildup, oi_walls, expected_range, smart_strikes, greeks_for)
# Advanced reads (gamma regime, IV skew, pin risk).
from pipelines.options.chain_advanced import gamma_exposure, iv_skew, pin_risk
# Strategy auto-selector (best multi-leg structure for the read).
from pipelines.options.strategy_selector import select_for_chain, structure_summary

# ── Risk config ───────────────────────────────────────
CAPITAL        = 1_000_000
RISK_PCT       = 0.01          # risk 1% of capital per trade
SMALL_FACTOR   = 0.5           # "small" position = half the risk budget
SL_PREMIUM_PCT = 0.35          # stop when option premium drops 35% (buyer)
TARGET_R       = 1.8           # target = 1.8× the risk taken
LOT_SIZE       = {"NIFTY": 75, "BANKNIFTY": 35}   # update if NSE revises lots


# ── Probability → action ──────────────────────────────
def action_from_probability(prob_up):
    if prob_up > 0.68:
        return {"action": "BUY_CE", "leg": "CE", "size_factor": 1.0,
                "conviction": "high"}
    if prob_up >= 0.55:
        return {"action": "SMALL_CE", "leg": "CE", "size_factor": SMALL_FACTOR,
                "conviction": "moderate"}
    if prob_up > 0.45:
        return {"action": "NO_TRADE", "leg": None, "size_factor": 0.0,
                "conviction": "none"}
    if prob_up >= 0.32:
        return {"action": "SMALL_PE", "leg": "PE", "size_factor": SMALL_FACTOR,
                "conviction": "moderate"}
    return {"action": "BUY_PE", "leg": "PE", "size_factor": 1.0,
            "conviction": "high"}


# ── Risk-based position sizing ────────────────────────
def position_size(symbol, entry_premium, size_factor,
                  capital=CAPITAL, risk_pct=RISK_PCT, sl_pct=SL_PREMIUM_PCT):
    """Size by RISK, not by capital: the stop distance (sl_pct of premium)
    and the risk budget decide how many lots — the single most important
    discipline in options buying."""
    lot = LOT_SIZE.get(symbol, 75)
    risk_budget = capital * risk_pct * size_factor
    risk_per_lot = entry_premium * sl_pct * lot          # ₹ lost per lot at SL
    if risk_per_lot <= 0:
        return None
    lots = int(risk_budget // risk_per_lot)
    lots = max(0, lots)
    qty = lots * lot
    deployed = qty * entry_premium
    max_loss = qty * entry_premium * sl_pct
    stop_premium = round(entry_premium * (1 - sl_pct), 2)
    target_premium = round(entry_premium * (1 + sl_pct * TARGET_R), 2)
    return {
        "lot_size": lot, "lots": lots, "qty": qty,
        "entry_premium": round(entry_premium, 2),
        "stop_premium": stop_premium, "target_premium": target_premium,
        "capital_deployed": int(deployed), "max_loss": int(max_loss),
        "reward_risk": TARGET_R,
    }


# ── Full trade plan ───────────────────────────────────
def build_trade(symbol, prob_up, ce_premium, pe_premium, atm_strike,
                capital=CAPITAL):
    act = action_from_probability(prob_up)
    plan = {"symbol": symbol, "prob_up": round(float(prob_up), 3),
            "atm_strike": atm_strike, **act}
    if act["leg"] is None:
        plan["note"] = "Probability in the dead zone (0.40–0.60) — stand aside."
        return plan
    premium = ce_premium if act["leg"] == "CE" else pe_premium
    if not premium or premium <= 0:
        plan["note"] = "No valid option premium for the chosen leg."
        return plan
    sizing = position_size(symbol, premium, act["size_factor"], capital)
    if not sizing or sizing["lots"] == 0:
        plan["note"] = "Risk budget too small for one lot — skip or widen stop."
        return plan
    plan.update(sizing)
    plan["instrument"] = f"{symbol} {atm_strike} {act['leg']}"
    return plan


# ── INTERIM rule-based bias (until the ML model has data) ──
def interim_chain_bias(agg_row):
    """Rule-based P(up) from chain structure: PCR, OI build-up, Max-Pain,
    IV skew, gamma regime, volume ratio, and OI concentration. Each signal
    contributes independently; total range is roughly 0.10–0.90."""
    score = 0.0

    # 1. PCR — high PCR (put writing) = bullish support building
    pcr = agg_row.get("pcr_oi", 1.0)
    score += max(-0.20, min(0.20, (pcr - 1.0) * 0.40))

    # 2. Net OI build-up — puts adding (support) bullish, calls adding bearish
    net_oi = agg_row.get("pe_chg_oi", 0) - agg_row.get("ce_chg_oi", 0)
    denom = abs(agg_row.get("tot_ce_oi", 1)) + abs(agg_row.get("tot_pe_oi", 1)) + 1
    score += max(-0.18, min(0.18, net_oi / denom * 5))

    # 3. Max Pain pull — price tends toward max pain near expiry
    mp_dist = agg_row.get("max_pain_dist_pct", 0)
    score += max(-0.10, min(0.10, -mp_dist * 0.04))

    # 4. IV skew — steep put skew = fear = bearish bias
    iv_skew_val = agg_row.get("iv_skew", 0)
    if iv_skew_val:
        score += max(-0.10, min(0.10, -iv_skew_val * 0.03))

    # 5. Gamma regime — positive GEX = range (neutral), negative = trend-friendly
    gex_sign = agg_row.get("gex_sign", 0)
    score += 0.05 * gex_sign

    # 6. Volume ratio — high put volume vs call volume = hedging = mildly bearish
    vol_ratio = agg_row.get("vol_pcr", 0)
    if vol_ratio:
        score += max(-0.08, min(0.08, (vol_ratio - 1.0) * 0.15))

    # 7. OI concentration — heavy OI above spot = resistance (bearish), below = support (bullish)
    oi_imbalance = agg_row.get("oi_imbalance", 0)
    score += max(-0.08, min(0.08, oi_imbalance * 0.15))

    return max(0.05, min(0.95, 0.5 + score))


def _latest_agg(symbol):
    path = os.path.join(ROOT, "data", "option_chain", "agg", f"{symbol}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df.iloc[-1].to_dict() if len(df) else None


def _atm_premiums(symbol, atm):
    """Pull the latest ATM CE/PE LTP from the raw strike-level snapshot."""
    raws = sorted(glob.glob(os.path.join(ROOT, "data", "option_chain", "raw",
                                         symbol, "*.csv")))
    if not raws:
        return 0, 0
    df = pd.read_csv(raws[-1])
    df = df[df["timestamp"] == df["timestamp"].max()]
    r = df.iloc[(df["strike"] - atm).abs().argmin()]
    return float(r.get("ce_ltp", 0)), float(r.get("pe_ltp", 0))


def _max_pain_dist(chain):
    df, spot = chain["df"], chain["spot"]
    strikes = df["strike"].values
    pains = [((k - df["strike"]).clip(lower=0) * df["ce_oi"]).sum() +
             ((df["strike"] - k).clip(lower=0) * df["pe_oi"]).sum() for k in strikes]
    if not len(pains):
        return 0.0
    mp = strikes[int(pd.Series(pains).idxmin())]
    return (spot - mp) / spot * 100


def chain_prob_up(chain):
    """Derive P(up) from the LIVE chain using all available signals:
    PCR, OI buildup, max pain, IV skew, gamma regime, volume, OI concentration."""
    df, spot, atm = chain["df"], chain["spot"], chain["atm"]
    tot_ce, tot_pe = df["ce_oi"].sum(), df["pe_oi"].sum()
    tot_ce_vol, tot_pe_vol = df["ce_vol"].sum(), df["pe_vol"].sum()

    # IV skew: OTM put IV vs OTM call IV near ATM
    otm_puts = df[df["strike"] < atm].tail(3)
    otm_calls = df[df["strike"] > atm].head(3)
    put_iv = otm_puts["pe_iv"].mean() if len(otm_puts) else 0
    call_iv = otm_calls["ce_iv"].mean() if len(otm_calls) else 0
    skew = (put_iv - call_iv) if (put_iv and call_iv) else 0

    # Gamma: use pre-computed dealer GEX if available
    try:
        gex = gamma_exposure(chain)
        gex_sign = 1 if gex["total_gex"] > 0 else -1
    except Exception:
        gex_sign = 0

    # OI concentration: pe_oi below spot vs ce_oi above spot
    below = df[df["strike"] < spot]
    above = df[df["strike"] > spot]
    pe_support = below["pe_oi"].sum() if len(below) else 0
    ce_resist = above["ce_oi"].sum() if len(above) else 0
    oi_total = pe_support + ce_resist + 1
    oi_imbalance = (pe_support - ce_resist) / oi_total

    agg = {"pcr_oi": tot_pe / tot_ce if tot_ce else 1.0,
           "pe_chg_oi": df["pe_chg_oi"].sum(), "ce_chg_oi": df["ce_chg_oi"].sum(),
           "tot_ce_oi": tot_ce, "tot_pe_oi": tot_pe,
           "max_pain_dist_pct": _max_pain_dist(chain),
           "iv_skew": skew,
           "gex_sign": gex_sign,
           "vol_pcr": tot_pe_vol / tot_ce_vol if tot_ce_vol else 1.0,
           "oi_imbalance": oi_imbalance}
    return interim_chain_bias(agg)


# ── The wired plan: live chain → bias → action → strike → risk ──
def live_trade_plan(symbol, capital=CAPITAL, prefer="ATM"):
    """End-to-end: pull the live chain, read it, decide the action, pick the
    cleanest strike, size by risk, and attach greeks + walls + expected range
    so the plan is self-contained and explainable."""
    chain = fetch_chain(symbol)
    if not chain:
        return {"symbol": symbol, "error": "chain fetch failed"}

    prob = chain_prob_up(chain)
    act = action_from_probability(prob)
    walls = oi_walls(chain); er = expected_range(chain); bu = buildup(chain)
    gex = gamma_exposure(chain); sk = iv_skew(chain); pin = pin_risk(chain)
    # Best multi-leg structure for this read (reuses the already-fetched chain).
    rec = select_for_chain(chain, prob, gex, sk, capital)
    base = {"symbol": symbol, "spot": chain["spot"], "expiry": chain["expiry"],
            "dte": chain["dte"], "atm": chain["atm"], "prob_up": round(prob, 3),
            "action": act["action"], "conviction": act["conviction"],
            "chain_bias": bu["aggregate_bias"], "walls": walls,
            "expected_range": er, "gamma_regime": gex, "iv_skew": sk,
            "pin_risk": pin, "recommended_structure": structure_summary(rec)}

    # Regime alignment: a directional BUY in a positive-gamma RANGE regime is
    # fighting mean-reversion → flag it and prefer the defined-risk spread.
    is_directional = act["leg"] is not None
    pos_gamma = gex["total_gex"] > 0
    if is_directional and pos_gamma:
        base["regime_alignment"] = ("⚠ directional view fights positive-gamma "
            "RANGE regime — prefer the defined-risk spread or trim size")
    elif is_directional and not pos_gamma:
        base["regime_alignment"] = ("✓ negative-gamma TREND regime supports a "
            "directional long-premium trade")
    else:
        base["regime_alignment"] = "neutral action aligns with range regime"

    if act["leg"] is None:
        base["note"] = "Probability in the 0.40–0.60 dead zone — stand aside."
        return base

    view = "bullish" if act["leg"] == "CE" else "bearish"
    ss = smart_strikes(chain, view)
    pick = ss["directional"]["picks"].get(prefer) or ss["directional"]["picks"]["ATM"]
    leg = act["leg"].lower()
    premium = pick["ltp"]
    if not premium or premium <= 0:
        base["note"] = "No valid premium on chosen strike."; return base

    sizing = position_size(symbol, premium, act["size_factor"], capital)
    if not sizing or sizing["lots"] == 0:
        base["note"] = "Risk budget too small for one lot — skip or widen stop."
        return base

    g = greeks_for(chain, pick["strike"], leg, lots=sizing["lots"])
    # Underlying invalidation = nearest opposite OI wall.
    if act["leg"] == "CE":
        inval = max([w["strike"] for w in walls["support"] if w["strike"] < chain["spot"]],
                    default=walls["support"][0]["strike"])
        und_target = er["daily_range"][1] if er["daily_range"] else None
    else:
        inval = min([w["strike"] for w in walls["resistance"] if w["strike"] > chain["spot"]],
                    default=walls["resistance"][0]["strike"])
        und_target = er["daily_range"][0] if er["daily_range"] else None

    base.update({
        "instrument": f"{symbol} {pick['strike']} {act['leg']}",
        "strike_choice": prefer, "leg_delta": pick["delta"], "leg_theta": pick["theta"],
        "liquidity": pick["liquidity"]["rating"],
        "entry_premium": sizing["entry_premium"], "stop_premium": sizing["stop_premium"],
        "target_premium": sizing["target_premium"],
        "lots": sizing["lots"], "qty": sizing["qty"],
        "capital_deployed": sizing["capital_deployed"], "max_loss": sizing["max_loss"],
        "reward_risk": sizing["reward_risk"],
        "position_greeks": {k: g[k] for k in
                            ("position_delta", "position_theta", "position_vega")},
        "underlying_invalidation": int(inval),
        "underlying_target": und_target,
        "spread_alt": ss.get("spread"),
    })
    return base


def demo(symbols=("NIFTY", "BANKNIFTY"), prefer="ATM"):
    print("=" * 70)
    print("  OPTIONS ACTION ENGINE  (live chain → bias → action → risk plan)")
    print("  ⚠ Bias is interim rule-based, NOT a trained-model edge. Greeks are")
    print("    Black-Scholes (feed has none); verify fills on your live chain.")
    print("=" * 70)
    for sym in symbols:
        p = live_trade_plan(sym, prefer=prefer)
        if p.get("error"):
            print(f"\n  {sym}: {p['error']}"); continue
        print(f"\n  ── {sym}  spot {p['spot']:.1f} | {p['chain_bias']} | "
              f"expiry {p['expiry']} ({p['dte']}d) ──")
        print(f"    P(up) {p['prob_up']}  →  {p['action']} ({p['conviction']})")
        print(f"    Gamma regime       : {p['gamma_regime']['regime']}")
        print(f"    Regime alignment   : {p['regime_alignment']}")
        rs = p["recommended_structure"]
        print(f"    ▶ BEST STRUCTURE   : {rs['kind'].replace('_',' ').upper()} "
              f"({rs['flow']} ₹{abs(rs['net_premium_per_share'])}/sh)")
        print(f"        legs   : {' , '.join(rs['legs'])}")
        ml = f"₹{rs['max_loss_rupees']:,}" if rs['max_loss_rupees'] is not None else "UNDEFINED"
        mp = f"₹{rs['max_profit_rupees']:,}" if rs['max_profit_rupees'] else "large"
        print(f"        {rs['lots']} lot(s) | BE {rs['breakevens']} | "
              f"maxL {ml} / maxP {mp} | "
              f"Δ{rs['net_greeks']['delta']} θ{rs['net_greeks']['theta']} "
              f"vega{rs['net_greeks']['vega']}")
        if rs["sizing_note"]:
            print(f"        ⚠ {rs['sizing_note']}")
        sk = p["iv_skew"]
        if sk.get("skew") is not None:
            print(f"    IV skew            : {sk['skew']} — {sk['interpretation']}")
        print(f"    Pin risk           : {p['pin_risk']['pin_risk']}")
        rng = p["expected_range"]
        print(f"    Expected day range : {rng['daily_range']} "
              f"(±{rng['daily_1sigma_pts']}, ATM IV {rng['atm_iv']}%)")
        print(f"    Resistance/Support : "
              f"{[w['strike'] for w in p['walls']['resistance']]} / "
              f"{[w['strike'] for w in p['walls']['support']]}")
        if not p.get("qty"):
            print(f"    {p.get('note','')}"); continue
        print(f"    Instrument         : {p['instrument']}  ({p['strike_choice']}, "
              f"Δ{p['leg_delta']}, liq {p['liquidity']})")
        print(f"    Premium E/S/T      : {p['entry_premium']} / {p['stop_premium']} "
              f"/ {p['target_premium']}  (R:R {p['reward_risk']})")
        print(f"    Size               : {p['lots']} lot(s) = {p['qty']} qty | "
              f"deploy ₹{p['capital_deployed']:,} | max loss ₹{p['max_loss']:,}")
        pg = p["position_greeks"]
        print(f"    Position greeks    : Δ{pg['position_delta']} "
              f"θ{pg['position_theta']} vega{pg['position_vega']}")
        print(f"    Underlying invalid : spot beyond {p['underlying_invalidation']} "
              f"| range target ~{p['underlying_target']}")
        sp = p.get("spread_alt")
        if sp:
            print(f"    Defined-risk alt   : {sp['type']} {sp['buy']}/{sp['sell']} "
                  f"net ₹{sp['net_debit']} max ₹{sp['max_profit']}")


if __name__ == "__main__":
    import sys as _s
    args = [a for a in _s.argv[1:] if not a.startswith("-")]
    pref = "OTM" if "--otm" in _s.argv else "ITM" if "--itm" in _s.argv else "ATM"
    demo(tuple(args) if args else ("NIFTY", "BANKNIFTY"), prefer=pref)

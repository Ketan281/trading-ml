"""
Options strategy auto-selector — the "what do I actually put on" brain.

Combines four reads into ONE recommended structure, then builds it with real
strikes, net greeks, breakevens and risk-based sizing:

  • DIRECTION   (from chain bias / P(up))  → bullish / bearish / neutral
  • GAMMA REGIME (positive=range, negative=trend) → buy vs SELL premium
  • IV SKEW      → which side's premium is rich (sell the rich side)
  • DTE / EXPIRY → 0–1 DTE crushes long premium (theta) and whips it (gamma),
                   so near expiry we favour defined-risk credit and cut size

Structures supported: long call/put, bull-call / bear-put DEBIT spreads,
bull-put / bear-call CREDIT spreads, iron condor, long & short straddle,
long strangle. Each returns multi-leg net Δ/θ/vega, breakevens, max P/L, and a
risk-sized lot count.

HONEST SCOPE: selection logic is professional rule-of-thumb, not a trained
model; greeks are Black-Scholes (the feed has none). It picks a sound STRUCTURE
for the current read — it does not promise the trade wins.
"""

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from pipelines.options.chain_live_intel import fetch_chain, bs_greeks
from pipelines.options.chain_advanced import gamma_exposure, iv_skew

CAPITAL  = 1_000_000
RISK_PCT = 0.01
LOT = {"NIFTY": 75, "BANKNIFTY": 35}


# ── Expiry bucket → sizing / behaviour ────────────────
def expiry_profile(dte):
    if dte <= 1:
        return {"bucket": "0-1 DTE (expiry)", "size_mult": 0.5,
                "warn": "extreme gamma/theta — long premium decays fast; "
                        "prefer defined-risk credit, manage actively"}
    if dte <= 3:
        return {"bucket": "near expiry (2-3 DTE)", "size_mult": 0.8,
                "warn": "elevated theta — favour spreads over naked longs"}
    if dte <= 7:
        return {"bucket": "weekly (4-7 DTE)", "size_mult": 1.0, "warn": ""}
    return {"bucket": "far (>7 DTE)", "size_mult": 1.0,
            "warn": "more time value — debit structures viable"}


# ── Leg + structure builders ──────────────────────────
def _row(chain, strike):
    df = chain["df"]
    return df.iloc[(df["strike"] - strike).abs().argmin()]


# Target deltas place strikes sensibly regardless of index level / DTE.
DELTA_SHORT_DIR  = 0.30      # directional credit-spread short leg
DELTA_SHORT_COND = 0.20      # iron-condor / strangle short leg
DELTA_SELL_DEBIT = 0.30      # the sold leg of a debit spread


def _strike_by_delta(chain, leg, target_abs):
    """Strike whose |BS delta| is closest to target_abs (self-scales across
    NIFTY/BANKNIFTY and any DTE — fixes fixed-step strikes sitting inside the
    expected move)."""
    df, spot, dte = chain["df"], chain["spot"], chain["dte"]
    best, best_gap = None, 1e9
    for r in df.itertuples():
        iv = r.ce_iv if leg == "ce" else r.pe_iv
        if not iv:
            continue
        d = abs(bs_greeks(spot, float(r.strike), dte, iv, leg)["delta"])
        if abs(d - target_abs) < best_gap:
            best, best_gap = int(r.strike), abs(d - target_abs)
    return best if best is not None else chain["atm"]


WING_PCT = 0.006             # protective wing ≈ 0.6% of spot from the short leg


def _wing(chain, short_strike, leg):
    """Protective wing a FIXED ~0.6%-of-spot width beyond the short leg, so the
    spread's max loss stays sane and risk-sizable (delta-placed wings get too
    wide on high-priced indices and blow the per-trade risk budget)."""
    step = chain["step"]
    offset = max(2 * step, round(chain["spot"] * WING_PCT / step) * step)
    return short_strike + offset if leg == "ce" else short_strike - offset


def _leg(chain, strike, leg, side, lots):
    """side: +1 long / -1 short. Returns a leg with per-lot premium & greeks."""
    r = _row(chain, strike); k = int(r["strike"]); lot = LOT[chain["symbol"]]
    ltp = float(r[f"{leg}_ltp"]); iv = r[f"{leg}_iv"]
    g = bs_greeks(chain["spot"], k, chain["dte"], iv, leg)
    qty = lots * lot
    return {"strike": k, "leg": leg.upper(), "side": "long" if side > 0 else "short",
            "ltp": ltp, "qty": qty, "sign": side,
            "delta": g["delta"], "theta": g["theta"],
            "gamma": g["gamma"], "vega": g["vega"]}


def _net_greeks(legs):
    out = {}
    for g in ("delta", "theta", "gamma", "vega"):
        out["net_" + g] = round(sum(l["sign"] * l["qty"] * l[g] for l in legs), 2)
    return out


def _net_premium(legs):
    """Per-share net cost: +ve = net debit (you pay), -ve = net credit (you get)."""
    return round(sum(l["sign"] * l["ltp"] for l in legs), 2)


def build_structure(chain, kind):
    """Return a fully-specified structure (1 lot baseline) for the given kind.
    Spread/condor short strikes are DELTA-placed so they sit outside the
    expected move; wings are the next lower-delta strike."""
    atm = chain["atm"]
    L = lambda k, leg, side: _leg(chain, k, leg, side, 1)
    sd = lambda leg, t: _strike_by_delta(chain, leg, t)

    if kind == "long_call":
        legs = [L(atm, "ce", +1)]
    elif kind == "long_put":
        legs = [L(atm, "pe", +1)]
    elif kind == "bull_call_debit":
        sell = sd("ce", DELTA_SELL_DEBIT)
        legs = [L(atm, "ce", +1), L(sell, "ce", -1)]
    elif kind == "bear_put_debit":
        sell = sd("pe", DELTA_SELL_DEBIT)
        legs = [L(atm, "pe", +1), L(sell, "pe", -1)]
    elif kind == "bull_put_credit":
        sp = sd("pe", DELTA_SHORT_DIR)
        legs = [L(sp, "pe", -1), L(_wing(chain, sp, "pe"), "pe", +1)]
    elif kind == "bear_call_credit":
        sc = sd("ce", DELTA_SHORT_DIR)
        legs = [L(sc, "ce", -1), L(_wing(chain, sc, "ce"), "ce", +1)]
    elif kind == "iron_condor":
        sc = sd("ce", DELTA_SHORT_COND); sp = sd("pe", DELTA_SHORT_COND)
        legs = [L(sc, "ce", -1), L(_wing(chain, sc, "ce"), "ce", +1),
                L(sp, "pe", -1), L(_wing(chain, sp, "pe"), "pe", +1)]
    elif kind == "long_straddle":
        legs = [L(atm, "ce", +1), L(atm, "pe", +1)]
    elif kind == "short_straddle":
        legs = [L(atm, "ce", -1), L(atm, "pe", -1)]
    elif kind == "long_strangle":
        legs = [L(sd("ce", DELTA_SHORT_COND), "ce", +1),
                L(sd("pe", DELTA_SHORT_COND), "pe", +1)]
    else:
        raise ValueError(kind)

    net = _net_premium(legs)
    bes, maxp, maxl, undefined = _payoff(kind, legs, net)
    return {"kind": kind, "legs": legs, "net_premium": net,
            "breakevens": bes, "max_profit_pts": maxp, "max_loss_pts": maxl,
            "undefined_risk": undefined, **_net_greeks(legs)}


def _payoff(kind, legs, net):
    """Per-share breakevens and max P/L, widths derived from actual legs.
    net>0 debit (you pay), net<0 credit (you receive)."""
    undefined = False
    if kind == "long_call":
        be = [round(legs[0]["strike"] + net, 1)]; maxp = None; maxl = net
    elif kind == "long_put":
        be = [round(legs[0]["strike"] - net, 1)]
        maxp = legs[0]["strike"] - net; maxl = net
    elif kind in ("bull_call_debit", "bear_put_debit"):
        width = abs(legs[0]["strike"] - legs[1]["strike"])
        anchor = legs[0]["strike"]
        be = [round(anchor + net if kind == "bull_call_debit" else anchor - net, 1)]
        maxp = round(width - net, 2); maxl = net
    elif kind in ("bull_put_credit", "bear_call_credit"):
        credit = -net
        short_k = legs[0]["strike"]; width = abs(legs[0]["strike"] - legs[1]["strike"])
        be = [round(short_k - credit if kind == "bull_put_credit"
                    else short_k + credit, 1)]
        maxp = round(credit, 2); maxl = round(width - credit, 2)
    elif kind == "iron_condor":
        credit = -net
        sc = max(l["strike"] for l in legs if l["leg"] == "CE" and l["side"] == "short")
        lc = max(l["strike"] for l in legs if l["leg"] == "CE" and l["side"] == "long")
        sp = min(l["strike"] for l in legs if l["leg"] == "PE" and l["side"] == "short")
        lp = min(l["strike"] for l in legs if l["leg"] == "PE" and l["side"] == "long")
        width = max(abs(lc - sc), abs(sp - lp))     # worst side
        be = [round(sp - credit, 1), round(sc + credit, 1)]
        maxp = round(credit, 2); maxl = round(width - credit, 2)
    elif kind == "long_straddle":
        k = legs[0]["strike"]; be = [round(k - net, 1), round(k + net, 1)]
        maxp = None; maxl = net
    elif kind == "short_straddle":
        k = legs[0]["strike"]; credit = -net
        be = [round(k - credit, 1), round(k + credit, 1)]; maxp = round(credit, 2)
        maxl = None; undefined = True
    elif kind == "long_strangle":
        ck = max(l["strike"] for l in legs); pk = min(l["strike"] for l in legs)
        be = [round(pk - net, 1), round(ck + net, 1)]; maxp = None; maxl = net
    else:
        be, maxp, maxl = [], None, None
    return be, maxp, maxl, undefined


# ── Risk-based sizing ─────────────────────────────────
def size_structure(chain, struct, size_mult=1.0,
                   capital=CAPITAL, risk_pct=RISK_PCT):
    lot = LOT[chain["symbol"]]
    budget = capital * risk_pct * size_mult
    if struct["undefined_risk"] or struct["max_loss_pts"] is None:
        lots = 1                                  # can't risk-size unlimited loss
        note = "UNDEFINED RISK — sized to 1 lot; margin/again-st-trend risk high"
    else:
        loss_per_lot = struct["max_loss_pts"] * lot
        if loss_per_lot <= 0:
            lots, note = 0, ""
        else:
            lots = int(budget // loss_per_lot)
            if lots == 0 and loss_per_lot <= budget * 1.5:
                lots = 1
                note = (f"1 lot risks ₹{round(loss_per_lot):,} "
                        f"(>{int(risk_pct*100)}% of capital) — smallest tradable size")
            elif lots == 0:
                note = (f"1 lot risks ₹{round(loss_per_lot):,} — exceeds risk "
                        f"budget; tighten width or raise capital")
            else:
                note = ""
    qty = lots * lot
    rupee = {"lots": lots, "qty": qty,
             "net_premium_rupees": round(struct["net_premium"] * qty),
             "max_loss_rupees": (None if struct["max_loss_pts"] is None
                                 else round(struct["max_loss_pts"] * qty)),
             "max_profit_rupees": (None if struct["max_profit_pts"] is None
                                   else round(struct["max_profit_pts"] * qty)),
             "sizing_note": note}
    # scale net greeks to the sized position
    for g in ("net_delta", "net_theta", "net_gamma", "net_vega"):
        rupee[g] = round(struct[g] * lots, 2)
    return rupee


# ── The selector ──────────────────────────────────────
def select_kind(prob_up, regime_positive, skew, dte):
    """direction × gamma-regime × skew × dte → one structure kind + reason."""
    direction = ("bullish" if prob_up > 0.60 else
                 "bearish" if prob_up < 0.40 else "neutral")
    regime = "range" if regime_positive else "trend"
    reasons = [f"{direction} view", f"{regime} regime"]

    if direction == "bullish":
        kind = "bull_put_credit" if regime == "range" else "bull_call_debit"
    elif direction == "bearish":
        kind = "bear_call_credit" if regime == "range" else "bear_put_debit"
    else:
        kind = "iron_condor" if regime == "range" else "long_straddle"

    # Skew nuance: rich premium is better SOLD.
    if skew is not None:
        if skew > 3 and direction == "bullish":
            reasons.append("steep put skew → selling puts is rich (credit favoured)")
        elif skew < -1 and direction == "bearish":
            reasons.append("call skew → selling calls is rich (credit favoured)")

    # DTE override: near expiry, swap long premium → defined-risk credit.
    if dte <= 1:
        if kind == "bull_call_debit":
            kind = "bull_put_credit"; reasons.append("0-1 DTE: long premium decays → credit instead")
        elif kind == "bear_put_debit":
            kind = "bear_call_credit"; reasons.append("0-1 DTE: long premium decays → credit instead")
        elif kind == "long_straddle":
            kind = "iron_condor"; reasons.append("0-1 DTE: buying vol bleeds → sell defined-risk instead")
    return kind, direction, regime, "; ".join(reasons)


def select_for_chain(chain, prob_up, gex=None, sk=None, capital=CAPITAL):
    """Pure selection on an ALREADY-fetched chain (no extra NSE call). Pass the
    gamma/skew reads if you already computed them to avoid recomputing."""
    if gex is None:
        gex = gamma_exposure(chain)
    if sk is None:
        sk = iv_skew(chain)
    skew = sk.get("skew")
    exp = expiry_profile(chain["dte"])
    kind, direction, regime, reason = select_kind(
        prob_up, gex["total_gex"] > 0, skew, chain["dte"])
    struct = build_structure(chain, kind)
    sizing = size_structure(chain, struct, exp["size_mult"], capital)
    return {"symbol": chain["symbol"], "spot": chain["spot"], "dte": chain["dte"],
            "prob_up": round(float(prob_up), 3), "kind": kind,
            "direction": direction, "regime": regime, "reason": reason,
            "expiry_profile": exp, "structure": struct, "sizing": sizing,
            "gamma": gex, "skew": sk}


def structure_summary(rec):
    """Compact, serialisable summary for embedding in a trade plan."""
    s, z = rec["structure"], rec["sizing"]
    return {
        "kind": rec["kind"], "reason": rec["reason"],
        "expiry_bucket": rec["expiry_profile"]["bucket"],
        "legs": [f"{l['side'].upper()} {l['strike']} {l['leg']} @ {l['ltp']}"
                 for l in s["legs"]],
        "net_premium_per_share": s["net_premium"],
        "flow": "debit" if s["net_premium"] > 0 else "credit",
        "breakevens": s["breakevens"],
        "lots": z["lots"], "qty": z["qty"],
        "max_loss_rupees": z["max_loss_rupees"], "max_profit_rupees": z["max_profit_rupees"],
        "net_greeks": {"delta": z["net_delta"], "theta": z["net_theta"],
                       "vega": z["net_vega"], "gamma": z["net_gamma"]},
        "sizing_note": z["sizing_note"], "undefined_risk": s["undefined_risk"],
    }


def print_recommendation(rec):
    s, z, exp = rec["structure"], rec["sizing"], rec["expiry_profile"]
    sym = rec["symbol"]
    print("=" * 70)
    print(f"  STRATEGY SELECTOR — {sym}  spot {rec['spot']:.1f} | "
          f"{exp['bucket']} | P(up) {rec['prob_up']:.2f}")
    print("=" * 70)
    print(f"  Read        : {rec['reason']}")
    print(f"  Gamma       : {rec['gamma']['regime']}")
    if rec["skew"].get("skew") is not None:
        print(f"  IV skew     : {rec['skew']['skew']} ({rec['skew']['interpretation']})")
    if exp["warn"]:
        print(f"  Expiry note : {exp['warn']}")
    print(f"\n  ▶ STRUCTURE : {rec['kind'].replace('_', ' ').upper()}")
    for l in s["legs"]:
        print(f"      {l['side'].upper():<5} {l['strike']} {l['leg']} @ {l['ltp']}")
    npm = s["net_premium"]
    print(f"  Net          : {'debit ₹' if npm > 0 else 'credit ₹'}{abs(npm)} /sh "
          f" | breakevens {s['breakevens']}")
    print(f"  Sized        : {z['lots']} lot(s) = {z['qty']} qty")
    if z["max_loss_rupees"] is not None:
        mp = f"₹{z['max_profit_rupees']:,}" if z["max_profit_rupees"] else "large"
        print(f"     max loss ₹{z['max_loss_rupees']:,} | max profit {mp}")
    print(f"  Net greeks   : Δ{z['net_delta']} θ{z['net_theta']} "
          f"vega{z['net_vega']} γ{z['net_gamma']}")
    if z["sizing_note"]:
        print(f"  ⚠ {z['sizing_note']}")


def recommend(symbol, capital=CAPITAL):
    chain = fetch_chain(symbol)
    if not chain:
        return None
    from pipelines.options_action_engine import chain_prob_up
    rec = select_for_chain(chain, chain_prob_up(chain), capital=capital)
    print_recommendation(rec)
    return rec


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["NIFTY", "BANKNIFTY"]):
        recommend(s); print()

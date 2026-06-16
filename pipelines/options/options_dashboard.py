"""
Unified options dashboard — the whole options brain in one command.

Fetches the live chain ONCE and runs every options engine against it, so you
get a single coherent read:

  CONTEXT   : spot, expiry, P(up), IV percentile (rich/cheap)
  STRUCTURE : gamma regime, dealer vanna/charm, IV skew, OI walls, pin risk
  FLOW      : option-chain order-flow proxy, intraday regime
  RANGE     : IV-1σ + straddle-implied expected move
  DECISION  : the action call + the single best risk-sized structure

Honest reminders carried through: interim rule-based P(up), Black-Scholes greeks
(free feed has none), dealer-positioning heuristic, proxy order-flow, and
IV-percentile depth still growing as the collector runs.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from pipelines.options.chain_live_intel import (
    fetch_chain, oi_walls, expected_range, buildup)
from pipelines.options.chain_advanced import (
    gamma_exposure, iv_skew, pin_risk, oi_iv_velocity)
from pipelines.options.dealer_exposure import dealer_exposure
from pipelines.options.order_flow import chain_flow, tape_flow
from pipelines.options.strategy_selector import select_for_chain, structure_summary
from pipelines.options_action_engine import chain_prob_up, action_from_probability
from infra.iv_surface import iv_percentile
from pipelines.intraday_regime import classify as intraday_regime


def dashboard(symbol):
    chain = fetch_chain(symbol)
    if not chain:
        print(f"  {symbol}: chain fetch failed."); return None

    # one fetch → feed everything
    prob = chain_prob_up(chain)
    act = action_from_probability(prob)
    gex = gamma_exposure(chain); sk = iv_skew(chain); pin = pin_risk(chain)
    walls = oi_walls(chain); er = expected_range(chain); bu = buildup(chain)
    de = dealer_exposure(chain)
    cf = chain_flow(symbol, chain=chain)
    vel = oi_iv_velocity(symbol)
    ivp = iv_percentile(symbol)
    rec = select_for_chain(chain, prob, gex, sk)
    ireg = intraday_regime(symbol)
    tf = tape_flow(symbol)

    spot = chain["spot"]; dte = chain["dte"]
    line = "─" * 70
    print("=" * 72)
    print(f"  OPTIONS DASHBOARD — {symbol}   spot {spot:.1f} | expiry "
          f"{chain['expiry']} ({dte}d)")
    print("=" * 72)

    # CONTEXT
    print("  CONTEXT")
    iv_txt = (f"{ivp['current_atm_iv']}% (pctile {ivp['iv_percentile']}, "
              f"{ivp['read']})" if "current_atm_iv" in ivp
              else f"ATM IV {er['atm_iv']}% (history thin: {ivp.get('snapshots',0)} snaps)")
    print(f"    P(up) {prob:.2f} → {act['action']} | chain bias: {bu['aggregate_bias']}")
    print(f"    IV: {iv_txt}")
    print(f"  {line}")

    # STRUCTURE
    print("  STRUCTURE")
    print(f"    Gamma : {gex['regime']}  (flip {gex['gamma_flip']}, "
          f"call-wall {gex['call_wall']}, put-wall {gex['put_wall']})")
    print(f"    Dealer: vanna {de['vanna_intensity']:+} ({de['vanna_read'].split('—')[0].strip()})"
          f" | charm {de['charm_intensity']:+}")
    if sk.get("skew") is not None:
        print(f"    Skew  : {sk['skew']} — {sk['interpretation']}")
    print(f"    Walls : R {[w['strike'] for w in walls['resistance']]} | "
          f"S {[w['strike'] for w in walls['support']]}")
    print(f"    Pin   : {pin['pin_risk']}")
    print(f"  {line}")

    # FLOW
    print("  FLOW")
    if cf:
        print(f"    Chain flow    : {cf['imbalance']:+} → {cf['read']}")
    if vel and "sentiment" in vel:
        print(f"    OI/IV velocity: {vel['sentiment']} | {vel['iv_note']} "
              f"({vel['snapshots']} snaps)")
    print(f"    Intraday regime: {ireg['regime'].upper()}"
          + (f" → {ireg['stance']}" if 'stance' in ireg else ""))
    if tf:
        print(f"    Tape flow     : {tf['read']}")
    print(f"  {line}")

    # RANGE
    print("  EXPECTED RANGE")
    print(f"    Day ±{er['daily_1sigma_pts']} → {er['daily_range']} | "
          f"straddle-implied ±{er['straddle_implied_move_pts']}")
    print(f"  {line}")

    # DECISION
    rs = structure_summary(rec)
    print("  DECISION")
    print(f"    ▶ {rs['kind'].replace('_',' ').upper()}  ({rs['flow']} "
          f"₹{abs(rs['net_premium_per_share'])}/sh)")
    print(f"      legs: {' , '.join(rs['legs'])}")
    ml = f"₹{rs['max_loss_rupees']:,}" if rs['max_loss_rupees'] is not None else "UNDEFINED"
    mp = f"₹{rs['max_profit_rupees']:,}" if rs['max_profit_rupees'] else "large"
    print(f"      {rs['lots']} lot(s) | BE {rs['breakevens']} | maxL {ml} / maxP {mp} | "
          f"Δ{rs['net_greeks']['delta']} θ{rs['net_greeks']['theta']} "
          f"vega{rs['net_greeks']['vega']}")
    print(f"      why: {rec['reason']}")
    if rs["sizing_note"]:
        print(f"      ⚠ {rs['sizing_note']}")
    print("=" * 72)
    print("  ⚠ Interim P(up), BS greeks, dealer heuristic, proxy flow — verify "
          "on your live chain.")
    return {"symbol": symbol, "prob_up": prob, "action": act["action"],
            "iv_percentile": ivp, "gamma": gex, "dealer": de, "skew": sk,
            "flow": cf, "intraday_regime": ireg, "structure": rs}


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["NIFTY", "BANKNIFTY"]):
        dashboard(s); print()

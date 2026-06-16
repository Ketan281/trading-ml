"""
Wallet-aware capital allocator — how many rupees actually go on each trade.

Combines every sizing input into one decision:

  • ACCOUNT SIZE     small accounts trade smaller (a tier multiplier), so the
                     same % risk doesn't over-concentrate a tiny wallet
  • REGIME POLICY    per-trade risk %, gross cap, max positions, sector cap
                     (from risk_policy) — exposure adapts to bull/bear/volatile
  • DRAWDOWN GUARD   the circuit-breaker scales gross toward zero as equity falls
  • META-LEARNING    the learned (source × regime) size multiplier — upsize what
                     has worked, downsize what hasn't
  • LEVERAGE CAP     gross deployed ≤ account × gross_eff × leverage(segment)
  • RISK BUDGET      qty solved from the STOP distance, then clamped by the
                     per-name and gross caps

Output: a full, auditable breakdown the Portfolio Manager / executor uses. In
paper mode equity-delivery leverage is 1.0 (no margin); intraday allows a cap.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipelines.risk_policy import effective_limits
from aos.meta_learning import sizing_multiplier

LEVERAGE = {"equity_delivery": 1.0, "equity_intraday": 4.0, "options": 1.0}
SL_OPT_PCT = 0.35      # option premium stop, for risk-per-unit on options


def account_tier(account_size):
    if account_size < 50_000:
        return "micro", 0.6
    if account_size < 200_000:
        return "small", 0.8
    if account_size < 1_000_000:
        return "standard", 1.0
    return "large", 1.0


def allocate(account_size, regime, candidate, *, segment="equity_delivery",
             source="ensemble", confidence=None, breadth_score=None,
             drawdown=0.0, current_gross=0.0, current_positions=0):
    """candidate needs entry + stop (equity) or entry premium (options)."""
    lim = effective_limits(regime, drawdown, breadth_score)
    tier, tier_mult = account_tier(account_size)

    if lim["gross_effective"] <= 0:
        return {"approved": False, "reason": "drawdown circuit-breaker: gross 0%",
                "tier": tier, "regime": regime}
    if current_positions >= lim["max_positions"]:
        return {"approved": False, "reason": f"max positions "
                f"({lim['max_positions']}) reached", "tier": tier}

    entry = float(candidate["entry"])
    if segment == "options":
        risk_per_unit = entry * SL_OPT_PCT
    else:
        stop = float(candidate["stop"])
        if stop >= entry:
            return {"approved": False, "reason": "invalid stop ≥ entry"}
        risk_per_unit = entry - stop

    meta = sizing_multiplier(source, regime, confidence)
    risk_budget = account_size * lim["per_trade_risk_pct"] * tier_mult * meta

    qty = int(risk_budget / risk_per_unit) if risk_per_unit > 0 else 0
    deploy = qty * entry

    # Per-name cap (sector cap proxies single-name concentration).
    name_cap = account_size * lim["max_sector_pct"]
    if deploy > name_cap:
        qty = int(name_cap / entry); deploy = qty * entry
        clamp = "per_name_cap"
    else:
        clamp = None

    # Gross / leverage cap.
    gross_cap = account_size * lim["gross_effective"] * LEVERAGE.get(segment, 1.0)
    if current_gross + deploy > gross_cap:
        room = max(0, gross_cap - current_gross)
        qty = int(room / entry); deploy = qty * entry
        clamp = "gross_cap"

    approved = qty >= 1
    return {
        "approved": approved,
        "reason": None if approved else "size < 1 unit after caps",
        "tier": tier, "regime": regime,
        "per_trade_risk_pct": lim["per_trade_risk_pct"],
        "tier_mult": tier_mult, "meta_mult": meta,
        "risk_budget": round(risk_budget),
        "qty": qty, "deploy": round(deploy),
        "risk_rupees": round(qty * risk_per_unit),
        "gross_cap": round(gross_cap), "gross_after": round(current_gross + deploy),
        "max_positions": lim["max_positions"], "name_cap": round(name_cap),
        "leverage": LEVERAGE.get(segment, 1.0), "clamp": clamp,
    }


if __name__ == "__main__":
    cand = {"entry": 1000, "stop": 950}        # 5% stop
    print("=" * 74)
    print("  CAPITAL ALLOCATOR — same trade, different accounts & regimes")
    print("=" * 74)
    print(f"  Candidate: entry {cand['entry']} stop {cand['stop']} (5% risk/unit)\n")
    print(f"  {'ACCOUNT':>10} {'REGIME':<10}{'TIER':<10}{'RISK%':>7}{'metaX':>7}"
          f"{'QTY':>6}{'DEPLOY':>12}{'RISK ₹':>9}  CLAMP")
    for acct in (50_000, 300_000, 2_000_000):
        for reg in ("bull", "bear", "volatile"):
            a = allocate(acct, reg, cand)
            if not a.get("approved"):
                print(f"  {acct:>10,} {reg:<10}{a.get('tier','-'):<10}  → blocked: {a['reason']}")
                continue
            print(f"  {acct:>10,} {reg:<10}{a['tier']:<10}{a['per_trade_risk_pct']:>7.2%}"
                  f"{a['meta_mult']:>7}{a['qty']:>6}{a['deploy']:>12,}"
                  f"{a['risk_rupees']:>9,}  {a['clamp'] or '-'}")

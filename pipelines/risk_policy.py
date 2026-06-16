"""
Regime-aware risk policy + portfolio drawdown guard.

The portfolio's RISK BUDGET should breathe with the market. In a bull market
you run more gross, more names, more risk per trade; in a bear or volatile
regime you cut all three. This module maps the 4-class regime (from
models/regime_classifier) into concrete exposure limits, and adds a drawdown
circuit-breaker that throttles risk as equity falls.

It complements pipelines/portfolio_risk.py (which checks live open positions)
by setting the LIMITS the portfolio optimizer builds within.
"""

# regime → exposure policy
POLICY = {
    "bull":     {"gross_max": 1.00, "max_positions": 12, "max_sector_pct": 0.30,
                 "per_trade_risk_pct": 0.010, "stop_bias": "trail/ATR"},
    "sideways": {"gross_max": 0.60, "max_positions": 8,  "max_sector_pct": 0.25,
                 "per_trade_risk_pct": 0.0075, "stop_bias": "tight/swing-low"},
    "bear":     {"gross_max": 0.30, "max_positions": 5,  "max_sector_pct": 0.20,
                 "per_trade_risk_pct": 0.005, "stop_bias": "tight, A+ only"},
    "volatile": {"gross_max": 0.40, "max_positions": 6,  "max_sector_pct": 0.20,
                 "per_trade_risk_pct": 0.005, "stop_bias": "wide ATR, small size"},
    "unknown":  {"gross_max": 0.50, "max_positions": 6,  "max_sector_pct": 0.25,
                 "per_trade_risk_pct": 0.0075, "stop_bias": "default"},
}

# Drawdown circuit-breaker: as the book's drawdown deepens, scale gross down.
DD_BANDS = [(-0.05, 1.0), (-0.10, 0.75), (-0.15, 0.5), (-0.20, 0.25), (-1.0, 0.0)]

MAX_PORTFOLIO_HEAT = 0.06     # sum of per-position risk ≤ 6% of capital


def regime_policy(regime):
    return dict(POLICY.get(regime, POLICY["unknown"]))


def drawdown_scaler(current_dd):
    """current_dd ≤ 0 (e.g., -0.08). Returns a gross multiplier in [0,1]."""
    for thresh, mult in DD_BANDS:
        if current_dd >= thresh:
            return mult
    return 0.0


def effective_limits(regime, current_dd=0.0, breadth_score=None):
    """Combine regime policy + drawdown guard (+ optional breadth haircut)
    into the limits the optimizer must respect."""
    pol = regime_policy(regime)
    dd_mult = drawdown_scaler(current_dd)
    gross = pol["gross_max"] * dd_mult
    # Weak breadth trims gross further (narrow markets are fragile).
    if breadth_score is not None and breadth_score < 40:
        gross *= 0.8
    pol["gross_effective"] = round(gross, 3)
    pol["dd_scaler"] = dd_mult
    pol["max_portfolio_heat"] = MAX_PORTFOLIO_HEAT
    pol["new_positions_allowed"] = gross > 0
    return pol


def portfolio_heat(positions):
    """positions: list of dicts with 'risk_rupees' and a shared capital base.
    Returns total heat as a fraction (needs 'capital' on each or pass total)."""
    total_risk = sum(p.get("risk_rupees", 0) for p in positions)
    cap = positions[0].get("capital", 1) if positions else 1
    return round(total_risk / cap, 4) if cap else 0.0


if __name__ == "__main__":
    print("=" * 60)
    print("  REGIME-AWARE RISK POLICY")
    print("=" * 60)
    for reg in ("bull", "sideways", "bear", "volatile"):
        p = effective_limits(reg, current_dd=0.0)
        print(f"  {reg:<9} gross {p['gross_max']:.0%} | {p['max_positions']} pos | "
              f"sector ≤{p['max_sector_pct']:.0%} | risk/trade "
              f"{p['per_trade_risk_pct']:.2%} | stops: {p['stop_bias']}")
    print("\n  Drawdown circuit-breaker (bear regime, gross 30%):")
    for dd in (-0.03, -0.08, -0.12, -0.18, -0.25):
        e = effective_limits("bear", current_dd=dd)
        print(f"     dd {dd:>5.0%} → ×{e['dd_scaler']:.2f} → "
              f"effective gross {e['gross_effective']:.0%}")

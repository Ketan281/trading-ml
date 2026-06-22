"""
Advanced Capital Allocation Engine — dynamic position sizing.

Combines Kelly criterion, conviction grading, drawdown state,
and portfolio awareness into optimal position sizes.

Sizing flow:
  regime risk budget → conviction multiplier → drawdown scaler →
  portfolio constraints → Kelly cap → final position size
"""

import os
import sys
import math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

KELLY_FRACTION = 0.25
MAX_SINGLE_POSITION_PCT = 0.05
MIN_POSITION_PCT = 0.005
LOT_SIZE_MAP = {
    "NIFTY": 25, "BANKNIFTY": 15, "FINNIFTY": 25,
    "RELIANCE": 250, "TCS": 175, "INFY": 300,
    "HDFCBANK": 550, "ICICIBANK": 700, "SBIN": 750,
}
DEFAULT_LOT_SIZE = 1


# ── Sub-computations ───────────────────────────────────

def kelly_size(win_rate, avg_win, avg_loss):
    """Raw Kelly criterion: f* = (p*b - q) / b."""
    if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    b = abs(avg_win / avg_loss)
    p = win_rate
    q = 1 - p
    f = (p * b - q) / b
    return max(0.0, f)


def conviction_multiplier(grade, conviction_score=None):
    """Scale position size by grade. A+ = full, C = quarter."""
    grade_map = {"A+": 1.0, "A": 0.75, "B": 0.5, "C": 0.25, "NO_TRADE": 0.0}
    mult = grade_map.get(grade, 0.0)
    if conviction_score is not None and isinstance(conviction_score, (int, float)):
        conv_adj = conviction_score / 100
        mult *= (0.5 + 0.5 * conv_adj)
    return round(mult, 4)


def drawdown_multiplier(current_dd, psychology_state=None):
    """Combine drawdown bands with psychology state."""
    try:
        from pipelines.risk_policy import drawdown_scaler
        dd_mult = drawdown_scaler(current_dd)
    except ImportError:
        if current_dd >= -0.05:
            dd_mult = 1.0
        elif current_dd >= -0.10:
            dd_mult = 0.75
        elif current_dd >= -0.15:
            dd_mult = 0.5
        elif current_dd >= -0.20:
            dd_mult = 0.25
        else:
            dd_mult = 0.0

    if psychology_state:
        risk_state = psychology_state.get("risk_state", "normal")
        psych_mult = {"normal": 1.0, "caution": 0.75,
                      "restricted": 0.5, "halt": 0.0}
        dd_mult *= psych_mult.get(risk_state, 1.0)

    return round(dd_mult, 4)


def portfolio_awareness(existing_positions, new_signal, capital):
    """Check portfolio constraints for the new trade.

    Returns {sector_room, correlation_penalty, gross_room, heat_room, can_add}.
    """
    positions = existing_positions or []
    symbol = new_signal.get("symbol", "")
    sector = new_signal.get("sector", "unknown")

    sector_exposure = {}
    gross_exposure = 0.0
    total_heat = 0.0

    for p in positions:
        s = p.get("sector", "unknown")
        val = abs(p.get("capital_allocated", p.get("value", 0)))
        sector_exposure[s] = sector_exposure.get(s, 0) + val
        gross_exposure += val
        total_heat += abs(p.get("risk_rupees", val * 0.02))

    gross_pct = gross_exposure / capital if capital > 0 else 0
    try:
        from pipelines.risk_policy import regime_policy
        policy = regime_policy("unknown")
        gross_limit = policy.get("gross_max", 1.0)
        sector_limit = policy.get("max_sector_pct", 0.25)
        heat_limit = policy.get("max_portfolio_heat", 0.06)
    except ImportError:
        gross_limit = 1.0
        sector_limit = 0.25
        heat_limit = 0.06

    gross_room = max(0, gross_limit - gross_pct)
    sector_val = sector_exposure.get(sector, 0)
    sector_pct = sector_val / capital if capital > 0 else 0
    sector_room = max(0, sector_limit - sector_pct)
    heat_pct = total_heat / capital if capital > 0 else 0
    heat_room = max(0, heat_limit - heat_pct)

    corr_penalty = 1.0
    same_sector_count = sum(1 for p in positions if p.get("sector") == sector)
    if same_sector_count >= 3:
        corr_penalty = 0.5
    elif same_sector_count >= 2:
        corr_penalty = 0.75

    can_add = gross_room > 0 and sector_room > 0 and heat_room > 0

    return {
        "sector_room": round(sector_room, 4),
        "correlation_penalty": round(corr_penalty, 4),
        "gross_room": round(gross_room, 4),
        "heat_room": round(heat_room, 4),
        "can_add": can_add,
    }


# ── Main Sizing Function ──────────────────────────────

def compute_position_size(signal, grade, conviction, trade_quality,
                          regime="unknown", portfolio=None, capital=1_000_000,
                          psychology_state=None, win_rate=0.55,
                          avg_win=1.5, avg_loss=1.0):
    """THE SIZING FUNCTION.

    Returns {shares, lots, capital_allocated, capital_pct, risk_rupees,
             risk_pct, sizing_method, multipliers, effective_kelly, reason}.
    """
    if grade == "NO_TRADE":
        return {
            "shares": 0, "lots": 0, "capital_allocated": 0,
            "capital_pct": 0, "risk_rupees": 0, "risk_pct": 0,
            "sizing_method": "blocked", "multipliers": {},
            "effective_kelly": 0, "reason": "Grade is NO_TRADE",
        }

    symbol = signal.get("symbol", "")
    price = signal.get("price")
    if isinstance(price, str):
        try:
            price = float(price.split("-")[0])
        except (ValueError, IndexError):
            price = None
    if not price or price <= 0:
        return {
            "shares": 0, "lots": 0, "capital_allocated": 0,
            "capital_pct": 0, "risk_rupees": 0, "risk_pct": 0,
            "sizing_method": "no_price", "multipliers": {},
            "effective_kelly": 0, "reason": "No valid price",
        }

    try:
        from pipelines.risk_policy import regime_policy
        policy = regime_policy(regime)
        base_risk_pct = policy.get("per_trade_risk_pct", 0.01)
    except ImportError:
        base_risk_pct = 0.01

    raw_kelly = kelly_size(win_rate, avg_win, avg_loss)
    eff_kelly = raw_kelly * KELLY_FRACTION

    conv_mult = conviction_multiplier(grade, conviction)

    current_dd = 0.0
    if portfolio and isinstance(portfolio, dict):
        current_dd = portfolio.get("current_drawdown", 0)
    dd_mult = drawdown_multiplier(current_dd, psychology_state)

    port_check = portfolio_awareness(
        portfolio.get("positions", []) if isinstance(portfolio, dict) else [],
        signal, capital)
    port_mult = port_check["correlation_penalty"] if port_check["can_add"] else 0.0

    quality_mult = min(trade_quality / 100, 1.0) if trade_quality else 0.5

    if psychology_state:
        try:
            from engines.psychology_engine import dynamic_risk_reduction
            base_risk_pct = dynamic_risk_reduction(base_risk_pct, psychology_state)
        except ImportError:
            pass

    effective_risk = base_risk_pct * conv_mult * dd_mult * port_mult * quality_mult
    effective_risk = min(effective_risk, eff_kelly) if eff_kelly > 0 else effective_risk
    effective_risk = max(min(effective_risk, MAX_SINGLE_POSITION_PCT), 0)

    if effective_risk < MIN_POSITION_PCT:
        return {
            "shares": 0, "lots": 0, "capital_allocated": 0,
            "capital_pct": 0, "risk_rupees": 0, "risk_pct": 0,
            "sizing_method": "too_small",
            "multipliers": {"conviction": conv_mult, "drawdown": dd_mult,
                            "portfolio": port_mult, "quality": quality_mult},
            "effective_kelly": round(eff_kelly, 6),
            "reason": f"Effective risk {effective_risk:.4%} below minimum {MIN_POSITION_PCT:.2%}",
        }

    risk_rupees = capital * effective_risk

    stop = signal.get("stop_loss")
    if isinstance(stop, str):
        try:
            stop = float(stop)
        except ValueError:
            stop = None
    if stop and price:
        stop_dist = abs(price - stop)
        if stop_dist > 0:
            shares = int(risk_rupees / stop_dist)
        else:
            shares = int(risk_rupees / (price * 0.02))
    else:
        shares = int(risk_rupees / (price * 0.02))

    lot = LOT_SIZE_MAP.get(symbol, DEFAULT_LOT_SIZE)
    if lot > 1:
        lots = max(1, shares // lot)
        shares = lots * lot
    else:
        lots = shares

    allocated = shares * price
    alloc_pct = allocated / capital if capital > 0 else 0

    if alloc_pct > MAX_SINGLE_POSITION_PCT:
        shares = int((capital * MAX_SINGLE_POSITION_PCT) / price)
        if lot > 1:
            lots = max(1, shares // lot)
            shares = lots * lot
        else:
            lots = shares
        allocated = shares * price
        alloc_pct = allocated / capital

    return {
        "shares": shares,
        "lots": lots if lot > 1 else shares,
        "capital_allocated": round(allocated, 2),
        "capital_pct": round(alloc_pct, 4),
        "risk_rupees": round(risk_rupees, 2),
        "risk_pct": round(effective_risk, 6),
        "sizing_method": "kelly_conviction_drawdown",
        "multipliers": {
            "conviction": conv_mult,
            "drawdown": dd_mult,
            "portfolio": port_mult,
            "quality": quality_mult,
        },
        "effective_kelly": round(eff_kelly, 6),
        "reason": f"Grade {grade}, conv_mult {conv_mult:.2f}, dd_mult {dd_mult:.2f}",
    }


# ── CLI ────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  ADVANCED CAPITAL ALLOCATION ENGINE")
    print("=" * 60)

    sig = {"symbol": "RELIANCE", "price": 2800, "stop_loss": 2755,
           "sector": "energy"}

    for g in ("A+", "A", "B", "C", "NO_TRADE"):
        r = compute_position_size(sig, grade=g, conviction=75,
                                  trade_quality=70, capital=1_000_000)
        print(f"\n  Grade {g:8}: {r['shares']:>5} shares, "
              f"₹{r['capital_allocated']:>10,.0f} ({r['capital_pct']:.2%}), "
              f"risk ₹{r['risk_rupees']:>8,.0f} — {r['reason']}")

    print(f"\n  Kelly raw: {kelly_size(0.55, 1.5, 1.0):.4f}")
    print(f"  Kelly quarter: {kelly_size(0.55, 1.5, 1.0) * KELLY_FRACTION:.4f}")

    p_state = {"risk_state": "restricted", "consecutive_losses": 2}
    r = compute_position_size(sig, "A", 70, 65, psychology_state=p_state)
    print(f"\n  Restricted state: {r['shares']} shares, "
          f"risk ₹{r['risk_rupees']:,.0f} — {r['reason']}")

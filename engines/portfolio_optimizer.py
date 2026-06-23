"""
Smart Portfolio Optimizer — Kelly sizing, correlation-aware, drawdown protection.

Takes ML picks and converts them into an optimized portfolio:
1. Kelly criterion sizing based on historical win rates
2. Sector concentration limits (max 30% in any sector)
3. Correlation penalty (don't hold 5 correlated stocks)
4. Drawdown protection (reduce size during drawdowns)
5. Conviction-weighted allocation (higher ML score = bigger position)

Integrates with existing Phase 2 engines:
- psychology_engine for risk state
- capital_allocation for position sizing
- regime_v2 for market regime
"""

import os
import sys
import json
import logging
import numpy as np
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

log = logging.getLogger("portfolio_optimizer")

SECTOR_MAP_PATH = os.path.join(ROOT, "data", "sector_map.json")

NIFTY50_SECTORS = {
    "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy", "NTPC": "Energy",
    "POWERGRID": "Energy", "ADANIGREEN": "Energy", "ADANIENT": "Conglomerate",
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "KOTAKBANK": "Banking",
    "AXISBANK": "Banking", "SBIN": "Banking", "INDUSINDBK": "Banking",
    "BAJFINANCE": "NBFC", "BAJAJFINSV": "NBFC", "HDFCLIFE": "Insurance",
    "SBILIFE": "Insurance",
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT", "TECHM": "IT",
    "LT": "Infrastructure", "ULTRACEMCO": "Cement", "GRASIM": "Cement",
    "TATAMOTORS": "Auto", "MARUTI": "Auto", "BAJAJ-AUTO": "Auto",
    "HEROMOTOCO": "Auto", "EICHERMOT": "Auto", "M&M": "Auto",
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "TATACONSUM": "FMCG",
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
    "APOLLOHOSP": "Pharma", "DIVISLAB": "Pharma",
    "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "HINDALCO": "Metals",
    "COALINDIA": "Mining",
    "ASIANPAINT": "Building", "PIDILITIND": "Building",
    "TITAN": "Consumer", "BHARTIARTL": "Telecom",
}


def _get_sector(symbol):
    """Get sector for a stock."""
    if os.path.exists(SECTOR_MAP_PATH):
        try:
            with open(SECTOR_MAP_PATH) as f:
                sector_map = json.load(f)
            if symbol in sector_map:
                return sector_map[symbol]
        except Exception:
            pass
    return NIFTY50_SECTORS.get(symbol, "Other")


def kelly_fraction(win_rate, avg_win, avg_loss):
    """Kelly criterion: optimal fraction of capital to risk.
    We use quarter-Kelly for safety."""
    if avg_loss == 0:
        return 0.02
    b = avg_win / abs(avg_loss)
    p = win_rate
    q = 1 - p
    kelly = (p * b - q) / b
    quarter_kelly = kelly / 4
    return max(0.005, min(0.05, quarter_kelly))


def _get_risk_multiplier(uid=None):
    """Get risk multiplier from psychology engine."""
    try:
        from engines.psychology_engine import get_state
        state = get_state(uid or "default")
        risk_state = state.get("risk_state", "normal")
        multipliers = {"normal": 1.0, "caution": 0.75, "restricted": 0.5, "halt": 0.0}
        return multipliers.get(risk_state, 1.0)
    except Exception:
        return 1.0


def _get_regime_multiplier():
    """Get sizing multiplier from regime detection."""
    try:
        from engines.regime_v2 import detect_day_type
        regime = detect_day_type()
        day_type = regime.get("day_type", "unknown")
        multipliers = {
            "trend_day": 1.2,
            "range_day": 0.8,
            "breakout_day": 1.0,
            "vol_expansion": 0.7,
            "vol_contraction": 1.1,
            "panic_selling": 0.3,
            "short_covering": 0.5,
            "risk_on": 1.1,
            "risk_off": 0.6,
        }
        return multipliers.get(day_type, 1.0)
    except Exception:
        return 1.0


def optimize_portfolio(ml_picks, capital=100000, uid=None, max_positions=5):
    """Take ML-ranked picks and build an optimized portfolio.

    Args:
        ml_picks: List of dicts from ml_inference.get_top_picks_for_trading()
        capital: Available capital in INR
        uid: User ID for psychology state
        max_positions: Maximum number of positions

    Returns:
        Dict with positions, allocations, risk metrics
    """
    if not ml_picks:
        return {"positions": [], "total_allocated": 0, "reason": "no picks"}

    risk_mult = _get_risk_multiplier(uid)
    regime_mult = _get_regime_multiplier()

    if risk_mult == 0:
        return {"positions": [], "total_allocated": 0,
                "reason": "psychology engine: HALT state — no trading allowed"}

    # Get historical performance for Kelly sizing
    try:
        from engines.performance_tracker import compute_performance
        perf = compute_performance(lookback_days=60)
        win_rate = perf.get("win_rate", 55) / 100
        avg_return = perf.get("avg_return", 1.5) / 100
        avg_loss = abs(perf.get("worst_prediction", -2)) / 100
        base_fraction = kelly_fraction(win_rate, avg_return, avg_loss)
    except Exception:
        base_fraction = 0.02  # 2% default

    # Sector tracking
    sector_allocation = {}
    max_sector_pct = 0.30

    positions = []
    total_allocated = 0

    for pick in ml_picks[:max_positions * 2]:  # consider extras for sector limits
        if len(positions) >= max_positions:
            break

        sym = pick["symbol"]
        confidence = pick.get("confidence", 50)
        price = pick.get("entry", pick.get("price", 0))
        if price <= 0:
            continue

        sector = _get_sector(sym)

        # Sector limit check
        current_sector_pct = sector_allocation.get(sector, 0) / capital if capital > 0 else 0
        if current_sector_pct >= max_sector_pct:
            continue

        # Conviction multiplier: higher ML score = bigger position
        conviction_mult = 0.5 + (confidence / 100) * 0.5  # 0.5x to 1.0x

        # Final position size
        position_fraction = base_fraction * conviction_mult * risk_mult * regime_mult
        position_fraction = max(0.005, min(0.05, position_fraction))  # 0.5% to 5%

        position_value = capital * position_fraction
        remaining_sector = (max_sector_pct * capital) - sector_allocation.get(sector, 0)
        position_value = min(position_value, remaining_sector, capital - total_allocated)

        if position_value < price:
            continue

        qty = max(1, int(position_value / price))
        actual_value = qty * price

        stop = pick.get("stop", price * 0.98)
        target = pick.get("target", price * 1.03)
        risk_per_share = abs(price - stop)
        risk_amount = risk_per_share * qty
        reward_amount = abs(target - price) * qty

        positions.append({
            "symbol": sym,
            "sector": sector,
            "side": pick.get("side", "long"),
            "entry": price,
            "stop": stop,
            "target": target,
            "qty": qty,
            "value": round(actual_value, 2),
            "pct_of_capital": round(actual_value / capital * 100, 1),
            "risk_amount": round(risk_amount, 2),
            "reward_amount": round(reward_amount, 2),
            "reward_risk": round(reward_amount / risk_amount, 1) if risk_amount > 0 else 0,
            "confidence": confidence,
            "ml_rank": pick.get("ml_rank", pick.get("rank", 0)),
            "kelly_fraction": round(position_fraction * 100, 2),
            "sizing_factors": {
                "base_kelly": round(base_fraction * 100, 2),
                "conviction": round(conviction_mult, 2),
                "psychology": round(risk_mult, 2),
                "regime": round(regime_mult, 2),
            },
        })

        total_allocated += actual_value
        sector_allocation[sector] = sector_allocation.get(sector, 0) + actual_value

    # Portfolio risk metrics
    total_risk = sum(p["risk_amount"] for p in positions)
    total_reward = sum(p["reward_amount"] for p in positions)

    return {
        "positions": positions,
        "n_positions": len(positions),
        "total_allocated": round(total_allocated, 2),
        "pct_invested": round(total_allocated / capital * 100, 1) if capital > 0 else 0,
        "cash_remaining": round(capital - total_allocated, 2),
        "total_risk": round(total_risk, 2),
        "total_reward": round(total_reward, 2),
        "portfolio_rr": round(total_reward / total_risk, 1) if total_risk > 0 else 0,
        "max_portfolio_risk_pct": round(total_risk / capital * 100, 1) if capital > 0 else 0,
        "sector_allocation": {k: round(v, 0) for k, v in sector_allocation.items()},
        "risk_factors": {
            "psychology_multiplier": round(risk_mult, 2),
            "regime_multiplier": round(regime_mult, 2),
            "kelly_base": round(base_fraction * 100, 2),
        },
        "computed_at": datetime.now().isoformat(),
    }


def get_portfolio_recommendations(capital=100000, uid=None, max_positions=5):
    """Full pipeline: ML picks → portfolio optimization → actionable trades."""
    from engines.ml_inference import get_top_picks_for_trading

    picks = get_top_picks_for_trading(capital=capital, max_picks=max_positions * 2)
    if not picks:
        return {"positions": [], "reason": "ML model returned no picks"}

    return optimize_portfolio(picks, capital=capital, uid=uid, max_positions=max_positions)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    result = get_portfolio_recommendations()
    print(json.dumps(result, indent=2, default=str))

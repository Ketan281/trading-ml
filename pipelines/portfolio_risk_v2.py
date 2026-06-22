"""
Portfolio risk intelligence — institutional-grade risk engine.

Extends the basic portfolio_risk.py with:
  - Sector concentration risk (Herfindahl index)
  - Correlation risk (rolling pairwise correlation matrix)
  - Exposure risk (net/gross, directional)
  - Portfolio volatility (variance-covariance)
  - Portfolio drawdown risk (historical max-DD, current DD)
  - Capital allocation multiplier (risk-regime-aware)
  - Risk regime classification (5 regimes)

All rule-based. No ML.
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.cross_sectional import load_prices

INDUSTRIES_PATH = os.path.join(ROOT, "data", "historical", "industries.json")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_CAPITAL = 500_000
LOOKBACK_CORR = 60
LOOKBACK_VOL = 20
LOOKBACK_DD = 252

RISK_REGIMES = {
    "very_low":  {"threshold": 20, "alloc_mult": 1.3, "max_positions": 8},
    "low":       {"threshold": 35, "alloc_mult": 1.1, "max_positions": 6},
    "normal":    {"threshold": 55, "alloc_mult": 1.0, "max_positions": 5},
    "elevated":  {"threshold": 75, "alloc_mult": 0.7, "max_positions": 4},
    "extreme":   {"threshold": 100, "alloc_mult": 0.4, "max_positions": 2},
}


def _load_industries():
    if os.path.exists(INDUSTRIES_PATH):
        with open(INDUSTRIES_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _returns_matrix(symbols, prices=None, lookback=60):
    if prices is None:
        prices = load_prices(universe=set(symbols))
    rets = {}
    for s in symbols:
        df = prices.get(s)
        if df is not None and len(df) > lookback:
            rets[s] = df["Close"].pct_change().iloc[-lookback:]
    return pd.DataFrame(rets).dropna(how="all")


# ── Sector Concentration Risk ───────────────────────

def sector_concentration_risk(positions, industries=None):
    if not positions:
        return {"herfindahl": 0, "top_sector": None, "top_pct": 0,
                "risk_level": "none", "sector_breakdown": {}}
    industries = industries or _load_industries()
    sector_values = {}
    total = 0.0
    for pos in positions:
        sym = pos.get("symbol", "")
        val = abs(pos.get("value", 0))
        total += val
        sector = industries.get(sym, "Other")
        sector_values[sector] = sector_values.get(sector, 0) + val

    if total == 0:
        return {"herfindahl": 0, "top_sector": None, "top_pct": 0,
                "risk_level": "none", "sector_breakdown": {}}

    weights = {s: v / total for s, v in sector_values.items()}
    hhi = sum(w ** 2 for w in weights.values())

    top_sector = max(weights, key=weights.get)
    top_pct = round(weights[top_sector] * 100, 1)

    if hhi > 0.5:
        risk = "extreme"
    elif hhi > 0.3:
        risk = "high"
    elif hhi > 0.18:
        risk = "moderate"
    else:
        risk = "low"

    return {
        "herfindahl": round(hhi, 4),
        "top_sector": top_sector,
        "top_pct": top_pct,
        "risk_level": risk,
        "sector_breakdown": {s: round(w * 100, 1) for s, w in weights.items()},
    }


# ── Correlation Risk ────────────────────────────────

def correlation_risk(positions, prices=None):
    symbols = [p.get("symbol", "") for p in positions if p.get("symbol")]
    if len(symbols) < 2:
        return {"avg_correlation": 0, "max_pair": None, "max_corr": 0,
                "risk_level": "low", "high_corr_pairs": []}

    ret_df = _returns_matrix(symbols, prices, LOOKBACK_CORR)
    if ret_df.shape[1] < 2:
        return {"avg_correlation": 0, "max_pair": None, "max_corr": 0,
                "risk_level": "low", "high_corr_pairs": []}

    corr = ret_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    upper = corr.where(mask)
    pairs = upper.stack().reset_index()
    pairs.columns = ["sym1", "sym2", "corr"]
    pairs = pairs.sort_values("corr", ascending=False)

    avg_corr = float(pairs["corr"].mean())
    high_pairs = pairs[pairs["corr"] > 0.6].to_dict("records")
    max_row = pairs.iloc[0] if len(pairs) > 0 else None

    if avg_corr > 0.7:
        risk = "extreme"
    elif avg_corr > 0.5:
        risk = "high"
    elif avg_corr > 0.3:
        risk = "moderate"
    else:
        risk = "low"

    return {
        "avg_correlation": round(avg_corr, 3),
        "max_pair": f"{max_row['sym1']}/{max_row['sym2']}" if max_row is not None else None,
        "max_corr": round(float(max_row["corr"]), 3) if max_row is not None else 0,
        "risk_level": risk,
        "high_corr_pairs": [{
            "pair": f"{r['sym1']}/{r['sym2']}",
            "corr": round(r["corr"], 3),
        } for r in high_pairs[:5]],
    }


# ── Exposure Risk ───────────────────────────────────

def exposure_risk(positions, capital=DEFAULT_CAPITAL):
    if not positions:
        return {"gross_exposure": 0, "net_exposure": 0, "long_pct": 0,
                "short_pct": 0, "risk_level": "none"}

    long_val = sum(p.get("value", 0) for p in positions
                   if p.get("direction", p.get("action", "buy")) in ("buy", "long"))
    short_val = sum(abs(p.get("value", 0)) for p in positions
                    if p.get("direction", p.get("action", "")) in ("sell", "short"))

    gross = long_val + short_val
    net = long_val - short_val
    gross_pct = gross / capital * 100
    net_pct = net / capital * 100

    if gross_pct > 150:
        risk = "extreme"
    elif gross_pct > 100:
        risk = "high"
    elif gross_pct > 60:
        risk = "moderate"
    else:
        risk = "low"

    return {
        "gross_exposure": round(gross, 2),
        "gross_pct": round(gross_pct, 1),
        "net_exposure": round(net, 2),
        "net_pct": round(net_pct, 1),
        "long_pct": round(long_val / capital * 100, 1),
        "short_pct": round(short_val / capital * 100, 1),
        "risk_level": risk,
    }


# ── Portfolio Volatility ────────────────────────────

def portfolio_volatility(positions, prices=None, capital=DEFAULT_CAPITAL):
    symbols = [p.get("symbol", "") for p in positions if p.get("symbol")]
    if len(symbols) < 1:
        return {"daily_vol": 0, "annualized_vol": 0, "risk_level": "low"}

    ret_df = _returns_matrix(symbols, prices, LOOKBACK_VOL)
    if ret_df.empty:
        return {"daily_vol": 0, "annualized_vol": 0, "risk_level": "low"}

    total_val = sum(abs(p.get("value", 0)) for p in positions)
    if total_val == 0:
        return {"daily_vol": 0, "annualized_vol": 0, "risk_level": "low"}

    weights = []
    used_syms = []
    for p in positions:
        s = p.get("symbol", "")
        if s in ret_df.columns:
            weights.append(abs(p.get("value", 0)) / total_val)
            used_syms.append(s)

    if not used_syms:
        return {"daily_vol": 0, "annualized_vol": 0, "risk_level": "low"}

    sub = ret_df[used_syms].dropna()
    if len(sub) < 5:
        return {"daily_vol": 0, "annualized_vol": 0, "risk_level": "low"}

    w = np.array(weights)
    cov = sub.cov().values
    port_var = w @ cov @ w
    daily_vol = np.sqrt(port_var)
    ann_vol = daily_vol * np.sqrt(252)

    if ann_vol > 0.40:
        risk = "extreme"
    elif ann_vol > 0.25:
        risk = "high"
    elif ann_vol > 0.15:
        risk = "moderate"
    else:
        risk = "low"

    return {
        "daily_vol": round(daily_vol * 100, 2),
        "annualized_vol": round(ann_vol * 100, 2),
        "risk_level": risk,
    }


# ── Drawdown Risk ───────────────────────────────────

def drawdown_risk(positions, prices=None):
    symbols = [p.get("symbol", "") for p in positions if p.get("symbol")]
    if not symbols:
        return {"max_dd": 0, "current_dd": 0, "risk_level": "low"}

    ret_df = _returns_matrix(symbols, prices, LOOKBACK_DD)
    if ret_df.empty or ret_df.shape[1] < 1:
        return {"max_dd": 0, "current_dd": 0, "risk_level": "low"}

    eq_weight_ret = ret_df.mean(axis=1)
    cumulative = (1 + eq_weight_ret).cumprod()
    running_max = cumulative.cummax()
    drawdowns = (cumulative - running_max) / running_max
    max_dd = float(drawdowns.min()) * 100
    current_dd = float(drawdowns.iloc[-1]) * 100

    if max_dd < -25:
        risk = "extreme"
    elif max_dd < -15:
        risk = "high"
    elif max_dd < -8:
        risk = "moderate"
    else:
        risk = "low"

    return {
        "max_dd": round(max_dd, 2),
        "current_dd": round(current_dd, 2),
        "risk_level": risk,
    }


# ── Risk Regime Classification ──────────────────────

def classify_risk_regime(sector_risk, corr_risk, exposure_risk_data,
                         vol_data, dd_data):
    risk_scores = {
        "none": 0, "low": 1, "moderate": 2, "high": 3, "extreme": 4
    }
    components = [
        sector_risk.get("risk_level", "low"),
        corr_risk.get("risk_level", "low"),
        exposure_risk_data.get("risk_level", "low"),
        vol_data.get("risk_level", "low"),
        dd_data.get("risk_level", "low"),
    ]
    avg_score = np.mean([risk_scores.get(c, 1) for c in components])
    composite = avg_score / 4.0 * 100

    regime = "normal"
    alloc_mult = 1.0
    max_pos = 5
    for name, cfg in sorted(RISK_REGIMES.items(), key=lambda x: x[1]["threshold"]):
        if composite <= cfg["threshold"]:
            regime = name
            alloc_mult = cfg["alloc_mult"]
            max_pos = cfg["max_positions"]
            break

    return {
        "regime": regime,
        "composite_risk_score": round(composite, 1),
        "capital_allocation_multiplier": alloc_mult,
        "max_positions_allowed": max_pos,
        "component_risks": dict(zip(
            ["sector", "correlation", "exposure", "volatility", "drawdown"],
            components
        )),
    }


# ── Full Portfolio Risk Read ────────────────────────

def portfolio_risk_v2_read(positions, capital=DEFAULT_CAPITAL, prices=None):
    industries = _load_industries()
    sector = sector_concentration_risk(positions, industries)
    corr = correlation_risk(positions, prices)
    exp = exposure_risk(positions, capital)
    vol = portfolio_volatility(positions, prices, capital)
    dd = drawdown_risk(positions, prices)
    regime = classify_risk_regime(sector, corr, exp, vol, dd)

    result = {
        "timestamp": datetime.now().isoformat(),
        "n_positions": len(positions),
        "capital": capital,
        "sector_concentration": sector,
        "correlation": corr,
        "exposure": exp,
        "volatility": vol,
        "drawdown": dd,
        "risk_regime": regime,
    }

    path = os.path.join(OUTPUT_DIR, f"portfolio_risk_v2_{datetime.now():%Y%m%d_%H%M}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


if __name__ == "__main__":
    test_positions = [
        {"symbol": "RELIANCE", "value": 100_000, "action": "buy"},
        {"symbol": "HDFCBANK", "value": 80_000, "action": "buy"},
        {"symbol": "ICICIBANK", "value": 60_000, "action": "buy"},
        {"symbol": "TCS", "value": 70_000, "action": "buy"},
        {"symbol": "INFY", "value": 50_000, "action": "buy"},
    ]

    print("=" * 64)
    print("  PORTFOLIO RISK INTELLIGENCE v2")
    print("=" * 64)

    r = portfolio_risk_v2_read(test_positions)

    print(f"\n  Positions: {r['n_positions']}  Capital: {r['capital']:,}")
    print(f"\n  Sector Concentration:")
    print(f"    HHI: {r['sector_concentration']['herfindahl']}")
    print(f"    Top: {r['sector_concentration']['top_sector']} ({r['sector_concentration']['top_pct']}%)")
    print(f"    Risk: {r['sector_concentration']['risk_level']}")

    print(f"\n  Correlation:")
    print(f"    Avg: {r['correlation']['avg_correlation']}")
    print(f"    Max: {r['correlation']['max_pair']} ({r['correlation']['max_corr']})")
    print(f"    Risk: {r['correlation']['risk_level']}")

    print(f"\n  Exposure:")
    print(f"    Gross: {r['exposure']['gross_pct']}%  Net: {r['exposure']['net_pct']}%")
    print(f"    Risk: {r['exposure']['risk_level']}")

    print(f"\n  Volatility:")
    print(f"    Daily: {r['volatility']['daily_vol']}%  Annual: {r['volatility']['annualized_vol']}%")
    print(f"    Risk: {r['volatility']['risk_level']}")

    print(f"\n  Drawdown:")
    print(f"    Max: {r['drawdown']['max_dd']}%  Current: {r['drawdown']['current_dd']}%")
    print(f"    Risk: {r['drawdown']['risk_level']}")

    print(f"\n  RISK REGIME: {r['risk_regime']['regime'].upper()}")
    print(f"    Composite Score: {r['risk_regime']['composite_risk_score']}")
    print(f"    Allocation Mult: {r['risk_regime']['capital_allocation_multiplier']}x")
    print(f"    Max Positions: {r['risk_regime']['max_positions_allowed']}")

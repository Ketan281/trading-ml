import os
import sys
import json
import numpy as np
import pandas as pd
from datetime        import datetime
from scipy.stats     import norm

ROOT = os.path.dirname(os.path.dirname(
       os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

sys.path.insert(0, os.path.join(ROOT, "pipelines", "options"))
from options_chain import (fetch_options_chain,
                            get_strike_step,
                            round_to_strike)

OUTPUT_DIR = os.path.join(ROOT, "data", "options")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Black Scholes Greeks ──────────────────────────────
def black_scholes_greeks(S, K, T, r, sigma, opt_type="CE"):
    """
    S     = Spot price
    K     = Strike price
    T     = Time to expiry in years
    r     = Risk free rate (use 0.065 for India)
    sigma = Implied volatility (as decimal e.g. 0.15)
    """
    try:
        if T <= 0 or sigma <= 0:
            return None

        d1 = (
            np.log(S / K) + (r + 0.5 * sigma ** 2) * T
        ) / (sigma * np.sqrt(T))

        d2 = d1 - sigma * np.sqrt(T)

        if opt_type == "CE":
            # Call Greeks
            delta = norm.cdf(d1)
            theta = (
                -(S * norm.pdf(d1) * sigma)
                / (2 * np.sqrt(T))
                - r * K * np.exp(-r * T) * norm.cdf(d2)
            ) / 365
            price = (
                S * norm.cdf(d1)
                - K * np.exp(-r * T) * norm.cdf(d2)
            )
        else:
            # Put Greeks
            delta = norm.cdf(d1) - 1
            theta = (
                -(S * norm.pdf(d1) * sigma)
                / (2 * np.sqrt(T))
                + r * K * np.exp(-r * T)
                * norm.cdf(-d2)
            ) / 365
            price = (
                K * np.exp(-r * T) * norm.cdf(-d2)
                - S * norm.cdf(-d1)
            )

        # Common Greeks
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega  = S * norm.pdf(d1) * np.sqrt(T) / 100

        return {
            "delta": round(delta, 4),
            "gamma": round(gamma, 6),
            "theta": round(theta, 4),
            "vega" : round(vega,  4),
            "price": round(price, 2),
            "d1"   : round(d1,    4),
            "d2"   : round(d2,    4)
        }

    except Exception as e:
        return None

# ── Greeks from Chain Data ────────────────────────────
def extract_chain_greeks(df, symbol):
    spot = df["spot"].iloc[0]
    step = get_strike_step(symbol)
    atm  = round_to_strike(spot, step)

    greeks_data = []

    for _, row in df.iterrows():
        strike = row["strike"]

        entry = {
            "strike"   : strike,
            "moneyness": row.get("moneyness", ""),
            "ce": {
                "ltp"  : row.get("ce_ltp",   0),
                "delta": row.get("ce_delta",  0),
                "gamma": row.get("ce_gamma",  0),
                "theta": row.get("ce_theta",  0),
                "vega" : row.get("ce_vega",   0),
                "iv"   : row.get("ce_iv",     0)
            },
            "pe": {
                "ltp"  : row.get("pe_ltp",   0),
                "delta": row.get("pe_delta",  0),
                "gamma": row.get("pe_gamma",  0),
                "theta": row.get("pe_theta",  0),
                "vega" : row.get("pe_vega",   0),
                "iv"   : row.get("pe_iv",     0)
            }
        }
        greeks_data.append(entry)

    return greeks_data

# ── Compute BS Greeks For All Strikes ─────────────────
def compute_bs_greeks(df, symbol, dte=7):
    spot    = df["spot"].iloc[0]
    step    = get_strike_step(symbol)
    atm     = round_to_strike(spot, step)
    T       = dte / 365
    r       = 0.065  # India risk free rate

    bs_results = []

    for _, row in df.iterrows():
        strike = row["strike"]
        ce_iv  = row.get("ce_iv", 0) / 100
        pe_iv  = row.get("pe_iv", 0) / 100

        entry = {"strike": strike}

        if ce_iv > 0:
            ce_greeks = black_scholes_greeks(
                spot, strike, T, r, ce_iv, "CE"
            )
            if ce_greeks:
                entry["ce_bs"] = ce_greeks

        if pe_iv > 0:
            pe_greeks = black_scholes_greeks(
                spot, strike, T, r, pe_iv, "PE"
            )
            if pe_greeks:
                entry["pe_bs"] = pe_greeks

        bs_results.append(entry)

    return bs_results

# ── ATM Greeks Summary ────────────────────────────────
def atm_greeks_summary(df, symbol, dte=7):
    spot   = df["spot"].iloc[0]
    step   = get_strike_step(symbol)
    atm    = round_to_strike(spot, step)
    T      = dte / 365
    r      = 0.065

    atm_row = df[df["strike"] == atm]
    if atm_row.empty:
        # Get nearest strike
        df_copy = df.copy()
        df_copy["dist"] = abs(df_copy["strike"] - atm)
        atm_row = df_copy.nsmallest(1, "dist")

    row    = atm_row.iloc[0]
    ce_iv  = row.get("ce_iv", 15) / 100
    pe_iv  = row.get("pe_iv", 15) / 100

    ce_g   = black_scholes_greeks(
                 spot, atm, T, r, ce_iv, "CE"
             ) or {}
    pe_g   = black_scholes_greeks(
                 spot, atm, T, r, pe_iv, "PE"
             ) or {}

    return {
        "symbol"     : symbol,
        "spot"       : spot,
        "atm"        : atm,
        "dte"        : dte,
        "atm_ce": {
            "ltp"  : row.get("ce_ltp",   0),
            "iv"   : row.get("ce_iv",    0),
            "delta": ce_g.get("delta",   0),
            "gamma": ce_g.get("gamma",   0),
            "theta": ce_g.get("theta",   0),
            "vega" : ce_g.get("vega",    0)
        },
        "atm_pe": {
            "ltp"  : row.get("pe_ltp",   0),
            "iv"   : row.get("pe_iv",    0),
            "delta": pe_g.get("delta",   0),
            "gamma": pe_g.get("gamma",   0),
            "theta": pe_g.get("theta",   0),
            "vega" : pe_g.get("vega",    0)
        }
    }

# ── Greeks Risk Interpretation ────────────────────────
def interpret_greeks(atm_summary):
    ce = atm_summary["atm_ce"]
    pe = atm_summary["atm_pe"]

    insights = []

    # Delta insight
    ce_delta = abs(ce.get("delta", 0))
    if ce_delta > 0.6:
        insights.append(
            "Deep ITM calls — high delta risk"
        )
    elif ce_delta > 0.45:
        insights.append(
            "ATM calls — balanced delta exposure"
        )
    else:
        insights.append(
            "OTM calls — low delta, needs big move"
        )

    # Theta insight
    ce_theta = abs(ce.get("theta", 0))
    if ce_theta > 50:
        insights.append(
            f"High theta decay ₹{ce_theta}/day — "
            f"time is working against buyers"
        )
    elif ce_theta > 20:
        insights.append(
            f"Moderate theta ₹{ce_theta}/day"
        )
    else:
        insights.append(
            f"Low theta ₹{ce_theta}/day — "
            f"time decay manageable"
        )

    # Gamma insight
    ce_gamma = ce.get("gamma", 0)
    if ce_gamma > 0.005:
        insights.append(
            "High gamma — large P&L swings near expiry"
        )
    else:
        insights.append(
            "Normal gamma — stable delta movement"
        )

    # Vega insight
    ce_vega = ce.get("vega", 0)
    if ce_vega > 50:
        insights.append(
            "High vega — position very sensitive to IV"
        )
    else:
        insights.append(
            "Normal vega — manageable IV sensitivity"
        )

    return insights

# ── Strategy Greeks ───────────────────────────────────
def strategy_greeks(df, symbol, strategy,
                    strikes, dte=7):
    """
    Calculate net Greeks for common strategies
    strategy: straddle, strangle, bull_spread,
              bear_spread, iron_condor
    strikes : list of strikes involved
    """
    spot = df["spot"].iloc[0]
    T    = dte / 365
    r    = 0.065

    net_delta = 0
    net_gamma = 0
    net_theta = 0
    net_vega  = 0
    net_cost  = 0

    legs = []

    if strategy == "straddle":
        atm    = strikes[0]
        atm_row = df[df["strike"] == atm]
        if atm_row.empty:
            return None

        row    = atm_row.iloc[0]
        ce_iv  = row.get("ce_iv", 15) / 100
        pe_iv  = row.get("pe_iv", 15) / 100

        ce_g   = black_scholes_greeks(
                     spot, atm, T, r, ce_iv, "CE"
                 ) or {}
        pe_g   = black_scholes_greeks(
                     spot, atm, T, r, pe_iv, "PE"
                 ) or {}

        # Long straddle = buy CE + buy PE
        legs = [
            {"type": "CE", "strike": atm,
             "action": "buy", "greeks": ce_g},
            {"type": "PE", "strike": atm,
             "action": "buy", "greeks": pe_g}
        ]

        net_delta = (
            ce_g.get("delta", 0) +
            pe_g.get("delta", 0)
        )
        net_gamma = (
            ce_g.get("gamma", 0) +
            pe_g.get("gamma", 0)
        )
        net_theta = (
            ce_g.get("theta", 0) +
            pe_g.get("theta", 0)
        )
        net_vega  = (
            ce_g.get("vega", 0) +
            pe_g.get("vega", 0)
        )
        net_cost  = (
            row.get("ce_ltp", 0) +
            row.get("pe_ltp", 0)
        )

    return {
        "strategy" : strategy,
        "symbol"   : symbol,
        "spot"     : spot,
        "legs"     : legs,
        "net_greeks": {
            "delta": round(net_delta, 4),
            "gamma": round(net_gamma, 6),
            "theta": round(net_theta, 4),
            "vega" : round(net_vega,  4)
        },
        "net_cost" : round(net_cost, 2),
        "breakeven": {
            "upper": round(
                strikes[0] + net_cost, 2
            ),
            "lower": round(
                strikes[0] - net_cost, 2
            )
        }
    }

# ── Full Greeks Analysis ──────────────────────────────
def full_greeks_analysis(symbol):
    print(f"\n{'=' * 55}")
    print(f"  Greeks Engine — {symbol}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 55}")

    # Fetch chain
    result = fetch_options_chain(symbol)
    if not result or result["df"].empty:
        print(f"  ❌ Could not fetch chain for {symbol}")
        return None

    df     = result["df"]
    spot   = result["spot"]
    atm    = result["atm"]
    expiry = result["expiry"]

    # Days to expiry
    try:
        exp_date = datetime.strptime(expiry, "%d-%b-%Y")
        dte      = max((exp_date - datetime.now()).days, 1)
    except Exception:
        dte = 7

    print(f"  Spot : ₹{spot} | ATM : {atm} | DTE : {dte}")

    # ATM Greeks
    atm_summary = atm_greeks_summary(df, symbol, dte)
    insights    = interpret_greeks(atm_summary)

    # Print ATM Greeks
    print(f"\n  🎯 ATM Greeks Summary:")
    print(f"\n  {'GREEK':<10} {'ATM CALL':<15} {'ATM PUT'}")
    print("  " + "─" * 38)

    greeks_list = ["delta", "gamma", "theta", "vega"]
    for g in greeks_list:
        ce_val = atm_summary["atm_ce"].get(g, 0)
        pe_val = atm_summary["atm_pe"].get(g, 0)
        print(
            f"  {g.upper():<10} "
            f"{str(ce_val):<15} "
            f"{pe_val}"
        )

    print(f"\n  ATM CE LTP : ₹{atm_summary['atm_ce']['ltp']}")
    print(f"  ATM PE LTP : ₹{atm_summary['atm_pe']['ltp']}")
    print(f"  ATM CE IV  : {atm_summary['atm_ce']['iv']}%")
    print(f"  ATM PE IV  : {atm_summary['atm_pe']['iv']}%")

    # Greeks insights
    print(f"\n  💡 Greeks Insights:")
    for insight in insights:
        print(f"     → {insight}")

    # Straddle analysis
    print(f"\n  📐 Straddle Analysis (ATM):")
    straddle = strategy_greeks(
        df, symbol, "straddle", [atm], dte
    )
    if straddle:
        ng = straddle["net_greeks"]
        print(f"     Net Cost     : ₹{straddle['net_cost']}")
        print(f"     Breakeven Up : "
              f"₹{straddle['breakeven']['upper']}")
        print(f"     Breakeven Dn : "
              f"₹{straddle['breakeven']['lower']}")
        print(f"     Net Delta    : {ng['delta']}")
        print(f"     Net Gamma    : {ng['gamma']}")
        print(f"     Net Theta    : {ng['theta']}")
        print(f"     Net Vega     : {ng['vega']}")

    # Greeks table for nearby strikes
    print(f"\n  📊 Greeks Table (ATM ± 3 strikes):")
    step = get_strike_step(symbol)
    print(f"  {'STRIKE':<10} {'CE Δ':<10} {'CE Θ':<10} "
          f"{'CE Γ':<10} {'PE Δ':<10} {'PE Θ'}")
    print("  " + "─" * 58)

    T = dte / 365
    r = 0.065

    nearby = df[
        (df["strike"] >= atm - 3 * step) &
        (df["strike"] <= atm + 3 * step)
    ]

    for _, row in nearby.iterrows():
        strike = row["strike"]
        ce_iv  = row.get("ce_iv", 15) / 100
        pe_iv  = row.get("pe_iv", 15) / 100

        ce_g   = black_scholes_greeks(
                     spot, strike, T, r, ce_iv, "CE"
                 ) or {}
        pe_g   = black_scholes_greeks(
                     spot, strike, T, r, pe_iv, "PE"
                 ) or {}

        marker = " ◄" if strike == atm else ""
        print(
            f"  {int(strike):<10} "
            f"{str(ce_g.get('delta', 0)):<10} "
            f"{str(ce_g.get('theta', 0)):<10} "
            f"{str(ce_g.get('gamma', 0)):<10} "
            f"{str(pe_g.get('delta', 0)):<10} "
            f"{pe_g.get('theta', 0)}"
            f"{marker}"
        )

    # Build report
    report = {
        "symbol"     : symbol,
        "timestamp"  : datetime.now().isoformat(),
        "spot"       : spot,
        "atm"        : atm,
        "dte"        : dte,
        "atm_summary": atm_summary,
        "insights"   : insights,
        "straddle"   : straddle
    }

    # Save
    path = os.path.join(
        OUTPUT_DIR,
        f"{symbol}_greeks_"
        f"{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    )
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  ✅ Greeks analysis saved → {path}")
    return report

# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    # Install scipy if needed
    try:
        from scipy.stats import norm
    except ImportError:
        import subprocess
        subprocess.run(
            ["pip", "install", "scipy"],
            check=True
        )
        from scipy.stats import norm

    for symbol in ["NIFTY", "BANKNIFTY"]:
        full_greeks_analysis(symbol)

    print("\n  ✅ Greeks Engine complete!")
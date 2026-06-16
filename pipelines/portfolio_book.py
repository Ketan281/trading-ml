"""
Portfolio book builder — ONE command → a regime-scaled, diversified,
explained, risk-managed book.

This is the capstone that wires the whole equity-intelligence stack together:

  market context   regime classifier + market breadth
        │
  selection        ensemble meta-model (momentum+quality+sector+intra-RS)
        │
  gates            fundamental quality gate  +  liquidity gate
        │
  construction     portfolio optimizer (sector caps, correlation caps,
                   inverse-vol weights, gross scaled by regime + drawdown)
        │
  per-name risk    dynamic stop-loss engine → shares, stop, per-trade risk
        │
  explainability   ranker SHAP + ensemble breakdown → plain-English reason
        │
  portfolio risk   heat check vs the regime risk policy

Output: a complete book you could actually trade, with a why for every name.
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.cross_sectional   import load_prices
from models.ensemble          import ensemble_score
from models.fundamentals      import quality_scores
from models.regime_classifier import classify
from models.explain           import explain, narrative
from pipelines.breadth        import breadth_read
from pipelines.portfolio_optimizer import optimize
from pipelines.stops          import dynamic_stops
from pipelines.risk_policy    import effective_limits, MAX_PORTFOLIO_HEAT

OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_QUALITY   = 35.0
MIN_TURNOVER  = 3e7        # ₹3 cr median daily turnover
TURNOVER_DAYS = 60
CAND_POOL     = 40         # quality/liquid names handed to the optimizer
CAPITAL       = 1_000_000


def _liquid(df):
    r = df.tail(TURNOVER_DAYS)
    return float((r["Close"] * r["Volume"]).median())


def build_book(capital=CAPITAL, current_dd=0.0):
    print("=" * 78)
    print("  PORTFOLIO BOOK BUILDER")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)

    # ── 1. Market context ────────────────────────────────
    regime = classify("NIFTY"); breadth = breadth_read()
    print(f"  Regime   : {regime['regime'].upper()} (conf {regime.get('confidence')})"
          f"  →  {regime.get('hint','')}")
    print(f"  Breadth  : {breadth['score']}/100 — {breadth['signal']}")
    lim = effective_limits(regime["regime"], current_dd, breadth.get("score"))
    print(f"  Policy   : gross ≤{lim['gross_effective']:.0%} | "
          f"{lim['max_positions']} pos | sector ≤{lim['max_sector_pct']:.0%} | "
          f"per-trade risk {lim['per_trade_risk_pct']:.2%}")
    if lim["gross_effective"] <= 0:
        print("\n  ⛔ Drawdown circuit-breaker: gross 0% — NO new positions.")
        return None

    # ── 2. Ensemble selection ────────────────────────────
    ens = ensemble_score(pool=500)
    if ens is None:
        print("  ⚠ Ranker not trained."); return None

    # ── 3. Gates: quality + liquidity ────────────────────
    q = quality_scores(symbols=set(ens["symbol"]))
    if q is not None:
        gate = q[["pass_gate", "quality_score"]]
        ens = ens.merge(gate, left_on="symbol", right_index=True, how="left",
                        suffixes=("", "_g"))
        ens["pass_gate"] = ens["pass_gate"].fillna(True)
        keep = ens["pass_gate"] & (ens["quality_score"].fillna(50) >= MIN_QUALITY)
        dropped = ens[~keep]["symbol"].head(10).tolist()
        ens = ens[keep]
        if dropped:
            print(f"  🧹 Quality gate dropped: {', '.join(dropped)}")

    prices = load_prices(universe=set(ens["symbol"].head(120)))
    liq_ok = [s for s in ens["symbol"] if s in prices and _liquid(prices[s]) >= MIN_TURNOVER]
    cand = ens[ens["symbol"].isin(liq_ok)].head(CAND_POOL)
    # Optimizer ranks on 'score' → feed it the ENSEMBLE score (clean frame).
    opt_input = pd.DataFrame({"symbol": cand["symbol"].values,
                              "score": cand["ensemble_score"].values,
                              "sector": cand["sector"].values})
    print(f"  Candidates after gates: {len(opt_input)} (from ensemble top, "
          f"quality≥{MIN_QUALITY:.0f}, turnover≥₹{MIN_TURNOVER/1e7:.0f}cr)")

    # ── 4. Diversified construction ──────────────────────
    book_res = optimize(candidates=opt_input, capital=capital,
                        regime=regime["regime"], current_dd=current_dd)
    if not book_res or not book_res.get("book"):
        print("  ⚠ No names passed diversification constraints."); return None

    # ── 5. Per-name stops + sizing + 6. explanations ─────
    syms = [b["symbol"] for b in book_res["book"]]
    exp = explain(syms)
    holdings = []
    total_risk = 0.0
    for b in book_res["book"]:
        s = b["symbol"]; df = prices.get(s)
        price = float(df["Close"].iloc[-1]) if df is not None else None
        st = dynamic_stops(df) if df is not None else None
        shares = int(b["capital"] / price) if price else 0
        risk_rs = round(shares * (price - st["stop"])) if (st and price) else None
        if risk_rs:
            total_risk += risk_rs
        holdings.append({
            "symbol": s, "sector": b["sector"], "weight_pct": b["weight_pct"],
            "capital": b["capital"], "price": round(price, 2) if price else None,
            "shares": shares,
            "stop": st["stop"] if st else None,
            "stop_method": st["recommended_method"] if st else None,
            "stop_risk_pct": st["risk_pct"] if st else None,
            "risk_rupees": risk_rs,
            "ensemble_score": exp.get(s, {}).get("ensemble_score"),
            "why": narrative(exp.get(s, {})),
        })

    heat = round(total_risk / capital, 4)
    diag = {
        "regime": regime["regime"], "breadth_score": breadth["score"],
        "gross_book_pct": book_res["gross_book_pct"],
        "positions": book_res["positions"],
        "sector_exposure": book_res["sector_exposure"],
        "avg_pairwise_corr": book_res["avg_pairwise_corr"],
        "portfolio_heat_pct": round(heat * 100, 2),
        "heat_limit_pct": round(MAX_PORTFOLIO_HEAT * 100, 1),
        "heat_ok": heat <= MAX_PORTFOLIO_HEAT,
    }

    _print_book(holdings, diag, book_res)
    report = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "context": {"regime": regime, "breadth": breadth, "policy": lim},
              "holdings": holdings, "diagnostics": diag,
              "skipped": book_res.get("skipped")}
    path = os.path.join(OUTPUT_DIR, "portfolio_book.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  ✅ Book saved → {path}")
    return report


def _print_book(holdings, diag, book_res):
    print("\n  📒 CONSTRUCTED BOOK")
    print("  " + "─" * 74)
    print(f"  {'SYMBOL':<12}{'WT%':>6}{'CAPITAL':>11}{'SH':>6}{'ENTRY':>9}"
          f"{'STOP':>9}{'RISK%':>6} {'STOPTYPE':<11} SECTOR")
    for h in holdings:
        print(f"  {h['symbol']:<12}{h['weight_pct']:>6}{h['capital']:>11,}"
              f"{h['shares']:>6}{(h['price'] or 0):>9.1f}{(h['stop'] or 0):>9.1f}"
              f"{(h['stop_risk_pct'] or 0):>6.1f} {(h['stop_method'] or '-'):<11} "
              f"{h['sector'] or '-'}")
    print("  " + "─" * 74)
    print(f"  Gross {diag['gross_book_pct']}% | {diag['positions']} positions | "
          f"avg corr {diag['avg_pairwise_corr']} | "
          f"portfolio heat {diag['portfolio_heat_pct']}% "
          f"(limit {diag['heat_limit_pct']}%) "
          f"{'✓' if diag['heat_ok'] else '⚠ OVER'}")
    print(f"  Sector exposure: {diag['sector_exposure']}")
    print("\n  🧠 WHY EACH NAME (ranker SHAP + ensemble blend)")
    print("  " + "─" * 74)
    for h in holdings:
        print(f"   • {h['why']}")


if __name__ == "__main__":
    build_book()

"""
End-to-end stock screener: ranking model + rule engine.

This ties the two halves together into the headline deliverable:

    1. The cross-sectional model ranks the whole universe and tells us
       WHICH stocks have the strongest relative-strength edge.
    2. A liquidity filter drops names too thin to trade cleanly.
    3. The deterministic rule engine runs on each survivor to decide
       WHEN to enter and sets the ATR stop-loss and target.

Output = an actionable shortlist: "buy this stock, here is the entry zone,
the smart stop-loss and the target", plus a watchlist of strong names that
are not yet at a clean entry.
"""

import os
import sys
import json
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.cross_sectional   import rank_today, load_prices
from models.fundamentals       import quality_scores
from pipelines.indicators      import compute_indicators
from pipelines.patterns        import detect_patterns
from pipelines.decision_engine import decide
from pipelines.trader          import build_playbook, playbook_text
from models.calibrate          import calibrate_scores

OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

POOL_SIZE      = 60          # how many top-ranked names to examine
FINAL_SIZE     = 20          # how many to report after liquidity filter
MIN_TURNOVER   = 3e7         # ₹3 crore median daily turnover minimum
TURNOVER_DAYS  = 60

# How the final shortlist is ordered when fundamentals are available:
# blend price-momentum rank with fundamental quality (both z-scored).
MOM_WEIGHT     = 0.60
QUAL_WEIGHT     = 0.40
MIN_QUALITY    = 35.0        # drop bottom-third quality names from the shortlist


# ── Liquidity ─────────────────────────────────────────
def _median_turnover(df, days=TURNOVER_DAYS):
    recent = df.tail(days)
    return float((recent["Close"] * recent["Volume"]).median())


# ── Fundamentals: gate + blend ────────────────────────
def _apply_fundamentals(ranked):
    """Attach fundamental quality to the ranked pool, drop junk via the
    quality gate, and re-order by a blend of momentum rank + quality.
    Falls back to pure momentum if no fundamentals are cached."""
    import numpy as np

    qdf = quality_scores(symbols=set(ranked["symbol"]))
    if qdf is None:
        print("  ⚠ No fundamentals cached — ranking on price-momentum only.")
        print("    (run: python training/fetch_fundamentals.py)")
        ranked = ranked.copy()
        ranked["quality_score"] = float("nan")
        ranked["pass_gate"]     = True
        ranked["blended"]       = ranked["score"]
        ranked["fund_ok"]       = False
        return ranked

    q = qdf[["quality_score", "quality_z", "pass_gate"]].copy()
    q.index.name = "symbol"
    merged = ranked.merge(q, left_on="symbol", right_index=True, how="left")

    # Stocks with no fundamentals: keep, treat as neutral quality (don't gate out).
    merged["pass_gate"]     = merged["pass_gate"].fillna(True)
    merged["quality_z"]     = merged["quality_z"].fillna(0.0)
    merged["quality_score"] = merged["quality_score"].fillna(50.0)

    # Drop balance-sheet junk and the weakest-quality third.
    keep = merged["pass_gate"] & (merged["quality_score"] >= MIN_QUALITY)
    dropped = merged[~keep]
    if len(dropped):
        names = ", ".join(dropped["symbol"].head(12))
        print(f"  🧹 Quality filter dropped {len(dropped)} name(s): {names}")
    merged = merged[keep].copy()

    # Blend momentum (z-scored prob) with fundamental quality (z).
    s  = merged["score"]
    mz = (s - s.mean()) / (s.std() + 1e-9)
    merged["blended"] = MOM_WEIGHT * mz + QUAL_WEIGHT * merged["quality_z"]
    merged = merged.sort_values("blended", ascending=False).reset_index(drop=True)
    merged["rank"] = merged.index + 1
    merged["fund_ok"] = True
    return merged


# ── Screen ────────────────────────────────────────────
def screen(pool=POOL_SIZE, final=FINAL_SIZE, min_turnover=MIN_TURNOVER):
    print("=" * 70)
    print("  TRADING AI — STOCK SCREENER  (rank → liquidity → entry/stop)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    ranked = rank_today(top_n=pool)
    if ranked is None or ranked.empty:
        print("  ❌ Ranking model unavailable. Train it first:")
        print("     python models/cross_sectional.py")
        return None

    # Layer fundamental quality on top of price-momentum (gate + blend).
    ranked = _apply_fundamentals(ranked)

    top_symbols = list(ranked["symbol"])
    prices = load_prices(universe=set(top_symbols))

    candidates = []
    for _, row in ranked.iterrows():
        sym = row["symbol"]
        df  = prices.get(sym)
        if df is None:
            continue

        # Liquidity gate
        turnover = _median_turnover(df)
        if turnover < min_turnover:
            continue

        ind = compute_indicators(df, sym)
        if ind is None:
            continue

        # Price-action read for entry timing, then the rule engine decides.
        pats = detect_patterns(df)
        d    = decide(ind, patterns=pats)

        cand = {
            "rank_in_pool": int(row["rank"]),
            "symbol":       sym,
            "price":        ind["price"],
            "rank_score":   round(float(row["score"]), 4),
            # Calibrated P(out-perform): honest confidence (raw score is
            # over-confident at the extremes — see models/calibrate.py).
            "rank_conf":    round(float(calibrate_scores([row["score"]])[0]), 3),
            "quality":      round(float(row["quality_score"]), 1),
            "action":       d["action"],
            "engine_conf":  d["confidence"],
            "risk_level":   d["risk_level"],
            "entry_signal": d["entry_signal"],
            "pattern":      d["pattern"] or "-",
            "pattern_context": d["pattern_context"],
            "entry_zone":   d["entry_zone"],
            "stop_loss":    d["stop_loss"],
            "target":       d["target"],
            "position_size":d["position_size"],
            "turnover_cr":  round(turnover / 1e7, 1),
            "trend":        ind["trend"],
            "rsi":          ind["rsi"],
        }
        # 20-year-trader synthesis: grade, conviction, sizing, trade plan.
        plan = build_playbook(cand)
        cand["grade"]      = plan["grade"]
        cand["conviction"] = plan["conviction"]
        cand["reward_risk"] = plan["reward_risk"]
        cand["playbook"]   = plan
        candidates.append(cand)
        if len(candidates) >= final:
            break

    # ACTIONABLE = strong rank + quality + a BUY whose entry trigger has
    # actually fired on the latest bars. Everything else strong = watchlist.
    buys  = [c for c in candidates
             if c["action"] == "buy" and c["entry_signal"] == "trigger"]
    watch = [c for c in candidates
             if not (c["action"] == "buy" and c["entry_signal"] == "trigger")]
    # A seasoned trader works the best setups first — sort by conviction.
    buys.sort(key=lambda c: c["conviction"], reverse=True)
    watch.sort(key=lambda c: c["conviction"], reverse=True)

    _print_table("🎯 ACTIONABLE — strong rank, quality, AND entry trigger fired",
                 buys, show_levels=True)
    _print_table("👀 WATCHLIST — strong rank, waiting for a clean entry",
                 watch, show_levels=False)

    # Full written trade plans for the highest-conviction actionable setups.
    top_plans = [c for c in buys if c["grade"] in ("A+", "A")][:5]
    if top_plans:
        print("\n  📋 TRADE PLANS — highest-conviction setups")
        print("  " + "─" * 66)
        for c in top_plans:
            print(playbook_text(c["playbook"]))
            print()

    report = {
        "date":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pool_size":   pool,
        "min_turnover_cr": min_turnover / 1e7,
        "fundamentals_applied": bool(ranked["fund_ok"].any()),
        "blend":       {"momentum": MOM_WEIGHT, "quality": QUAL_WEIGHT,
                        "min_quality": MIN_QUALITY},
        "actionable":  buys,
        "watchlist":   watch,
    }
    out_path = os.path.join(OUTPUT_DIR, "screener_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  ✅ Full report → {out_path}\n")
    return report


def _print_table(title, rows, show_levels):
    print(f"\n  {title}")
    print("  " + "─" * 66)
    if not rows:
        print("     (none)")
        return
    if show_levels:
        print(f"  {'GRD':<4}{'CONV':>5} {'SYMBOL':<13}{'PRICE':>9} {'QUAL':>5} {'R:R':>4} "
              f"{'PATTERN':>18} {'ENTRY':>15} {'STOP':>9} {'TARGET':>9}")
        for c in rows:
            rr = c.get("reward_risk")
            print(f"  {c['grade']:<4}{c['conviction']:>5.0f} {c['symbol']:<13}"
                  f"{c['price']:>9.1f} {c['quality']:>5.0f} "
                  f"{(rr if rr is not None else 0):>4.1f} {c['pattern']:>18} "
                  f"{c['entry_zone']:>15} {c['stop_loss']:>9} {c['target']:>9}")
    else:
        print(f"  {'#':<3}{'SYMBOL':<13}{'PRICE':>10} {'QUAL':>5} "
              f"{'ACTION':>7} {'PATTERN':>20} {'CONTEXT':>20} {'RSI':>6}")
        for c in rows:
            print(f"  {c['rank_in_pool']:<3}{c['symbol']:<13}"
                  f"{c['price']:>10.1f} {c['quality']:>5.0f} "
                  f"{c['action']:>7} {c['pattern']:>20} "
                  f"{c['pattern_context']:>20} {c['rsi']:>6.1f}")


if __name__ == "__main__":
    screen()

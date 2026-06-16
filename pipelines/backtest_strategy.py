"""
Full-system walk-forward backtest — the evidence layer (roadmap #1).

The ranker has a measured cross-sectional IC, but IC is a statistical
correlation, not money. This answers the question that actually matters:
if you had traded the ranker's picks, non-overlapping, month after month,
NET of realistic costs — would you have beaten just holding the universe?

What is (and isn't) tested
--------------------------
TESTED   : the cross-sectional ranker (top-quintile and top-N long books)
           and the price-action ENTRY OVERLAY (only hold a pick once a
           bullish pattern trigger fires). Both are computable point-in-time
           from price history, so the walk-forward is leak-free.
NOT TESTED: the fundamental quality tilt. Those values are a CURRENT
           snapshot — using them on 2017 dates would leak the future. That
           is exactly why point-in-time fundamentals are roadmap step #3;
           until then the fundamental layer stays out of the backtest.

Method
------
• Build the same factor panel the ranker uses (per-date z-scored features,
  realised forward return `fwd`, beat-median `label`).
• Rebalance every HORIZON days (non-overlapping → clean compounding).
• At each rebalance, train the model ONLY on data whose label window closes
  before the rebalance date (strict walk-forward embargo), score the live
  cross-section, build the book.
• Charge COST_PER_SIDE on the turnover each rebalance (sells + buys).
• Report gross & net CAGR, Sharpe, max drawdown, hit-rate vs the universe,
  and average turnover.
"""

import os
import sys
import json

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.cross_sectional import (
    load_prices, build_panel, _make_model,
    FEATURES_Z, HORIZON, MIN_NAMES,
)
from pipelines.patterns import detect_patterns
from pipelines.market_regime import _load_index, regime_at

OUTPUT_DIR = os.path.join(ROOT, "outputs", "backtests")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Realistic round-trip is ~2× this. 20 bps/side ≈ STT + impact + charges
# for liquid NSE delivery names. Bump up to stress-test illiquid books.
COST_PER_SIDE  = 0.0020
TOP_N          = 15      # concentrated book (≈ what the screener surfaces)
QUINTILE       = 0.20    # diversified top-quintile book
RETRAIN_EVERY  = 3       # retrain the model every N rebalances (~quarterly)
PERIODS_PER_YR = 252.0 / HORIZON


# ── metrics ───────────────────────────────────────────
def _metrics(period_rets):
    r = np.asarray(period_rets, dtype=float)
    if len(r) == 0:
        return {}
    equity = np.cumprod(1 + r)
    years  = len(r) / PERIODS_PER_YR
    cagr   = equity[-1] ** (1 / years) - 1 if years > 0 and equity[-1] > 0 else float("nan")
    vol    = r.std(ddof=1) * np.sqrt(PERIODS_PER_YR) if len(r) > 1 else float("nan")
    sharpe = (r.mean() * PERIODS_PER_YR) / vol if vol and vol == vol else float("nan")
    peak   = np.maximum.accumulate(equity)
    maxdd  = float((equity / peak - 1).min())
    return {
        "periods":   int(len(r)),
        "cagr":      round(float(cagr) * 100, 2),
        "vol":       round(float(vol) * 100, 2),
        "sharpe":    round(float(sharpe), 2),
        "max_dd":    round(maxdd * 100, 2),
        "final_mult": round(float(equity[-1]), 2),
    }


def _turnover(prev, new):
    if not new:
        return 0.0
    overlap = len(set(prev) & set(new)) / len(new)
    return 1.0 - overlap     # one-way fraction changed


# ── the backtest ──────────────────────────────────────
def run_backtest(top_n=TOP_N, quintile=QUINTILE, cost=COST_PER_SIDE,
                 retrain_every=RETRAIN_EVERY, pattern_overlay=True):
    print("=" * 68)
    print("  FULL-SYSTEM WALK-FORWARD BACKTEST (net of costs)")
    print("=" * 68)

    prices = load_prices()
    print(f"  Universe: {len(prices)} symbols | horizon {HORIZON}d | "
          f"cost {cost*1e4:.0f} bps/side")
    panel = build_panel(prices)
    dates = np.sort(panel["date"].unique())
    rb    = list(dates[::HORIZON])      # non-overlapping rebalance dates
    print(f"  Panel rows: {len(panel):,} | rebalances: {len(rb)}\n")

    # group rows by rebalance date once (fast lookup)
    by_date = {d: g for d, g in panel.groupby("date")}

    # Market-regime overlay: scale the top-N book's gross exposure by the
    # point-in-time NIFTY regime, parking the rest in cash (0 return).
    index_df = _load_index("NIFTY")

    model = None
    prev_q, prev_n, prev_p = set(), set(), set()
    rets = {"market": [], "quintile_net": [], "topN_net": [],
            "topN_gross": [], "overlay_net": [], "topN_regime": []}
    turns = []
    used_dates = []

    for i, d in enumerate(rb):
        cs = by_date.get(d)
        if cs is None or len(cs) < MIN_NAMES:
            continue
        if i == 0:                       # need history to train
            continue

        # Walk-forward train: only labels that close before this date.
        cutoff = rb[i - 1]
        if model is None or (i % retrain_every == 0):
            tr = panel[panel["date"] < cutoff]
            if len(tr) < 1000:
                continue
            model = _make_model()
            model.fit(tr[FEATURES_Z], tr["label"])

        cs = cs.copy()
        cs["score"] = model.predict_proba(cs[FEATURES_Z])[:, 1]
        cs = cs.sort_values("score", ascending=False)

        # Books
        mkt = float(cs["fwd"].mean())
        nq  = max(1, int(len(cs) * quintile))
        qsel = cs.head(nq)
        nsel = cs.head(top_n)
        q_ret = float(qsel["fwd"].mean())
        n_ret = float(nsel["fwd"].mean())
        q_names = list(qsel["symbol"]); n_names = list(nsel["symbol"])

        # Costs on turnover (sell old + buy new ≈ 2× one-way turnover).
        q_to = _turnover(prev_q, q_names); n_to = _turnover(prev_n, n_names)
        q_net = q_ret - 2 * q_to * cost
        n_net = n_ret - 2 * n_to * cost

        # ── Pattern entry overlay on the top-N book ──
        ov_net = n_net
        if pattern_overlay:
            kept = []
            for sym in n_names:
                dfp = prices.get(sym)
                if dfp is None:
                    continue
                hist = dfp.loc[:d]               # point-in-time slice
                r = detect_patterns(hist)
                if r["entry_trigger"] == "long" or r["pattern_score"] > 0.2:
                    kept.append(sym)
            if kept:
                ov_ret = float(cs[cs["symbol"].isin(kept)]["fwd"].mean())
                ov_to  = _turnover(prev_p, kept)
                ov_net = ov_ret - 2 * ov_to * cost
                prev_p = set(kept)
            else:
                ov_net = 0.0                     # nothing triggered → cash
                prev_p = set()

        # Regime-scaled top-N: exposure e in [0,1] at this rebalance date,
        # (1-e) sits in cash. Point-in-time (index history up to d only).
        if index_df is not None:
            e = regime_at(index_df, as_of=d).get("exposure", 1.0)
        else:
            e = 1.0
        regime_net = e * n_net           # cash earns 0 over the period

        rets["market"].append(mkt)
        rets["quintile_net"].append(q_net)
        rets["topN_gross"].append(n_ret)
        rets["topN_net"].append(n_net)
        rets["overlay_net"].append(ov_net)
        rets["topN_regime"].append(regime_net)
        turns.append(n_to)
        used_dates.append(pd.Timestamp(d).strftime("%Y-%m-%d"))
        prev_q, prev_n = set(q_names), set(n_names)

    # ── Results ───────────────────────────────────────
    books = {
        "Universe (equal-wt)":      _metrics(rets["market"]),
        "Top-quintile (net)":       _metrics(rets["quintile_net"]),
        f"Top-{top_n} (gross)":     _metrics(rets["topN_gross"]),
        f"Top-{top_n} (net)":       _metrics(rets["topN_net"]),
        f"Top-{top_n} + patterns":  _metrics(rets["overlay_net"]),
        f"Top-{top_n} + regime":    _metrics(rets["topN_regime"]),
    }

    hit_q = np.mean(np.array(rets["quintile_net"]) > np.array(rets["market"]))
    hit_n = np.mean(np.array(rets["topN_net"])     > np.array(rets["market"]))

    print(f"  {'BOOK':<26}{'CAGR':>8}{'VOL':>8}{'SHARPE':>8}"
          f"{'MAXDD':>9}{'×':>7}")
    print("  " + "─" * 64)
    for name, m in books.items():
        if not m:
            continue
        print(f"  {name:<26}{m['cagr']:>7.1f}%{m['vol']:>7.1f}%"
              f"{m['sharpe']:>8.2f}{m['max_dd']:>8.1f}%{m['final_mult']:>7.2f}")

    print("\n  " + "─" * 64)
    print(f"  Periods                : {books['Universe (equal-wt)'].get('periods')}"
          f"  ({books['Universe (equal-wt)'].get('periods', 0)/PERIODS_PER_YR:.1f} yrs)")
    print(f"  Avg turnover / rebal   : {np.mean(turns)*100:.0f}%  (top-{top_n})")
    print(f"  Hit-rate vs universe   : quintile {hit_q*100:.0f}% | "
          f"top-{top_n} {hit_n*100:.0f}%")

    # verdict on the NET top-N vs the market
    mkt_m = books["Universe (equal-wt)"]
    net_m = books[f"Top-{top_n} (net)"]
    edge_cagr = net_m["cagr"] - mkt_m["cagr"]
    verdict = ("REAL EDGE net of costs" if edge_cagr > 1.0 and net_m["sharpe"] > mkt_m["sharpe"]
               else "MARGINAL — edge thin after costs" if edge_cagr > 0
               else "NO EDGE after costs")
    print(f"  Net top-{top_n} vs universe : {edge_cagr:+.1f}% CAGR  →  {verdict}")
    print("  (Reminder: fundamentals tilt NOT included — snapshot data would")
    print("   leak. That validation arrives with point-in-time data, step #3.)")

    out = {"books": books, "hit_rate": {"quintile": round(float(hit_q), 3),
           "topN": round(float(hit_n), 3)}, "avg_turnover": round(float(np.mean(turns)), 3),
           "edge_cagr_net": round(edge_cagr, 2), "verdict": verdict,
           "config": {"top_n": top_n, "quintile": quintile, "cost_per_side": cost,
                      "horizon": HORIZON, "pattern_overlay": pattern_overlay},
           "equity": {"dates": used_dates,
                      "topN_net": list(np.cumprod(1+np.array(rets["topN_net"])).round(4)),
                      "market":   list(np.cumprod(1+np.array(rets["market"])).round(4))}}
    path = os.path.join(OUTPUT_DIR, "strategy_backtest.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  ✅ Saved → {path}")
    return out


if __name__ == "__main__":
    run_backtest()

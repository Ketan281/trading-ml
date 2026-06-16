"""
Fundamental quality scoring (goal #4).

Turns the raw fundamentals snapshot (training/fetch_fundamentals.py →
data/historical/fundamentals.json) into a single, comparable QUALITY SCORE
per stock, plus a pass/fail quality gate. This is how a long-term trader
separates "good business at a fair price" from "cheap junk / hype".

How the score is built
----------------------
1. Each raw field has a DIRECTION (+1 = higher is better, e.g. ROE;
   -1 = lower is better, e.g. debt/equity, PE) and a CATEGORY.
2. Per field, compute a cross-sectional z-score ACROSS THE UNIVERSE, then
   multiply by its direction so "+z = better" universally. Z-scores are
   winsorised at ±3 so one outlier can't dominate.
3. Average the (sign-adjusted) z-scores within each category, so a company
   with one missing field isn't unfairly punished.
4. Combine category scores with weights into a composite, then map to a
   0–100 percentile rank for an intuitive "quality" number.

The score is RELATIVE (a percentile within the universe), which is exactly
what we want for a cross-sectional shortlist.

Point-in-time note: these are current-snapshot fundamentals, so this module
is used only at SCREENING time — never inside the historical backtest.
"""

import os
import json

import numpy as np
import pandas as pd

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUND_PATH = os.path.join(ROOT, "data", "historical", "fundamentals.json")
IND_PATH  = os.path.join(ROOT, "data", "historical", "industries.json")

# field -> (direction, category).  direction +1 = higher better, -1 = lower better
FIELD_SPEC = {
    # Valuation (cheaper = better)
    "trailingPE":                   (-1, "valuation"),
    "forwardPE":                    (-1, "valuation"),
    "priceToBook":                  (-1, "valuation"),
    "pegRatio":                     (-1, "valuation"),
    "enterpriseToEbitda":           (-1, "valuation"),
    "priceToSalesTrailing12Months": (-1, "valuation"),
    # Profitability (higher = better)
    "returnOnEquity":               (+1, "profitability"),
    "returnOnAssets":               (+1, "profitability"),
    "profitMargins":                (+1, "profitability"),
    "operatingMargins":             (+1, "profitability"),
    "grossMargins":                 (+1, "profitability"),
    # Growth (higher = better)
    "earningsGrowth":               (+1, "growth"),
    "revenueGrowth":                (+1, "growth"),
    "earningsQuarterlyGrowth":      (+1, "growth"),
    # Financial health
    "debtToEquity":                 (-1, "health"),
    "currentRatio":                 (+1, "health"),
    "quickRatio":                   (+1, "health"),
    # Shareholder / size (higher = better/safer)
    "dividendYield":                (+1, "quality"),
    "freeCashflow":                 (+1, "quality"),
    "marketCap":                    (+1, "quality"),
}

# How much each category counts toward the composite.
CATEGORY_WEIGHTS = {
    "profitability": 0.30,
    "growth":        0.20,
    "health":        0.20,
    "valuation":     0.20,
    "quality":       0.10,
}

# Hard quality gate — drops balance-sheet junk regardless of momentum.
# NaN-aware (a name we can't VERIFY isn't waved through) and sector-relative
# on leverage (so banks/NBFCs aren't punished for structurally high D/E).
GATE = {
    "min_market_cap":    1.0e10,   # ₹1,000 cr floor (drop micro-caps)
    "min_op_margin":    -0.02,     # core operations must ~break even
    "min_roe":          -0.02,     # drop real loss-makers (when ROE known)
    "min_roa":          -0.05,
    "require_pos_book":  True,     # priceToBook > 0  → no negative equity
    "min_core_fields":   6,        # data sufficiency: must be verifiable
    "de_sector_mult":    2.5,      # fail only if D/E > 2.5× its sector median
    "de_abs_floor":      150.0,    # ...but never fail below this absolute D/E
}

# Fields that must be reasonably populated to TRUST a quality verdict.
CORE_FIELDS = [
    "trailingPE", "returnOnEquity", "returnOnAssets", "profitMargins",
    "operatingMargins", "debtToEquity", "revenueGrowth", "currentRatio",
    "priceToBook",
]


def load_industries():
    if os.path.exists(IND_PATH):
        try:
            with open(IND_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _sector_de_median():
    """Median debt/equity PER SECTOR across the whole cached universe, so the
    leverage cap is judged relative to a stock's own industry norm."""
    fund = load_fundamentals()
    ind  = load_industries()
    if not fund:
        return {}, float("nan")
    rows = [{"sector": ind.get(s, "?"), "de": v.get("debtToEquity")}
            for s, v in fund.items()]
    df = pd.DataFrame(rows).dropna(subset=["de"])
    by_sector = df.groupby("sector")["de"].median().to_dict()
    overall   = float(df["de"].median()) if len(df) else float("nan")
    return by_sector, overall


# ── Load ──────────────────────────────────────────────
def load_fundamentals():
    if not os.path.exists(FUND_PATH):
        return {}
    try:
        with open(FUND_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _winsor_z(s):
    """Cross-sectional z-score, winsorised at ±3."""
    mu, sd = s.mean(), s.std()
    if not sd or sd != sd:
        return pd.Series(0.0, index=s.index)
    z = (s - mu) / sd
    return z.clip(-3, 3)


# ── Core: score the universe ──────────────────────────
def quality_scores(symbols=None):
    """Return a DataFrame indexed by symbol with per-category scores, a
    composite z, a 0–100 quality percentile, and a pass_gate flag.
    Scores are RELATIVE to whichever `symbols` are scored together."""
    fund = load_fundamentals()
    if not fund:
        return None

    rows = {s: v for s, v in fund.items()
            if (symbols is None or s in symbols)}
    if len(rows) < 5:
        return None

    raw = pd.DataFrame.from_dict(rows, orient="index")

    # Per-field sign-adjusted z-scores.
    cat_scores = {c: [] for c in CATEGORY_WEIGHTS}
    for field, (direction, cat) in FIELD_SPEC.items():
        if field not in raw.columns:
            continue
        z = _winsor_z(raw[field]) * direction
        cat_scores[cat].append(z)

    # Average within each category (ignoring missing fields per stock).
    df = pd.DataFrame(index=raw.index)
    for cat, zlist in cat_scores.items():
        if zlist:
            df[f"{cat}_score"] = pd.concat(zlist, axis=1).mean(axis=1)
        else:
            df[f"{cat}_score"] = 0.0

    # Weighted composite across categories.
    composite = sum(df[f"{cat}_score"].fillna(0) * w
                    for cat, w in CATEGORY_WEIGHTS.items())
    df["quality_z"]     = composite
    df["quality_score"] = (composite.rank(pct=True) * 100).round(1)

    # ── Hard quality gate (NaN-aware + sector-relative leverage) ──
    industries          = load_industries()
    sector_de, overall_de = _sector_de_median()

    def _gate_row(sym, r):
        """Return (pass: bool, reason: str). A name we can't verify, or with
        a clear balance-sheet/operating red flag, fails."""
        g = lambda k: r.get(k, np.nan)
        present = sum(1 for c in CORE_FIELDS if pd.notna(g(c)))

        # Size floor (missing market cap => can't size => fail).
        if pd.isna(g("marketCap")) or g("marketCap") < GATE["min_market_cap"]:
            return False, "micro/unknown cap"
        # Data sufficiency: too sparse to trust a quality verdict.
        if present < GATE["min_core_fields"]:
            return False, "insufficient data"
        # Negative equity (book value < 0) — catches names like IDEA.
        if GATE["require_pos_book"] and pd.notna(g("priceToBook")) \
                and g("priceToBook") <= 0:
            return False, "negative book value"
        # Core operations must roughly break even (use op margin, not the
        # noisy profitMargins field which can be polluted by one-offs).
        if pd.notna(g("operatingMargins")) and g("operatingMargins") < GATE["min_op_margin"]:
            return False, "operating loss"
        # Profitability floors when known.
        if pd.notna(g("returnOnEquity")) and g("returnOnEquity") < GATE["min_roe"]:
            return False, "negative ROE"
        if pd.notna(g("returnOnAssets")) and g("returnOnAssets") < GATE["min_roa"]:
            return False, "negative ROA"
        # Leverage — sector-relative. Cap = max(floor, mult × sector median).
        de = g("debtToEquity")
        if pd.notna(de):
            sec = industries.get(sym, "?")
            med = sector_de.get(sec, overall_de)
            if not (med == med):       # NaN guard
                med = overall_de
            cap = max(GATE["de_abs_floor"], GATE["de_sector_mult"] * (med or 0))
            if de > cap:
                return False, f"over-levered vs {sec} (D/E {de:.0f}>{cap:.0f})"
        return True, "ok"

    results = {s: _gate_row(s, raw.loc[s]) for s in raw.index}
    df["pass_gate"]   = pd.Series({s: ok for s, (ok, _) in results.items()})
    df["gate_reason"] = pd.Series({s: rs for s, (_, rs) in results.items()})

    # Keep raw fields handy for reporting.
    for f in ("trailingPE", "returnOnEquity", "debtToEquity",
              "revenueGrowth", "marketCap", "profitMargins"):
        if f in raw.columns:
            df[f] = raw[f]

    return df.sort_values("quality_score", ascending=False)


# ── CLI ───────────────────────────────────────────────
if __name__ == "__main__":
    df = quality_scores()
    if df is None:
        print("  ⚠ No fundamentals cached. Run training/fetch_fundamentals.py first.")
    else:
        print(f"  Scored {len(df)} stocks on fundamental quality.\n")
        cols = ["quality_score", "pass_gate", "profitability_score",
                "growth_score", "health_score", "valuation_score",
                "trailingPE", "returnOnEquity", "debtToEquity"]
        cols = [c for c in cols if c in df.columns]
        print("  TOP 15 BY QUALITY")
        print(df[cols].head(15).to_string())
        print("\n  BOTTOM 10 BY QUALITY")
        print(df[cols].tail(10).to_string())
        print(f"\n  Passing quality gate: {int(df['pass_gate'].sum())}/{len(df)}")

"""
Point-in-time fundamental FEATURES — roadmap #3 (feature layer).

Takes the raw quarterly statements harvested by
training/fetch_fundamentals_pit.py and turns them into ratio features that
are legitimately point-in-time:

  • Each quarter's numbers become "known" to the market only AFTER results
    are filed. We model that with REPORTING_LAG_DAYS, so the feature carries
    an `available_date = period_end + lag`. Any backtest that as-of joins on
    `available_date <= trade_date` is therefore leak-free.
  • Growth features are year-over-year (period t vs the same quarter a year
    earlier) so they are not distorted by seasonality.

The headline entry point is `pit_feature_panel(symbols)`, which returns a
tidy frame [symbol, available_date, <features...>] ready to be as-of merged
onto the price panel by models/cross_sectional or the backtest.
"""

import os
import json

import numpy as np
import pandas as pd

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST_DIR = os.path.join(ROOT, "data", "historical")
PIT_PATH = os.path.join(HIST_DIR, "fundamentals_pit.json")

# Indian listed companies must file quarterly results within 45 days of the
# quarter end (SEBI LODR). We use that as the lag before a quarter's numbers
# are tradable information. Annual (Q4/March) gets 60 days.
REPORTING_LAG_DAYS    = 45
ANNUAL_LAG_DAYS       = 60

# The point-in-time features we expose. Names are deliberately distinct from
# the snapshot module so the two can coexist.
PIT_FEATURES = [
    "pit_profit_margin", "pit_op_margin", "pit_gross_margin",
    "pit_roe", "pit_roa", "pit_debt_to_equity", "pit_current_ratio",
    "pit_rev_growth_yoy", "pit_earnings_growth_yoy", "pit_fcf_margin",
]


def _safe_div(a, b):
    if a is None or b is None:
        return np.nan
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return np.nan
    if b == 0 or b != b or a != a:
        return np.nan
    return a / b


def _load_pit():
    if not os.path.exists(PIT_PATH):
        return {}
    try:
        with open(PIT_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _available_date(period_end):
    """When the market could first act on this quarter's results."""
    pe = pd.Timestamp(period_end)
    lag = ANNUAL_LAG_DAYS if pe.month == 3 else REPORTING_LAG_DAYS
    return pe + pd.Timedelta(days=lag)


def symbol_pit_features(periods):
    """periods: {period_end_str: {line_item: value}} for one symbol.
    Returns a DataFrame indexed by available_date with PIT_FEATURES, or an
    empty frame if there is not enough to compute anything."""
    if not periods:
        return pd.DataFrame()

    # Order quarters chronologically.
    rows = []
    for pe in sorted(periods):
        v = periods[pe]
        rows.append({"period_end": pd.Timestamp(pe), **v})
    q = pd.DataFrame(rows).set_index("period_end").sort_index()

    rev = q.get("revenue")
    ni  = q.get("net_income")

    feat = pd.DataFrame(index=q.index)
    feat["pit_profit_margin"]  = _series_div(ni,                rev)
    feat["pit_op_margin"]      = _series_div(q.get("operating_income"), rev)
    feat["pit_gross_margin"]   = _series_div(q.get("gross_profit"),     rev)
    feat["pit_roe"]            = _series_div(ni,                q.get("equity"))
    feat["pit_roa"]            = _series_div(ni,                q.get("total_assets"))
    feat["pit_debt_to_equity"] = _series_div(q.get("total_debt"), q.get("equity"))
    feat["pit_current_ratio"]  = _series_div(q.get("current_assets"),
                                             q.get("current_liab"))
    feat["pit_fcf_margin"]     = _series_div(q.get("free_cash_flow"), rev)

    # Year-over-year growth: same quarter one year earlier ≈ 4 quarters back.
    # Use a 4-period shift only when the gap really is ~1 year, so an irregular
    # statement history doesn't create a bogus growth number.
    feat["pit_rev_growth_yoy"]      = _yoy_growth(rev)
    feat["pit_earnings_growth_yoy"] = _yoy_growth(ni)

    # Stamp with the date the market could first know each quarter.
    feat = feat.copy()
    feat["available_date"] = [_available_date(pe) for pe in feat.index]
    feat = feat.reset_index(drop=True)
    # Drop rows that are entirely empty of features.
    keep = feat[PIT_FEATURES].notna().any(axis=1)
    return feat[keep].reset_index(drop=True)


def _series_div(num, den):
    if num is None or den is None:
        return np.nan
    return num / den.replace(0, np.nan)


def _yoy_growth(s):
    """t vs ~4 quarters earlier, only if that earlier point is 270-460 days
    back (i.e. genuinely a year, not a data gap)."""
    if s is None:
        return np.nan
    s = s.astype(float)
    idx = s.index
    out = pd.Series(np.nan, index=idx)
    for i in range(len(idx)):
        if i < 1:
            continue
        # find the row closest to ~365 days before idx[i]
        target = idx[i] - pd.Timedelta(days=365)
        gaps = np.abs((idx[:i] - target).days.to_numpy())
        if len(gaps) == 0:
            continue
        j = int(np.argmin(gaps))
        days_back = (idx[i] - idx[j]).days
        prev = s.iloc[j]
        if 270 <= days_back <= 460 and prev == prev:   # prev not NaN
            base = abs(prev)
            if base > 0:
                out.iloc[i] = (s.iloc[i] - prev) / base
    return out


def pit_feature_panel(symbols=None):
    """Tidy long frame [symbol, available_date, <PIT_FEATURES>] across the
    universe — the object an as-of join consumes."""
    raw = _load_pit()
    if not raw:
        return pd.DataFrame(columns=["symbol", "available_date"] + PIT_FEATURES)

    frames = []
    for sym, periods in raw.items():
        if symbols is not None and sym not in symbols:
            continue
        f = symbol_pit_features(periods)
        if f.empty:
            continue
        f.insert(0, "symbol", sym)
        frames.append(f)

    if not frames:
        return pd.DataFrame(columns=["symbol", "available_date"] + PIT_FEATURES)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["symbol", "available_date"]).reset_index(drop=True)


def asof_join(price_panel, pit_panel=None, date_col="date", sym_col="symbol"):
    """For every (symbol, date) row in price_panel, attach the most recent
    fundamental snapshot whose available_date <= date. Leak-free by
    construction. Returns price_panel with PIT_FEATURES columns added (NaN
    where no fundamental is yet available)."""
    if pit_panel is None:
        pit_panel = pit_feature_panel()
    if pit_panel.empty:
        out = price_panel.copy()
        for c in PIT_FEATURES:
            out[c] = np.nan
        return out

    left = price_panel.sort_values(date_col).copy()
    left[date_col] = pd.to_datetime(left[date_col])
    right = pit_panel.sort_values("available_date").copy()
    right["available_date"] = pd.to_datetime(right["available_date"])

    merged = pd.merge_asof(
        left, right,
        left_on=date_col, right_on="available_date",
        by=sym_col, direction="backward",
    )
    return merged.drop(columns=["available_date"])


# ── CLI: quick sanity / coverage report ───────────────
if __name__ == "__main__":
    panel = pit_feature_panel()
    if panel.empty:
        print("  ⚠ No PIT fundamentals cached.")
        print("    Run: python training/fetch_fundamentals_pit.py")
    else:
        n_sym = panel["symbol"].nunique()
        print("=" * 60)
        print("  Point-in-time fundamental feature panel")
        print("=" * 60)
        print(f"  Symbols with features : {n_sym}")
        print(f"  Feature rows (q-stamps): {len(panel)}")
        print(f"  Date span             : "
              f"{panel['available_date'].min().date()} → "
              f"{panel['available_date'].max().date()}")
        cov = panel[PIT_FEATURES].notna().mean().sort_values(ascending=False)
        print("\n  Coverage per feature (non-NaN share):")
        for f, c in cov.items():
            print(f"     {f:<26} {c*100:5.1f}%")
        print("\n  Sample (latest 3 rows):")
        print(panel.tail(3).to_string(index=False))

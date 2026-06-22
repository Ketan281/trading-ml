"""
Relative Strength engine -- stock-level RS for the full tracked universe.

Computes multi-window relative strength (stock vs NIFTY, stock vs sector),
percentile ranks across the universe, and a composite RS score that blends
short / swing / trend horizons.  Designed to run efficiently over 450+ stocks
by vectorising with pandas where possible.

Depends on:
    models.cross_sectional.load_prices  -- {symbol: DataFrame} with OHLCV
    data/historical/industries.json     -- {symbol: sector_name}
    data/historical/NIFTY.csv (or data/NIFTY_daily.csv) -- index benchmark
"""

import os
import sys
import json

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.cross_sectional import load_prices

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
IND_PATH    = os.path.join(ROOT, "data", "historical", "industries.json")
NIFTY_HIST  = os.path.join(ROOT, "data", "historical", "NIFTY.csv")
NIFTY_DAILY = os.path.join(ROOT, "data", "NIFTY_daily.csv")

# ---------------------------------------------------------------------------
# RS windows and composite weights
# ---------------------------------------------------------------------------
WINDOWS = [5, 20, 60]                      # trading days
COMPOSITE_WEIGHTS = {5: 0.20, 20: 0.40, 60: 0.40}

# Regime thresholds (percentile boundaries)
REGIME_BINS   = [0, 20, 40, 60, 80, 100]
REGIME_LABELS = ["Laggard", "Weak", "Neutral", "Strong", "Leader"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_industries():
    """Return {symbol: sector_name} or empty dict."""
    if os.path.exists(IND_PATH):
        with open(IND_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load_nifty():
    """Load NIFTY index Close series with DatetimeIndex."""
    for path in (NIFTY_HIST, NIFTY_DAILY):
        if os.path.exists(path):
            df = pd.read_csv(path, index_col="Date", parse_dates=True)
            if "Close" in df.columns:
                return df["Close"].sort_index()
    return None


def _rolling_return(close, window):
    """Simple rolling return over *window* trading days."""
    return close / close.shift(window) - 1


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_rs_universe(prices=None):
    """Compute relative strength for all stocks in the universe.

    Returns a DataFrame (one row per symbol, latest date) with columns:
        symbol, sector,
        ret_5d, ret_20d, ret_60d,
        nifty_ret_5d, nifty_ret_20d, nifty_ret_60d,
        rs_vs_nifty_5d, rs_vs_nifty_20d, rs_vs_nifty_60d,
        rs_vs_sector_5d, rs_vs_sector_20d, rs_vs_sector_60d,
        rs_pctile_5d, rs_pctile_20d, rs_pctile_60d,
        composite_rs, rs_percentile, rs_regime
    """
    if prices is None:
        prices = load_prices()

    industries = _load_industries()
    nifty_close = _load_nifty()

    # Pre-compute NIFTY rolling returns at each window.
    nifty_rets = {}
    if nifty_close is not None:
        for w in WINDOWS:
            nifty_rets[w] = _rolling_return(nifty_close, w)

    # ------------------------------------------------------------------
    # Pass 1: gather per-stock rolling returns (latest row only)
    # ------------------------------------------------------------------
    rows = []
    for sym, df in prices.items():
        close = df["Close"]
        if len(close) < max(WINDOWS) + 5:
            continue

        rec = {"symbol": sym, "sector": industries.get(sym, "Unknown")}

        for w in WINDOWS:
            ret_col = "ret_%dd" % w
            ret = _rolling_return(close, w)
            latest_ret = ret.iloc[-1]
            rec[ret_col] = latest_ret

            # RS vs NIFTY: ratio of stock return to index return
            nifty_col = "nifty_ret_%dd" % w
            rs_nifty_col = "rs_vs_nifty_%dd" % w
            if w in nifty_rets:
                # Align by date -- use the stock's last date
                last_date = close.index[-1]
                nifty_r = nifty_rets[w].asof(last_date)
                rec[nifty_col] = nifty_r
                # Ratio: handle near-zero nifty return gracefully
                if pd.notna(nifty_r) and abs(nifty_r) > 1e-8:
                    rec[rs_nifty_col] = latest_ret / nifty_r
                else:
                    rec[rs_nifty_col] = np.nan
            else:
                rec[nifty_col] = np.nan
                rec[rs_nifty_col] = np.nan

        rows.append(rec)

    if not rows:
        return pd.DataFrame()

    df_all = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Pass 2: RS vs sector (stock return minus sector median return)
    # ------------------------------------------------------------------
    for w in WINDOWS:
        ret_col = "ret_%dd" % w
        sec_col = "rs_vs_sector_%dd" % w
        sector_median = df_all.groupby("sector")[ret_col].transform("median")
        df_all[sec_col] = df_all[ret_col] - sector_median

    # ------------------------------------------------------------------
    # Pass 3: percentile ranks within the universe at each window
    # ------------------------------------------------------------------
    for w in WINDOWS:
        ret_col = "ret_%dd" % w
        pctile_col = "rs_pctile_%dd" % w
        df_all[pctile_col] = df_all[ret_col].rank(pct=True) * 100

    # ------------------------------------------------------------------
    # Composite RS score: weighted blend of percentile ranks
    # ------------------------------------------------------------------
    df_all["composite_rs"] = 0.0
    for w, weight in COMPOSITE_WEIGHTS.items():
        pctile_col = "rs_pctile_%dd" % w
        df_all["composite_rs"] += weight * df_all[pctile_col]

    # RS percentile of the composite itself
    df_all["rs_percentile"] = df_all["composite_rs"].rank(pct=True) * 100

    # RS regime classification
    df_all["rs_regime"] = pd.cut(
        df_all["rs_percentile"],
        bins=REGIME_BINS,
        labels=REGIME_LABELS,
        include_lowest=True,
    )

    # Order columns nicely
    col_order = ["symbol", "sector"]
    for w in WINDOWS:
        col_order += [
            "ret_%dd" % w,
            "nifty_ret_%dd" % w,
            "rs_vs_nifty_%dd" % w,
            "rs_vs_sector_%dd" % w,
            "rs_pctile_%dd" % w,
        ]
    col_order += ["composite_rs", "rs_percentile", "rs_regime"]
    # Only include columns that exist (nifty cols may be missing)
    col_order = [c for c in col_order if c in df_all.columns]
    df_all = df_all[col_order].sort_values(
        "composite_rs", ascending=False
    ).reset_index(drop=True)

    return df_all


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def rs_for_symbol(symbol, prices=None):
    """Get RS data for a single stock.  Returns a one-row DataFrame or None."""
    universe = compute_rs_universe(prices)
    if universe.empty:
        return None
    match = universe[universe["symbol"] == symbol]
    if match.empty:
        return None
    return match


def rs_leaders(n=20, prices=None):
    """Top N stocks by composite RS (strongest momentum)."""
    universe = compute_rs_universe(prices)
    if universe.empty:
        return pd.DataFrame()
    return universe.head(n)


def rs_laggards(n=20, prices=None):
    """Bottom N stocks by composite RS (weakest momentum)."""
    universe = compute_rs_universe(prices)
    if universe.empty:
        return pd.DataFrame()
    return universe.tail(n).iloc[::-1].reset_index(drop=True)


def rs_improving(n=20, prices=None):
    """Stocks whose short-term RS is much higher than their trend RS --
    momentum is *turning* upward.  Useful for catching early rotations.

    Measured as rs_pctile_5d - rs_pctile_60d (biggest gap = fastest
    improvement).
    """
    universe = compute_rs_universe(prices)
    if universe.empty:
        return pd.DataFrame()
    universe = universe.copy()
    universe["rs_improvement"] = (
        universe["rs_pctile_5d"] - universe["rs_pctile_60d"]
    )
    return (
        universe
        .sort_values("rs_improvement", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def rs_time_series(symbol, prices=None, days=120):
    """Rolling RS percentile time series for a stock (for charting).

    Returns a DataFrame indexed by date with columns:
        ret_5d, ret_20d, ret_60d, composite_rs, rs_percentile
    for the last *days* trading days.
    """
    if prices is None:
        prices = load_prices()

    if symbol not in prices:
        return pd.DataFrame()

    # We need the full universe to compute percentile ranks on each date.
    # Build a close-price panel (date x symbol) for efficiency.
    sym_list = [s for s, df in prices.items() if len(df) > max(WINDOWS) + 5]
    if symbol not in sym_list:
        return pd.DataFrame()

    # Build aligned close matrix
    closes = {}
    for s in sym_list:
        closes[s] = prices[s]["Close"]
    close_panel = pd.DataFrame(closes)

    # Trim to the last (days + max_window + buffer) rows
    lookback = days + max(WINDOWS) + 10
    close_panel = close_panel.iloc[-lookback:]

    # Rolling returns for every stock at each window
    ret_panels = {}
    for w in WINDOWS:
        ret_panels[w] = close_panel.pct_change(w)

    # For each date in the last *days*, compute the percentile rank of
    # the target symbol within the universe, plus the composite RS.
    target_dates = close_panel.index[-days:]
    records = []
    for dt in target_dates:
        rec = {"date": dt}
        pctiles = {}
        for w in WINDOWS:
            rp = ret_panels[w]
            if dt not in rp.index:
                continue
            row = rp.loc[dt].dropna()
            if symbol not in row.index or len(row) < 10:
                continue
            rec["ret_%dd" % w] = row[symbol]
            rank_pct = row.rank(pct=True)
            pctiles[w] = rank_pct[symbol] * 100
            rec["rs_pctile_%dd" % w] = pctiles[w]

        if pctiles:
            comp = sum(
                COMPOSITE_WEIGHTS.get(w, 0) * pctiles[w]
                for w in pctiles
            )
            rec["composite_rs"] = comp
            # Percentile of the composite is just the composite itself
            # (it is already 0-100 from the underlying pctile ranks).
            rec["rs_percentile"] = comp
        records.append(rec)

    if not records:
        return pd.DataFrame()

    ts = pd.DataFrame(records).set_index("date").sort_index()
    return ts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fmt_pct(val, width=8):
    """Format a float as percentage string, handling NaN."""
    if pd.isna(val):
        return " " * (width - 3) + "N/A"
    return ("%*.1f%%" % (width - 1, val * 100))


def _fmt_score(val, width=6):
    if pd.isna(val):
        return " " * (width - 3) + "N/A"
    return "%*.1f" % (width, val)


if __name__ == "__main__":
    print("=" * 78)
    print("  RELATIVE STRENGTH ENGINE -- Universe Scan")
    print("=" * 78)

    universe = compute_rs_universe()
    if universe.empty:
        print("  No price data available. Run training/fetch_historical.py first.")
        sys.exit(1)

    total = len(universe)
    print("  Stocks analysed: %d" % total)
    print()

    # --- Top 20 Leaders ---
    print("  TOP 20 LEADERS (highest composite RS)")
    print("  " + "-" * 74)
    print("  %-12s %-22s %8s %8s %8s %7s %s" % (
        "SYMBOL", "SECTOR", "Ret5d", "Ret20d", "Ret60d", "RS", "REGIME"))
    leaders = universe.head(20)
    for _, r in leaders.iterrows():
        print("  %-12s %-22s %s %s %s %s  %s" % (
            r["symbol"],
            str(r["sector"])[:22],
            _fmt_pct(r.get("ret_5d")),
            _fmt_pct(r.get("ret_20d")),
            _fmt_pct(r.get("ret_60d")),
            _fmt_score(r.get("rs_percentile")),
            r.get("rs_regime", ""),
        ))

    print()

    # --- Bottom 20 Laggards ---
    print("  BOTTOM 20 LAGGARDS (lowest composite RS)")
    print("  " + "-" * 74)
    print("  %-12s %-22s %8s %8s %8s %7s %s" % (
        "SYMBOL", "SECTOR", "Ret5d", "Ret20d", "Ret60d", "RS", "REGIME"))
    laggards = universe.tail(20).iloc[::-1]
    for _, r in laggards.iterrows():
        print("  %-12s %-22s %s %s %s %s  %s" % (
            r["symbol"],
            str(r["sector"])[:22],
            _fmt_pct(r.get("ret_5d")),
            _fmt_pct(r.get("ret_20d")),
            _fmt_pct(r.get("ret_60d")),
            _fmt_score(r.get("rs_percentile")),
            r.get("rs_regime", ""),
        ))

    print()

    # --- Improving ---
    print("  TOP 20 IMPROVING (short-term RS rising fastest)")
    print("  " + "-" * 74)
    improving = rs_improving(20)
    if not improving.empty:
        print("  %-12s %-22s %8s %8s %8s %9s" % (
            "SYMBOL", "SECTOR", "RS_5d", "RS_60d", "Gap", "REGIME"))
        for _, r in improving.iterrows():
            gap = r.get("rs_improvement", np.nan)
            print("  %-12s %-22s %s %s %s  %s" % (
                r["symbol"],
                str(r["sector"])[:22],
                _fmt_score(r.get("rs_pctile_5d")),
                _fmt_score(r.get("rs_pctile_60d")),
                _fmt_score(gap) if pd.notna(gap) else "   N/A",
                r.get("rs_regime", ""),
            ))

    # --- Regime distribution ---
    print()
    print("  REGIME DISTRIBUTION")
    print("  " + "-" * 40)
    regime_counts = universe["rs_regime"].value_counts()
    for regime in REGIME_LABELS[::-1]:
        count = regime_counts.get(regime, 0)
        bar = "#" * int(count / max(1, total) * 50)
        print("  %-10s %4d  %s" % (regime, count, bar))

    print()
    print("  Done.")

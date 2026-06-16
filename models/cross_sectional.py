"""
Cross-sectional stock ranking model.

Instead of predicting a single stock's absolute direction (near-impossible
on daily data), this learns to RANK the universe: which stocks will
out-perform the others over the next horizon. Relative performance is far
more learnable than absolute direction, and it directly answers the real
question — "which stocks should I be in right now?".

Pipeline
--------
1. For every stock, compute classic factor features (momentum, reversal,
   volatility, trend) from price history.
2. On each date, the label is whether the stock beats the universe median
   forward return — a purely RELATIVE target (market beta removed).
3. Features are cross-sectionally normalised PER DATE (z-scores), so they
   are comparable across stocks on any given day.
4. The model is validated walk-forward BY DATE, with an embargo gap so a
   stock's forward-return label never leaks into the test window.
5. Edge is measured the way quants measure it:
      • Rank IC  — rank correlation of prediction vs actual relative return
      • L/S spread — top-quintile minus bottom-quintile actual return
   Positive, consistent values across folds = genuine predictive edge.
"""

import os
import json
import glob
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST_DIR   = os.path.join(ROOT, "data", "historical")
DATA_DIR   = os.path.join(ROOT, "data")
MODELS_DIR = os.path.dirname(__file__)

HORIZON   = 21      # forward trading days the ranking targets (~1 month)
MIN_NAMES = 10      # minimum stocks on a date to compute cross-section

# Indices are not stocks — never rank them in the cross-section.
EXCLUDE = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"}

FEATURES = [
    "mom_21", "mom_63", "mom_126", "mom_252",
    "rev_5", "vol_21", "rsi_14", "dist_high", "ma_ratio",
]
FEATURES_Z = [c + "_z" for c in FEATURES]


# ── Per-symbol factor features ────────────────────────
def _symbol_factors(df):
    close = df["Close"]
    f = pd.DataFrame(index=df.index)

    # Momentum over multiple lookbacks (the strongest equity factor)
    f["mom_21"]  = close.pct_change(21)
    f["mom_63"]  = close.pct_change(63)
    f["mom_126"] = close.pct_change(126)
    f["mom_252"] = close.pct_change(252)

    # Short-term reversal (usually a negative predictor)
    f["rev_5"]   = close.pct_change(5)

    # Realised volatility
    f["vol_21"]  = close.pct_change().rolling(21).std()

    # RSI(14)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi_14"]  = 100 - 100 / (1 + gain / loss)

    # Distance from 1-year high, and price vs 200d MA (trend)
    f["dist_high"] = close / close.rolling(252).max() - 1
    f["ma_ratio"]  = close / close.rolling(200).mean()

    return f


# ── Load the universe ─────────────────────────────────
def load_prices(universe=None):
    """Load OHLCV per symbol. Prefers data/historical/*.csv (the 10-year
    set from training/fetch_historical.py); falls back to data/*_daily.csv."""
    prices = {}
    src = HIST_DIR if os.path.isdir(HIST_DIR) and glob.glob(
        os.path.join(HIST_DIR, "*.csv")) else DATA_DIR

    for path in glob.glob(os.path.join(src, "*.csv")):
        name = os.path.basename(path).replace(".csv", "")
        name = name.replace("_daily", "")
        if name.lower() in ("manifest",) or name in EXCLUDE:
            continue
        if universe and name not in universe:
            continue
        try:
            df = pd.read_csv(path, index_col="Date", parse_dates=True)
            if {"Close", "High", "Low"}.issubset(df.columns) and len(df) > 260:
                prices[name] = df.sort_index()
        except Exception:
            pass
    return prices


# ── Build the cross-sectional panel ───────────────────
def build_panel(prices, horizon=HORIZON):
    frames = []
    for sym, df in prices.items():
        f = _symbol_factors(df)
        f["fwd"]    = df["Close"].shift(-horizon) / df["Close"] - 1
        f["symbol"] = sym
        f["date"]   = f.index
        frames.append(f)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=FEATURES + ["fwd"])

    # Keep only dates with enough names to form a real cross-section.
    counts = panel.groupby("date")["symbol"].transform("count")
    panel  = panel[counts >= MIN_NAMES].copy()

    # RELATIVE forward return (remove the market move on each date).
    panel["rel_fwd"] = panel.groupby("date")["fwd"].transform(
        lambda x: x - x.mean())

    # Label: did this stock beat the universe median that day?
    panel["label"] = (
        panel.groupby("date")["fwd"].rank(pct=True) > 0.5
    ).astype(int)

    # Cross-sectional z-scores of each feature, per date.
    for c in FEATURES:
        panel[c + "_z"] = panel.groupby("date")[c].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9))

    return panel.sort_values("date").reset_index(drop=True)


# ── Quant evaluation metrics ──────────────────────────
def _rank_ic(g):
    """Spearman (rank) correlation of prediction vs actual relative return,
    computed without scipy."""
    ics = []
    for _, grp in g.groupby("date"):
        if len(grp) < MIN_NAMES:
            continue
        pr = grp["pred"].rank().values
        ar = grp["rel_fwd"].rank().values
        if pr.std() == 0 or ar.std() == 0:
            continue
        ics.append(np.corrcoef(pr, ar)[0, 1])
    return float(np.mean(ics)) if ics else float("nan")


def _long_short(g, q=0.2):
    """Average (top-quintile minus bottom-quintile) actual relative return."""
    spreads = []
    for _, grp in g.groupby("date"):
        if len(grp) < MIN_NAMES:
            continue
        n   = max(1, int(len(grp) * q))
        s   = grp.sort_values("pred", ascending=False)
        top = s.head(n)["rel_fwd"].mean()
        bot = s.tail(n)["rel_fwd"].mean()
        spreads.append(top - bot)
    return float(np.mean(spreads)) if spreads else float("nan")


def _make_model():
    return xgb.XGBClassifier(
        n_estimators     = 300,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        eval_metric      = "logloss",
        random_state     = 42,
        verbosity        = 0,
    )


# ── Train + walk-forward validate ─────────────────────
def train(universe=None, n_splits=5, horizon=HORIZON):
    print("=" * 60)
    print("  Cross-Sectional Ranking Model — Training")
    print("=" * 60)

    prices = load_prices(universe)
    print(f"  Universe loaded: {len(prices)} symbols")
    if len(prices) < MIN_NAMES:
        print(f"  ⚠ Need at least {MIN_NAMES} symbols. "
              f"Run training/fetch_historical.py first.")
        return None

    panel = build_panel(prices, horizon)
    dates = np.sort(panel["date"].unique())
    print(f"  Panel rows: {len(panel):,} | "
          f"dates: {len(dates)} | features: {len(FEATURES)}")

    # Walk-forward by date with an embargo equal to the label horizon.
    fold_size = len(dates) // (n_splits + 1)
    embargo   = horizon
    ic_scores, ls_scores = [], []

    for k in range(1, n_splits + 1):
        train_end  = fold_size * k
        test_start = train_end + embargo
        test_end   = min(fold_size * (k + 1), len(dates))
        if test_start >= test_end:
            continue

        train_dates = set(dates[:train_end])
        test_dates  = set(dates[test_start:test_end])

        tr = panel[panel["date"].isin(train_dates)]
        te = panel[panel["date"].isin(test_dates)].copy()
        if len(tr) < 500 or len(te) < 100:
            continue

        model = _make_model()
        model.fit(tr[FEATURES_Z], tr["label"])
        te["pred"] = model.predict_proba(te[FEATURES_Z])[:, 1]

        ic = _rank_ic(te)
        ls = _long_short(te)
        ic_scores.append(ic)
        ls_scores.append(ls)
        print(f"  Fold {k}: Rank IC = {ic:+.4f} | "
              f"L/S spread = {ls:+.4%}")

    mean_ic = float(np.nanmean(ic_scores)) if ic_scores else float("nan")
    mean_ls = float(np.nanmean(ls_scores)) if ls_scores else float("nan")

    print("\n  " + "-" * 50)
    print(f"  Mean Rank IC        : {mean_ic:+.4f}")
    print(f"  Mean L/S spread     : {mean_ls:+.4%}  (per {horizon}d)")
    verdict = ("EDGE" if (mean_ic > 0.02 and mean_ls > 0)
               else "WEAK/NONE")
    print(f"  Verdict             : {verdict}")
    print("  (Rank IC > ~0.03 sustained is considered a real signal.)")

    # ── Final model on all data ───────────────────────
    final = _make_model()
    final.fit(panel[FEATURES_Z], panel["label"])

    imp = pd.Series(final.feature_importances_, index=FEATURES_Z)
    imp = imp.sort_values(ascending=False)
    print("\n  Top features:")
    for feat, val in imp.head(6).items():
        print(f"     {feat:<14} {'█' * int(val * 50)} {val:.3f}")

    model_path = os.path.join(MODELS_DIR, "cross_sectional_xgb.pkl")
    meta_path  = os.path.join(MODELS_DIR, "cross_sectional_meta.json")
    with open(model_path, "wb") as f:
        pickle.dump(final, f)
    with open(meta_path, "w") as f:
        json.dump({
            "type":      "cross_sectional_ranker",
            "horizon":   horizon,
            "features":  FEATURES,
            "n_symbols": len(prices),
            "mean_ic":   round(mean_ic, 4),
            "mean_ls":   round(mean_ls, 4),
            "verdict":   verdict,
            "trained":   datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\n  ✅ Model saved → {model_path}")
    return {"mean_ic": mean_ic, "mean_ls": mean_ls, "verdict": verdict}


# ── Rank the universe today ───────────────────────────
def rank_today(top_n=15, universe=None):
    """Return the model's current top-ranked stocks (best relative-strength
    candidates to be long). Computes the latest cross-section, z-scores it,
    and scores every symbol with the saved model."""
    model_path = os.path.join(MODELS_DIR, "cross_sectional_xgb.pkl")
    if not os.path.exists(model_path):
        print("  ⚠ No cross-sectional model trained yet.")
        return None
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    prices = load_prices(universe)
    rows = []
    for sym, df in prices.items():
        f = _symbol_factors(df).dropna()
        if f.empty:
            continue
        last = f.iloc[-1]
        rows.append({"symbol": sym, "price": float(df["Close"].iloc[-1]),
                     **{c: float(last[c]) for c in FEATURES}})

    cs = pd.DataFrame(rows)
    if len(cs) < MIN_NAMES:
        print(f"  ⚠ Only {len(cs)} symbols available — need {MIN_NAMES}+.")
        return None

    # Cross-sectional z-score across today's universe.
    for c in FEATURES:
        cs[c + "_z"] = (cs[c] - cs[c].mean()) / (cs[c].std() + 1e-9)

    cs["score"] = model.predict_proba(cs[FEATURES_Z])[:, 1]
    cs = cs.sort_values("score", ascending=False).reset_index(drop=True)
    cs["rank"] = cs.index + 1

    return cs[["rank", "symbol", "price", "score",
               "mom_63", "mom_252", "rsi_14"]].head(top_n)


# ── CLI ───────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if "--rank" in sys.argv:
        out = rank_today(top_n=15)
        if out is not None:
            print("\n  🏆 TOP RANKED STOCKS (relative-strength candidates)\n")
            print(out.to_string(index=False))
    else:
        train()

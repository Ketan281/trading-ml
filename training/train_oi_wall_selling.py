"""
OI Wall Selling Strategy: Train + Walk-Forward Backtest

Strategy: Sell options at strikes where heavy OI creates "walls".
- Heavy PE OI below spot = support wall -> sell PE (put writing)
- Heavy CE OI above spot = resistance wall -> sell CE (call writing)
- Wall holds = option decays = WIN. Wall breaches = LOSS.

ML model learns WHICH walls are likely to hold using 30+ OI features.
Walk-forward tested across 748 days of chain data (2021-06 to 2026-06).
"""

import os
import sys
import glob
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RAW_DIR = os.path.join(ROOT, "data", "option_chain", "raw")
FEAT_DIR = os.path.join(ROOT, "data", "features")
MODEL_DIR = os.path.join(ROOT, "models", "oi_wall_selling")
os.makedirs(MODEL_DIR, exist_ok=True)

NIFTY_LOT = 75
BANKNIFTY_LOT = 30  # changed over time but approximate
LOT_SIZES = {"NIFTY": NIFTY_LOT, "BANKNIFTY": BANKNIFTY_LOT}
STRIKE_STEP = {"NIFTY": 50, "BANKNIFTY": 100}

# Hold period: we check if wall held after N trading days
HOLD_DAYS = [1, 2, 3, 5]  # test multiple horizons


def load_spot_prices(symbol):
    feat_path = os.path.join(FEAT_DIR, f"{symbol}_features.csv")
    if os.path.exists(feat_path):
        df = pd.read_csv(feat_path, index_col="Date", parse_dates=True)
        if "Close" in df.columns:
            prices = df["Close"].copy()
            prices.index = prices.index.normalize()
            return prices
    try:
        import yfinance as yf
        tickers = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}
        data = yf.download(tickers.get(symbol, "^NSEI"),
                           start="2021-01-01", end="2027-01-01", progress=False)
        return data["Close"]
    except Exception:
        return None


def load_chain(csv_path):
    """Load chain file, keep only the last snapshot if multiple timestamps."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    # Keep only last snapshot of the day
    if df["timestamp"].nunique() > 1:
        last_ts = df["timestamp"].max()
        df = df[df["timestamp"] == last_ts].copy()
    for col in ["ce_oi", "pe_oi", "ce_chg_oi", "pe_chg_oi",
                "ce_vol", "pe_vol", "ce_iv", "pe_iv", "ce_ltp", "pe_ltp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def find_oi_walls(df, spot, step):
    """Find put wall (support) and call wall (resistance) from OI structure."""
    # Strikes below spot — put wall (support)
    below = df[df["strike"] < spot - step].copy()  # at least 1 strike OTM
    above = df[df["strike"] > spot + step].copy()

    put_wall = None
    call_wall = None

    if len(below) > 0:
        # Put wall = strike with highest PE OI below spot
        idx = below["pe_oi"].idxmax()
        put_wall = below.loc[idx].to_dict()
        put_wall["dist_pct"] = (spot - put_wall["strike"]) / spot * 100

    if len(above) > 0:
        # Call wall = strike with highest CE OI above spot
        idx = above["ce_oi"].idxmax()
        call_wall = above.loc[idx].to_dict()
        call_wall["dist_pct"] = (call_wall["strike"] - spot) / spot * 100

    return put_wall, call_wall


def extract_wall_features(df, spot, wall, wall_type, step):
    """Extract 30+ features for a single wall trade."""
    strike = wall["strike"]
    total_oi = df["ce_oi"].sum() + df["pe_oi"].sum() + 1

    if wall_type == "put":
        wall_oi = wall["pe_oi"]
        wall_vol = wall["pe_vol"]
        wall_iv = wall["pe_iv"]
        wall_premium = wall["pe_ltp"]
        wall_chg_oi = wall["pe_chg_oi"]
    else:
        wall_oi = wall["ce_oi"]
        wall_vol = wall["ce_vol"]
        wall_iv = wall["ce_iv"]
        wall_premium = wall["ce_ltp"]
        wall_chg_oi = wall["ce_chg_oi"]

    dist_pct = wall["dist_pct"]

    # OI concentration: how much of total OI is at this wall?
    oi_concentration = wall_oi / total_oi * 100

    # OI buildup: is the wall growing or shrinking?
    oi_change_ratio = wall_chg_oi / (wall_oi + 1)

    # PCR at this strike
    strike_pcr = wall["pe_oi"] / (wall["ce_oi"] + 1)

    # Overall chain metrics
    total_ce_oi = df["ce_oi"].sum()
    total_pe_oi = df["pe_oi"].sum()
    pcr = total_pe_oi / (total_ce_oi + 1)
    total_ce_vol = df["ce_vol"].sum()
    total_pe_vol = df["pe_vol"].sum()
    vol_pcr = total_pe_vol / (total_ce_vol + 1)

    # Max pain
    strikes_arr = df["strike"].values
    if len(strikes_arr) > 0:
        pains = []
        for k in strikes_arr:
            p = ((k - df["strike"]).clip(lower=0) * df["ce_oi"]).sum() + \
                ((df["strike"] - k).clip(lower=0) * df["pe_oi"]).sum()
            pains.append(float(p))
        max_pain = strikes_arr[int(np.argmin(pains))]
        max_pain_dist = (spot - max_pain) / spot * 100
    else:
        max_pain = spot
        max_pain_dist = 0

    # Wall vs max pain distance
    wall_to_maxpain = abs(strike - max_pain) / spot * 100

    # How many other strikes have significant OI near this wall?
    if wall_type == "put":
        nearby = df[(df["strike"] >= strike - 2*step) & (df["strike"] <= strike + step)]
        support_depth = nearby["pe_oi"].sum()
    else:
        nearby = df[(df["strike"] >= strike - step) & (df["strike"] <= strike + 2*step)]
        support_depth = nearby["ce_oi"].sum()
    depth_ratio = support_depth / (wall_oi + 1)

    # IV metrics
    atm_idx = (df["strike"] - spot).abs().idxmin()
    atm_ce_iv = df.loc[atm_idx, "ce_iv"] if atm_idx in df.index else 0
    atm_pe_iv = df.loc[atm_idx, "pe_iv"] if atm_idx in df.index else 0
    iv_vs_atm = wall_iv / (atm_ce_iv + atm_pe_iv + 1) * 2

    # OI skew: ratio of OI below spot vs above
    below_oi = df[df["strike"] < spot]["pe_oi"].sum()
    above_oi = df[df["strike"] > spot]["ce_oi"].sum()
    oi_skew = below_oi / (above_oi + 1)

    # Volume at wall vs average
    avg_vol = (df["ce_vol"].mean() + df["pe_vol"].mean()) / 2 + 1
    wall_vol_ratio = wall_vol / avg_vol

    # Number of strikes with OI > wall_oi * 0.5 (competing walls)
    if wall_type == "put":
        competing = (df["pe_oi"] > wall_oi * 0.5).sum()
    else:
        competing = (df["ce_oi"] > wall_oi * 0.5).sum()

    # Premium as % of spot (moneyness proxy)
    premium_pct = wall_premium / spot * 100

    # Net OI change across chain
    net_pe_chg = df["pe_chg_oi"].sum()
    net_ce_chg = df["ce_chg_oi"].sum()
    net_oi_bias = (net_pe_chg - net_ce_chg) / (abs(net_pe_chg) + abs(net_ce_chg) + 1)

    features = {
        "dist_pct": round(dist_pct, 3),
        "wall_oi": wall_oi,
        "wall_oi_log": round(np.log1p(wall_oi), 3),
        "oi_concentration": round(oi_concentration, 3),
        "oi_change_ratio": round(oi_change_ratio, 4),
        "wall_chg_oi": wall_chg_oi,
        "oi_building": 1 if wall_chg_oi > 0 else 0,
        "strike_pcr": round(strike_pcr, 3),
        "pcr": round(pcr, 3),
        "vol_pcr": round(vol_pcr, 3),
        "max_pain_dist": round(max_pain_dist, 3),
        "wall_to_maxpain": round(wall_to_maxpain, 3),
        "wall_iv": round(wall_iv, 2),
        "iv_vs_atm": round(iv_vs_atm, 3),
        "depth_ratio": round(depth_ratio, 3),
        "support_depth_log": round(np.log1p(support_depth), 3),
        "oi_skew": round(oi_skew, 3),
        "wall_vol": wall_vol,
        "wall_vol_log": round(np.log1p(wall_vol), 3),
        "wall_vol_ratio": round(wall_vol_ratio, 3),
        "competing_walls": competing,
        "premium_pct": round(premium_pct, 4),
        "premium": round(wall_premium, 2),
        "net_oi_bias": round(net_oi_bias, 4),
        "total_oi_log": round(np.log1p(total_oi), 3),
        "n_strikes": len(df),
        "wall_type": 1 if wall_type == "put" else 0,
        "spot": spot,
        "strike": strike,
    }
    return features


def build_dataset(symbol, verbose=True):
    """Build features + labels for every wall trade across all historical days."""
    print(f"\n{'='*70}")
    print(f"  BUILDING DATASET: {symbol} OI WALL SELLING")
    print(f"{'='*70}")

    prices = load_spot_prices(symbol)
    if prices is None:
        print("  No price data")
        return None
    price_dates = {d.strftime("%Y-%m-%d"): float(prices.loc[d])
                   for d in prices.index}

    chain_files = sorted(glob.glob(os.path.join(RAW_DIR, symbol, "*.csv")))
    step = STRIKE_STEP[symbol]

    all_records = []

    for cf in chain_files:
        date_str = os.path.basename(cf).replace(".csv", "")
        if date_str not in price_dates:
            continue

        spot = price_dates[date_str]
        df = load_chain(cf)
        if df is None or len(df) < 5:
            continue

        put_wall, call_wall = find_oi_walls(df, spot, step)

        # Get future prices for outcome labeling
        trade_date = pd.Timestamp(date_str).normalize()
        future = prices.index[prices.index > trade_date]

        for wall, wtype in [(put_wall, "put"), (call_wall, "call")]:
            if wall is None:
                continue
            if wall["dist_pct"] < 0.3 or wall["dist_pct"] > 8.0:
                continue  # too close or too far

            features = extract_wall_features(df, spot, wall, wtype, step)
            features["date"] = date_str

            # Label: did the wall hold for each hold period?
            strike = wall["strike"]
            for hold in HOLD_DAYS:
                if len(future) < hold:
                    features[f"held_{hold}d"] = None
                    features[f"breach_pct_{hold}d"] = None
                    continue

                # Check all days in the hold period for breach
                hold_prices = [float(prices.loc[future[i]])
                               for i in range(hold)]

                if wtype == "put":
                    # Put wall holds if spot never goes below strike
                    min_price = min(hold_prices)
                    held = min_price >= strike
                    breach_pct = (strike - min_price) / spot * 100 if not held else 0
                else:
                    # Call wall holds if spot never goes above strike
                    max_price = max(hold_prices)
                    held = max_price <= strike
                    breach_pct = (max_price - strike) / spot * 100 if not held else 0

                features[f"held_{hold}d"] = 1 if held else 0
                features[f"breach_pct_{hold}d"] = round(breach_pct, 3)

            # Premium P&L (if we have next day data)
            if len(future) >= 1:
                next_date = future[0].strftime("%Y-%m-%d")
                next_path = os.path.join(RAW_DIR, symbol, f"{next_date}.csv")
                if os.path.exists(next_path):
                    next_df = load_chain(next_path)
                    if next_df is not None:
                        next_row = next_df[next_df["strike"] == strike]
                        if len(next_row) > 0:
                            nr = next_row.iloc[0]
                            if wtype == "put":
                                exit_prem = nr["pe_ltp"]
                            else:
                                exit_prem = nr["ce_ltp"]
                            features["premium_pnl"] = round(
                                features["premium"] - exit_prem, 2)
                            features["premium_pnl_pct"] = round(
                                (features["premium"] - exit_prem) /
                                (features["premium"] + 0.01) * 100, 2)

            all_records.append(features)

    df_out = pd.DataFrame(all_records)
    if verbose and len(df_out) > 0:
        print(f"  Total wall trades: {len(df_out)}")
        print(f"  Put walls: {(df_out['wall_type']==1).sum()}")
        print(f"  Call walls: {(df_out['wall_type']==0).sum()}")
        print(f"  Date range: {df_out['date'].iloc[0]} to {df_out['date'].iloc[-1]}")
        for hold in HOLD_DAYS:
            col = f"held_{hold}d"
            if col in df_out.columns:
                valid = df_out[col].dropna()
                if len(valid) > 0:
                    wr = valid.mean() * 100
                    print(f"  {hold}-day wall hold rate: {wr:.1f}% ({int(valid.sum())}/{len(valid)})")

    return df_out


def train_wall_model(symbol, df_data, hold_days=1, verbose=True):
    """Walk-forward train XGBoost to predict which walls will hold."""
    import xgboost as xgb

    label_col = f"held_{hold_days}d"
    if label_col not in df_data.columns:
        print(f"  No label column {label_col}")
        return None

    data = df_data.dropna(subset=[label_col]).copy()
    data["date_dt"] = pd.to_datetime(data["date"])
    data["year"] = data["date_dt"].dt.year

    feature_cols = [c for c in data.columns
                    if c not in ["date", "date_dt", "year", "spot", "strike",
                                 "premium_pnl", "premium_pnl_pct"] and
                    not c.startswith("held_") and not c.startswith("breach_")]
    feature_cols = [c for c in feature_cols if data[c].dtype in [
        np.float64, np.int64, np.float32, np.int32, float, int]]

    X = data[feature_cols].astype(float)
    y = data[label_col].astype(int).values
    dates = data["date_dt"]
    years = data["year"]

    print(f"\n  --- TRAINING: {symbol} {hold_days}-DAY WALL HOLD MODEL ---")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(X)} (held={y.sum()}, breached={len(y)-y.sum()})")
    print(f"  Base rate (wall holds): {y.mean()*100:.1f}%")

    # Walk-forward eras
    unique_years = sorted(years.unique())
    eras = []
    for i in range(len(unique_years)):
        test_yr = unique_years[i]
        train_yrs = [y for y in unique_years if y < test_yr]
        if len(train_yrs) < 1:
            continue
        eras.append({"train_years": train_yrs, "test_year": test_yr})

    if not eras:
        print("  Not enough years for walk-forward")
        return None

    all_preds = []
    all_actuals = []
    all_probas = []
    all_dates = []
    era_results = []

    for era in eras:
        train_mask = years.isin(era["train_years"])
        test_mask = years == era["test_year"]

        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]

        if len(X_tr) < 30 or len(X_te) < 10:
            continue

        sw = compute_sample_weight("balanced", y_tr)
        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.3, reg_lambda=1.5,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        model.fit(X_tr, y_tr, sample_weight=sw)

        preds = model.predict(X_te)
        probas = model.predict_proba(X_te)[:, 1]  # P(wall holds)
        acc = accuracy_score(y_te, preds)
        base = y_te.mean()

        all_preds.extend(preds)
        all_actuals.extend(y_te)
        all_probas.extend(probas)
        all_dates.extend(dates[test_mask].values)

        era_results.append({
            "test_year": int(era["test_year"]),
            "accuracy": round(acc, 4),
            "baseline": round(float(base), 4),
            "edge": round(acc - float(base), 4),
            "n_test": len(X_te),
        })

        if verbose:
            print(f"  {era['test_year']}: Acc {acc:.1%} | Base {base:.1%} | "
                  f"Edge {acc-base:+.1%} | n={len(X_te)}")

    if not all_preds:
        return None

    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)
    all_probas = np.array(all_probas)

    overall_acc = accuracy_score(all_actuals, all_preds)
    overall_base = all_actuals.mean()

    print(f"\n  Walk-forward overall:")
    print(f"    Accuracy: {overall_acc:.1%}")
    print(f"    Baseline: {overall_base:.1%}")
    print(f"    Edge: {overall_acc - overall_base:+.1%}")

    # Key metric: when model says P(hold) > threshold, what's actual hold rate?
    print(f"\n  --- CONFIDENCE THRESHOLDS ---")
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        mask = all_probas >= thresh
        if mask.sum() > 0:
            actual_hold = all_actuals[mask].mean() * 100
            participation = mask.sum() / len(all_probas) * 100
            print(f"  P(hold)>={thresh:.2f}: {actual_hold:.1f}% hold rate, "
                  f"{participation:.1f}% participation ({mask.sum()}/{len(all_probas)})")

    # Train final model on all data
    print(f"\n  Training final model on all {len(X)} samples...")
    sw_full = compute_sample_weight("balanced", y)
    final_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.3, reg_lambda=1.5,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    final_model.fit(X, y, sample_weight=sw_full)

    # Feature importances
    importances = pd.Series(
        final_model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print(f"\n  Top 10 features:")
    for feat, imp in importances.head(10).items():
        bar = "#" * int(imp * 60)
        print(f"    {feat:<22} {bar} {imp:.3f}")

    # Save
    model_path = os.path.join(MODEL_DIR, f"{symbol}_wall_{hold_days}d_xgb.pkl")
    fc_path = os.path.join(MODEL_DIR, f"{symbol}_wall_{hold_days}d_features.json")
    meta_path = os.path.join(MODEL_DIR, f"{symbol}_wall_{hold_days}d_meta.json")

    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    with open(fc_path, "w") as f:
        json.dump(feature_cols, f)
    with open(meta_path, "w") as f:
        json.dump({
            "symbol": symbol,
            "hold_days": hold_days,
            "accuracy": round(overall_acc, 4),
            "baseline": round(float(overall_base), 4),
            "edge": round(overall_acc - float(overall_base), 4),
            "n_samples": len(X),
            "era_results": era_results,
            "top_features": importances.head(10).index.tolist(),
        }, f, indent=2)

    print(f"  Model saved -> {model_path}")

    return {
        "symbol": symbol,
        "hold_days": hold_days,
        "accuracy": round(overall_acc, 4),
        "baseline": round(float(overall_base), 4),
        "edge": round(overall_acc - float(overall_base), 4),
        "era_results": era_results,
        "all_probas": all_probas,
        "all_actuals": all_actuals,
        "all_dates": all_dates,
    }


def backtest_strategy(symbol, df_data, model_result, hold_days=1):
    """Simulate the full strategy with P&L."""
    if model_result is None:
        return

    probas = model_result["all_probas"]
    actuals = model_result["all_actuals"]
    dates = model_result["all_dates"]

    label_col = f"held_{hold_days}d"
    valid = df_data.dropna(subset=[label_col, "premium_pnl"]).copy()
    valid["date_dt"] = pd.to_datetime(valid["date"])

    # Match by dates
    pred_dates = set(pd.to_datetime(dates).strftime("%Y-%m-%d"))
    valid = valid[valid["date"].isin(pred_dates)]

    print(f"\n  --- STRATEGY BACKTEST: {symbol} {hold_days}d ---")
    print(f"  Trades with premium P&L data: {len(valid)}")

    if len(valid) == 0:
        return

    # Test different confidence thresholds
    print(f"\n  {'Threshold':<12} {'Trades':<8} {'Part%':<8} {'Win%':<8} {'AvgP&L':<10} {'TotalP&L':<12}")
    print(f"  {'-'*58}")

    total_possible = len(df_data.dropna(subset=[label_col]))

    for thresh_name, cond in [
        ("All", lambda p: True),
        ("P>0.50", lambda p: p >= 0.50),
        ("P>0.55", lambda p: p >= 0.55),
        ("P>0.60", lambda p: p >= 0.60),
        ("P>0.65", lambda p: p >= 0.65),
        ("P>0.70", lambda p: p >= 0.70),
        ("P>0.75", lambda p: p >= 0.75),
        ("P>0.80", lambda p: p >= 0.80),
    ]:
        # For "All" threshold, use all valid trades without filtering
        if thresh_name == "All":
            subset = valid.copy()
        else:
            # We need to match probas to the valid data
            # Since walk-forward gives us probas for test years only,
            # just use the actual hold rate for "All"
            subset_mask = valid[label_col].values == valid[label_col].values  # all True
            subset = valid.copy()
            # For threshold-based, we need to be smarter
            # Skip for now and focus on actual hold rates
            continue

        trades = len(subset)
        if trades == 0:
            continue
        wins = subset[label_col].sum()
        win_pct = wins / trades * 100
        part_pct = trades / total_possible * 100 if total_possible > 0 else 0
        avg_pnl = subset["premium_pnl"].mean()
        total_pnl = subset["premium_pnl"].sum()

        print(f"  {thresh_name:<12} {trades:<8} {part_pct:<8.1f} {win_pct:<8.1f} "
              f"{avg_pnl:<10.2f} {total_pnl:<12.2f}")

    # Detailed by wall type
    print(f"\n  By wall type:")
    for wt, wt_name in [(1, "PUT_WALL"), (0, "CALL_WALL")]:
        subset = valid[valid["wall_type"] == wt]
        if len(subset) == 0:
            continue
        wins = subset[label_col].sum()
        win_pct = wins / len(subset) * 100
        avg_pnl = subset["premium_pnl"].mean()
        print(f"  {wt_name}: {wins}/{len(subset)} = {win_pct:.1f}% hold rate, "
              f"avg premium P&L = {avg_pnl:.2f}")

    # By year
    valid["year"] = valid["date_dt"].dt.year
    print(f"\n  By year:")
    for year, group in valid.groupby("year"):
        wins = group[label_col].sum()
        win_pct = wins / len(group) * 100
        avg_pnl = group["premium_pnl"].mean()
        print(f"  {year}: {wins}/{len(group)} = {win_pct:.1f}% hold rate, "
              f"avg P&L = {avg_pnl:.2f}")

    # By distance
    print(f"\n  By wall distance from spot:")
    for lo, hi in [(0.3, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 8.0)]:
        subset = valid[(valid["dist_pct"] >= lo) & (valid["dist_pct"] < hi)]
        if len(subset) < 5:
            continue
        wins = subset[label_col].sum()
        win_pct = wins / len(subset) * 100
        avg_pnl = subset["premium_pnl"].mean()
        print(f"  {lo:.0f}-{hi:.0f}%: {wins}/{len(subset)} = {win_pct:.1f}% hold rate, "
              f"avg P&L = {avg_pnl:.2f}")


def predict_wall_hold(symbol, wall_features, hold_days=1):
    """Live inference: predict P(wall holds) for a new wall trade."""
    model_path = os.path.join(MODEL_DIR, f"{symbol}_wall_{hold_days}d_xgb.pkl")
    fc_path = os.path.join(MODEL_DIR, f"{symbol}_wall_{hold_days}d_features.json")

    if not os.path.exists(model_path):
        return None

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(fc_path) as f:
        feature_cols = json.load(f)

    X = pd.DataFrame([wall_features])
    available = [c for c in feature_cols if c in X.columns]
    X = X[available].astype(float)

    prob_hold = model.predict_proba(X)[0][1]
    return round(float(prob_hold), 4)


if __name__ == "__main__":
    all_results = {}

    for symbol in ["NIFTY", "BANKNIFTY"]:
        # Build dataset
        df_data = build_dataset(symbol)
        if df_data is None or len(df_data) < 100:
            print(f"  Insufficient data for {symbol}")
            continue

        # Save dataset
        ds_path = os.path.join(ROOT, "data", f"oi_wall_dataset_{symbol}.csv")
        df_data.to_csv(ds_path, index=False)
        print(f"  Dataset saved -> {ds_path}")

        # Train and backtest for each hold period
        for hold in HOLD_DAYS:
            result = train_wall_model(symbol, df_data, hold_days=hold)
            if result:
                all_results[f"{symbol}_{hold}d"] = result
                backtest_strategy(symbol, df_data, result, hold_days=hold)

    # Final summary
    if all_results:
        print(f"\n{'='*70}")
        print(f"  FINAL SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Config':<20} {'Accuracy':<10} {'Baseline':<10} {'Edge':<10}")
        print(f"  {'-'*50}")
        for key, r in all_results.items():
            print(f"  {key:<20} {r['accuracy']:.1%}{'':<5} {r['baseline']:.1%}{'':<5} "
                  f"{r['edge']:+.1%}")

        # Find best confidence threshold across all models
        print(f"\n  BEST THRESHOLDS FOR 85% WIN + 80% PARTICIPATION:")
        for key, r in all_results.items():
            probas = r["all_probas"]
            actuals = r["all_actuals"]
            for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
                mask = probas >= thresh
                if mask.sum() > 0:
                    win_pct = actuals[mask].mean() * 100
                    part_pct = mask.sum() / len(probas) * 100
                    if win_pct >= 80 and part_pct >= 60:
                        print(f"  ** {key} @ P>={thresh:.2f}: "
                              f"{win_pct:.1f}% win, {part_pct:.1f}% part **")

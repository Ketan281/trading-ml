import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime
from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import TimeSeriesSplit
from sklearn.metrics          import classification_report, accuracy_score
from sklearn.preprocessing    import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.dirname(__file__)
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Feature Engineering ───────────────────────────────
def compute_features(df, symbol=None):
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    features = pd.DataFrame(index=df.index)

    # Price features
    features["returns"]      = close.pct_change()
    features["returns_5d"]   = close.pct_change(5)
    features["returns_10d"]  = close.pct_change(10)
    features["returns_20d"]  = close.pct_change(20)

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    features["rsi"] = 100 - (100 / (1 + gain / loss))

    # MACD
    ema12  = close.ewm(span=12).mean()
    ema26  = close.ewm(span=26).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    features["macd_hist"] = macd - signal

    # EMAs
    features["ema20"]       = close.ewm(span=20).mean()
    features["ema50"]       = close.ewm(span=50).mean()
    features["ema_ratio"]   = features["ema20"] / features["ema50"]
    features["price_ema20"] = close / features["ema20"]

    # Bollinger Bands
    ma20  = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    features["bb_position"] = (close - ma20) / (2 * std20)

    # ATR / Volatility
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    features["atr"]        = tr.rolling(14).mean()
    features["volatility"] = close.rolling(20).std() / close.rolling(20).mean()

    # Volume
    features["volume_ratio"] = volume / volume.rolling(20).mean()

    # Momentum
    features["momentum_10"] = close / close.shift(10)
    features["momentum_20"] = close / close.shift(20)

    # Position within recent range (0 = at 20d low, 1 = at 20d high)
    roll_high = high.rolling(20).max()
    roll_low  = low.rolling(20).min()
    features["range_pos_20"] = (
        (close - roll_low) / (roll_high - roll_low)
    )

    # Short trend slope: normalised 10-day linear regression slope
    def _slope(x):
        idx = np.arange(len(x))
        return np.polyfit(idx, x, 1)[0]
    features["trend_slope_10"] = (
        close.rolling(10).apply(_slope, raw=True) / close
    )

    # Distance from 20d EMA in ATR units (how stretched price is)
    tr2 = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr20 = tr2.rolling(14).mean()
    features["stretch_atr"] = (close - features["ema20"]) / atr20

    # ── Real intermarket + institutional features ─────
    # MFI — real money flow from price + volume
    typical = (high + low + close) / 3
    mf = typical * volume
    pos_mf = pd.Series(np.where(typical > typical.shift(1), mf, 0), index=df.index)
    neg_mf = pd.Series(np.where(typical < typical.shift(1), mf, 0), index=df.index)
    mfr = pos_mf.rolling(14).sum() / neg_mf.rolling(14).sum().replace(0, np.nan)
    features["mfi_14"] = 100 - (100 / (1 + mfr))

    # Volume-price confirmation (delivery signal)
    ret_5d = close.pct_change(5)
    vol_5d = volume.rolling(5).mean()
    vol_20d_avg = volume.rolling(20).mean()
    vol_surge = vol_5d / vol_20d_avg
    features["vol_price_confirm"] = np.where(
        (ret_5d > 0) & (vol_surge > 1.5), 1,
        np.where((ret_5d < 0) & (vol_surge > 1.5), -1, 0)
    ).astype(float)

    # S/R proximity
    high_20 = high.rolling(20).max()
    low_20 = low.rolling(20).min()
    range_20 = high_20 - low_20
    features["sr_proximity"] = np.where(range_20 > 0, (close - low_20) / range_20, 0.5)

    # Fibonacci level (52-week)
    high_52 = high.rolling(252).max()
    low_52 = low.rolling(252).min()
    range_52 = high_52 - low_52
    features["fib_level"] = np.where(range_52 > 0, (high_52 - close) / range_52, 0.5)

    # A/D momentum (smart money flow)
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    ad = (clv.fillna(0) * volume).cumsum()
    ad_ema5 = ad.ewm(span=5).mean()
    ad_ema20 = ad.ewm(span=20).mean()
    features["ad_momentum"] = np.where(ad_ema20 != 0, (ad_ema5 / ad_ema20 - 1) * 100, 0)

    # Sharpe-like ratio
    ret_60 = close.pct_change(60)
    vol_60 = close.pct_change().rolling(60).std()
    features["sharpe_60d"] = np.where(vol_60 > 0, ret_60 / vol_60, 0)

    # Up/down volume ratio (institutional conviction)
    ret_1d = close.pct_change()
    up_vol = pd.Series(np.where(ret_1d > 0, volume, 0), index=df.index)
    dn_vol = pd.Series(np.where(ret_1d < 0, volume, 0), index=df.index)
    features["up_down_vol_ratio"] = up_vol.rolling(20).sum() / dn_vol.rolling(20).sum().replace(0, np.nan)

    # ── Join REAL intermarket data (crude, DXY, VIX, gold, FII) ──
    intel_path = os.path.join(DATA_DIR, "market_intel", "intermarket_features.csv")
    if os.path.exists(intel_path):
        try:
            im = pd.read_csv(intel_path, index_col="Date", parse_dates=True)
            for col in im.columns:
                if col not in features.columns:
                    aligned = im[col].reindex(df.index, method="ffill")
                    if aligned.notna().sum() > len(df) * 0.3:
                        features[col] = aligned
        except Exception:
            pass

    # ── Join earnings event features (if available) ──
    earn_dir = os.path.join(DATA_DIR, "earnings")
    earn_sym = symbol

    if earn_sym and os.path.isdir(earn_dir):
        earn_path = os.path.join(earn_dir, f"{earn_sym}_earnings.csv")
        if os.path.exists(earn_path):
            try:
                edf = pd.read_csv(earn_path, parse_dates=["earnings_date"])
                edf = edf.sort_values("earnings_date").drop_duplicates("earnings_date")
                earn_dates_ts = pd.to_datetime(edf["earnings_date"].values)
                surprises = edf["surprise_pct"].values

                days_since = np.full(len(df.index), np.nan)
                days_to = np.full(len(df.index), np.nan)
                last_surprise = np.full(len(df.index), np.nan)
                streak_arr = np.full(len(df.index), 0.0)

                for i, dt in enumerate(df.index):
                    past = earn_dates_ts[earn_dates_ts <= dt]
                    if len(past) > 0:
                        days_since[i] = (dt - past[-1]).days
                        idx = len(past) - 1
                        if idx < len(surprises):
                            last_surprise[i] = surprises[idx]
                        streak = 0
                        for j in range(len(past)-1, max(len(past)-5, -1), -1):
                            if j < len(surprises) and not np.isnan(surprises[j]):
                                if surprises[j] > 0: streak += 1
                                elif surprises[j] < 0: streak -= 1
                                else: break
                        streak_arr[i] = streak
                    future = earn_dates_ts[earn_dates_ts > dt]
                    if len(future) > 0:
                        days_to[i] = (future[0] - dt).days

                features["days_since_earnings"] = days_since
                features["days_to_earnings"] = days_to
                features["last_surprise_pct"] = last_surprise
                features["surprise_streak"] = streak_arr
                features["near_earnings"] = (
                    (pd.Series(days_to, index=df.index).fillna(999) <= 5) |
                    (pd.Series(days_since, index=df.index).fillna(999) <= 2)
                ).astype(float)
            except Exception:
                pass

    return features

# ── Create Labels (Triple-Barrier) ────────────────────
def create_labels(df, horizon=10, atr_mult=1.5):
    """
    Triple-barrier labelling — the way a trader actually frames a trade.

    For each day, set an upper barrier at +atr_mult*ATR and a lower barrier
    at -atr_mult*ATR. Look forward up to `horizon` days:
        • upper hit first  -> "uptrend"   (a long would have won)
        • lower hit first  -> "downtrend" (a long would have stopped out)
        • neither (timeout)-> "sideways"
    Rows whose forward window runs past the end of the data are left as
    NaN so we never train on labels we can't actually determine.

    Barriers scale with volatility (ATR), so the label means the same
    thing for a calm stock and a wild one. Symmetric barriers keep the
    three classes reasonably balanced.
    """
    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values

    # ATR(14) in price units
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().values

    n   = len(close)
    out = np.empty(n, dtype=object)

    for i in range(n):
        a = atr[i]
        if np.isnan(a) or a <= 0:
            out[i] = None
            continue

        upper = close[i] + atr_mult * a
        lower = close[i] - atr_mult * a
        end   = min(i + horizon, n - 1)

        label = None
        for j in range(i + 1, end + 1):
            hit_up = high[j] >= upper
            hit_dn = low[j]  <= lower
            if hit_up and hit_dn:
                label = "sideways"      # both in one bar -> ambiguous
                break
            if hit_up:
                label = "uptrend"
                break
            if hit_dn:
                label = "downtrend"
                break

        if label is None:
            # Window fully elapsed with no barrier touched -> sideways.
            # Window incomplete (too close to today) -> undetermined.
            label = "sideways" if (i + horizon) <= (n - 1) else None

        out[i] = label

    return pd.Series(out, index=df.index)

# ── Model Factories ───────────────────────────────────
def _make_xgb():
    return xgb.XGBClassifier(
        n_estimators     = 300,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        eval_metric      = "mlogloss",
        random_state     = 42,
        verbosity        = 0,
    )

def _make_rf():
    return RandomForestClassifier(
        n_estimators = 300,
        max_depth    = 6,
        class_weight = "balanced",
        random_state = 42,
        n_jobs       = -1,
    )


# ── Era-Based Walk-Forward Windows ────────────────────
# Each era trains on one market regime, tests on a different one.
# This validates that the model generalises across regimes, not just
# recent data — the way a 20-year trader builds conviction.
ERA_WINDOWS = [
    {"name": "GFC_era",          "train_start": 2006, "train_end": 2010, "test_year": 2011},
    {"name": "bull_run",         "train_start": 2012, "train_end": 2015, "test_year": 2016},
    {"name": "pre_covid",        "train_start": 2016, "train_end": 2019, "test_year": 2020},
    {"name": "post_covid",       "train_start": 2020, "train_end": 2025, "test_year": 2026},
]


# ── Train Model (era-based walk-forward + balanced) ───
def train_model(symbol):
    print(f"\n  [TRAIN] Training model for {symbol}...")

    # Try both naming conventions for historical data
    path = os.path.join(DATA_DIR, f"{symbol}_daily.csv")
    if not os.path.exists(path):
        path = os.path.join(DATA_DIR, "historical", f"{symbol}.csv")
    if not os.path.exists(path):
        print(f"  [WARN] No data found for {symbol}")
        return None

    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    if len(df) < 300:
        print(f"  [WARN] Not enough data for {symbol} ({len(df)} rows). Need 300+")
        return None

    # Features + triple-barrier labels, aligned and cleaned.
    features = compute_features(df, symbol=symbol)
    labels   = create_labels(df)
    data = features.copy()
    data["label"] = labels
    data.dropna(inplace=True)
    if len(data) < 200:
        print(f"  [WARN] Not enough clean rows ({len(data)})")
        return None

    feature_cols = [c for c in data.columns if c != "label"]
    X = data[feature_cols]
    y_text = data["label"]
    dates = data.index

    # Show class balance — confirms labels aren't degenerate.
    dist = y_text.value_counts(normalize=True).to_dict()
    print("     Label balance: " + ", ".join(
        f"{k}={v:.0%}" for k, v in dist.items()))
    print(f"     Data range: {dates[0].date()} -> {dates[-1].date()} ({len(data)} rows)")

    le = LabelEncoder()
    y  = le.fit_transform(y_text)

    # ── Era-based walk-forward cross-validation ──────
    # Train on one market era, test on the next — validates regime
    # generalisation the way a 20-year veteran builds conviction.
    xgb_scores, rf_scores, baseline_scores = [], [], []
    era_results = []
    last_fold = None

    for era in ERA_WINDOWS:
        train_mask = (dates.year >= era["train_start"]) & (dates.year <= era["train_end"])
        test_mask  = dates.year == era["test_year"]

        train_idx = np.array(train_mask)
        test_idx  = np.array(test_mask)
        X_tr = X[train_idx].reset_index(drop=True)
        X_te = X[test_idx].reset_index(drop=True)
        y_tr = y[train_idx]
        y_te = y[test_idx]

        if len(X_tr) < 50 or len(X_te) < 10:
            print(f"     {era['name']}: skipped (train={len(X_tr)}, test={len(X_te)})")
            continue

        sw = compute_sample_weight("balanced", y_tr)
        xm = _make_xgb(); xm.fit(X_tr, y_tr, sample_weight=sw)
        rm = _make_rf();  rm.fit(X_tr, y_tr)

        xgb_acc = accuracy_score(y_te, xm.predict(X_te))
        rf_acc  = accuracy_score(y_te, rm.predict(X_te))
        majority = np.bincount(y_tr).argmax()
        base_acc = accuracy_score(y_te, np.full_like(y_te, majority))

        xgb_scores.append(xgb_acc)
        rf_scores.append(rf_acc)
        baseline_scores.append(base_acc)
        last_fold = (X_te, y_te, xm)

        era_results.append({
            "era": era["name"],
            "train": f"{era['train_start']}-{era['train_end']}",
            "test": era["test_year"],
            "xgb": round(xgb_acc, 4),
            "rf": round(rf_acc, 4),
            "baseline": round(base_acc, 4),
            "edge": round(xgb_acc - base_acc, 4),
            "train_rows": len(X_tr),
            "test_rows": len(X_te),
        })
        print(f"     {era['name']}: Train {era['train_start']}-{era['train_end']} "
              f"({len(X_tr)} rows) -> Test {era['test_year']} ({len(X_te)} rows) "
              f"| XGB {xgb_acc:.1%} | RF {rf_acc:.1%} | Base {base_acc:.1%} "
              f"| Edge {xgb_acc - base_acc:+.1%}")

    # Fallback to TimeSeriesSplit if not enough era data
    if not xgb_scores:
        print("     No era windows had enough data — falling back to TimeSeriesSplit")
        X_reset = X.reset_index(drop=True)
        tscv = TimeSeriesSplit(n_splits=5)
        for tr_idx, te_idx in tscv.split(X_reset):
            X_tr, X_te = X_reset.iloc[tr_idx], X_reset.iloc[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            sw = compute_sample_weight("balanced", y_tr)
            xm = _make_xgb(); xm.fit(X_tr, y_tr, sample_weight=sw)
            rm = _make_rf();  rm.fit(X_tr, y_tr)
            xgb_scores.append(accuracy_score(y_te, xm.predict(X_te)))
            rf_scores.append(accuracy_score(y_te, rm.predict(X_te)))
            majority = np.bincount(y_tr).argmax()
            baseline_scores.append(accuracy_score(y_te, np.full_like(y_te, majority)))
            last_fold = (X_te, y_te, xm)

    xgb_cv      = float(np.mean(xgb_scores))
    rf_cv       = float(np.mean(rf_scores))
    baseline_cv = float(np.mean(baseline_scores))
    edge        = xgb_cv - baseline_cv

    print(f"\n  [OK] {symbol} era-based walk-forward results ({len(xgb_scores)} eras):")
    print(f"     XGBoost  CV accuracy : {xgb_cv:.2%}")
    print(f"     RandomForest CV acc  : {rf_cv:.2%}")
    print(f"     Majority baseline    : {baseline_cv:.2%}")
    print(f"     => Edge over guessing : {edge:+.2%}")

    # Out-of-sample report from the final (most recent) fold.
    X_te, y_te, xm_last = last_fold
    print(f"\n  Classification report (last fold, out-of-sample):")
    print(classification_report(
        y_te, xm_last.predict(X_te),
        labels=list(range(len(le.classes_))),
        target_names=le.classes_, zero_division=0))

    # ── Fit FINAL models on all data (for live prediction) ─
    X_final   = X.reset_index(drop=True)
    sw_full   = compute_sample_weight("balanced", y)
    xgb_model = _make_xgb(); xgb_model.fit(X_final, y, sample_weight=sw_full)
    rf_model  = _make_rf();  rf_model.fit(X_final, y)

    importances = pd.Series(
        xgb_model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print(f"\n  Top 5 important features:")
    for feat, imp in importances.head(5).items():
        bar = "#" * int(imp * 50)
        print(f"     {feat:<20} {bar} {imp:.3f}")

    # ── Save ──────────────────────────────────────────
    xgb_path = os.path.join(MODELS_DIR, f"{symbol}_xgb.pkl")
    rf_path  = os.path.join(MODELS_DIR, f"{symbol}_rf.pkl")
    le_path  = os.path.join(MODELS_DIR, f"{symbol}_le.pkl")
    fc_path  = os.path.join(MODELS_DIR, f"{symbol}_features.json")
    meta_path = os.path.join(MODELS_DIR, f"{symbol}_meta.json")
    with open(xgb_path, "wb") as f: pickle.dump(xgb_model, f)
    with open(rf_path,  "wb") as f: pickle.dump(rf_model, f)
    with open(le_path,  "wb") as f: pickle.dump(le, f)
    with open(fc_path,  "w")  as f: json.dump(feature_cols, f)
    # Persist the honest walk-forward edge so the engine can decide
    # whether this model has earned the right to influence decisions.
    with open(meta_path, "w") as f:
        json.dump({
            "symbol":   symbol,
            "xgb_cv":   round(xgb_cv, 4),
            "baseline": round(baseline_cv, 4),
            "edge":     round(edge, 4),
            "trained":  datetime.now().isoformat(),
            "data_range": f"{dates[0].date()} -> {dates[-1].date()}",
            "total_rows": len(data),
            "era_results": era_results,
        }, f, indent=2)
    print(f"\n  [OK] Models saved -> {xgb_path}")

    return {
        "symbol"       : symbol,
        "xgb_accuracy" : round(xgb_cv, 4),
        "rf_accuracy"  : round(rf_cv, 4),
        "baseline"     : round(baseline_cv, 4),
        "edge"         : round(edge, 4),
        "top_features" : importances.head(5).index.tolist(),
        "xgb_path"     : xgb_path,
        "rf_path"      : rf_path
    }

# ── Predict With Saved Model ──────────────────────────
def predict(symbol, latest_indicators=None):
    """
    Predict the near-term regime for `symbol` using its saved model.

    The feature row is computed from the SAME price history and the SAME
    compute_features() used during training, then the most recent valid
    row is fed to the model. This guarantees the inputs at prediction time
    match the inputs the model was trained on. (`latest_indicators` is
    accepted for backwards compatibility but is no longer used.)
    """
    xgb_path = os.path.join(MODELS_DIR, f"{symbol}_xgb.pkl")
    le_path  = os.path.join(MODELS_DIR, f"{symbol}_le.pkl")
    fc_path  = os.path.join(MODELS_DIR, f"{symbol}_features.json")

    if not os.path.exists(xgb_path):
        print(f"  [WARN] No trained model found for {symbol}")
        return None

    with open(xgb_path, "rb") as f: model = pickle.load(f)
    with open(le_path,  "rb") as f: le    = pickle.load(f)
    with open(fc_path,  "r")  as f: cols  = json.load(f)

    # ── Compute REAL features from price history ──────
    data_path = os.path.join(DATA_DIR, f"{symbol}_daily.csv")
    if not os.path.exists(data_path):
        print(f"  [WARN] No price data for {symbol} — cannot build features")
        return None

    df = pd.read_csv(data_path, index_col="Date", parse_dates=True)
    features = compute_features(df, symbol=symbol)

    # Keep only the columns the model was trained on, in the same order,
    # then take the most recent fully-populated row.
    features = features.reindex(columns=cols)
    features = features.dropna()
    if features.empty:
        print(f"  [WARN] Not enough history to compute features for {symbol}")
        return None

    X    = features.iloc[[-1]]            # latest real feature row
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    label      = le.inverse_transform([pred])[0]
    confidence = round(float(max(prob)), 4)

    # Attach the model's proven out-of-sample edge (if recorded), so the
    # decision layer can refuse to trust a model that has no edge.
    edge = None
    meta_path = os.path.join(MODELS_DIR, f"{symbol}_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                edge = json.load(f).get("edge")
        except Exception:
            pass

    return {
        "symbol"    : symbol,
        "ml_regime" : label,
        "confidence": confidence,
        "edge"      : edge
    }

# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Trading AI — ML Model Training")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS"]
    results = []

    for symbol in symbols:
        result = train_model(symbol)
        if result:
            results.append(result)

    # Summary
    print("\n" + "=" * 55)
    print("  ML Training Summary")
    print("=" * 55)
    print(f"  {'SYMBOL':<12} {'XGB CV':<10} {'BASELINE':<10} {'EDGE'}")
    print("  " + "-" * 45)
    for r in results:
        flag = "[OK]" if r.get("edge", 0) > 0 else "[WARN]"
        print(f"  {r['symbol']:<12} "
              f"{r['xgb_accuracy']:.2%}{'':>3} "
              f"{r.get('baseline', 0):.2%}{'':>3} "
              f"{r.get('edge', 0):+.2%} {flag}")

    print(f"\n  [OK] {len(results)} models trained and saved to /models")
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
def compute_features(df):
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


# ── Train Model (walk-forward + balanced) ─────────────
def train_model(symbol):
    print(f"\n  📊 Training model for {symbol}...")

    path = os.path.join(DATA_DIR, f"{symbol}_daily.csv")
    if not os.path.exists(path):
        print(f"  ⚠ No data found for {symbol}")
        return None

    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    if len(df) < 300:
        print(f"  ⚠ Not enough data for {symbol} ({len(df)} rows). Need 300+")
        return None

    # Features + triple-barrier labels, aligned and cleaned.
    features = compute_features(df)
    labels   = create_labels(df)
    data = features.copy()
    data["label"] = labels
    data.dropna(inplace=True)
    if len(data) < 200:
        print(f"  ⚠ Not enough clean rows ({len(data)})")
        return None

    feature_cols = [c for c in data.columns if c != "label"]
    X = data[feature_cols].reset_index(drop=True)
    y_text = data["label"]

    # Show class balance — confirms labels aren't degenerate.
    dist = y_text.value_counts(normalize=True).to_dict()
    print("     Label balance: " + ", ".join(
        f"{k}={v:.0%}" for k, v in dist.items()))

    le = LabelEncoder()
    y  = le.fit_transform(y_text)

    # ── Walk-forward cross-validation ─────────────────
    # Train only on the PAST, test on the immediate FUTURE, repeatedly.
    # This is the only honest accuracy estimate for a time series.
    tscv = TimeSeriesSplit(n_splits=5)
    xgb_scores, rf_scores, baseline_scores = [], [], []
    last_fold = None

    for tr_idx, te_idx in tscv.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Balance classes so the model can't win by always guessing one.
        sw = compute_sample_weight("balanced", y_tr)

        xm = _make_xgb(); xm.fit(X_tr, y_tr, sample_weight=sw)
        rm = _make_rf();  rm.fit(X_tr, y_tr)

        xgb_scores.append(accuracy_score(y_te, xm.predict(X_te)))
        rf_scores.append(accuracy_score(y_te, rm.predict(X_te)))

        # Baseline: always predict the most common class seen in training.
        majority = np.bincount(y_tr).argmax()
        baseline_scores.append(
            accuracy_score(y_te, np.full_like(y_te, majority)))

        last_fold = (X_te, y_te, xm)

    xgb_cv      = float(np.mean(xgb_scores))
    rf_cv       = float(np.mean(rf_scores))
    baseline_cv = float(np.mean(baseline_scores))
    edge        = xgb_cv - baseline_cv      # >0 means real predictive value

    print(f"\n  ✅ {symbol} walk-forward results (5 folds):")
    print(f"     XGBoost  CV accuracy : {xgb_cv:.2%}")
    print(f"     RandomForest CV acc  : {rf_cv:.2%}")
    print(f"     Majority baseline    : {baseline_cv:.2%}")
    print(f"     ⇒ Edge over guessing : {edge:+.2%}")

    # Out-of-sample report from the final (most recent) fold.
    X_te, y_te, xm_last = last_fold
    print(f"\n  Classification report (last fold, out-of-sample):")
    print(classification_report(
        y_te, xm_last.predict(X_te),
        labels=list(range(len(le.classes_))),
        target_names=le.classes_, zero_division=0))

    # ── Fit FINAL models on all data (for live prediction) ─
    sw_full   = compute_sample_weight("balanced", y)
    xgb_model = _make_xgb(); xgb_model.fit(X, y, sample_weight=sw_full)
    rf_model  = _make_rf();  rf_model.fit(X, y)

    importances = pd.Series(
        xgb_model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print(f"\n  Top 5 important features:")
    for feat, imp in importances.head(5).items():
        bar = "█" * int(imp * 50)
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
        }, f, indent=2)
    print(f"\n  ✅ Models saved → {xgb_path}")

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
        print(f"  ⚠ No trained model found for {symbol}")
        return None

    with open(xgb_path, "rb") as f: model = pickle.load(f)
    with open(le_path,  "rb") as f: le    = pickle.load(f)
    with open(fc_path,  "r")  as f: cols  = json.load(f)

    # ── Compute REAL features from price history ──────
    data_path = os.path.join(DATA_DIR, f"{symbol}_daily.csv")
    if not os.path.exists(data_path):
        print(f"  ⚠ No price data for {symbol} — cannot build features")
        return None

    df = pd.read_csv(data_path, index_col="Date", parse_dates=True)
    features = compute_features(df)

    # Keep only the columns the model was trained on, in the same order,
    # then take the most recent fully-populated row.
    features = features.reindex(columns=cols)
    features = features.dropna()
    if features.empty:
        print(f"  ⚠ Not enough history to compute features for {symbol}")
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
        flag = "✅" if r.get("edge", 0) > 0 else "⚠"
        print(f"  {r['symbol']:<12} "
              f"{r['xgb_accuracy']:.2%}{'':>3} "
              f"{r.get('baseline', 0):.2%}{'':>3} "
              f"{r.get('edge', 0):+.2%} {flag}")

    print(f"\n  ✅ {len(results)} models trained and saved to /models")
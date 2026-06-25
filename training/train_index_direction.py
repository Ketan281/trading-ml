"""
Train NIFTY / BANKNIFTY next-day direction models using the full 104-feature set.

Target: next-day direction (up / down / flat) for index options trading.
Uses walk-forward validation across market regimes.
Saves models to models/index_direction/ for live inference by the options engine.
"""

import os
import sys
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FEATURE_DIR = os.path.join(ROOT, "data", "features")
MODEL_DIR = os.path.join(ROOT, "models", "index_direction")
os.makedirs(MODEL_DIR, exist_ok=True)

INDICES = ["NIFTY", "BANKNIFTY"]

# Forward return horizon and threshold for binary labelling
FWD_HORIZON = 5     # 5-day forward return captures weekly trend
UP_THRESH = 0.5     # >+0.5% over 5 days = bullish move


def load_features(symbol):
    path = os.path.join(FEATURE_DIR, f"{symbol}_features.csv")
    if not os.path.exists(path):
        print(f"  [WARN] No feature file for {symbol}")
        return None
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df


def create_direction_labels(df):
    """Binary: will the index be higher in FWD_HORIZON days?"""
    fwd_ret = df["Close"].pct_change(FWD_HORIZON).shift(-FWD_HORIZON) * 100
    labels = pd.Series(index=df.index, dtype=object)
    labels[fwd_ret > UP_THRESH] = "up"
    labels[fwd_ret <= UP_THRESH] = "not_up"
    labels[fwd_ret.isna()] = None
    return labels


ERA_WINDOWS = [
    {"name": "era_2014_2016", "train_start": 2014, "train_end": 2016, "test_year": 2017},
    {"name": "era_2015_2017", "train_start": 2015, "train_end": 2017, "test_year": 2018},
    {"name": "era_2016_2018", "train_start": 2016, "train_end": 2018, "test_year": 2019},
    {"name": "era_2017_2019", "train_start": 2017, "train_end": 2019, "test_year": 2020},
    {"name": "era_2018_2020", "train_start": 2018, "train_end": 2020, "test_year": 2021},
    {"name": "era_2019_2021", "train_start": 2019, "train_end": 2021, "test_year": 2022},
    {"name": "era_2020_2022", "train_start": 2020, "train_end": 2022, "test_year": 2023},
    {"name": "era_2021_2023", "train_start": 2021, "train_end": 2023, "test_year": 2024},
]


def train_index(symbol, verbose=True):
    print(f"\n{'='*60}")
    print(f"  TRAINING INDEX DIRECTION MODEL: {symbol}")
    print(f"{'='*60}")

    df = load_features(symbol)
    if df is None or len(df) < 500:
        print(f"  Insufficient data for {symbol}")
        return None

    labels = create_direction_labels(df)
    feature_cols = [c for c in df.columns if c != "Close"]
    data = df[feature_cols].copy()

    # Drop non-numeric columns XGBoost can't handle
    num_cols = data.select_dtypes(include=["number", "bool"]).columns.tolist()
    str_cols = [c for c in data.columns if c not in num_cols]
    if str_cols:
        print(f"  Dropping non-numeric columns: {str_cols}")
        data = data[num_cols]
        feature_cols = [c for c in feature_cols if c in num_cols]

    data["label"] = labels
    data.dropna(inplace=True)

    if len(data) < 300:
        print(f"  Not enough clean rows ({len(data)})")
        return None

    # Ensure feature_cols matches what's actually in data
    feature_cols = [c for c in feature_cols if c in data.columns]
    X = data[feature_cols].astype(float)
    y_text = data["label"]
    dates = data.index

    dist = y_text.value_counts(normalize=True).to_dict()
    print(f"  Label balance: {', '.join(f'{k}={v:.0%}' for k, v in dist.items())}")
    print(f"  Data: {dates[0].date()} -> {dates[-1].date()} ({len(data)} rows)")
    print(f"  Features: {len(feature_cols)}")

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y_text)

    xgb_scores, baseline_scores = [], []
    era_results = []
    last_fold = None

    for era in ERA_WINDOWS:
        train_mask = (dates.year >= era["train_start"]) & (dates.year <= era["train_end"])
        test_mask = dates.year == era["test_year"]
        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]

        if len(X_tr) < 100 or len(X_te) < 30:
            continue

        sw = compute_sample_weight("balanced", y_tr)
        model = xgb.XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="mlogloss", random_state=42, verbosity=0,
        )
        model.fit(X_tr, y_tr, sample_weight=sw)

        acc = accuracy_score(y_te, model.predict(X_te))
        majority = np.bincount(y_tr).argmax()
        base = accuracy_score(y_te, np.full_like(y_te, majority))

        xgb_scores.append(acc)
        baseline_scores.append(base)
        last_fold = (X_te, y_te, model)

        era_results.append({
            "era": era["name"], "test_year": era["test_year"],
            "accuracy": round(acc, 4), "baseline": round(base, 4),
            "edge": round(acc - base, 4),
            "train_rows": len(X_tr), "test_rows": len(X_te),
        })
        if verbose:
            print(f"  {era['name']}: test {era['test_year']} "
                  f"| Acc {acc:.1%} | Base {base:.1%} | Edge {acc-base:+.1%}")

    if not xgb_scores:
        print(f"  No valid eras for {symbol}")
        return None

    mean_acc = float(np.mean(xgb_scores))
    mean_base = float(np.mean(baseline_scores))
    edge = mean_acc - mean_base

    print(f"\n  Walk-forward results ({len(xgb_scores)} eras):")
    print(f"    Mean accuracy : {mean_acc:.2%}")
    print(f"    Mean baseline : {mean_base:.2%}")
    print(f"    Edge          : {edge:+.2%}")

    if last_fold:
        X_te, y_te, m = last_fold
        print(f"\n  Last era classification report:")
        print(classification_report(y_te, m.predict(X_te),
              target_names=le.classes_, zero_division=0))

    # When model predicts "up" with high confidence, how often correct?
    if last_fold:
        X_te, y_te, m = last_fold
        probas = m.predict_proba(X_te)
        up_idx = list(le.classes_).index("up") if "up" in le.classes_ else 0
        p_ups = probas[:, up_idx]
        actuals = le.inverse_transform(y_te)
        for thresh in [0.55, 0.60, 0.65]:
            mask = p_ups > thresh
            if mask.sum() > 0:
                correct = sum(1 for a in actuals[mask] if a == "up")
                acc = correct / mask.sum()
                print(f"  P(up)>{thresh:.2f}: {acc:.1%} correct "
                      f"({mask.sum()}/{len(p_ups)} = {mask.sum()/len(p_ups)*100:.0f}% participation)")

    # Train final model on all data
    print(f"\n  Training final model on all {len(X)} rows...")
    sw_full = compute_sample_weight("balanced", y)
    final_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="mlogloss", random_state=42, verbosity=0,
    )
    final_model.fit(X, y, sample_weight=sw_full)

    importances = pd.Series(
        final_model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print(f"\n  Top 10 features:")
    for feat, imp in importances.head(10).items():
        bar = "#" * int(imp * 80)
        print(f"    {feat:<22} {bar} {imp:.3f}")

    # Save
    xgb_path = os.path.join(MODEL_DIR, f"{symbol}_direction_xgb.pkl")
    le_path = os.path.join(MODEL_DIR, f"{symbol}_direction_le.pkl")
    fc_path = os.path.join(MODEL_DIR, f"{symbol}_direction_features.json")
    meta_path = os.path.join(MODEL_DIR, f"{symbol}_direction_meta.json")

    with open(xgb_path, "wb") as f:
        pickle.dump(final_model, f)
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    with open(fc_path, "w") as f:
        json.dump(feature_cols, f)
    with open(meta_path, "w") as f:
        json.dump({
            "symbol": symbol,
            "model_type": "index_direction",
            "accuracy": round(mean_acc, 4),
            "baseline": round(mean_base, 4),
            "edge": round(edge, 4),
            "n_features": len(feature_cols),
            "n_rows": len(data),
            "data_range": f"{dates[0].date()} -> {dates[-1].date()}",
            "trained": datetime.now().isoformat(),
            "era_results": era_results,
            "top_features": importances.head(10).index.tolist(),
            "thresholds": {"up_pct": UP_THRESH, "horizon_days": FWD_HORIZON},
        }, f, indent=2)

    print(f"\n  [OK] Model saved -> {xgb_path}")
    return {
        "symbol": symbol, "accuracy": round(mean_acc, 4),
        "baseline": round(mean_base, 4), "edge": round(edge, 4),
        "n_features": len(feature_cols),
    }


def predict_direction(symbol):
    """Predict next-day direction for an index using the trained model.
    Returns P(up), P(down), P(flat) and the predicted direction."""
    xgb_path = os.path.join(MODEL_DIR, f"{symbol}_direction_xgb.pkl")
    le_path = os.path.join(MODEL_DIR, f"{symbol}_direction_le.pkl")
    fc_path = os.path.join(MODEL_DIR, f"{symbol}_direction_features.json")

    if not os.path.exists(xgb_path):
        return None

    with open(xgb_path, "rb") as f:
        model = pickle.load(f)
    with open(le_path, "rb") as f:
        le = pickle.load(f)
    with open(fc_path) as f:
        feature_cols = json.load(f)

    feat_path = os.path.join(FEATURE_DIR, f"{symbol}_features.csv")
    if not os.path.exists(feat_path):
        return None

    df = pd.read_csv(feat_path, index_col="Date", parse_dates=True)
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    str_cols = [c for c in X.columns if X[c].dtype == object]
    if str_cols:
        X.drop(columns=str_cols, inplace=True)
    X = X.astype(float).dropna()
    if X.empty:
        return None

    row = X.iloc[[-1]]
    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0]

    classes = le.classes_
    prob_dict = {c: round(float(p), 4) for c, p in zip(classes, proba)}

    direction = le.inverse_transform([pred])[0]
    confidence = round(float(max(proba)), 4)

    # Binary model: P(up) is directly the "up" class probability
    prob_up = prob_dict.get("up", 0.5)

    meta_path = os.path.join(MODEL_DIR, f"{symbol}_direction_meta.json")
    edge = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                edge = json.load(f).get("edge")
        except Exception:
            pass

    return {
        "symbol": symbol,
        "direction": direction,
        "confidence": confidence,
        "prob_up": round(prob_up, 4),
        "probabilities": prob_dict,
        "edge": edge,
        "date": str(row.index[0].date()),
    }


if __name__ == "__main__":
    results = {}
    for sym in INDICES:
        r = train_index(sym)
        if r:
            results[sym] = r

    if results:
        print(f"\n{'='*60}")
        print(f"  SUMMARY")
        print(f"{'='*60}")
        for sym, r in results.items():
            print(f"  {sym}: acc={r['accuracy']:.2%} base={r['baseline']:.2%} "
                  f"edge={r['edge']:+.2%} features={r['n_features']}")

        # Quick test inference
        print(f"\n  Testing inference...")
        for sym in results:
            pred = predict_direction(sym)
            if pred:
                print(f"  {sym}: {pred['direction']} "
                      f"(P(up)={pred['prob_up']:.3f}, conf={pred['confidence']:.3f}, "
                      f"edge={pred['edge']})")

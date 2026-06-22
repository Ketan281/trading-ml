"""
Index options strategy model -- NIFTY / BANKNIFTY (daily -> options structure).

Trained on 20 years of REAL market data (2006-2026) with intermarket features:
crude oil, DXY, CBOE VIX, India VIX, S&P 500, US 10Y yield, gold, USD/INR,
FII/DII flows. Era-based walk-forward validation across 4 market regimes.

Two prediction heads:
    1. DIRECTION : up / down / flat over weekly horizon
    2. VOLATILITY: will realised vol EXPAND or CONTRACT?

Maps the joint call to the textbook options play:
    dir x vol -> spread / straddle / condor / long option

HONEST LIMITS:
- Weekly horizon (~5 trading days), NOT intraday
- Option P&L not backtested (no historical option prices)
- Direction is barely above baseline for indices; vol view has real edge
"""

import os
import sys
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "data")
HIST_DIR   = os.path.join(ROOT, "data", "historical")
MODELS_DIR = os.path.dirname(__file__)
OUT_DIR    = os.path.join(ROOT, "outputs", "options_model")
os.makedirs(OUT_DIR, exist_ok=True)

HORIZON   = 5        # trading days = one weekly expiry
N_SPLITS  = 6
TRADING_DAYS = 252
INTEL_DIR  = os.path.join(ROOT, "data", "market_intel")

DIR_CONF_MIN = 0.50

ERA_WINDOWS = [
    {"name": "GFC_era",    "train_start": 2006, "train_end": 2010, "test_year": 2011},
    {"name": "bull_run",   "train_start": 2012, "train_end": 2015, "test_year": 2016},
    {"name": "pre_covid",  "train_start": 2016, "train_end": 2019, "test_year": 2020},
    {"name": "post_covid", "train_start": 2020, "train_end": 2025, "test_year": 2026},
]

# Base features (from price/volume) -- intermarket features added dynamically
BASE_FEATURES = [
    "mom_5", "mom_10", "mom_20", "mom_60",
    "rsi_14", "macd_hist", "ema_gap_20", "ema_gap_50",
    "dist_high_20", "dist_high_50",
    "rv_5", "rv_10", "rv_20", "vol_ratio", "atr_pct", "bb_width",
    "range_pct", "gap", "skew_20", "streak", "dow",
    "mfi_14", "ad_momentum", "sr_proximity", "vol_rsi",
    "vol_price_confirm", "sharpe_60d",
]

_intermarket_cache = None

def _load_intermarket():
    global _intermarket_cache
    if _intermarket_cache is not None:
        return _intermarket_cache
    path = os.path.join(INTEL_DIR, "intermarket_features.csv")
    if os.path.exists(path):
        _intermarket_cache = pd.read_csv(path, index_col="Date", parse_dates=True)
    return _intermarket_cache


# -- Data -------------------------------------------------
def load_index(symbol):
    """Merge the two daily files we keep per index into the longest clean
    daily series (deduped by date)."""
    frames = []
    for path in (os.path.join(HIST_DIR, f"{symbol}.csv"),
                 os.path.join(DATA_DIR, f"{symbol}_daily.csv")):
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.columns = [c.title() for c in df.columns]
            dcol = "Date" if "Date" in df.columns else df.columns[0]
            df[dcol] = pd.to_datetime(df[dcol])
            df = df.rename(columns={dcol: "Date"}).set_index("Date")
            frames.append(df[["Open", "High", "Low", "Close", "Volume"]])
    if not frames:
        return None
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.dropna(subset=["Close"])


# --Features ------------------------------------------
def build_features(df):
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    f = pd.DataFrame(index=df.index)
    r1 = c.pct_change()

    for n in (5, 10, 20, 60):
        f[f"mom_{n}"] = c.pct_change(n)

    d = c.diff()
    g = d.clip(lower=0).rolling(14).mean(); los = (-d.clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - 100 / (1 + g / los)

    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9).mean()) / c

    f["ema_gap_20"] = c / c.ewm(span=20).mean() - 1
    f["ema_gap_50"] = c / c.ewm(span=50).mean() - 1
    f["dist_high_20"] = c / c.rolling(20).max() - 1
    f["dist_high_50"] = c / c.rolling(50).max() - 1

    f["rv_5"]  = r1.rolling(5).std()  * np.sqrt(TRADING_DAYS)
    f["rv_10"] = r1.rolling(10).std() * np.sqrt(TRADING_DAYS)
    f["rv_20"] = r1.rolling(20).std() * np.sqrt(TRADING_DAYS)
    f["vol_ratio"] = f["rv_5"] / (f["rv_20"] + 1e-9)

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    f["atr_pct"] = tr.rolling(14).mean() / c
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_width"] = (4 * sd20) / ma20
    f["range_pct"] = (h - l) / c
    f["gap"] = df["Open"] / c.shift() - 1
    f["skew_20"] = r1.rolling(20).skew()

    sign = np.sign(r1.fillna(0))
    streak = sign.groupby((sign != sign.shift()).cumsum()).cumcount() + 1
    f["streak"] = (streak * sign).clip(-10, 10)
    f["dow"] = df.index.dayofweek

    # MFI -- real money flow from price + volume
    typical = (h + l + c) / 3
    mf = typical * v
    pos_mf = pd.Series(np.where(typical > typical.shift(1), mf, 0), index=df.index)
    neg_mf = pd.Series(np.where(typical < typical.shift(1), mf, 0), index=df.index)
    mfr = pos_mf.rolling(14).sum() / neg_mf.rolling(14).sum().replace(0, np.nan)
    f["mfi_14"] = 100 - (100 / (1 + mfr))

    # A/D momentum
    clv = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    ad = (clv.fillna(0) * v).cumsum()
    ad5, ad20 = ad.ewm(span=5).mean(), ad.ewm(span=20).mean()
    f["ad_momentum"] = np.where(ad20 != 0, (ad5 / ad20 - 1) * 100, 0)

    # S/R proximity
    h20, l20 = h.rolling(20).max(), l.rolling(20).min()
    rng20 = h20 - l20
    f["sr_proximity"] = np.where(rng20 > 0, (c - l20) / rng20, 0.5)

    # Volume-weighted RSI
    vg = (d.clip(lower=0) * v).rolling(14).sum()
    vl = (-d.clip(upper=0) * v).rolling(14).sum()
    f["vol_rsi"] = np.where(vl != 0, 100 - (100 / (1 + vg / vl)), 50)

    # Volume-price confirmation
    r5 = c.pct_change(5)
    v5, v20 = v.rolling(5).mean(), v.rolling(20).mean()
    vsurge = v5 / v20
    f["vol_price_confirm"] = np.where(
        (r5 > 0) & (vsurge > 1.5), 1,
        np.where((r5 < 0) & (vsurge > 1.5), -1, 0)
    ).astype(float)

    # Sharpe ratio
    r60 = c.pct_change(60)
    v60 = c.pct_change().rolling(60).std()
    f["sharpe_60d"] = np.where(v60 > 0, r60 / v60, 0)

    # Join REAL intermarket data (crude, DXY, VIX, India VIX, gold, FII)
    im = _load_intermarket()
    if im is not None and len(im) > 0:
        for col in im.columns:
            if col not in f.columns:
                aligned = im[col].reindex(df.index, method="ffill")
                if aligned.notna().sum() > len(df) * 0.3:
                    f[col] = aligned

    return f


def build_labels(df, horizon=HORIZON):
    c = df["Close"]; r1 = c.pct_change()
    fwd = c.shift(-horizon) / c - 1
    # Adaptive flat-band: ~0.4 of the expected H-day move (regime aware).
    rv20 = r1.rolling(20).std()
    band = 0.4 * rv20 * np.sqrt(horizon)
    direction = pd.Series(1, index=c.index)            # 1 = flat
    direction[fwd > band]  = 2                          # 2 = up
    direction[fwd < -band] = 0                          # 0 = down

    # Volatility expansion: realised vol over the NEXT horizon vs the last 5d.
    fwd_rv = r1.shift(-horizon).rolling(horizon).std() * np.sqrt(TRADING_DAYS)
    cur_rv = r1.rolling(5).std() * np.sqrt(TRADING_DAYS)
    vol_expand = (fwd_rv > cur_rv).astype(int)          # 1 = expand, 0 = contract
    return direction, vol_expand, fwd


# --Models --------------------------------------------
def _xgb_clf(n_classes):
    obj = "multi:softprob" if n_classes > 2 else "binary:logistic"
    return xgb.XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.04,
                             subsample=0.8, colsample_bytree=0.8,
                             objective=obj, num_class=n_classes if n_classes > 2 else None,
                             eval_metric="mlogloss" if n_classes > 2 else "logloss",
                             random_state=42, verbosity=0)


def _torch():
    import torch, torch.nn as nn
    return torch, nn


def _train_mlp_multiclass(Xtr, ytr, n_classes, epochs=60):
    torch, nn = _torch(); torch.manual_seed(42)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 48), nn.BatchNorm1d(48), nn.ReLU(),
                        nn.Dropout(0.3), nn.Linear(48, 24), nn.ReLU(),
                        nn.Dropout(0.3), nn.Linear(24, n_classes))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        loss = lossf(net(Xt), yt); loss.backward(); opt.step()
    net.eval()
    return net


def _mlp_predict(net, X):
    torch, _ = _torch()
    with torch.no_grad():
        return net(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy()


# -- Walk-forward evaluation ----------------------------
def _standardize(tr, te):
    mu, sd = tr.mean(0), tr.std(0) + 1e-9
    return (tr - mu) / sd, (te - mu) / sd


def _get_feature_cols(data):
    """Get all valid feature columns (base + intermarket)."""
    cols = [c for c in BASE_FEATURES if c in data.columns]
    im = _load_intermarket()
    if im is not None:
        for c in im.columns:
            if c in data.columns:
                cols.append(c)
    return cols


def evaluate(symbol, try_dl=True):
    df = load_index(symbol)
    if df is None or len(df) < 500:
        print(f"  [WARN] Not enough data for {symbol}."); return None
    feats = build_features(df)
    direction, vol_expand, fwd = build_labels(df)

    data = feats.copy()
    data["dir"] = direction; data["vol"] = vol_expand; data["fwd"] = fwd

    feat_cols = _get_feature_cols(data)
    data = data.dropna(subset=feat_cols + ["dir", "vol", "fwd"])
    dates = data.index
    n = len(data)

    dir_xgb, dir_dl, dir_base, vol_xgb, vol_base = [], [], [], [], []
    era_results = []
    dl_ok = try_dl

    # Era-based walk-forward (primary)
    for era in ERA_WINDOWS:
        train_mask = (dates.year >= era["train_start"]) & (dates.year <= era["train_end"])
        test_mask  = dates.year == era["test_year"]
        tr_idx, te_idx = np.array(train_mask), np.array(test_mask)

        tr, te = data[tr_idx], data[te_idx]
        if len(tr) < 100 or len(te) < 20:
            print(f"     {era['name']}: skipped (train={len(tr)}, test={len(te)})")
            continue

        Xtr, Xte = tr[feat_cols].to_numpy(), te[feat_cols].to_numpy()

        # Direction (3-class)
        mx = _xgb_clf(3); mx.fit(Xtr, tr["dir"])
        px = mx.predict(Xte)
        dacc = float((px == te["dir"].to_numpy()).mean())
        dbase = float(te["dir"].value_counts(normalize=True).max())
        dir_xgb.append(dacc)
        dir_base.append(dbase)

        if dl_ok:
            try:
                Xtr_s, Xte_s = _standardize(Xtr, Xte)
                net = _train_mlp_multiclass(Xtr_s, tr["dir"].to_numpy(), 3)
                pd_ = _mlp_predict(net, Xte_s)
                dir_dl.append(float((pd_ == te["dir"].to_numpy()).mean()))
            except Exception:
                dl_ok = False

        # Volatility (binary)
        mv = _xgb_clf(2); mv.fit(Xtr, tr["vol"])
        pv = mv.predict(Xte)
        vacc = float((pv == te["vol"].to_numpy()).mean())
        vbase = float(te["vol"].value_counts(normalize=True).max())
        vol_xgb.append(vacc)
        vol_base.append(vbase)

        era_results.append({
            "era": era["name"],
            "train": f"{era['train_start']}-{era['train_end']}",
            "test": era["test_year"],
            "dir_acc": round(dacc, 4), "dir_base": round(dbase, 4),
            "vol_acc": round(vacc, 4), "vol_base": round(vbase, 4),
            "train_rows": len(tr), "test_rows": len(te),
        })
        print(f"     {era['name']}: Train {era['train_start']}-{era['train_end']} "
              f"({len(tr)}) -> Test {era['test_year']} ({len(te)}) "
              f"| Dir {dacc:.1%} (base {dbase:.1%}) "
              f"| Vol {vacc:.1%} (base {vbase:.1%})")

    # Fallback to rolling split if era windows insufficient
    if not dir_xgb:
        print("     No era windows had enough data -- falling back to rolling split")
        fold = n // (N_SPLITS + 1)
        for k in range(1, N_SPLITS + 1):
            tr_end = fold * k
            te_s, te_e = tr_end + HORIZON, min(fold * (k + 1), n)
            if te_s >= te_e: continue
            tr, te = data.iloc[:tr_end], data.iloc[te_s:te_e]
            if len(tr) < 200 or len(te) < 40: continue
            Xtr, Xte = tr[feat_cols].to_numpy(), te[feat_cols].to_numpy()
            mx = _xgb_clf(3); mx.fit(Xtr, tr["dir"]); px = mx.predict(Xte)
            dir_xgb.append((px == te["dir"].to_numpy()).mean())
            dir_base.append(te["dir"].value_counts(normalize=True).max())
            mv = _xgb_clf(2); mv.fit(Xtr, tr["vol"]); pv = mv.predict(Xte)
            vol_xgb.append((pv == te["vol"].to_numpy()).mean())
            vol_base.append(te["vol"].value_counts(normalize=True).max())

    res = {
        "symbol": symbol, "rows": int(n),
        "n_features": len(feat_cols),
        "span": [str(df.index.min().date()), str(df.index.max().date())],
        "dir_acc_xgb":  round(float(np.mean(dir_xgb)), 4),
        "dir_acc_dl":   round(float(np.mean(dir_dl)), 4) if dir_dl else None,
        "dir_baseline": round(float(np.mean(dir_base)), 4),
        "vol_acc_xgb":  round(float(np.mean(vol_xgb)), 4),
        "vol_baseline": round(float(np.mean(vol_base)), 4),
        "era_results":  era_results,
    }
    res["dir_winner"] = ("DL" if (dir_dl and res["dir_acc_dl"] > res["dir_acc_xgb"])
                         else "XGB")
    return res


# --Strategy mapping (the 20-yr trader's playbook) ----
STRATEGY = {
    ("up", True):    ("Long Call / Bull Call debit spread",
                      "Directional up + vol expanding -> buy premium, defined risk"),
    ("up", False):   ("Bull Put credit spread (sell puts)",
                      "Up but vol contracting -> collect theta below support"),
    ("down", True):  ("Long Put / Bear Put debit spread",
                      "Directional down + vol expanding -> buy premium for the fall"),
    ("down", False): ("Bear Call credit spread (sell calls)",
                      "Down but vol contracting -> collect theta above resistance"),
    ("flat", True):  ("Long Straddle / Strangle",
                      "No clear direction + vol expanding -> buy volatility"),
    ("flat", False): ("Iron Condor / Short Straddle",
                      "Range-bound + vol contracting -> sell volatility, collect theta"),
}
DIR_NAME = {0: "down", 1: "flat", 2: "up"}


def train_final_and_recommend(symbol):
    """Fit on all history with XGB+RF ensemble, save models, emit current
    weekly view + options structure."""
    df = load_index(symbol)
    feats = build_features(df)
    direction, vol_expand, _ = build_labels(df)
    data = feats.copy(); data["dir"] = direction; data["vol"] = vol_expand

    feat_cols = _get_feature_cols(data)
    train = data.dropna(subset=feat_cols + ["dir", "vol"])
    print(f"     Training final models on {len(train)} rows x {len(feat_cols)} features")

    from sklearn.ensemble import RandomForestClassifier

    # Direction: XGB + RF ensemble
    mdir_xgb = _xgb_clf(3); mdir_xgb.fit(train[feat_cols], train["dir"])
    mdir_rf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=10,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    mdir_rf.fit(train[feat_cols], train["dir"])

    # Volatility: XGB + RF ensemble
    mvol_xgb = _xgb_clf(2); mvol_xgb.fit(train[feat_cols], train["vol"])
    mvol_rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=10,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    mvol_rf.fit(train[feat_cols], train["vol"])

    # Save all models + feature list
    with open(os.path.join(MODELS_DIR, f"opt_{symbol}_dir.pkl"), "wb") as f:
        pickle.dump({"xgb": mdir_xgb, "rf": mdir_rf}, f)
    with open(os.path.join(MODELS_DIR, f"opt_{symbol}_vol.pkl"), "wb") as f:
        pickle.dump({"xgb": mvol_xgb, "rf": mvol_rf}, f)
    with open(os.path.join(MODELS_DIR, f"opt_{symbol}_features.json"), "w") as f:
        json.dump(feat_cols, f, indent=2)

    # Feature importance
    imp = dict(zip(feat_cols, mdir_xgb.feature_importances_))
    top5 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"     Top 5 direction features:")
    for name, score in top5:
        bar = "#" * max(1, int(score * 50))
        print(f"       {name:20s} {bar} {score:.3f}")

    # Current recommendation (ensemble average)
    latest = feats.dropna(subset=feat_cols).iloc[[-1]]
    dprob_xgb = mdir_xgb.predict_proba(latest[feat_cols])[0]
    dprob_rf  = mdir_rf.predict_proba(latest[feat_cols])[0]
    dprob = (dprob_xgb + dprob_rf) / 2
    vprob_xgb = mvol_xgb.predict_proba(latest[feat_cols])[0]
    vprob_rf  = mvol_rf.predict_proba(latest[feat_cols])[0]
    vprob = (vprob_xgb + vprob_rf) / 2

    dcls  = int(np.argmax(dprob)); vexp = int(np.argmax(vprob))
    dname = DIR_NAME[dcls]
    dir_conf = float(dprob[dcls])
    eff_dir = dname if (dname != "flat" and dir_conf >= DIR_CONF_MIN) else "flat"
    strat, why = STRATEGY[(eff_dir, bool(vexp))]
    if eff_dir != dname:
        why += " | direction too weak to trust -> defaulting to the vol view"
    return {
        "symbol": symbol, "as_of": str(df.index.max().date()),
        "spot": round(float(df["Close"].iloc[-1]), 1),
        "n_features": len(feat_cols),
        "direction_raw": dname, "direction_conf": round(dir_conf, 2),
        "direction_used": eff_dir,
        "vol_view": "expand" if vexp else "contract",
        "vol_conf": round(float(vprob[vexp]), 2),
        "strategy": strat, "rationale": why,
    }


def run(symbols=("NIFTY", "BANKNIFTY")):
    print("=" * 70)
    print("  INDEX OPTIONS MODEL -- daily -> weekly options structure")
    print("  [WARN] Weekly horizon (not intraday). Option P&L not backtested")
    print("         (no historical option prices) -- view & structure only.")
    print("  Real intermarket features: crude, DXY, VIX, India VIX, gold,")
    print("  S&P500, US10Y, USD/INR, FII/DII | Era-based walk-forward")
    print("=" * 70)

    report = {}
    for sym in symbols:
        print(f"\n  -- {sym} --")
        ev = evaluate(sym)
        if not ev:
            continue
        print(f"\n  {sym}  ({ev['span'][0]} -> {ev['span'][1]}, "
              f"{ev['rows']} days, {ev['n_features']} features)")
        print(f"    Direction acc : XGB {ev['dir_acc_xgb']:.3f}"
              + (f" | DL {ev['dir_acc_dl']:.3f}" if ev['dir_acc_dl'] else "")
              + f"  (baseline {ev['dir_baseline']:.3f})  -> {ev['dir_winner']}")
        print(f"    Vol-regime acc: XGB {ev['vol_acc_xgb']:.3f}  "
              f"(baseline {ev['vol_baseline']:.3f})")
        rec = train_final_and_recommend(sym)
        used = rec["direction_used"].upper()
        if rec["direction_used"] != rec["direction_raw"]:
            used += f" (raw {rec['direction_raw']}, too weak @ {rec['direction_conf']})"
        print(f"    >> Current view: dir {used} | vol {rec['vol_view'].upper()} "
              f"(conf {rec['vol_conf']})")
        print(f"    >> Play: {rec['strategy']}")
        print(f"       {rec['rationale']}")
        report[sym] = {"eval": ev, "recommendation": rec}

    path = os.path.join(OUT_DIR, "index_options_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  [OK] Saved -> {path}")
    print("  Verdict guide: direction acc must beat baseline to be useful;")
    print("  if approx baseline, the index is efficient -- trust vol view more.")

    # Save meta
    meta = {
        "trained": datetime.now().isoformat(),
        "symbols": list(symbols),
        "validation": "era_based_walk_forward",
        "eras": ERA_WINDOWS,
        "results": {k: {"eval": v["eval"], "rec": v["recommendation"]}
                    for k, v in report.items()},
    }
    meta_path = os.path.join(MODELS_DIR, "opt_training_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  [OK] Meta -> {meta_path}")
    return report


if __name__ == "__main__":
    run()

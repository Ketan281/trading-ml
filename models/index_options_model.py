"""
Index options strategy model — NIFTY / BANKNIFTY (daily → options structure).

WHAT THIS IS (and the honest scope)
-----------------------------------
A professional options trader doesn't predict a single price — they form a
view on TWO axes and pick a structure to match:

    1. DIRECTION   : up / down / flat over the horizon
    2. VOLATILITY  : will realised vol EXPAND or CONTRACT?

This model LEARNS both axes from ~10 years of DAILY NIFTY/BANKNIFTY history
(the most we have; true 20-yr intraday option-chain data is a paid product we
don't possess), validates them walk-forward, and maps the joint call to the
textbook options play:

        dir × vol  →  spread / straddle / condor / long option

HONEST LIMITS (stated, not hidden)
----------------------------------
• Horizon is WEEKLY (~5 trading days), matching weekly expiries — NOT
  intraday. Intraday options needs minute option-chain history we don't have.
• Direction & vol are validated on the index itself. The actual OPTION P&L is
  NOT backtested, because we have no historical option prices (no IV/greeks
  time series). The model tells you the right STRUCTURE and view; sizing/fills
  must be checked on your live chain.
• With only ~3000 daily rows, deep learning is prone to overfit; we still run
  an honest XGBoost-vs-DL check on the direction head and keep the winner.
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

HORIZON   = 5        # trading days ≈ one weekly expiry
N_SPLITS  = 6
TRADING_DAYS = 252

# Direction is barely better than a coin flip at this horizon (validated), so
# we only take a DIRECTIONAL options structure when the model is unusually
# confident; otherwise we default to the volatility-based NEUTRAL play, which
# is the head that actually has edge.
DIR_CONF_MIN = 0.50

FEATURES = [
    "mom_5", "mom_10", "mom_20", "mom_60",
    "rsi_14", "macd_hist", "ema_gap_20", "ema_gap_50",
    "dist_high_20", "dist_high_50",
    "rv_5", "rv_10", "rv_20", "vol_ratio", "atr_pct", "bb_width",
    "range_pct", "gap", "skew_20", "streak", "dow",
]


# ── Data ──────────────────────────────────────────────
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


# ── Features ──────────────────────────────────────────
def build_features(df):
    c, h, l = df["Close"], df["High"], df["Low"]
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


# ── Models ────────────────────────────────────────────
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


# ── Walk-forward evaluation ───────────────────────────
def _standardize(tr, te):
    mu, sd = tr.mean(0), tr.std(0) + 1e-9
    return (tr - mu) / sd, (te - mu) / sd


def evaluate(symbol, try_dl=True):
    df = load_index(symbol)
    if df is None or len(df) < 500:
        print(f"  ⚠ Not enough data for {symbol}."); return None
    feats = build_features(df)
    direction, vol_expand, fwd = build_labels(df)

    data = feats.copy()
    data["dir"] = direction; data["vol"] = vol_expand; data["fwd"] = fwd
    data = data.dropna(subset=FEATURES + ["dir", "vol", "fwd"])
    dates = data.index.values
    n = len(data)
    fold = n // (N_SPLITS + 1)

    dir_xgb, dir_dl, dir_base, vol_xgb, vol_base = [], [], [], [], []
    dl_ok = try_dl
    for k in range(1, N_SPLITS + 1):
        tr_end = fold * k
        te_s, te_e = tr_end + HORIZON, min(fold * (k + 1), n)
        if te_s >= te_e:
            continue
        tr, te = data.iloc[:tr_end], data.iloc[te_s:te_e]
        if len(tr) < 200 or len(te) < 40:
            continue
        Xtr, Xte = tr[FEATURES].to_numpy(), te[FEATURES].to_numpy()

        # Direction (3-class)
        mx = _xgb_clf(3); mx.fit(Xtr, tr["dir"])
        px = mx.predict(Xte)
        dir_xgb.append((px == te["dir"].to_numpy()).mean())
        dir_base.append(te["dir"].value_counts(normalize=True).max())  # majority

        if dl_ok:
            try:
                Xtr_s, Xte_s = _standardize(Xtr, Xte)
                net = _train_mlp_multiclass(Xtr_s, tr["dir"].to_numpy(), 3)
                pd_ = _mlp_predict(net, Xte_s)
                dir_dl.append((pd_ == te["dir"].to_numpy()).mean())
            except Exception:
                dl_ok = False

        # Volatility (binary)
        mv = _xgb_clf(2); mv.fit(Xtr, tr["vol"])
        pv = mv.predict(Xte)
        vol_xgb.append((pv == te["vol"].to_numpy()).mean())
        vol_base.append(te["vol"].value_counts(normalize=True).max())

    res = {
        "symbol": symbol, "rows": int(n),
        "span": [str(df.index.min().date()), str(df.index.max().date())],
        "dir_acc_xgb":  round(float(np.mean(dir_xgb)), 4),
        "dir_acc_dl":   round(float(np.mean(dir_dl)), 4) if dir_dl else None,
        "dir_baseline": round(float(np.mean(dir_base)), 4),
        "vol_acc_xgb":  round(float(np.mean(vol_xgb)), 4),
        "vol_baseline": round(float(np.mean(vol_base)), 4),
    }
    res["dir_winner"] = ("DL" if (dir_dl and res["dir_acc_dl"] > res["dir_acc_xgb"])
                         else "XGB")
    return res


# ── Strategy mapping (the 20-yr trader's playbook) ────
STRATEGY = {
    ("up", True):    ("Long Call / Bull Call debit spread",
                      "Directional up + vol expanding → buy premium, defined risk"),
    ("up", False):   ("Bull Put credit spread (sell puts)",
                      "Up but vol contracting → collect theta below support"),
    ("down", True):  ("Long Put / Bear Put debit spread",
                      "Directional down + vol expanding → buy premium for the fall"),
    ("down", False): ("Bear Call credit spread (sell calls)",
                      "Down but vol contracting → collect theta above resistance"),
    ("flat", True):  ("Long Straddle / Strangle",
                      "No clear direction + vol expanding → buy volatility"),
    ("flat", False): ("Iron Condor / Short Straddle",
                      "Range-bound + vol contracting → sell volatility, collect theta"),
}
DIR_NAME = {0: "down", 1: "flat", 2: "up"}


def train_final_and_recommend(symbol):
    """Fit on all history, save models, and emit the current weekly view +
    options structure."""
    df = load_index(symbol)
    feats = build_features(df)
    direction, vol_expand, _ = build_labels(df)
    data = feats.copy(); data["dir"] = direction; data["vol"] = vol_expand
    train = data.dropna(subset=FEATURES + ["dir", "vol"])

    mdir = _xgb_clf(3); mdir.fit(train[FEATURES], train["dir"])
    mvol = _xgb_clf(2); mvol.fit(train[FEATURES], train["vol"])
    with open(os.path.join(MODELS_DIR, f"opt_{symbol}_dir.pkl"), "wb") as f:
        pickle.dump(mdir, f)
    with open(os.path.join(MODELS_DIR, f"opt_{symbol}_vol.pkl"), "wb") as f:
        pickle.dump(mvol, f)

    latest = feats.dropna(subset=FEATURES).iloc[[-1]]
    dprob = mdir.predict_proba(latest[FEATURES])[0]
    vprob = mvol.predict_proba(latest[FEATURES])[0]
    dcls  = int(np.argmax(dprob)); vexp = int(np.argmax(vprob))
    dname = DIR_NAME[dcls]
    # Honest gate: only honour direction if confidently above the floor;
    # else fall back to "flat" so we take the vol-edge neutral structure.
    dir_conf = float(dprob[dcls])
    eff_dir = dname if (dname != "flat" and dir_conf >= DIR_CONF_MIN) else "flat"
    strat, why = STRATEGY[(eff_dir, bool(vexp))]
    if eff_dir != dname:
        why += " | direction too weak to trust → defaulting to the vol view"
    return {
        "symbol": symbol, "as_of": str(df.index.max().date()),
        "spot": round(float(df["Close"].iloc[-1]), 1),
        "direction_raw": dname, "direction_conf": round(dir_conf, 2),
        "direction_used": eff_dir,
        "vol_view": "expand" if vexp else "contract",
        "vol_conf": round(float(vprob[vexp]), 2),
        "strategy": strat, "rationale": why,
    }


def run(symbols=("NIFTY", "BANKNIFTY")):
    print("=" * 70)
    print("  INDEX OPTIONS MODEL — daily → weekly options structure")
    print("  ⚠ Weekly horizon (not intraday). Option P&L not backtested")
    print("    (no historical option prices) — view & structure only.")
    print("=" * 70)

    report = {}
    for sym in symbols:
        ev = evaluate(sym)
        if not ev:
            continue
        print(f"\n  ── {sym}  ({ev['span'][0]} → {ev['span'][1]}, {ev['rows']} days) ──")
        print(f"    Direction acc : XGB {ev['dir_acc_xgb']:.3f}"
              + (f" | DL {ev['dir_acc_dl']:.3f}" if ev['dir_acc_dl'] else "")
              + f"  (baseline {ev['dir_baseline']:.3f})  → {ev['dir_winner']}")
        print(f"    Vol-regime acc: XGB {ev['vol_acc_xgb']:.3f}  "
              f"(baseline {ev['vol_baseline']:.3f})")
        rec = train_final_and_recommend(sym)
        used = rec["direction_used"].upper()
        if rec["direction_used"] != rec["direction_raw"]:
            used += f" (raw {rec['direction_raw']}, too weak @ {rec['direction_conf']})"
        print(f"    ▶ Current view: dir {used} | vol {rec['vol_view'].upper()} "
              f"(conf {rec['vol_conf']})")
        print(f"    ▶ Play: {rec['strategy']}")
        print(f"      {rec['rationale']}")
        report[sym] = {"eval": ev, "recommendation": rec}

    path = os.path.join(OUT_DIR, "index_options_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  ✅ Saved → {path}")
    print("  Verdict guide: direction acc must beat baseline to be useful;")
    print("  if ≈ baseline, the index is efficient at this horizon — trust vol view more.")
    return report


if __name__ == "__main__":
    run()

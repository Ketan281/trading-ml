"""
Pre-compute cache — make API responses instant.

The engines hit live NSE + models (a few seconds each). For a responsive
frontend we PRE-COMPUTE the common answers on a schedule and serve them from a
disk cache; the API only computes live if the cache is stale or missing.

  • run_precompute()        → refresh all cached views (schedule this, ~5 min)
  • cached_dashboard(sym)   → fresh options dashboard from cache or live
  • cached_book / _screen   → same for the equity views

Scheduled like the data collectors (weekdays, market hours), the dashboards are
always warm and `/query` returns in milliseconds.
"""

import os
import io
import sys
import json
import time
import contextlib
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CACHE_DIR = os.path.join(ROOT, "data", "api_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

SERVE_TTL = 600        # serve cached result if younger than 10 min
INDICES = ["NIFTY", "BANKNIFTY"]


def _silent(fn, *a, **k):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            return fn(*a, **k)
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {str(e)[:160]}"}


def _path(key):
    return os.path.join(CACHE_DIR, f"{key}.json")


def save(key, data):
    json.dump({"ts": time.time(), "at": datetime.now().isoformat(), "data": data},
              open(_path(key), "w"), default=str)


def load_fresh(key, ttl=SERVE_TTL):
    p = _path(key)
    if not os.path.exists(p):
        return None
    try:
        blob = json.load(open(p))
    except Exception:
        return None
    if time.time() - blob.get("ts", 0) > ttl:
        return None
    return blob.get("data")


def cached(key, builder, ttl=SERVE_TTL):
    d = load_fresh(key, ttl)
    if d is not None:
        return d
    d = builder()
    save(key, d)
    return d


# ── Cache-aware accessors the API uses ────────────────
def cached_dashboard(symbol):
    from pipelines.options.options_dashboard import dashboard
    return cached(f"options_{symbol.upper()}",
                  lambda: _silent(dashboard, symbol.upper()))


def cached_book():
    from pipelines.portfolio_book import build_book
    return cached("book", lambda: _silent(build_book))


def cached_screen():
    from pipelines.screener import screen
    return cached("screen", lambda: _silent(screen))


# ── The scheduled refresh ─────────────────────────────
def run_precompute():
    print("=" * 60)
    print(f"  API PRE-COMPUTE  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    from pipelines.options.options_dashboard import dashboard
    for sym in INDICES:
        save(f"options_{sym}", _silent(dashboard, sym))
        print(f"  ✓ options_{sym}")
    from pipelines.screener import screen
    save("screen", _silent(screen)); print("  ✓ screen")
    from pipelines.portfolio_book import build_book
    save("book", _silent(build_book)); print("  ✓ book")
    print(f"  Cache → {CACHE_DIR}")


RECO_CACHE_KEY = "recommendations"


def _equity_picks_lean(pool=30, final=15):
    """Rank stocks one-at-a-time to stay within 1GB RAM."""
    import gc
    import pickle
    import numpy as np
    import pandas as pd
    import glob as _glob

    model_path = os.path.join(ROOT, "models", "cross_sectional_xgb.pkl")
    if not os.path.exists(model_path):
        return []

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    hist_dir = os.path.join(ROOT, "data", "historical")
    src = hist_dir if os.path.isdir(hist_dir) and _glob.glob(
        os.path.join(hist_dir, "*.csv")) else os.path.join(ROOT, "data")

    EXCLUDE = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"}
    FEATURES = ["mom_21", "mom_63", "mom_126", "mom_252",
                "rev_5", "vol_21", "rsi_14", "dist_high", "ma_ratio"]

    rows = []
    for path in _glob.glob(os.path.join(src, "*.csv")):
        name = os.path.basename(path).replace(".csv", "").replace("_daily", "")
        if name.lower() in ("manifest",) or name in EXCLUDE:
            continue
        try:
            df = pd.read_csv(path, index_col="Date", parse_dates=True,
                             usecols=["Date", "Open", "High", "Low", "Close", "Volume"])
            if len(df) < 260:
                del df; continue

            close = df["Close"]
            feat = {}
            feat["mom_21"] = float(close.pct_change(21).iloc[-1])
            feat["mom_63"] = float(close.pct_change(63).iloc[-1])
            feat["mom_126"] = float(close.pct_change(126).iloc[-1])
            feat["mom_252"] = float(close.pct_change(252).iloc[-1])
            feat["rev_5"] = float(close.pct_change(5).iloc[-1])
            feat["vol_21"] = float(close.pct_change().rolling(21).std().iloc[-1])
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
            loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
            feat["rsi_14"] = float(100 - 100 / (1 + gain / (loss + 1e-9)))
            feat["dist_high"] = float(close.iloc[-1] / close.rolling(252).max().iloc[-1] - 1)
            ma200 = close.rolling(200).mean().iloc[-1]
            feat["ma_ratio"] = float(close.iloc[-1] / ma200) if ma200 > 0 else 1.0

            if any(np.isnan(v) for v in feat.values()):
                del df; continue

            turnover = float((df["Close"].tail(60) * df["Volume"].tail(60)).median())
            price = float(close.iloc[-1])
            atr = float((df["High"] - df["Low"]).tail(14).mean())
            rows.append({"symbol": name, "price": price, "turnover": turnover,
                         "atr": atr, **feat})
            del df
        except Exception:
            pass
        if len(rows) % 50 == 0:
            gc.collect()

    gc.collect()
    if len(rows) < 10:
        return []

    cs = pd.DataFrame(rows)
    for c in FEATURES:
        cs[c + "_z"] = (cs[c] - cs[c].mean()) / (cs[c].std() + 1e-9)
    cs["score"] = model.predict_proba(cs[[c + "_z" for c in FEATURES]])[:, 1]
    cs = cs.sort_values("score", ascending=False).reset_index(drop=True)
    del model; gc.collect()

    picks = []
    for _, row in cs.head(pool).iterrows():
        if row["turnover"] < 3e7:
            continue
        price, atr = row["price"], row["atr"]
        stop = round(price - 1.5 * atr, 2)
        target = round(price + 2.5 * atr, 2)
        risk = price - stop
        rr = round((target - price) / risk, 2) if risk > 0 else 0
        conf = round(float(row["score"]) * 100, 1)
        trend = "bullish" if row["mom_21"] > 0 and row["ma_ratio"] > 1 else \
                "bearish" if row["mom_21"] < 0 and row["ma_ratio"] < 1 else "neutral"
        grade = "A+" if conf >= 85 else "A" if conf >= 70 else "B" if conf >= 55 else "C"
        picks.append({
            "segment": "equity_intraday", "symbol": row["symbol"],
            "action": "BUY", "entry": round(price, 2), "stop": stop, "target": target,
            "confidence": conf, "grade": grade, "reward_risk": rr, "trend": trend,
            "rsi": round(row["rsi_14"], 1),
            "reason": f"{row['symbol']} — score {row['score']:.3f}, R:R {rr}:1, "
                      f"{trend}, RSI {row['rsi_14']:.0f}",
        })
        if len(picks) >= final:
            break

    del cs; gc.collect()
    picks.sort(key=lambda x: x["confidence"], reverse=True)
    return picks


def _options_picks_lean(capital=100_000):
    """Fetch live NIFTY/BANKNIFTY chains and compute best trade."""
    import gc
    picks = []
    for sym in ("BANKNIFTY", "NIFTY"):
        try:
            from pipelines.options_action_engine import live_trade_plan
            plan = live_trade_plan(sym, capital)
            gc.collect()
            if not plan or plan.get("error") or plan.get("note"):
                continue
            conf_map = {"high": 85, "moderate": 60, "none": 0}
            conf = conf_map.get(plan.get("conviction", "none"), 0)
            prob = plan.get("prob_up", 0.5)
            eff_conf = round(conf * 0.6 + abs(prob - 0.5) * 200 * 0.4, 1)
            picks.append({
                "segment": "options",
                "symbol": plan.get("instrument", sym), "underlying": sym,
                "action": plan.get("action", "NO_TRADE"),
                "entry": plan.get("entry_premium", 0),
                "stop": plan.get("stop_premium", 0),
                "target": plan.get("target_premium", 0),
                "confidence": eff_conf, "prob_up": round(prob, 3),
                "conviction": plan.get("conviction", "none"),
                "lots": plan.get("lots", 0), "qty": plan.get("qty", 0),
                "capital_deployed": plan.get("capital_deployed", 0),
                "max_loss": plan.get("max_loss", 0),
                "reward_risk": plan.get("reward_risk", 0),
                "leg": plan.get("action", "").replace("BUY_", "").replace("SMALL_", ""),
                "reason": f"{plan.get('instrument', sym)} — {plan.get('conviction', '')} "
                          f"({eff_conf:.0f}%), P(up) {prob:.1%}, "
                          f"R:R {plan.get('reward_risk', 0):.1f}:1",
            })
        except Exception:
            gc.collect()
    picks.sort(key=lambda x: x["confidence"], reverse=True)
    return picks


def compute_recommendations():
    """Full pipeline: equity + options + swing. Memory-safe for 1GB instances."""
    import gc
    print(f"  [precompute] recommendations starting at {datetime.now()}")
    equity = _silent(_equity_picks_lean) or []
    gc.collect()
    options = _silent(_options_picks_lean) or []
    gc.collect()
    swing = []
    for p in equity[:10]:
        if p.get("confidence", 0) >= 55 and p.get("reward_risk", 0) >= 1.5:
            sp = dict(p)
            sp["segment"] = "swing"
            sp["target"] = round(p["entry"] + (p["entry"] - p["stop"]) * 3, 2)
            sp["reward_risk"] = round((sp["target"] - p["entry"]) / max(p["entry"] - p["stop"], 0.01), 2)
            sp["holding_period"] = "2–10 days"
            swing.append(sp)
    result = {
        "equity_intraday": equity, "options": options, "swing": swing,
        "best_per_segment": {
            "equity_intraday": equity[0] if equity else None,
            "options": options[0] if options else None,
            "swing": swing[0] if swing else None,
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    save(RECO_CACHE_KEY, result)
    print(f"  [precompute] done — {len(equity)} equity, {len(options)} options, {len(swing)} swing")
    gc.collect()
    return result


def get_cached_recommendations():
    """Serve recommendations from cache (instant, zero memory)."""
    data = load_fresh(RECO_CACHE_KEY, ttl=600)
    if data:
        return data
    return {
        "equity_intraday": [], "options": [], "swing": [],
        "best_per_segment": {"equity_intraday": None, "options": None, "swing": None},
        "note": "Recommendations computing — check back in a few minutes",
    }


def _bg_reco_loop():
    """Background: refresh recommendations every 5 min during market hours."""
    import gc
    while True:
        try:
            now = datetime.now()
            is_market = now.weekday() < 5 and 9 <= now.hour < 16
            if is_market or not os.path.exists(_path(RECO_CACHE_KEY)):
                compute_recommendations()
                gc.collect()
            time.sleep(300 if is_market else 1800)
        except Exception as e:
            print(f"  [precompute] bg error: {e}")
            time.sleep(120)


def start_reco_background():
    import threading
    t = threading.Thread(target=_bg_reco_loop, daemon=True, name="reco-precompute")
    t.start()
    print("  [precompute] recommendation background loop started (5min interval)")


if __name__ == "__main__":
    run_precompute()

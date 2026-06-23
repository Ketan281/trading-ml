"""
Production scheduler for 1GB Oracle Cloud micro instance.

Pipeline order (rule #25):
    Data Collection → Breadth → Relative Strength → Sector Rotation
    → Options Flow → Stock Filtering → ML Inference (top tier only)
    → Recommendation Engine → Cache Results

Memory rules:
    - Pre-market (08:30–09:00): run full 434-stock scan, cache to SQLite
    - Market hours: NEVER run all 434 — use cached tier + lightweight updates
    - Process stocks sequentially, gc.collect() between batches
    - Max ~350MB working set (leave headroom for OS + FastAPI)

Tiered universe (rule #16):
    Tier A: Top 50 by composite (breadth + RS + sector + volume) — full ML
    Tier B: Next 100 — lightweight scoring only
    Tier C: Remaining — skip during market hours
"""

import gc
import os
import sys
import json
import time
import sqlite3
import logging
import pickle
import threading
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CACHE_DIR = os.path.join(ROOT, "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
DB_PATH = os.path.join(CACHE_DIR, "scheduler.db")

log = logging.getLogger("scheduler")
log.setLevel(logging.WARNING)

_lock = threading.Lock()

# ── SQLite cache (rule #8, #24) ──────────────────────────

def _db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_cache_db():
    conn = _db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS ml_predictions (
            symbol TEXT, score REAL, rank INTEGER,
            features TEXT, computed_at TEXT,
            PRIMARY KEY (symbol)
        );
        CREATE TABLE IF NOT EXISTS tier_universe (
            symbol TEXT, tier TEXT, composite REAL,
            breadth_score REAL, rs_score REAL, sector_score REAL,
            volume_score REAL, computed_at TEXT,
            PRIMARY KEY (symbol)
        );
        CREATE TABLE IF NOT EXISTS cached_recommendations (
            segment TEXT, data TEXT, computed_at TEXT,
            PRIMARY KEY (segment)
        );
        CREATE TABLE IF NOT EXISTS scheduler_state (
            key TEXT PRIMARY KEY, value TEXT, updated_at TEXT
        );
    """)
    conn.close()


def _cache_get(key):
    # Try SQLite first
    try:
        conn = _db()
        row = conn.execute(
            "SELECT value, updated_at FROM scheduler_state WHERE key=?", (key,)
        ).fetchone()
        conn.close()
        if row:
            return json.loads(row[0])
    except Exception:
        pass
    # Fallback: JSON file
    fp = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(fp):
        try:
            with open(fp) as f:
                data = json.load(f)
            if time.time() - data.get("_ts", 0) < 3600:
                data.pop("_ts", None)
                return data
        except Exception:
            pass
    return None


def _cache_set(key, value):
    try:
        conn = _db()
        conn.execute(
            "INSERT OR REPLACE INTO scheduler_state (key, value, updated_at) VALUES (?,?,?)",
            (key, json.dumps(value, default=str), datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.error("_cache_set(%s) SQLite failed: %s", key, e)
    # Also write JSON fallback
    try:
        fp = os.path.join(CACHE_DIR, f"{key}.json")
        value["_ts"] = time.time()
        with open(fp, "w") as f:
            json.dump(value, f, default=str)
    except Exception:
        pass


# ── Data loading utilities ───────────────────────────────

def _get_data_dir():
    hist = os.path.join(ROOT, "data", "historical")
    import glob
    if os.path.isdir(hist) and glob.glob(os.path.join(hist, "*.csv")):
        return hist
    return os.path.join(ROOT, "data")


EXCLUDE = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"}


def _load_single_stock(path):
    """Load one stock CSV, compute basic features, return dict or None."""
    name = os.path.basename(path).replace(".csv", "").replace("_daily", "")
    if name.lower() in ("manifest",) or name in EXCLUDE:
        return None
    try:
        df = pd.read_csv(path, index_col="Date", parse_dates=True,
                         usecols=["Date", "Open", "High", "Low", "Close", "Volume"])
        if len(df) < 60:
            return None
        df = df.sort_index()
        close = df["Close"]
        vol = df["Volume"]
        turnover = float((close.tail(60) * vol.tail(60)).median())
        price = float(close.iloc[-1])
        atr = float((df["High"] - df["Low"]).tail(14).mean())

        ret_5 = float(close.pct_change(5).iloc[-1]) if len(df) > 5 else 0
        ret_20 = float(close.pct_change(20).iloc[-1]) if len(df) > 20 else 0
        ret_60 = float(close.pct_change(60).iloc[-1]) if len(df) > 60 else 0
        vol_20 = float(close.pct_change().rolling(20).std().iloc[-1]) if len(df) > 20 else 0

        above_20ema = float(close.iloc[-1] > close.ewm(span=20).mean().iloc[-1])
        above_200ma = float(close.iloc[-1] > close.rolling(200).mean().iloc[-1]) if len(df) > 200 else 0.5

        avg_vol = float(vol.tail(20).mean())
        vol_ratio = float(vol.iloc[-1] / avg_vol) if avg_vol > 0 else 1

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean().iloc[-1] if len(df) > 14 else 0
        loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1] if len(df) > 14 else 0
        rsi = float(100 - 100 / (1 + gain / (loss + 1e-9))) if len(df) > 14 else 50

        return {
            "symbol": name, "price": price, "turnover": turnover, "atr": atr,
            "ret_5": ret_5, "ret_20": ret_20, "ret_60": ret_60,
            "vol_20": vol_20, "rsi": rsi,
            "above_20ema": above_20ema, "above_200ma": above_200ma,
            "vol_ratio": vol_ratio, "avg_vol": avg_vol,
            "has_history": len(df) >= 260,
        }
    except Exception:
        return None


# ── Step 1: Market Breadth (rule #23 — run first) ───────

def compute_breadth_score(stocks):
    """Lightweight breadth from pre-loaded stock data."""
    if not stocks:
        return {"score": 50, "advancing": 0, "declining": 0, "regime": "neutral"}
    adv = sum(1 for s in stocks if s["ret_5"] > 0)
    dec = sum(1 for s in stocks if s["ret_5"] < 0)
    total = len(stocks)
    ad_ratio = adv / max(total, 1)
    above_20 = sum(1 for s in stocks if s["above_20ema"]) / max(total, 1)
    above_200 = sum(1 for s in stocks if s["above_200ma"]) / max(total, 1)

    score = round(ad_ratio * 30 + above_20 * 35 + above_200 * 35, 1)

    if score >= 70:
        regime = "strong_bullish"
    elif score >= 55:
        regime = "bullish"
    elif score >= 40:
        regime = "neutral"
    elif score >= 25:
        regime = "bearish"
    else:
        regime = "strong_bearish"

    return {"score": score, "advancing": adv, "declining": dec,
            "total": total, "regime": regime,
            "pct_above_20ema": round(above_20 * 100, 1),
            "pct_above_200ma": round(above_200 * 100, 1)}


# ── Step 2: Relative Strength (rule #3, #15) ────────────

def compute_rs_scores(stocks):
    """Relative strength vs universe median — lightweight version."""
    if not stocks:
        return {}
    median_ret20 = np.median([s["ret_20"] for s in stocks])
    median_ret60 = np.median([s["ret_60"] for s in stocks if s.get("ret_60")])

    rs = {}
    for s in stocks:
        rs_20 = (s["ret_20"] - median_ret20) * 100
        rs_60 = (s.get("ret_60", 0) - median_ret60) * 100
        composite = rs_20 * 0.5 + rs_60 * 0.5
        rs[s["symbol"]] = round(composite, 2)
    return rs


# ── Step 3: Sector Rotation (rule #3) ───────────────────

def compute_sector_scores(stocks):
    """Per-sector composite from constituent performance."""
    ind_path = os.path.join(ROOT, "data", "historical", "industries.json")
    sector_map = {}
    if os.path.exists(ind_path):
        try:
            with open(ind_path) as f:
                sector_map = json.load(f)
        except Exception:
            pass

    sectors = {}
    for s in stocks:
        sec = sector_map.get(s["symbol"], "Other")
        sectors.setdefault(sec, []).append(s)

    scores = {}
    for sec, members in sectors.items():
        avg_ret = np.mean([m["ret_20"] for m in members])
        breadth = sum(1 for m in members if m["ret_5"] > 0) / max(len(members), 1)
        sc = round(avg_ret * 50 + breadth * 50, 2)
        for m in members:
            scores[m["symbol"]] = {"sector": sec, "sector_score": sc}
    return scores


# ── Step 4: Volume filter (rule #15) ────────────────────

def filter_liquid(stocks, min_turnover=3e7):
    """Only keep stocks with sufficient trading volume."""
    return [s for s in stocks if s["turnover"] >= min_turnover]


# ── Step 5: Build tiered universe (rule #16, #17) ───────

def build_tiers(stocks, rs_scores, sector_scores):
    """
    Tier A: Top 50 — full ML inference
    Tier B: Next 100 — lightweight scoring
    Tier C: Rest — skip during market hours
    """
    for s in stocks:
        sym = s["symbol"]
        rs = rs_scores.get(sym, 0)
        sec = sector_scores.get(sym, {}).get("sector_score", 0)
        vol_score = min(s["turnover"] / 1e8, 10) * 10
        composite = rs * 0.35 + sec * 0.25 + vol_score * 0.20 + s["rsi"] * 0.10 + s["ret_20"] * 100 * 0.10
        s["composite"] = round(composite, 2)
        s["rs_score"] = rs
        s["sector_score"] = sec

    stocks.sort(key=lambda x: x["composite"], reverse=True)

    tiers = {"A": [], "B": [], "C": []}
    for i, s in enumerate(stocks):
        if i < 50:
            s["tier"] = "A"
            tiers["A"].append(s)
        elif i < 150:
            s["tier"] = "B"
            tiers["B"].append(s)
        else:
            s["tier"] = "C"
            tiers["C"].append(s)

    return tiers


# ── Step 6: ML inference on Tier A only (rule #1,2,23) ──

def ml_score_tier_a(tier_a_stocks):
    """Load XGBoost model once, score Tier A stocks."""
    model_path = os.path.join(ROOT, "models", "cross_sectional_xgb.pkl")
    if not os.path.exists(model_path):
        log.warning("No cross_sectional_xgb.pkl found — skipping ML scoring")
        return tier_a_stocks

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        log.warning("Failed to load ML model: %s", e)
        return tier_a_stocks

    FEATURES = ["mom_21", "mom_63", "mom_126", "mom_252",
                "rev_5", "vol_21", "rsi_14", "dist_high", "ma_ratio"]

    src = _get_data_dir()
    stock_map = {}
    feat_rows = []
    for s in tier_a_stocks:
        path_candidates = [
            os.path.join(src, f"{s['symbol']}.csv"),
            os.path.join(src, f"{s['symbol']}_daily.csv"),
        ]
        path = next((p for p in path_candidates if os.path.exists(p)), None)
        if not path:
            continue
        try:
            df = pd.read_csv(path, index_col="Date", parse_dates=True,
                             usecols=["Date", "Open", "High", "Low", "Close", "Volume"])
            if len(df) < 260:
                del df; continue
            df = df.sort_index()
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

            sym = s["symbol"]
            feat_rows.append({"symbol": sym, **feat})
            stock_map[sym] = s
            del df
        except Exception:
            pass

    gc.collect()
    log.warning("ML scoring: %d stocks have features (of %d Tier A)", len(feat_rows), len(tier_a_stocks))

    if len(feat_rows) < 5:
        del model; gc.collect()
        return tier_a_stocks

    feat_df = pd.DataFrame(feat_rows)
    for c in FEATURES:
        feat_df[c + "_z"] = (feat_df[c] - feat_df[c].mean()) / (feat_df[c].std() + 1e-9)
    feat_z = [c + "_z" for c in FEATURES]

    try:
        feat_df["ml_score"] = model.predict_proba(feat_df[feat_z])[:, 1]
    except Exception as e:
        log.warning("ML predict failed: %s", e)
        del model; gc.collect()
        return tier_a_stocks

    del model; gc.collect()

    scored = []
    for _, row in feat_df.iterrows():
        sym = row["symbol"]
        s = stock_map.get(sym)
        if not s:
            continue
        s["ml_score"] = round(float(row["ml_score"]), 4)
        scored.append(s)

    scored.sort(key=lambda x: x["ml_score"], reverse=True)
    for i, s in enumerate(scored):
        s["ml_rank"] = i + 1

    try:
        conn = _db()
        now = datetime.now().isoformat()
        for s in scored:
            conn.execute(
                "INSERT OR REPLACE INTO ml_predictions (symbol, score, rank, features, computed_at) "
                "VALUES (?,?,?,?,?)",
                (s["symbol"], s["ml_score"], s["ml_rank"], json.dumps({
                    k: s.get(k) for k in ["ret_5", "ret_20", "ret_60", "rsi", "composite"]
                }), now)
            )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("Failed to cache ML scores: %s", e)

    return scored


# ── Step 7: Options flow (rule #6) ──────────────────────

def compute_options_picks(capital=100_000):
    """Lightweight options picks — HTTP calls to NSE only."""
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
                "symbol": plan.get("instrument", sym),
                "underlying": sym,
                "action": plan.get("action", "NO_TRADE"),
                "entry": plan.get("entry_premium", 0),
                "stop": plan.get("stop_premium", 0),
                "target": plan.get("target_premium", 0),
                "confidence": eff_conf,
                "prob_up": round(prob, 3),
                "conviction": plan.get("conviction", "none"),
                "lots": plan.get("lots", 0),
                "qty": plan.get("qty", 0),
                "capital_deployed": plan.get("capital_deployed", 0),
                "max_loss": plan.get("max_loss", 0),
                "reward_risk": plan.get("reward_risk", 0),
                "leg": plan.get("action", "").replace("BUY_", "").replace("SMALL_", ""),
                "reason": f"{plan.get('instrument', sym)} — {plan.get('conviction', '')} "
                          f"({eff_conf:.0f}%), P(up) {prob:.1%}, "
                          f"R:R {plan.get('reward_risk', 0):.1f}:1",
            })
        except Exception as e:
            log.warning("Options %s: %s", sym, e)
            gc.collect()
    return picks


# ── Step 8: Build recommendations (rule #24, #25) ───────

def build_equity_recommendations(scored_stocks, breadth):
    """Convert scored Tier A stocks into recommendation format."""
    log.warning("Building equity recos from %d scored stocks", len(scored_stocks))
    picks = []
    for s in scored_stocks[:20]:
        try:
            ml = s.get("ml_score", 0.5)
            price = s.get("price", 0)
            atr = s.get("atr", 0)
            if price <= 0 or atr <= 0:
                continue

            stop = round(price - 1.5 * atr, 2)
            target = round(price + 2.5 * atr, 2)
            risk = price - stop
            rr = round((target - price) / risk, 2) if risk > 0 else 0
            conf = round(ml * 100, 1)

            trend = "bullish" if s.get("ret_20", 0) > 0 and s.get("above_200ma", 0) else \
                    "bearish" if s.get("ret_20", 0) < 0 else "neutral"
            grade = "A+" if conf >= 85 else "A" if conf >= 70 else "B" if conf >= 55 else "C"

            picks.append({
                "segment": "equity_intraday",
                "symbol": s["symbol"],
                "action": "BUY",
                "entry": round(price, 2),
                "stop": stop,
                "target": target,
                "confidence": conf,
                "grade": grade,
                "reward_risk": rr,
                "trend": trend,
                "rsi": round(s.get("rsi", 50), 1),
                "tier": s.get("tier", "A"),
                "ml_rank": s.get("ml_rank", 0),
                "rs_score": round(s.get("rs_score", 0), 1),
                "sector_score": round(s.get("sector_score", 0), 1),
                "reason": f"{s['symbol']} — ML rank #{s.get('ml_rank', '?')}, "
                          f"score {ml:.3f}, R:R {rr}:1, {trend}, RSI {s.get('rsi', 50):.0f}",
            })
        except Exception as e:
            log.warning("Failed to build reco for %s: %s", s.get("symbol", "?"), e)
    picks.sort(key=lambda x: x["confidence"], reverse=True)
    log.warning("Built %d equity recommendations", len(picks))
    return picks


def build_swing_recommendations(equity_picks):
    """Derive swing from top equity — wider targets for multi-day holds."""
    swings = []
    for p in equity_picks[:10]:
        if p["confidence"] < 55 or p["reward_risk"] < 1.5:
            continue
        sp = dict(p)
        sp["segment"] = "swing"
        risk = p["entry"] - p["stop"]
        sp["target"] = round(p["entry"] + risk * 3, 2)
        sp["reward_risk"] = round((sp["target"] - p["entry"]) / max(risk, 0.01), 2)
        sp["holding_period"] = "2–10 days"
        swings.append(sp)
    return swings


# ── Full pipeline (rule #25) ────────────────────────────

def run_full_pipeline(is_premarket=False):
    """
    The orchestrated pipeline:
    1. Data Collection (load stock data sequentially)
    2. Breadth
    3. Relative Strength
    4. Sector Rotation
    5. Volume Filter
    6. Tier Universe
    7. ML Inference (Tier A only)
    8. Options Flow
    9. Build Recommendations
    10. Cache Results
    """
    with _lock:
        t0 = time.time()
        log.warning("Pipeline starting (premarket=%s)...", is_premarket)

        # Step 1: Load all stock data sequentially (rule #11)
        src = _get_data_dir()
        import glob as _glob
        files = _glob.glob(os.path.join(src, "*.csv"))
        if is_premarket:
            pass  # process all
        else:
            files.sort(key=lambda p: os.path.getsize(p), reverse=True)
            files = files[:200]

        all_stocks = []
        for i, path in enumerate(files):
            s = _load_single_stock(path)
            if s:
                all_stocks.append(s)
            if i % 100 == 0:
                gc.collect()

        gc.collect()
        log.warning("Step 1: Loaded %d stocks in %.1fs", len(all_stocks), time.time() - t0)

        # Step 2: Breadth (rule #23 — run first)
        breadth = compute_breadth_score(all_stocks)
        _cache_set("breadth", breadth)
        log.warning("Step 2: Breadth score %.1f (%s)", breadth["score"], breadth["regime"])

        # Step 3: Relative Strength
        rs_scores = compute_rs_scores(all_stocks)
        log.warning("Step 3: RS computed for %d stocks", len(rs_scores))

        # Step 4: Sector Rotation
        sector_scores = compute_sector_scores(all_stocks)
        log.warning("Step 4: Sector scores for %d stocks", len(sector_scores))

        # Step 5: Volume filter (rule #15)
        liquid = filter_liquid(all_stocks)
        log.warning("Step 5: %d liquid stocks (of %d)", len(liquid), len(all_stocks))

        # Step 6: Tier universe
        tiers = build_tiers(liquid, rs_scores, sector_scores)
        tier_summary = {t: len(v) for t, v in tiers.items()}
        _cache_set("tiers", tier_summary)
        log.warning("Step 6: Tiers — A:%d B:%d C:%d", *[tier_summary.get(t, 0) for t in "ABC"])

        # Save tier data to SQLite
        conn = _db()
        now = datetime.now().isoformat()
        for tier_name, members in tiers.items():
            for s in members:
                conn.execute(
                    "INSERT OR REPLACE INTO tier_universe "
                    "(symbol, tier, composite, breadth_score, rs_score, sector_score, volume_score, computed_at) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (s["symbol"], tier_name, s["composite"],
                     breadth["score"], s.get("rs_score", 0),
                     s.get("sector_score", 0), s["turnover"], now)
                )
        conn.commit()
        conn.close()

        del all_stocks  # free memory before ML
        gc.collect()

        # Step 7: ML inference — Tier A only (rule #1,2)
        try:
            scored = ml_score_tier_a(tiers["A"])
        except Exception as e:
            log.error("Step 7 ML failed: %s", e)
            scored = tiers["A"]
        gc.collect()
        log.warning("Step 7: ML scored %d Tier A stocks", len(scored))

        # Step 8: Options flow
        options = []
        try:
            options = compute_options_picks()
        except Exception as e:
            log.error("Step 8 Options failed: %s", e)
        gc.collect()
        log.warning("Step 8: %d options picks", len(options))

        # Step 9: Build recommendations
        equity = []
        swing = []
        try:
            equity = build_equity_recommendations(scored, breadth)
            swing = build_swing_recommendations(equity)
        except Exception as e:
            log.error("Step 9 Recommendations failed: %s", e)
        log.warning("Step 9: %d equity, %d swing recommendations", len(equity), len(swing))

        result = {
            "equity_intraday": equity,
            "options": options,
            "swing": swing,
            "market_context": {
                "breadth": breadth,
                "tiers": tier_summary,
            },
            "best_per_segment": {
                "equity_intraday": equity[0] if equity else None,
                "options": options[0] if options else None,
                "swing": swing[0] if swing else None,
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pipeline_time": round(time.time() - t0, 1),
        }

        # Step 10: Cache to SQLite (rule #24)
        try:
            conn = _db()
            now = datetime.now().isoformat()
            for seg in ("equity_intraday", "options", "swing"):
                conn.execute(
                    "INSERT OR REPLACE INTO cached_recommendations (segment, data, computed_at) "
                    "VALUES (?,?,?)",
                    (seg, json.dumps(result[seg], default=str), now)
                )
            conn.commit()
            conn.close()
        except Exception as e:
            log.error("Step 10 SQLite cache failed: %s", e)

        _cache_set("last_recommendations", result)

        elapsed = time.time() - t0
        log.warning("Pipeline COMPLETE in %.1fs — %d equity, %d options, %d swing",
                     elapsed, len(equity), len(options), len(swing))
        gc.collect()
        return result


def get_cached_recommendations():
    """Serve from SQLite cache — zero computation, instant response (rule #24).
    Falls back to old screener output if scheduler hasn't populated yet."""
    cached = _cache_get("last_recommendations")
    if cached:
        return cached

    # Fallback: read from old screener output + options engine
    fallback = _fallback_from_screener()
    if fallback:
        return fallback

    return {
        "equity_intraday": [], "options": [], "swing": [],
        "best_per_segment": {"equity_intraday": None, "options": None, "swing": None},
        "note": "Recommendations computing — check back in a few minutes",
    }


def _fallback_from_screener():
    """Convert old screener_report.json to recommendation format."""
    report_path = os.path.join(ROOT, "outputs", "screener_report.json")
    if not os.path.exists(report_path):
        return None
    try:
        age = time.time() - os.path.getmtime(report_path)
        if age > 3600:
            return None

        with open(report_path) as f:
            report = json.load(f)

        equity = []
        for c in (report.get("actionable", []) + report.get("watchlist", []))[:15]:
            entry = c.get("price", 0)
            stop = c.get("stop_loss", 0)
            target = c.get("target", 0)
            try:
                entry = float(str(entry).replace(",", "").split("-")[-1])
                stop = float(str(stop).replace(",", ""))
                target = float(str(target).replace(",", "")) if target else entry * 1.03
            except (ValueError, TypeError):
                continue
            if not entry or stop >= entry:
                continue
            risk = entry - stop
            rr = round((target - entry) / risk, 2) if risk > 0 else 0
            conf = c.get("conviction", 50)
            equity.append({
                "segment": "equity_intraday",
                "symbol": c.get("symbol", "?"),
                "action": "BUY",
                "entry": round(entry, 2),
                "stop": round(stop, 2),
                "target": round(target, 2),
                "confidence": round(conf, 1),
                "grade": c.get("grade", "B"),
                "reward_risk": rr,
                "trend": c.get("trend", "neutral"),
                "rsi": round(c.get("rsi", 50), 1),
                "reason": f"{c.get('symbol')} — grade {c.get('grade', '?')}, "
                          f"conviction {conf:.0f}%, R:R {rr}:1",
            })

        # Try to get options from the old api_cache
        options = []
        cache_path = os.path.join(ROOT, "data", "api_cache", "options_NIFTY.json")
        for sym in ("BANKNIFTY", "NIFTY"):
            cp = os.path.join(ROOT, "data", "api_cache", f"options_{sym}.json")
            if os.path.exists(cp):
                try:
                    with open(cp) as f:
                        blob = json.load(f)
                    if time.time() - blob.get("ts", 0) < 3600:
                        data = blob.get("data", {})
                        if isinstance(data, dict) and data.get("action"):
                            prob = data.get("prob_up", 0.5)
                            conf_map = {"high": 85, "moderate": 60, "none": 0}
                            conf = conf_map.get(data.get("conviction", "none"), 50)
                            eff_conf = round(conf * 0.6 + abs(prob - 0.5) * 200 * 0.4, 1)
                            options.append({
                                "segment": "options",
                                "symbol": data.get("instrument", sym),
                                "underlying": sym,
                                "action": data.get("action", "NO_TRADE"),
                                "entry": data.get("entry_premium", 0),
                                "stop": data.get("stop_premium", 0),
                                "target": data.get("target_premium", 0),
                                "confidence": eff_conf,
                                "reward_risk": data.get("reward_risk", 0),
                                "prob_up": round(prob, 3),
                                "conviction": data.get("conviction", "none"),
                                "lots": data.get("lots", 0),
                                "qty": data.get("qty", 0),
                                "reason": f"{data.get('instrument', sym)} — {data.get('conviction', '')} "
                                          f"({eff_conf:.0f}%), P(up) {prob:.1%}",
                            })
                except Exception:
                    pass

        if not equity and not options:
            return None

        return {
            "equity_intraday": equity,
            "options": options,
            "swing": [],
            "best_per_segment": {
                "equity_intraday": equity[0] if equity else None,
                "options": options[0] if options else None,
                "swing": None,
            },
            "timestamp": report.get("date", ""),
            "source": "screener_fallback",
        }
    except Exception:
        return None


def get_tier_universe():
    """Return current tiered universe from SQLite."""
    try:
        conn = _db()
        rows = conn.execute(
            "SELECT symbol, tier, composite, rs_score, sector_score FROM tier_universe "
            "ORDER BY composite DESC"
        ).fetchall()
        conn.close()
        return [{"symbol": r[0], "tier": r[1], "composite": r[2],
                 "rs_score": r[3], "sector_score": r[4]} for r in rows]
    except Exception:
        return []


# ── Background scheduler (rule #13, #14, #25) ───────────

def _should_run_premarket():
    """08:30–09:00 IST on weekdays."""
    now = datetime.now()
    return now.weekday() < 5 and now.hour == 8 and 30 <= now.minute < 60


def _should_run_market():
    """09:15–15:30 IST on weekdays — lighter updates every 10min."""
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    t = now.hour * 100 + now.minute
    return 915 <= t <= 1530


def _should_run_postmarket():
    """15:30–16:30 IST — one final scan."""
    now = datetime.now()
    return now.weekday() < 5 and now.hour == 15 and now.minute >= 30 or \
           now.weekday() < 5 and now.hour == 16 and now.minute < 30


_premarket_ran_today = None


def background_loop():
    """
    The production scheduler loop:
    - 08:30–09:00: Full 434-stock pre-market scan (rule #1)
    - 09:15–15:30: Lightweight Tier A + options every 10 min (rule #2,6)
    - 15:30–16:30: Final post-market scan
    - Overnight: sleep (rule #13)
    """
    global _premarket_ran_today
    time.sleep(30)  # let server stabilize

    # Initial run on startup
    try:
        run_full_pipeline(is_premarket=False)
    except Exception as e:
        log.error("Initial pipeline: %s", e)

    while True:
        try:
            now = datetime.now()
            today = now.strftime("%Y-%m-%d")

            if _should_run_premarket() and _premarket_ran_today != today:
                log.warning("Running PRE-MARKET full scan...")
                run_full_pipeline(is_premarket=True)
                _premarket_ran_today = today
                time.sleep(300)
            elif _should_run_market():
                run_full_pipeline(is_premarket=False)
                time.sleep(600)
            elif _should_run_postmarket():
                run_full_pipeline(is_premarket=False)
                time.sleep(1800)
            else:
                time.sleep(900)
        except Exception as e:
            log.error("Scheduler error: %s", e)
            time.sleep(300)


def start_background():
    init_cache_db()
    t = threading.Thread(target=background_loop, daemon=True, name="market-scheduler")
    t.start()
    log.warning("Market scheduler started")

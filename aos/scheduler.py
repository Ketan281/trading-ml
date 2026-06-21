"""
24/7 scheduler — the operational backbone that runs the OS continuously.

A single always-on process that fires every job on its schedule, survives
crashes (idempotent jobs + persisted last-run + state on disk), logs
everything, and catches up on jobs missed during downtime.

JOBS
  premarket    daily 08:30 (wkday)   global+regime+breadth briefing
  deliberate   every 15m, mkt hours  committee considers a NEW trade
  manage       every 2m,  mkt hours  tick open positions → stops/targets/booking
  user_ml_tick every 1m,  weekdays   tick ML-mode user books + auto-open setups
  precompute   every 5m,  mkt hours  warm API caches
  monitor      every 60m             health dashboard; logs WARN/ALERT
  postmarket   daily 16:00 (wkday)   self-review → lessons
  metalearn    daily 18:00           re-learn the signal×regime policy
  retrain      weekly Sat 07:00      walk-forward retrain (promote-if-better)

Recovery: last_run.json tracks success; on start, any daily job whose time has
already passed today (and didn't run) is run once (catch-up). Every job is
wrapped so one failure logs and never kills the loop. The TradeManager and
Trade Memory persist to disk, so a restart resumes mid-session.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, time as dtime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

LOG_DIR = os.path.join(ROOT, "logs", "aos")
STATE_DIR = os.path.join(ROOT, "data", "aos")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)
LAST_RUN = os.path.join(STATE_DIR, "last_run.json")

MARKET_OPEN, MARKET_CLOSE = dtime(9, 15), dtime(15, 30)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOG_DIR, "aos.log"), encoding="utf-8"),
              logging.StreamHandler()])
log = logging.getLogger("aos")


# ── shared, persistent TradeManager (one per process) ─
_TM = None
def _tm():
    global _TM
    if _TM is None:
        from agents.manager import TradeManager
        _TM = TradeManager(starting_cash=300_000)     # loads state.json
    return _TM


# ── job implementations (all idempotent, failure-safe) ─
def job_premarket():
    from aos.premarket import run_premarket
    b, _ = run_premarket(); return {"regime": b["regime"]["label"]}

def job_deliberate():
    from aos.orchestrator import Orchestrator
    d = Orchestrator(trade_manager=_tm()).deliberate({"index": "BANKNIFTY"})
    return {"symbol": d["symbol"], "final": d["final_action"], "executed": d["executed"]}

def job_manage():
    from agents.auto_trader import live_prices
    tm = _tm(); pm = live_prices(tm)
    fills = tm.on_prices(pm) if pm else []
    return {"ticked": len(pm), "fills": len(fills)}

def job_monitor():
    from aos.monitoring import monitor
    h = monitor(run_dq=False)
    if h["overall"] in ("WARN", "ALERT"):
        log.warning("HEALTH %s — drift=%s perf=%s agents=%s", h["overall"],
                    h["model_drift"]["level"], h["performance"]["level"],
                    h["agent_performance"]["level"])
    return {"overall": h["overall"]}

def job_postmarket():
    from aos.postmarket import review
    r = review(); return {"trades": r.get("n_trades", 0), "lessons": len(r.get("lessons", []))}

def job_metalearn():
    from aos.meta_learning import learn
    p = learn(); return {"status": p["status"], "n": p.get("n_outcomes", 0)}

def job_retrain():
    from infra.retrain_pipeline import run
    r = run(); return {"promoted": r.get("promoted"), "verdict": r.get("verdict")}

def job_precompute():
    from api.precompute import run_precompute
    run_precompute()
    return {"status": "warmed"}

def job_user_ml_tick():
    from aos import user_wallet as uw
    results = uw.ml_tick_all()
    return {"opened": len(results), "events": results[:10]}

def job_wallet_open():
    from aos.sim_wallet import start_daily_trade
    r = start_daily_trade()
    return {"status": r.get("status"), "symbol": (r.get("trade") or {}).get("symbol")}

def job_wallet_tick():
    from aos.sim_wallet import tick
    s = tick(); t = s.get("active_trade") or {}
    return {"trade_status": t.get("status"), "points": len(t.get("pnl_series", []))}


# ── schedule definitions ──────────────────────────────
# kind: 'daily' (at HH:MM), 'interval' (every N min), 'weekly' (weekday@HH:MM)
JOBS = {
    "premarket":  {"fn": job_premarket,  "kind": "daily",    "at": "08:30", "wkday": True},
    "deliberate": {"fn": job_deliberate, "kind": "interval", "every_min": 15, "market": True},
    "manage":     {"fn": job_manage,     "kind": "interval", "every_min": 2,  "market": True},
    "user_ml_tick": {"fn": job_user_ml_tick, "kind": "interval", "every_min": 1, "wkday": True},
    "precompute": {"fn": job_precompute, "kind": "interval", "every_min": 5, "market": True},
    "monitor":    {"fn": job_monitor,    "kind": "interval", "every_min": 60},
    "postmarket": {"fn": job_postmarket, "kind": "daily",    "at": "16:00", "wkday": True},
    "metalearn":  {"fn": job_metalearn,  "kind": "daily",    "at": "18:00"},
    "retrain":    {"fn": job_retrain,    "kind": "weekly",   "weekday": 5, "at": "07:00"},
    # Autonomous paper-trading wallet: open one call after the open, tick it live.
    "wallet_open": {"fn": job_wallet_open, "kind": "daily",    "at": "09:25", "wkday": True},
    "wallet_tick": {"fn": job_wallet_tick, "kind": "interval", "every_min": 1, "market": True},
}


def _load_last():
    if os.path.exists(LAST_RUN):
        try:
            return json.load(open(LAST_RUN))
        except Exception:
            pass
    return {}

def _save_last(d):
    json.dump(d, open(LAST_RUN, "w"), indent=2)


def last_run(name=None):
    data = _load_last()
    return data.get(name) if name else data


def market_open(now=None):
    now = now or datetime.now()
    return now.weekday() < 5 and MARKET_OPEN <= now.time() <= MARKET_CLOSE


def _hhmm(s):
    h, m = s.split(":"); return dtime(int(h), int(m))


def is_due(name, now, last):
    j = JOBS[name]; lr = last.get(name)
    last_dt = datetime.fromisoformat(lr) if lr else None
    if j.get("wkday") and now.weekday() >= 5:
        return False
    if j.get("market") and not market_open(now):
        return False
    if j["kind"] == "interval":
        return last_dt is None or (now - last_dt).total_seconds() >= j["every_min"] * 60
    if j["kind"] == "daily":
        if now.time() < _hhmm(j["at"]):
            return False
        return last_dt is None or last_dt.date() < now.date()    # once per day
    if j["kind"] == "weekly":
        if now.weekday() != j["weekday"] or now.time() < _hhmm(j["at"]):
            return False
        return last_dt is None or last_dt.date() < now.date()
    return False


def run_job(name):
    try:
        t0 = time.time()
        res = JOBS[name]["fn"]()
        log.info("JOB %s ok (%.1fs) → %s", name, time.time() - t0, res)
        last = _load_last(); last[name] = datetime.now().isoformat(); _save_last(last)
        return res
    except Exception as e:
        log.error("JOB %s FAILED: %s", name, e)
        return {"_error": str(e)}


# On a 1 GB micro box, skip the jobs that reload models / train (they OOM).
MICRO = os.getenv("AOS_PROFILE") == "micro"
MICRO_SKIP = {"retrain", "deliberate", "manage"}


def run_forever(tick=30):
    skip = MICRO_SKIP if MICRO else set()
    log.info("AOS scheduler starting (profile=%s). Jobs: %s",
             os.getenv("AOS_PROFILE", "full"), [j for j in JOBS if j not in skip])
    while True:
        now = datetime.now(); last = _load_last()
        for name in JOBS:
            if name in skip:
                continue
            if is_due(name, now, last):
                run_job(name)
        time.sleep(tick)


def status():
    last = _load_last(); now = datetime.now()
    print("=" * 64)
    print(f"  AOS SCHEDULER — {now:%Y-%m-%d %H:%M:%S}  | market_open={market_open(now)}")
    print("=" * 64)
    for name, j in JOBS.items():
        sched = (f"every {j['every_min']}m" if j["kind"] == "interval"
                 else f"{j['kind']} {j.get('at','')}")
        mh = " [mkt]" if j.get("market") else ""
        print(f"  {name:<11} {sched:<14}{mh:<6} last={last.get(name,'never')[:19]:<19} "
              f"due={is_due(name, now, last)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "run-forever":
        run_forever()
    elif cmd == "run-once" and len(sys.argv) > 2:
        print(run_job(sys.argv[2]))
    else:
        status()

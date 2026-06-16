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


if __name__ == "__main__":
    run_precompute()

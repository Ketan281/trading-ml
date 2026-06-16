"""
Feature store — compute features ONCE, reuse everywhere, consistently.

Models, the backtest and the screener all recompute the same factor panel /
breadth / sector features from raw prices. That is slow and, worse, risks the
training pipeline and the live pipeline computing them slightly differently.
A feature store fixes both: a named, versioned, content-hashed cache.

  • save(name, df, source_hash)  → persist a feature set + metadata
  • load(name)                    → get it back (or None)
  • cached(name, builder, hash)   → return cached if fresh & hash matches,
                                     else rebuild via `builder`, store, return

The content hash ties a cached feature set to the data it was built from, so a
stale cache is detected automatically when the underlying prices change.
"""

import os
import json
import time
import pickle
import hashlib
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE_DIR = os.path.join(ROOT, "data", "feature_store")
os.makedirs(STORE_DIR, exist_ok=True)


def _paths(name):
    return (os.path.join(STORE_DIR, f"{name}.pkl"),
            os.path.join(STORE_DIR, f"{name}.meta.json"))


def hash_inputs(*parts):
    """Stable content hash from arbitrary inputs (paths, mtimes, params)."""
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8", "ignore"))
    return h.hexdigest()[:16]


def universe_hash(prices):
    """Hash a {symbol: df} price dict by symbol set + last dates + row counts —
    changes whenever the underlying data is updated."""
    sig = sorted((s, str(df.index[-1].date()) if len(df) else "", len(df))
                 for s, df in prices.items())
    return hash_inputs(sig)


def save(name, df, source_hash=None, extra=None):
    pkl, meta = _paths(name)
    with open(pkl, "wb") as f:
        pickle.dump(df, f)
    info = {"name": name, "rows": int(getattr(df, "shape", [0])[0]),
            "cols": list(getattr(df, "columns", [])),
            "source_hash": source_hash, "saved": datetime.now().isoformat(),
            "saved_ts": time.time()}
    if extra:
        info.update(extra)
    with open(meta, "w") as f:
        json.dump(info, f, indent=2, default=str)
    return info


def load(name):
    pkl, meta = _paths(name)
    if not os.path.exists(pkl):
        return None, None
    with open(pkl, "rb") as f:
        df = pickle.load(f)
    m = json.load(open(meta)) if os.path.exists(meta) else {}
    return df, m


def is_fresh(name, source_hash=None, max_age_hours=None):
    _, m = _paths(name)
    if not os.path.exists(m):
        return False
    info = json.load(open(m))
    if source_hash is not None and info.get("source_hash") != source_hash:
        return False
    if max_age_hours is not None:
        age_h = (time.time() - info.get("saved_ts", 0)) / 3600
        if age_h > max_age_hours:
            return False
    return True


def cached(name, builder, source_hash=None, max_age_hours=24, rebuild=False):
    """Return the cached feature set if fresh & hash-matched; else build it via
    `builder()` (a zero-arg callable), store, and return."""
    if not rebuild and is_fresh(name, source_hash, max_age_hours):
        df, _ = load(name)
        if df is not None:
            print(f"  [feature_store] HIT  {name}")
            return df
    print(f"  [feature_store] BUILD {name}")
    df = builder()
    save(name, df, source_hash)
    return df


def list_features():
    out = []
    for fn in os.listdir(STORE_DIR):
        if fn.endswith(".meta.json"):
            out.append(json.load(open(os.path.join(STORE_DIR, fn))))
    return sorted(out, key=lambda x: x.get("saved", ""), reverse=True)


if __name__ == "__main__":
    feats = list_features()
    print("=" * 60)
    print(f"  FEATURE STORE  ({STORE_DIR})")
    print("=" * 60)
    if not feats:
        print("  (empty — features are cached on first model/backtest run)")
    for f in feats:
        print(f"  {f['name']:<28} rows {f.get('rows'):>8} | "
              f"hash {f.get('source_hash')} | {f.get('saved','')[:19]}")

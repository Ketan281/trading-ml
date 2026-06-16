"""
Model registry + experiment tracking (lightweight — no MLflow needed).

Every time a model is trained it should be versioned with the metrics and
data it was trained on, so you can compare experiments, promote the best, and
ROLL BACK if a fresh model degrades live. This is that, in plain JSON + pickle.

  • register(name, model, metrics, params, data_hash) → new immutable version
  • list_versions(name)                                → all experiments
  • best(name, metric)                                 → highest-metric version
  • get(name, "latest"|"best"|"production"|<v>)        → load a model
  • promote(name, version)                             → mark production (others
                                                          archived) — auditable

Used by the auto-retrain pipeline: a new model is registered, compared to the
production one, and only promoted if it clears the metric gate.
"""

import os
import json
import pickle
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG_DIR = os.path.join(ROOT, "models", "registry")
os.makedirs(REG_DIR, exist_ok=True)
INDEX = os.path.join(REG_DIR, "registry.json")


def _load_index():
    if os.path.exists(INDEX):
        try:
            return json.load(open(INDEX))
        except Exception:
            pass
    return {}


def _save_index(idx):
    with open(INDEX, "w") as f:
        json.dump(idx, f, indent=2, default=str)


def register(name, model, metrics=None, params=None, data_hash=None, tags=None):
    idx = _load_index()
    versions = idx.get(name, [])
    v = (max([e["version"] for e in versions]) + 1) if versions else 1
    path = os.path.join(REG_DIR, f"{name}_v{v}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    entry = {"version": v, "path": path, "metrics": metrics or {},
             "params": params or {}, "data_hash": data_hash, "tags": tags or [],
             "status": "candidate", "registered": datetime.now().isoformat()}
    versions.append(entry)
    idx[name] = versions
    _save_index(idx)
    print(f"  [registry] registered {name} v{v}  metrics={metrics}")
    return entry


def list_versions(name):
    return _load_index().get(name, [])


def best(name, metric="mean_ic"):
    vs = [v for v in list_versions(name) if metric in v.get("metrics", {})]
    return max(vs, key=lambda e: e["metrics"][metric], default=None)


def _find(name, selector):
    vs = list_versions(name)
    if not vs:
        return None
    if selector == "latest":
        return vs[-1]
    if selector == "best":
        return best(name)
    if selector == "production":
        return next((v for v in vs if v["status"] == "production"), None)
    return next((v for v in vs if v["version"] == int(selector)), None)


def get(name, selector="production"):
    """Load a model by selector. Falls back production→best→latest."""
    entry = _find(name, selector) or _find(name, "best") or _find(name, "latest")
    if not entry or not os.path.exists(entry["path"]):
        return None, None
    with open(entry["path"], "rb") as f:
        return pickle.load(f), entry


def promote(name, version):
    idx = _load_index()
    for e in idx.get(name, []):
        e["status"] = "production" if e["version"] == version else (
            "archived" if e["status"] == "production" else e["status"])
    _save_index(idx)
    print(f"  [registry] promoted {name} v{version} → production")


def compare(name, metric="mean_ic"):
    print(f"\n  Experiments for '{name}' (by {metric}):")
    print(f"  {'VER':<5}{'STATUS':<12}{metric:>10}  REGISTERED")
    for e in list_versions(name):
        m = e.get("metrics", {}).get(metric)
        print(f"  v{e['version']:<4}{e['status']:<12}"
              f"{(round(m,4) if m is not None else '-'):>10}  {e['registered'][:19]}")


if __name__ == "__main__":
    idx = _load_index()
    print("=" * 60)
    print("  MODEL REGISTRY")
    print("=" * 60)
    if not idx:
        print("  (empty — models register here on retrain)")
    for name in idx:
        compare(name)

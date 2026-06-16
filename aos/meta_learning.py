"""
Meta-Learning Layer — the system gets better from its own track record.

Trains on the Trade-Memory meta-dataset (every signal joined to its realised
outcome) to learn three things and feed them back into selection + sizing:

  1. SIGNAL × REGIME EDGE   which sources actually win in which regimes →
                            a recommended size multiplier per (source, regime)
  2. CONFIDENCE RELIABILITY per regime, is a HIGH confidence score actually
                            more likely to win? If not, confidence is untrusted
                            there and gets discounted.
  3. RECURRING LOSS CAUSES  the most common post-market lesson categories.

The learned policy is saved to data/aos/meta_policy.json; the orchestrator /
portfolio manager call `sizing_multiplier()` to adjust position size and
conviction. HONEST: this is statistics on the system's history, not magic — it
stays NEUTRAL until enough completed trades exist (MIN_TOTAL), and every group
needs MIN_GROUP samples before it influences sizing.
"""

import os
import sys
import json

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from aos import memory as mem

POLICY_PATH = os.path.join(ROOT, "data", "aos", "meta_policy.json")
MIN_TOTAL = 20      # activate meta-learning only with enough completed trades
MIN_GROUP = 8       # a (source,regime) group needs this many to influence sizing


def _neutral(reason, n=0):
    return {"status": "insufficient", "reason": reason, "n_outcomes": n,
            "need": MIN_TOTAL, "signal_regime_edge": {}, "confidence_reliability": {},
            "loss_causes": [], "contract": {
                "signal_regime_edge": "per (source,regime): win_rate, avg_ret → size multiplier",
                "confidence_reliability": "per regime: is high confidence actually better?",
                "loss_causes": "ranked post-market lesson categories"}}


def learn():
    data = mem.meta_dataset()
    if len(data) < MIN_TOTAL:
        pol = _neutral(f"only {len(data)} completed outcomes (<{MIN_TOTAL})", len(data))
        _save(pol)
        return pol
    df = pd.DataFrame(data)
    df["outcome_label"] = df["outcome_label"].fillna(
        (df["outcome_ret"] > 0).astype(int))

    # 1) signal × regime edge
    edge = {}
    g = df.groupby(["source", "regime"])
    for (src, reg), grp in g:
        n = len(grp)
        wr = float(grp["outcome_label"].mean())
        avg = float(grp["outcome_ret"].mean())
        mult = 1.0
        if n >= MIN_GROUP:
            mult = float(np.clip(1 + 2 * (wr - 0.5), 0.3, 1.5))   # edge → size
        edge[f"{src}|{reg}"] = {"n": n, "win_rate": round(wr, 3),
                                "avg_ret": round(avg, 4),
                                "size_multiplier": round(mult, 3),
                                "reliable": n >= MIN_GROUP}

    # 2) confidence reliability per regime
    conf_rel = {}
    for reg, grp in df.dropna(subset=["confidence"]).groupby("regime"):
        if len(grp) < MIN_GROUP:
            conf_rel[reg] = {"n": len(grp), "useful": None, "note": "too few"}
            continue
        med = grp["confidence"].median()
        hi = grp[grp["confidence"] >= med]["outcome_label"].mean()
        lo = grp[grp["confidence"] < med]["outcome_label"].mean()
        useful = bool(hi > lo + 0.05)        # high conf must clearly beat low
        conf_rel[reg] = {"n": len(grp), "hi_winrate": round(float(hi), 3),
                         "lo_winrate": round(float(lo), 3), "useful": useful,
                         "discount": 1.0 if useful else 0.7}

    # 3) recurring loss causes
    causes = mem.query("SELECT category, COUNT(*) AS c FROM lessons "
                       "GROUP BY category ORDER BY c DESC")

    pol = {"status": "active", "n_outcomes": len(df),
           "signal_regime_edge": edge, "confidence_reliability": conf_rel,
           "loss_causes": causes}
    _save(pol)
    return pol


def _save(pol):
    json.dump(pol, open(POLICY_PATH, "w"), indent=2, default=str)


def load_policy():
    if os.path.exists(POLICY_PATH):
        try:
            return json.load(open(POLICY_PATH))
        except Exception:
            pass
    return _neutral("no policy file")


def sizing_multiplier(source, regime, confidence=None, policy=None):
    """The factor the portfolio manager applies to position size, learned from
    history. Neutral (1.0) until enough evidence exists."""
    policy = policy or load_policy()
    if policy.get("status") != "active":
        return 1.0
    m = 1.0
    e = policy["signal_regime_edge"].get(f"{source}|{regime}")
    if e and e.get("reliable"):
        m *= e["size_multiplier"]
    cr = policy["confidence_reliability"].get(regime)
    if cr and cr.get("useful") is False:
        m *= cr.get("discount", 0.7)         # confidence untrusted here → trim
    return round(m, 3)


if __name__ == "__main__":
    pol = learn()
    print("=" * 66)
    print("  META-LEARNING LAYER")
    print("=" * 66)
    if pol["status"] != "active":
        print(f"  Status: INSUFFICIENT — {pol['reason']}")
        print(f"  Needs ≥{MIN_TOTAL} completed trade outcomes to activate.")
        print(f"  Learning contract: {list(pol['contract'].keys())}")
    else:
        print(f"  Active on {pol['n_outcomes']} completed outcomes.\n")
        print("  SIGNAL × REGIME EDGE (size multiplier):")
        for k, v in sorted(pol["signal_regime_edge"].items(),
                           key=lambda x: -x[1]["size_multiplier"]):
            tag = "" if v["reliable"] else "  (thin)"
            print(f"     {k:<22} n={v['n']:<3} win {v['win_rate']:.0%} "
                  f"→ ×{v['size_multiplier']}{tag}")
        print("\n  CONFIDENCE RELIABILITY (per regime):")
        for reg, v in pol["confidence_reliability"].items():
            print(f"     {reg:<10} hi {v.get('hi_winrate')} vs lo {v.get('lo_winrate')} "
                  f"→ useful={v.get('useful')}")
        print(f"\n  TOP LOSS CAUSES: {pol['loss_causes']}")

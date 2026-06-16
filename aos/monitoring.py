"""
Continuous monitoring — one health dashboard for the whole system.

Unifies four watchdogs into a single OK / WARN / ALERT read:

  • MODEL & FEATURE DRIFT  — reads the drift monitor (edge IC retention + PSI)
  • PERFORMANCE DEGRADATION — recent win-rate / net P&L vs the longer record
  • DATA QUALITY            — universe validation (OHLC / NaN / staleness)
  • AGENT PERFORMANCE       — does each agent's VOTE actually correlate with
                              winning trades? Agents that are consistently wrong
                              get a low score and should lose influence.

Agent scoring is the novel piece: it grades each agent on whether its stance
(approve→win, reject/veto→avoided-loss) matched the realised outcome, so the
committee can down-weight chronically wrong members. All from Trade Memory —
nothing fabricated; thin data returns "insufficient", not a fake score.
"""

import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from aos import memory as mem

MIN_TRADES = 10
DRIFT_REPORT = os.path.join(ROOT, "outputs", "monitoring", "drift_report.json")


def _level(*levels):
    order = {"OK": 0, "WARN": 1, "ALERT": 2, "UNKNOWN": 1}
    worst = max(levels, key=lambda x: order.get(x, 1))
    return worst


# ── 1. model + feature drift ──────────────────────────
def model_drift():
    if not os.path.exists(DRIFT_REPORT):
        return {"level": "UNKNOWN", "note": "drift monitor not run"}
    d = json.load(open(DRIFT_REPORT))
    return {"level": d.get("overall", "UNKNOWN"), "ic_retention": d.get("ic_retention"),
            "feature_status": d.get("feature_status"), "live_ic": d.get("live_ic")}


# ── 2. performance degradation ────────────────────────
def performance(window=10):
    tr = mem.query("SELECT net_pnl FROM trades WHERE status='closed' "
                   "AND net_pnl IS NOT NULL ORDER BY id")
    if len(tr) < MIN_TRADES:
        return {"level": "UNKNOWN", "n": len(tr), "note": "insufficient trades"}
    pnls = [t["net_pnl"] for t in tr]
    overall_wr = sum(1 for p in pnls if p > 0) / len(pnls)
    recent = pnls[-window:]
    recent_wr = sum(1 for p in recent if p > 0) / len(recent)
    net_recent = sum(recent)
    level = ("ALERT" if recent_wr < 0.35 or net_recent < 0 and recent_wr < 0.45
             else "WARN" if recent_wr < overall_wr - 0.15
             else "OK")
    return {"level": level, "n": len(pnls), "overall_winrate": round(overall_wr, 2),
            "recent_winrate": round(recent_wr, 2), "recent_net_pnl": round(net_recent),
            "total_net_pnl": round(sum(pnls))}


# ── 3. data quality ───────────────────────────────────
def data_quality(sample=60):
    try:
        from infra.data_quality import validate_universe
        reports, by = validate_universe(limit=sample)
        n = len(reports); fails = len(by["fail"])
        level = ("ALERT" if fails > n * 0.10 else "WARN" if fails > 0 else "OK")
        return {"level": level, "checked": n, "pass": len(by["pass"]),
                "warn": len(by["warn"]), "fail": fails}
    except Exception as e:
        return {"level": "UNKNOWN", "note": str(e)[:100]}


# ── 4. agent performance scoring ──────────────────────
def agent_scores():
    rows = mem.query(
        "SELECT ar.agent, ar.vote, t.net_pnl FROM agent_reports ar "
        "JOIN decisions d ON ar.decision_id = d.id "
        "JOIN trades t ON t.decision_id = d.id "
        "WHERE t.status='closed' AND t.net_pnl IS NOT NULL")
    if len(rows) < MIN_TRADES:
        return {"level": "UNKNOWN", "n": len(rows), "scores": {},
                "note": "insufficient graded votes"}
    agg = {}
    for r in rows:
        win = (r["net_pnl"] or 0) > 0
        v = r["vote"]
        if v == "approve":
            correct = win
        elif v in ("reject", "veto"):
            correct = not win
        else:
            continue                                  # neutral votes not graded
        a = agg.setdefault(r["agent"], {"correct": 0, "n": 0})
        a["correct"] += int(correct); a["n"] += 1
    scores = {a: {"accuracy": round(v["correct"] / v["n"], 2), "n": v["n"]}
              for a, v in agg.items() if v["n"] > 0}
    worst = min((s["accuracy"] for s in scores.values()), default=1.0)
    level = "ALERT" if worst < 0.35 else "WARN" if worst < 0.45 else "OK"
    return {"level": level, "scores": scores}


# ── unified ───────────────────────────────────────────
def monitor(run_dq=True):
    md = model_drift(); pf = performance()
    dq = data_quality() if run_dq else {"level": "UNKNOWN", "note": "skipped"}
    ag = agent_scores()
    overall = _level(md["level"], pf["level"], dq["level"], ag["level"])
    return {"overall": overall, "model_drift": md, "performance": pf,
            "data_quality": dq, "agent_performance": ag}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--no-dq", action="store_true")
    a = p.parse_args()
    h = monitor(run_dq=not a.no_dq)
    print("=" * 64)
    print(f"  CONTINUOUS MONITORING — OVERALL: {h['overall']}")
    print("=" * 64)
    md = h["model_drift"]
    print(f"  Model/feature drift : {md['level']}  "
          f"(IC retention {md.get('ic_retention')}, features {md.get('feature_status')})")
    pf = h["performance"]
    print(f"  Performance         : {pf['level']}  "
          + (f"recent WR {pf.get('recent_winrate')} vs {pf.get('overall_winrate')}, "
             f"net ₹{pf.get('total_net_pnl')}" if pf.get("n") else pf.get("note", "")))
    dq = h["data_quality"]
    print(f"  Data quality        : {dq['level']}  "
          + (f"{dq.get('pass')} pass / {dq.get('warn')} warn / {dq.get('fail')} fail"
             if "pass" in dq else dq.get("note", "")))
    ag = h["agent_performance"]
    print(f"  Agent performance   : {ag['level']}")
    for name, s in ag.get("scores", {}).items():
        print(f"     {name:<16} accuracy {s['accuracy']} (n={s['n']})")
    if not ag.get("scores"):
        print(f"     {ag.get('note','')}")

"""
Orchestration layer — where the agents collaborate, debate, and decide.

Runs the agents in rounds over a shared context, surfaces CONFLICTS (the
debate), lets the Risk Officer VETO, computes a final decision and conviction
from quantitative votes (never the LLM), executes via the paper manager, and
logs the entire deliberation to Trade Memory.

Rounds:
  1. INTELLIGENCE   market regime, sentiment, quant candidate, options view
  2. PROPOSAL       portfolio manager sizes the candidate to wallet + regime
  3. RISK REVIEW    risk officer enforces limits → may VETO
  4. EXECUTION      trade executed only if a candidate exists AND not vetoed

The LLM (failure-safe) only writes a prose summary of the already-decided
outcome — it cannot change the decision or any number.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from aos.base import APPROVE, REJECT, NEUTRAL, VETO
from aos.agents import (
    MarketIntelligenceAgent, EventAwarenessAgent, QuantDecisionAgent,
    OptionsStrategyAgent, PortfolioManagerAgent, RiskOfficerAgent,
    TradeExecutionAgent)
from aos import memory as mem


class Orchestrator:
    def __init__(self, trade_manager=None, enable_llm=False):
        self.tm = trade_manager
        self.enable_llm = enable_llm

    def deliberate(self, ctx=None):
        ctx = ctx or {}
        if self.tm is not None:
            ctx["trade_manager"] = self.tm
            ctx["wallet_cash"] = self.tm.wallet.cash
            ctx["open_risk"] = self._open_risk()
        reports = []

        # Round 1 — intelligence
        for A in (MarketIntelligenceAgent, EventAwarenessAgent,
                  QuantDecisionAgent, OptionsStrategyAgent):
            reports.append(A().run(ctx, self.enable_llm))
        # Round 2 — proposal
        reports.append(PortfolioManagerAgent().run(ctx, self.enable_llm))
        # Round 3 — risk review (may veto)
        ro = RiskOfficerAgent().run(ctx, self.enable_llm); reports.append(ro)
        vetoed = ro.vote == VETO
        ctx["vetoed"] = vetoed
        veto_reason = "; ".join(ro.flags) if vetoed else None

        # Decide final action
        cand = ctx.get("candidate")
        if not cand:
            proposed, final = "NONE", "NO_TRADE"
        elif vetoed:
            proposed, final = "BUY", "NO_TRADE"
        else:
            proposed, final = "BUY", "BUY"

        # Round 4 — execution (only when approved)
        executed = None
        if final == "BUY":
            ex = TradeExecutionAgent().run(ctx, self.enable_llm); reports.append(ex)
            executed = ctx.get("execution")

        conflicts = self._conflicts(reports)
        conviction = self._conviction(ctx, conflicts, vote_of=reports)
        market = ctx.get("market", {})

        decision = {
            "symbol": (cand or {}).get("symbol"),
            "asset": (cand or {}).get("asset"),
            "proposed_action": proposed, "final_action": final,
            "conviction": conviction, "regime": market.get("regime"),
            "vetoed": vetoed, "veto_reason": veto_reason,
            "conflicts": conflicts,
            "votes": {r.agent: r.vote for r in reports},
            "executed": bool(executed and executed.get("action") == "opened"),
            "reports": [r.to_dict() for r in reports],
        }
        decision_id = self._persist(decision, reports, ctx)
        decision["decision_id"] = decision_id
        decision["narration"] = self._narrate(decision) if self.enable_llm else ""
        return decision

    # ── debate: detect disagreement among the agents ────
    def _conflicts(self, reports):
        v = {r.agent: r.vote for r in reports}
        out = []
        if v.get("market_intel") == REJECT and v.get("quant") == APPROVE:
            out.append("Quant wants to BUY into a risk-off regime that Market "
                       "Intelligence rejects — trading against the trend.")
        if v.get("event_awareness") == REJECT and v.get("quant") == APPROVE:
            out.append("Quant signal conflicts with high-impact event risk — "
                       "calendar says reduce exposure.")
        if v.get("risk_officer") == VETO and v.get("quant") == APPROVE:
            out.append("Risk Officer vetoed a setup the Quant agent approved.")
        return out

    def _conviction(self, ctx, conflicts, vote_of):
        v = {r.agent: r.vote for r in vote_of}
        base = float((ctx.get("candidate") or {}).get("conviction") or 50.0)
        if v.get("market_intel") == REJECT:
            base *= 0.80
        if v.get("event_awareness") == REJECT:
            base *= 0.60
        elif v.get("event_awareness") == NEUTRAL:
            base *= 0.85
        ec = ctx.get("event_context", {})
        ps = ec.get("position_size", "full")
        if ps == "avoid":
            base = 0.0
        elif ps == "quarter":
            base *= 0.50
        elif ps == "half":
            base *= 0.75
        if conflicts:
            base *= 0.90
        return round(max(0.0, min(100.0, base)), 1)

    def _open_risk(self):
        risk = 0.0
        for ag in getattr(self.tm, "agents", []):
            p = ag.p
            if p.status == "open":
                risk += abs(p.entry - p.stop) * p.remaining
        return risk

    # ── persistence to Trade Memory ─────────────────────
    def _persist(self, decision, reports, ctx):
        market = ctx.get("market", {})
        ec = ctx.get("event_context", {})
        evidence = {**market}
        if ec:
            evidence["event_risk"] = ec.get("event_risk")
            evidence["is_event_day"] = ec.get("is_event_day")
            evidence["position_size"] = ec.get("position_size")
            evidence["alerts"] = ec.get("alerts", [])
        mem.record_regime(market.get("regime"), market.get("breadth_score"),
                          market.get("vol_pctile"), market.get("top_sectors"),
                          extra={"event_context": ec} if ec else None)
        did = mem.record_decision(
            decision["symbol"], decision["asset"], decision["proposed_action"],
            decision["final_action"], decision["conviction"], decision["regime"],
            decision["vetoed"], decision["veto_reason"], evidence=evidence)
        mem.record_reports(did, decision["reports"])
        cand = ctx.get("candidate")
        if cand:
            mem.record_signal(cand["symbol"], cand.get("asset"), "quant",
                              score=None, confidence=(cand.get("conviction") or 0) / 100,
                              regime=decision["regime"], sentiment=ctx.get("sentiment"),
                              snapshot=cand)
        ex = ctx.get("execution")
        if ex and ex.get("action") == "opened":
            mem.record_trade(did, ex["symbol"], ex["segment"], "long",
                             ex["entry"], ex["qty"], ex["stop"], ex.get("targets"))
        return did

    def _narrate(self, decision):
        try:
            from api.narrate import polish
            facts = (f"Summarise the trade committee outcome in 2 sentences, "
                     f"changing no numbers. Decision={decision['final_action']} on "
                     f"{decision['symbol']}, conviction {decision['conviction']}, "
                     f"regime {decision['regime']}, vetoed={decision['vetoed']} "
                     f"({decision['veto_reason']}), votes={decision['votes']}, "
                     f"conflicts={decision['conflicts']}.")
            text = polish(facts)
            return text if text and text != facts else ""
        except Exception:
            return ""


if __name__ == "__main__":
    from agents.manager import TradeManager
    print("=" * 70)
    print("  ORCHESTRATOR — committee deliberation (paper)")
    print("=" * 70)
    tm = TradeManager(starting_cash=300_000, load=False)
    orch = Orchestrator(trade_manager=tm, enable_llm=False)
    d = orch.deliberate({"index": "BANKNIFTY"})
    print(f"  Symbol      : {d['symbol']} ({d['asset']})")
    print(f"  Votes       : {d['votes']}")
    print(f"  Conflicts   : {d['conflicts'] or 'none'}")
    print(f"  Regime      : {d['regime']} | conviction {d['conviction']}")
    print(f"  Decision    : proposed {d['proposed_action']} → FINAL {d['final_action']}"
          + (f"  (VETO: {d['veto_reason']})" if d['vetoed'] else ""))
    print(f"  Executed    : {d['executed']} | decision_id {d['decision_id']}")
    print(f"  Memory      : {mem.stats()}")

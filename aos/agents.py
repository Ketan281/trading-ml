"""
The nine agents — each wraps a REAL quantitative engine and returns an
AgentReport whose numbers come only from that engine (never the LLM).

  MarketIntelligenceAgent  regime + breadth + sector rotation
  EventAwarenessAgent      calendar events, RBI/Fed/budget risk, results season
  QuantDecisionAgent       the proven equity ranker / screener → a candidate
  OptionsStrategyAgent     options dashboard → directional/structure call
  PortfolioManagerAgent    sizes the candidate within wallet + regime budget
  RiskOfficerAgent         enforces limits — CAN VETO
  TradeExecutionAgent      routes the approved trade to the paper manager
  TradeReviewAgent         post-market review of closed trades → lessons
  ModelImprovementAgent    reads drift health → recommends retrain

Every analyze() is failure-safe (engines make live calls); a failure yields a
NEUTRAL report with a flag, never a crash.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from aos.base import Agent, AgentReport, APPROVE, REJECT, NEUTRAL, VETO


def _safe(fn, default=None):
    try:
        return fn()
    except Exception as e:
        return default if default is not None else {"_error": str(e)[:120]}


# 1 ──────────────────────────────────────────────────────
class MarketIntelligenceAgent(Agent):
    name = "market_intel"; role = "Market Intelligence"

    def analyze(self, ctx):
        from models.regime_classifier import classify
        from pipelines.breadth import breadth_read
        from models.sector_strength import sector_strength
        reg = _safe(lambda: classify("NIFTY"), {"regime": "unknown"})
        br = _safe(breadth_read, {"score": None})
        sec, _ = _safe(lambda: sector_strength(), (None, None)) or (None, None)
        tops = list(sec.head(3)["sector"]) if sec is not None else []
        ev = {"regime": reg.get("regime"), "regime_conf": reg.get("confidence"),
              "breadth_score": br.get("score"), "vol_pctile": reg.get("vol_pctile"),
              "top_sectors": tops}
        ctx["market"] = ev
        regime = ev["regime"]
        vote = (APPROVE if regime in ("bull", "sideways") else
                NEUTRAL if regime == "volatile" else REJECT)  # bear → risk-off
        flags = ["risk_off_regime"] if regime in ("bear", "volatile") else []
        return AgentReport(self.name, self.role, vote=vote,
                           confidence=reg.get("confidence"), evidence=ev, flags=flags)


# 2 ──────────────────────────────────────────────────────
class EventAwarenessAgent(Agent):
    name = "event_awareness"; role = "Event & Calendar Risk"

    def analyze(self, ctx):
        from pipelines.event_awareness import build_event_context
        ec = _safe(build_event_context, {})
        risk = ec.get("event_risk", "normal")
        is_event_day = ec.get("is_event_day", False)
        pos_size = ec.get("position_size", "full")
        ctx["event_context"] = ec
        ctx["sentiment"] = 0.0

        if risk == "extreme" or pos_size == "avoid":
            vote = REJECT
            confidence = 0.95
        elif risk == "high" or pos_size == "quarter":
            vote = REJECT
            confidence = 0.75
        elif risk == "elevated" or pos_size == "half":
            vote = NEUTRAL
            confidence = 0.50
        else:
            vote = APPROVE
            confidence = 0.60

        flags = []
        if is_event_day:
            flags.append("event_day")
        if ec.get("results_season", {}).get("in_results_season"):
            flags.append("results_season")
        if ec.get("alerts"):
            flags.extend(ec["alerts"][:2])

        return AgentReport(self.name, self.role, vote=vote, confidence=confidence,
                           evidence=ec, flags=flags)


# 3 ──────────────────────────────────────────────────────
class QuantDecisionAgent(Agent):
    name = "quant"; role = "Quant Decision"

    def analyze(self, ctx):
        from api.precompute import cached_screen
        rep = _safe(cached_screen, {})
        buys = (rep or {}).get("actionable", [])
        if not buys:
            return AgentReport(self.name, self.role, vote=NEUTRAL,
                               evidence={"actionable": 0},
                               flags=["no_clean_entry"])
        c = buys[0]
        cand = {"symbol": c["symbol"], "asset": "equity",
                "segment": "equity_delivery", "entry": c.get("price"),
                "stop": c.get("stop_loss"), "target": c.get("target"),
                "grade": c.get("grade"), "conviction": c.get("conviction")}
        ctx["candidate"] = cand
        ev = {"symbol": c["symbol"], "grade": c.get("grade"),
              "conviction": c.get("conviction"), "rr": c.get("reward_risk"),
              "n_actionable": len(buys)}
        vote = APPROVE if c.get("grade") in ("A+", "A") else NEUTRAL
        return AgentReport(self.name, self.role, vote=vote,
                           confidence=(c.get("conviction") or 0) / 100.0, evidence=ev)


# 4 ──────────────────────────────────────────────────────
class OptionsStrategyAgent(Agent):
    name = "options"; role = "Options Strategy"

    def analyze(self, ctx):
        from api.precompute import cached_dashboard
        idx = ctx.get("index", "BANKNIFTY")
        d = _safe(lambda: cached_dashboard(idx), {})
        if not d or d.get("_error"):
            return AgentReport(self.name, self.role, vote=NEUTRAL,
                               evidence={"index": idx}, flags=["chain_unavailable"])
        s = d.get("structure", {})
        ev = {"index": idx, "prob_up": d.get("prob_up"), "action": d.get("action"),
              "structure": s.get("kind"), "max_loss": s.get("max_loss_rupees"),
              "regime": (d.get("gamma") or {}).get("regime")}
        ctx["options_view"] = ev
        # Options agent informs; it doesn't force a directional buy unless the
        # action engine produced one (not NO_TRADE).
        vote = APPROVE if d.get("action") not in ("NO_TRADE", None) else NEUTRAL
        return AgentReport(self.name, self.role, vote=vote,
                           confidence=d.get("prob_up"), evidence=ev)


# 5 ──────────────────────────────────────────────────────
class PortfolioManagerAgent(Agent):
    name = "portfolio_mgr"; role = "Portfolio Manager"

    def analyze(self, ctx):
        from pipelines.risk_policy import effective_limits
        cand = ctx.get("candidate")
        market = ctx.get("market", {})
        wallet_cash = ctx.get("wallet_cash", 300_000)
        if not cand or not cand.get("entry") or not cand.get("stop"):
            return AgentReport(self.name, self.role, vote=NEUTRAL,
                               evidence={}, flags=["no_candidate"])
        lim = effective_limits(market.get("regime", "unknown"),
                               breadth_score=market.get("breadth_score"))
        entry = float(cand["entry"]); stop = float(cand["stop"])
        risk = max(entry - stop, 1e-6)
        budget = wallet_cash * lim["per_trade_risk_pct"]
        qty = max(0, int(budget / risk))
        qty = min(qty, int(wallet_cash * lim["max_sector_pct"] / entry))
        deploy = qty * entry
        ev = {"symbol": cand["symbol"], "qty": qty, "deploy": round(deploy),
              "per_trade_risk_pct": lim["per_trade_risk_pct"],
              "gross_cap_pct": round(lim["gross_effective"] * 100, 1),
              "regime": market.get("regime")}
        ctx["proposal"] = {**cand, "qty": qty, "deploy": deploy, "limits": lim}
        vote = APPROVE if qty >= 1 else REJECT
        flags = [] if qty >= 1 else ["below_one_unit"]
        return AgentReport(self.name, self.role, vote=vote, evidence=ev, flags=flags)


# 6 ──────────────────────────────────────────────────────
class RiskOfficerAgent(Agent):
    name = "risk_officer"; role = "Risk Officer"

    def analyze(self, ctx):
        prop = ctx.get("proposal"); market = ctx.get("market", {})
        if not prop:
            return AgentReport(self.name, self.role, vote=NEUTRAL,
                               flags=["nothing_to_review"])
        from pipelines.risk_policy import MAX_PORTFOLIO_HEAT
        wallet_cash = ctx.get("wallet_cash", 300_000)
        lim = prop.get("limits", {})
        entry, stop, qty = float(prop["entry"]), float(prop["stop"]), prop["qty"]
        trade_risk = (entry - stop) * qty
        heat = (ctx.get("open_risk", 0.0) + trade_risk) / max(wallet_cash, 1)
        checks, veto_reasons = {}, []

        # Hard limits → veto.
        if lim.get("gross_effective", 1) <= 0:
            veto_reasons.append("drawdown circuit-breaker: gross 0%")
        if heat > MAX_PORTFOLIO_HEAT:
            veto_reasons.append(f"portfolio heat {heat:.1%} > {MAX_PORTFOLIO_HEAT:.0%}")
        if market.get("regime") == "bear" and prop.get("grade") not in ("A+", "A"):
            veto_reasons.append("bear regime: only A/A+ setups allowed")
        # Earnings event risk (equity only).
        if prop.get("asset") == "equity":
            er = _safe(lambda: __import__("models.earnings_risk", fromlist=["earnings_risk"])
                       .earnings_risk(prop["symbol"]), {})
            if er.get("action") == "avoid":
                veto_reasons.append(f"earnings in {er.get('days')}d — avoid")
            checks["earnings"] = er.get("action")

        checks.update({"portfolio_heat_pct": round(heat * 100, 2),
                       "heat_limit_pct": round(MAX_PORTFOLIO_HEAT * 100, 1),
                       "trade_risk": round(trade_risk)})
        if veto_reasons:
            return AgentReport(self.name, self.role, vote=VETO, evidence=checks,
                               flags=veto_reasons)
        return AgentReport(self.name, self.role, vote=APPROVE, evidence=checks)


# 7 ──────────────────────────────────────────────────────
class TradeExecutionAgent(Agent):
    name = "execution"; role = "Trade Execution"

    def analyze(self, ctx):
        if ctx.get("vetoed") or not ctx.get("proposal"):
            return AgentReport(self.name, self.role, vote=NEUTRAL,
                               flags=["no_execution"])
        prop = ctx["proposal"]
        tm = ctx.get("trade_manager")
        if tm is None:
            return AgentReport(self.name, self.role, vote=NEUTRAL,
                               evidence={"dry_run": True, "would_open": prop["symbol"]},
                               flags=["no_trade_manager"])
        entry, stop = float(prop["entry"]), float(prop["stop"])
        risk = entry - stop
        t2 = float(prop.get("target") or entry + 2 * risk)
        ev = tm.open_trade(prop["symbol"], prop["segment"], entry, prop["qty"], stop,
                           targets=[(entry + risk, 0.5), (t2, 0.5)], trail_pct=0.03,
                           meta={"source": "aos"})
        ctx["execution"] = ev
        vote = APPROVE if ev.get("action") == "opened" else REJECT
        return AgentReport(self.name, self.role, vote=vote, evidence=ev)


# 8 ──────────────────────────────────────────────────────
class TradeReviewAgent(Agent):
    name = "review"; role = "Trade Review"

    def analyze(self, ctx):
        from aos import memory as m
        closed = _safe(lambda: m.query(
            "SELECT symbol,net_pnl,fees,exit_reason FROM trades WHERE status='closed'"), [])
        if not closed:
            return AgentReport(self.name, self.role, vote=NEUTRAL,
                               evidence={"closed": 0}, flags=["no_closed_trades"])
        wins = [c for c in closed if (c["net_pnl"] or 0) > 0]
        worst = min(closed, key=lambda c: c["net_pnl"] or 0)
        ev = {"closed": len(closed), "win_rate_pct": round(len(wins) / len(closed) * 100, 1),
              "net_pnl": round(sum(c["net_pnl"] or 0 for c in closed)),
              "worst": {"symbol": worst["symbol"], "pnl": worst["net_pnl"],
                        "reason": worst["exit_reason"]}}
        ctx["review"] = ev
        return AgentReport(self.name, self.role, vote=NEUTRAL, evidence=ev)


# 9 ──────────────────────────────────────────────────────
class ModelImprovementAgent(Agent):
    name = "model_improve"; role = "Model Improvement"

    def analyze(self, ctx):
        import json
        path = os.path.join(ROOT, "outputs", "monitoring", "drift_report.json")
        if not os.path.exists(path):
            return AgentReport(self.name, self.role, vote=NEUTRAL,
                               evidence={"drift": "not_run"},
                               flags=["run_drift_monitor"])
        d = _safe(lambda: json.load(open(path)), {})
        overall = d.get("overall")
        ev = {"overall": overall, "live_ic": d.get("live_ic"),
              "ic_retention": d.get("ic_retention"),
              "feature_status": d.get("feature_status")}
        vote = REJECT if overall == "ALERT" else APPROVE
        flags = ["retrain_recommended"] if overall == "ALERT" else []
        return AgentReport(self.name, self.role, vote=vote, evidence=ev, flags=flags)


ALL_AGENTS = [MarketIntelligenceAgent, EventAwarenessAgent, QuantDecisionAgent,
              OptionsStrategyAgent, PortfolioManagerAgent, RiskOfficerAgent,
              TradeExecutionAgent, TradeReviewAgent, ModelImprovementAgent]


if __name__ == "__main__":
    # Smoke test: run each agent with a minimal shared context.
    ctx = {"index": "BANKNIFTY", "wallet_cash": 300_000}
    print("=" * 60)
    print("  AOS AGENTS — smoke test")
    print("=" * 60)
    for A in ALL_AGENTS:
        a = A()
        r = a.run(ctx, enable_llm=False)
        print(f"  {a.role:<22} vote={r.vote:<8} conf={r.confidence} "
              f"flags={r.flags}")

"""
Agentic OS — agent base class + the cardinal rule.

Every agent returns an AgentReport whose NUMBERS come only from quantitative
engines (models, chain, risk policy). The LLM is allowed to write the
`rationale` prose AND nothing else — never a price, probability, size, or
vote. This is enforced structurally: `analyze()` (quant) produces the vote,
evidence and confidence; `narrate()` (LLM, failure-safe) only fills rationale.
"""

import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Vote vocabulary used across the debate.
APPROVE, REJECT, NEUTRAL, VETO = "approve", "reject", "neutral", "veto"


@dataclass
class AgentReport:
    agent: str
    role: str
    vote: str = NEUTRAL                 # approve / reject / neutral / veto
    confidence: Optional[float] = None  # 0..1, from the model — NOT the LLM
    evidence: dict = field(default_factory=dict)   # quantitative facts
    flags: list = field(default_factory=list)      # risks / conflicts noticed
    rationale: str = ""                 # LLM prose ONLY (cosmetic, failure-safe)

    def to_dict(self):
        return asdict(self)


class Agent:
    name = "agent"
    role = "generic"

    def analyze(self, ctx) -> AgentReport:
        """MUST be implemented by subclasses. Returns a report whose vote /
        confidence / evidence are derived from quantitative engines only."""
        raise NotImplementedError

    def narrate(self, report: AgentReport, enable_llm=True) -> AgentReport:
        """Optionally add prose rationale via the local LLM. Failure-safe and
        STRICTLY cosmetic — it cannot change the vote/numbers."""
        if not enable_llm or report.rationale:
            return report
        try:
            from api.narrate import polish
            facts = (f"As the {self.role}, summarise this in one sentence for a "
                     f"trader. Do not change any number. Vote={report.vote}, "
                     f"confidence={report.confidence}, evidence={report.evidence}, "
                     f"flags={report.flags}.")
            text = polish(facts)
            # Guard: keep only if it didn't error to the same input.
            report.rationale = text if text and text != facts else ""
        except Exception:
            report.rationale = ""
        return report

    def run(self, ctx, enable_llm=False) -> AgentReport:
        rep = self.analyze(ctx)
        if not isinstance(rep, AgentReport):
            rep = AgentReport(self.name, self.role, flags=["bad_report"])
        return self.narrate(rep, enable_llm)

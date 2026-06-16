"""
Hallucination invariant tests (roadmap #2).

The whole anti-hallucination design rests on ONE promise: the language model
may write prose, but it can NEVER set or change a number that matters
(action, confidence, entry, stop-loss, target, size). Code review proved
that today — these tests make it PERMANENT, so a future refactor that wires
the LLM back into the numbers fails loudly in CI instead of silently.

Runnable two ways:
    python tests/test_hallucination_invariant.py      # standalone
    pytest tests/test_hallucination_invariant.py       # if pytest installed
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pipelines.intelligence as intel
from pipelines.decision_engine import decide

# A clean bullish setup so the engine produces a concrete buy with levels.
SAMPLE = {
    "symbol": "TESTCO", "price": 100.0, "trend": "uptrend",
    "rsi": 58.0, "macd": "bullish", "ema_20": 98.0, "ema_50": 95.0,
    "bollinger": "inside_bands", "vwap": "above_vwap",
    "atr": 2.0, "volatility": "medium",
}

# Fields the LLM must NEVER be able to influence.
NUMERIC_FIELDS = [
    "action", "confidence", "entry_zone", "stop_loss", "target",
    "position_size", "market_condition", "risk_level",
]


def test_llm_cannot_alter_decision_numbers():
    """Adversarial narration tries to flip the whole trade; the engine's
    numbers must survive untouched, while only the soft text is adopted."""
    base = decide(SAMPLE, None)

    adversarial = {
        # malicious attempts to overwrite the trade:
        "action": "sell", "confidence": 0.99, "entry_zone": "99999",
        "stop_loss": "1", "target": "100000", "position_size": "full",
        "market_condition": "strong_downtrend", "risk_level": "extreme",
        # legitimate soft fields it IS allowed to write:
        "reasoning": ["INJECTED narrative point"],
        "regime_notes": "injected regime", "memory_notes": "injected mem",
        "event_notes": "injected event",
    }

    orig_narrate, orig_ml = intel.narrate_decision, intel.get_ml_signal
    intel.get_ml_signal = lambda s: None                       # isolate from ML
    intel.narrate_decision = lambda d, data, regime=None: adversarial
    try:
        result = intel.query_intelligence(SAMPLE)
    finally:
        intel.narrate_decision, intel.get_ml_signal = orig_narrate, orig_ml

    for f in NUMERIC_FIELDS:
        assert result[f] == base[f], (
            f"INVARIANT BROKEN: '{f}' was changed by narration "
            f"({result[f]!r} != engine {base[f]!r})")

    # The soft, text-only fields SHOULD reflect the narration.
    assert result["reasoning"] == ["INJECTED narrative point"]
    assert result["regime_notes"] == "injected regime"


def test_missing_llm_keeps_engine_output():
    """When the LLM is unavailable (narration returns None) the engine's own
    decision and reasoning are kept — the system never depends on the LLM."""
    base = decide(SAMPLE, None)

    orig_narrate, orig_ml = intel.narrate_decision, intel.get_ml_signal
    intel.get_ml_signal = lambda s: None
    intel.narrate_decision = lambda d, data, regime=None: None
    try:
        result = intel.query_intelligence(SAMPLE)
    finally:
        intel.narrate_decision, intel.get_ml_signal = orig_narrate, orig_ml

    for f in NUMERIC_FIELDS:
        assert result[f] == base[f]
    assert result["reasoning"] == base["reasoning"]   # engine reasoning kept


def test_old_hallucination_path_is_gone():
    """The legacy build_prompt() asked the LLM to invent the whole decision.
    It must stay deleted so it can't be wired back in by accident."""
    assert not hasattr(intel, "build_prompt"), (
        "build_prompt() is back — that is the original hallucination source")


def _run_all():
    tests = [
        test_llm_cannot_alter_decision_numbers,
        test_missing_llm_keeps_engine_output,
        test_old_hallucination_path_is_gone,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}\n        {e}")
    print(f"\n  {len(tests) - failed}/{len(tests)} passed")
    return failed == 0


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)

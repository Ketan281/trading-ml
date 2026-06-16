"""
Optional LLM polish — rewrite the deterministic answer into nicer prose.

Consistent with the system's anti-hallucination rule: the LLM may ONLY rephrase
the already-computed answer; it must not add, drop or alter a single number,
symbol or fact. The structured `data` dict remains the source of truth (the
frontend renders that) — this prose is cosmetic. Fully failure-safe: if Ollama
is down, the model is missing, or anything errors, the original answer is
returned unchanged.
"""

import os

try:
    import ollama
except Exception:
    ollama = None

# Prefer 3b; fall back to whatever qwen is pulled, else disable.
PREFERRED = ["qwen2.5:3b", "qwen2.5:1.5b"]

SYSTEM = (
    "You rewrite trading summaries into clear, concise, professional prose for an "
    "experienced trader. ABSOLUTE RULES: do NOT add, remove, or change ANY number, "
    "price, strike, symbol, percentage or fact. Keep every figure exactly as given. "
    "Do not invent recommendations. Keep it to 2-4 sentences. Preserve any ⚠ warning."
)


def _pick_model():
    if ollama is None:
        return None
    try:
        names = [m.get("model") or m.get("name") for m in ollama.list().get("models", [])]
        for p in PREFERRED:
            if any((n or "").startswith(p) for n in names):
                return p
    except Exception:
        return None
    return None


def polish(answer, model=None):
    """Return an LLM-rephrased version of `answer`, or `answer` unchanged on any
    failure. Never raises."""
    if not answer or ollama is None:
        return answer
    model = model or _pick_model()
    if not model:
        return answer
    try:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": answer}],
            options={"temperature": 0.2, "num_predict": 140},
        )
        text = (resp.get("message", {}).get("content") or "").strip()
        # Guard: only accept if non-empty and not absurdly long.
        return text if 0 < len(text) <= len(answer) * 3 else answer
    except Exception:
        return answer


if __name__ == "__main__":
    sample = ("BANKNIFTY INTRADAY: regime TRENDING_UP → buy CE on dips to VWAP. "
              "Bias P(up) 0.48, IV 18.54% (cheap). ⚠ Rule-based read — confirm on live chain.")
    print("model:", _pick_model())
    print("raw  :", sample)
    print("prose:", polish(sample))

"""
Daily pre-market analysis — the briefing the agents wake up to.

A scheduled job (run ~08:30 IST) that assembles the market picture before the
open and writes it to Trade Memory + a briefing file the orchestrator consumes.

REAL sections (computed from data we have):
  • global_markets   overnight moves in major world indices (yfinance)
  • volatility       NIFTY realised vol + the vol-forecast model + regime pctile
  • regime           the 4-class market regime
  • breadth          participation across the ~470-stock universe
  • sector_rotation  leading vs lagging sectors

HONEST STUBS (no free feed — flagged, never fabricated):
  • macro_events · rbi_sebi · fii_dii · news_sentiment
  These return {status:"no_feed"} with a note so a data source can be wired in
  without changing the contract. We do NOT invent these numbers.
"""

import os
import sys
import json
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT_DIR = os.path.join(ROOT, "data", "aos", "premarket")
os.makedirs(OUT_DIR, exist_ok=True)

GLOBAL = {"S&P500": "^GSPC", "Nasdaq": "^IXIC", "Dow": "^DJI",
          "Nikkei": "^N225", "HangSeng": "^HSI", "FTSE": "^FTSE",
          "DAX": "^GDAXI"}


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


# ── REAL sections ─────────────────────────────────────
def global_markets():
    import yfinance as yf
    out = {}
    for name, tk in GLOBAL.items():
        def f():
            h = yf.Ticker(tk).history(period="5d")
            if len(h) >= 2:
                return round((h["Close"].iloc[-1] / h["Close"].iloc[-2] - 1) * 100, 2)
            return None
        out[name] = _safe(f)
    vals = [v for v in out.values() if v is not None]
    tone = ("positive" if vals and sum(vals) / len(vals) > 0.2 else
            "negative" if vals and sum(vals) / len(vals) < -0.2 else "mixed")
    return {"changes_pct": out, "tone": tone}


def volatility():
    from models.regime_classifier import classify
    reg = _safe(lambda: classify("NIFTY"), {})
    fc = _safe(lambda: __import__("models.vol_forecast", fromlist=["forecast"])
               .forecast("NIFTY"), {})
    return {"regime_vol_pctile": reg.get("vol_pctile"),
            "nifty_rv20": fc.get("current_rv20"),
            "vol_forecast_ml": fc.get("forecast_vol_ml"),
            "vol_direction": fc.get("direction")}


def sector_rotation():
    from models.sector_strength import sector_strength
    sec, _ = _safe(lambda: sector_strength(), (None, None)) or (None, None)
    if sec is None:
        return {"status": "unavailable"}
    lead = sec.head(3)[["sector", "sector_momentum"]]
    lag = sec.tail(3)[["sector", "sector_momentum"]]
    return {"leaders": [(r.sector, round(r.sector_momentum * 100, 1)) for r in lead.itertuples()],
            "laggards": [(r.sector, round(r.sector_momentum * 100, 1)) for r in lag.itertuples()]}


# ── HONEST STUBS (need a data feed) ───────────────────
def _no_feed(what, source_hint):
    return {"status": "no_feed", "note": f"{what} needs a data source ({source_hint}); "
            f"not fabricated"}


# ── Assemble ──────────────────────────────────────────
def run_premarket(enable_llm=False):
    from models.regime_classifier import classify
    from pipelines.breadth import breadth_read
    from aos import memory as mem

    reg = _safe(lambda: classify("NIFTY"), {"regime": "unknown"})
    br = _safe(breadth_read, {"score": None})
    gm = global_markets()
    vol = volatility()
    sect = sector_rotation()

    briefing = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "regime": {"label": reg.get("regime"), "confidence": reg.get("confidence"),
                   "hint": reg.get("hint")},
        "breadth": {"score": br.get("score"), "signal": br.get("signal"),
                    "pct_above_200dma": br.get("pct_above_200dma")},
        "global_markets": gm,
        "volatility": vol,
        "sector_rotation": sect,
        "macro_events": _no_feed("Macro economic calendar (CPI/Fed/RBI policy)",
                                 "economic-calendar API"),
        "rbi_sebi": _no_feed("RBI/SEBI announcements", "regulator RSS / news API"),
        "fii_dii": _no_feed("FII/DII cash flows", "NSE provisional / paid feed"),
        "news_sentiment": _no_feed("News sentiment summary", "news + NLP feed"),
    }

    # Deterministic summary (no LLM numbers).
    summary = (f"Regime {reg.get('regime','?').upper()} "
               f"(breadth {br.get('score')}, global {gm['tone']}). "
               f"Leaders: {', '.join(s[0] for s in sect.get('leaders', [])[:2])}. "
               f"{reg.get('hint','')}")
    briefing["summary"] = summary
    if enable_llm:
        briefing["narration"] = _safe(lambda: __import__("api.narrate", fromlist=["polish"])
                                      .polish(summary), "") or ""

    # Persist: file + Trade Memory regime snapshot.
    path = os.path.join(OUT_DIR, f"briefing_{briefing['date']}.json")
    json.dump(briefing, open(path, "w"), indent=2, default=str)
    mem.record_regime(reg.get("regime"), br.get("score"), reg.get("vol_pctile"),
                      [s[0] for s in sect.get("leaders", [])],
                      extra={"global_tone": gm["tone"]})
    return briefing, path


if __name__ == "__main__":
    b, path = run_premarket()
    print("=" * 68)
    print(f"  PRE-MARKET BRIEFING — {b['date']}")
    print("=" * 68)
    print(f"  Regime    : {b['regime']['label'].upper()} (conf {b['regime']['confidence']})")
    print(f"  Breadth   : {b['breadth']['score']} — {b['breadth']['signal']}")
    print(f"  Global    : {b['global_markets']['tone']}  "
          + " ".join(f"{k} {v:+}%" for k, v in b['global_markets']['changes_pct'].items() if v is not None))
    v = b["volatility"]
    print(f"  Volatility: RV20 {v['nifty_rv20']}% | forecast {v['vol_forecast_ml']}% "
          f"({v['vol_direction']}) | pctile {v['regime_vol_pctile']}")
    print(f"  Leaders   : {b['sector_rotation'].get('leaders')}")
    print(f"  Laggards  : {b['sector_rotation'].get('laggards')}")
    print(f"  No-feed   : macro / rbi_sebi / fii_dii / news_sentiment (flagged)")
    print(f"\n  SUMMARY: {b['summary']}")
    print(f"\n  Saved → {path}")

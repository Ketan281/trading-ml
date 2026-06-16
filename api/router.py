"""
Natural-language intent router — maps a frontend query to the right engine.

Deliberately RULE-BASED (not an LLM): routing must be deterministic and
reliable, the same discipline as the rest of the system. It detects the
symbol, asset class (options / equity / intraday) and intent from keywords,
calls the matching engine, and returns BOTH a human-readable answer and the
structured data the frontend renders. Engine prints are suppressed so the API
returns clean JSON.
"""

import io
import os
import re
import sys
import contextlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _silent(fn, *a, **k):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            return fn(*a, **k)
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {str(e)[:160]}"}


def detect_symbol(q):
    ql = q.lower()
    if "banknifty" in ql or "bank nifty" in ql:
        return "BANKNIFTY"
    if "finnifty" in ql:
        return "FINNIFTY"
    if "nifty" in ql:
        return "NIFTY"
    m = re.findall(r"\b[A-Z][A-Z&-]{2,}\b", q)        # an uppercase stock ticker
    bad = {"CE", "PE", "RBI", "FED", "CPI", "AI", "OI", "IV", "PCR"}
    m = [t for t in m if t not in bad]
    return m[0] if m else None


def parse_intent(q):
    ql = q.lower()
    sym = detect_symbol(q)
    is_options = any(w in ql for w in
                     ["option", " ce", " pe", "call", "put", "premium", "strike",
                      "straddle", "condor", "spread", "expiry", "iv", "oi"])
    is_intraday = any(w in ql for w in ["intraday", "today", "5m", "15m", "scalp", "now"])
    wants_book = any(w in ql for w in ["book", "portfolio", "allocate", "diversif"])
    wants_screen = any(w in ql for w in ["screen", "shortlist", "stocks to buy",
                                         "best stock", "swing", "which stock"])
    index = sym in ("NIFTY", "BANKNIFTY", "FINNIFTY")
    if is_options or index:
        intent = "options"
    elif wants_book:
        intent = "book"
    elif is_intraday and (wants_screen or sym):
        intent = "intraday_stock"
    elif wants_screen:
        intent = "screen"
    else:
        intent = "screen"
    return {"symbol": sym, "intent": intent, "is_intraday": is_intraday,
            "is_options": is_options}


# ── Engine answers ────────────────────────────────────
def _options_answer(symbol, is_intraday):
    from api.precompute import cached_dashboard
    d = cached_dashboard(symbol)             # cache-aware (instant if pre-computed)
    if not d or d.get("_error"):
        return {"answer": f"Could not read {symbol} options chain right now "
                f"(market closed or NSE unreachable).",
                "error": d.get("_error") if d else None, "data": None}
    s = d["structure"]; ir = d["intraday_regime"]; ivp = d["iv_percentile"]
    iv_txt = (f"IV {ivp['current_atm_iv']}% ({ivp['read']})"
              if "current_atm_iv" in ivp else "IV history thin")
    ml = f"₹{s['max_loss_rupees']:,}" if s["max_loss_rupees"] is not None else "undefined"
    struct_txt = (f"For a positional/weekly play, best structure: "
                  f"{s['kind'].replace('_', ' ').upper()} — {' , '.join(s['legs'])} "
                  f"({s['flow']} ₹{abs(s['net_premium_per_share'])}/sh, {s['lots']} lot, "
                  f"max loss {ml}, BE {s['breakevens']}).")

    if is_intraday:
        # Lead with the timeframe-appropriate intraday read.
        stance = ir.get("stance", "no clear intraday edge")
        reg = ir["regime"].upper()
        flow = d.get("flow") or {}
        answer = (
            f"{symbol} INTRADAY: regime {reg} → {stance}. "
            f"upside probability {d['prob_up']:.2f}, {iv_txt}"
            + (f", chain flow {flow.get('read')}" if flow.get("read") else "") + ". "
            + ("Trend is up — favour buying CE on dips toward VWAP, tight stop below VWAP. "
               if "trending_up" in ir["regime"] else
               "Trend is down — favour buying PE on pops toward VWAP, tight stop above VWAP. "
               if "trending_down" in ir["regime"] else
               "Range/choppy — premium-selling is favoured over directional buying. ")
            + struct_txt + " "
            + "⚠ Rule-based read (no proven intraday-ML edge yet) — confirm on your live "
              "chain and respect stops."
        )
    else:
        answer = (
            f"{symbol} options: upside probability {d['prob_up']:.2f} → action {d['action']}; "
            f"intraday regime {ir['regime'].upper()}"
            + (f" ({ir['stance']})" if ir.get("stance") else "") + ". "
            f"{iv_txt}. ▶ {struct_txt} Rationale: {s['reason']}. "
            f"⚠ Rule-based chain read — verify fills on your live chain."
        )
    return {"answer": answer, "intent": "options_recommendation",
            "symbol": symbol, "data": d, "is_intraday": is_intraday}


def _book_answer():
    from api.precompute import cached_book
    r = cached_book()
    if not r or r.get("_error"):
        return {"answer": "Could not build the portfolio book (model/data issue).",
                "error": r.get("_error") if r else None, "data": None}
    holds = ", ".join(f"{h['symbol']} {h['weight_pct']}%" for h in r["holdings"])
    diag = r["diagnostics"]; ctx = r["context"]["regime"]
    answer = (f"Regime {ctx['regime'].upper()}. Constructed book ({diag['positions']} "
              f"names, gross {diag['gross_book_pct']}%, heat {diag['portfolio_heat_pct']}%): "
              f"{holds}. Each name has an entry stop and a reason in the data.")
    return {"answer": answer, "intent": "portfolio_book", "data": r}


def _screen_answer():
    from api.precompute import cached_screen
    r = cached_screen()
    if not r or r.get("_error"):
        return {"answer": "Could not run the screener.",
                "error": r.get("_error") if r else None, "data": None}
    buys = r.get("actionable", [])
    if not buys:
        return {"answer": "No actionable swing buys right now (no clean entry "
                "triggers). Watchlist names are in the data.", "intent": "screen",
                "data": r}
    top = ", ".join(f"{c['symbol']} ({c['grade']}, conv {c['conviction']:.0f})"
                    for c in buys[:5])
    answer = f"Top actionable swing setups today: {top}. Full plans in the data."
    return {"answer": answer, "intent": "screen", "data": r}


def _intraday_stock_answer():
    from pipelines.intraday import scan_buckets
    r = _silent(scan_buckets)
    longs = [c for c in (r or []) if c.get("side") == "long"]
    longs.sort(key=lambda c: c["score"], reverse=True)
    if not longs:
        return {"answer": "No high-quality intraday long setups on the liquid "
                "universe right now.", "intent": "intraday_stock", "data": r}
    top = ", ".join(f"{c['symbol']} ({c['setup']}, score {c['score']})"
                    for c in longs[:5])
    answer = (f"Top intraday long setups (5m, 15m-confirmed): {top}. "
              f"⚠ Rule-based, not a validated-edge model.")
    return {"answer": answer, "intent": "intraday_stock", "data": r}


def route(query):
    """Main entry: NL query → {answer, intent, symbol, data}."""
    p = parse_intent(query)
    if p["intent"] == "options":
        return {**_options_answer(p["symbol"] or "NIFTY", p["is_intraday"]),
                "parsed": p}
    if p["intent"] == "book":
        return {**_book_answer(), "parsed": p}
    if p["intent"] == "intraday_stock":
        return {**_intraday_stock_answer(), "parsed": p}
    return {**_screen_answer(), "parsed": p}


if __name__ == "__main__":
    import sys, json
    q = " ".join(sys.argv[1:]) or "which is the best to enter in banknifty intraday option today"
    print("Q:", q)
    res = route(q)
    print("\nANSWER:\n", res["answer"])
    print("\nINTENT:", res.get("intent"), "| parsed:", res.get("parsed"))

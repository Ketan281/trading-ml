"""
Trading-AI API server — deploy on a server, consume from a frontend.

Run:
    uvicorn api.server:app --host 0.0.0.0 --port 8000
    # or: python api/server.py

Frontend usage:
    POST /query   {"q": "which is the best to enter in banknifty intraday option today"}
        → {"answer": "...", "intent": "...", "data": {...}}

Endpoints
    GET  /health
    POST /query                 natural-language command (the main one)
    GET  /options/{symbol}      full options dashboard
    GET  /book                  constructed portfolio book
    GET  /screen                swing-trade shortlist

NOTE: handlers call the live engines (NSE / model), so a request can take a few
seconds. The frontend should show a loading state. Responses are cached briefly
to keep repeated queries snappy.
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.router import route, _silent

app = FastAPI(title="Trading-AI", version="1.0",
              description="NSE equity + NIFTY/BANKNIFTY options intelligence")

# CORS lockdown: in production set ALLOWED_ORIGINS to your frontend origin(s),
# comma-separated (e.g. "https://app.example.com"). Defaults to localhost for
# dev. Never ship "*" with credentials in production.
_origins = os.getenv("ALLOWED_ORIGINS",
                     "http://localhost:3000,http://127.0.0.1:5500").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Tiny TTL cache so repeated identical queries don't re-hit NSE.
_CACHE = {}
_TTL = 60


def _cached(key, fn):
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < _TTL:
        return hit[1]
    val = fn()
    _CACHE[key] = (now, val)
    return val


class Query(BaseModel):
    q: str
    polish: bool = True          # LLM-rephrase the answer (failure-safe)


@app.get("/health")
def health():
    return {"status": "ok", "service": "trading-ai"}


@app.post("/query")
def query(body: Query):
    """The main endpoint: natural-language command → answer + structured data.

    The deterministic answer is instant (served from the pre-computed cache).
    `polish=true` adds an LLM rewrite that is slower on CPU (~seconds) but the
    polished result is cached, so repeats are instant. Set `polish=false` for a
    guaranteed-instant deterministic answer."""
    # On low-RAM hosts set AOS_DISABLE_LLM=1 so the LLM can never be invoked
    # (it would OOM a 1 GB box). Answers are still clean and deterministic.
    want_polish = body.polish and os.getenv("AOS_DISABLE_LLM") != "1"
    key = f"q::{int(want_polish)}::{body.q.lower().strip()}"

    def build():
        res = route(body.q)
        if want_polish and res.get("answer"):
            from api.narrate import polish
            res = {**res, "answer_raw": res["answer"], "answer": polish(res["answer"])}
        return res

    return _cached(key, build)


@app.get("/options/{symbol}")
def options(symbol: str):
    from pipelines.options.options_dashboard import dashboard
    return _cached(f"opt::{symbol.upper()}",
                   lambda: _silent(dashboard, symbol.upper()))


@app.get("/book")
def book():
    from pipelines.portfolio_book import build_book
    return _cached("book", lambda: _silent(build_book))


@app.get("/screen")
def screen_ep():
    from pipelines.screener import screen
    return _cached("screen", lambda: _silent(screen))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)

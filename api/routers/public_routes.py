import os

from fastapi import APIRouter, Depends

from api import auth
from api.cache import cached
from api.router import _silent, route
from api.schemas import Deposit, Query
from aos import user_wallet as uw

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok", "service": "trading-ai"}


@router.post("/query")
def query(body: Query):
    want_polish = body.polish and os.getenv("AOS_DISABLE_LLM") != "1"
    key = f"q::{int(want_polish)}::{body.q.lower().strip()}"

    def build():
        result = route(body.q)
        if want_polish and result.get("answer"):
            from api.narrate import polish
            result = {**result, "answer_raw": result["answer"], "answer": polish(result["answer"])}
        return result

    return cached(key, build)


@router.get("/options/{symbol}")
def options(symbol: str):
    from pipelines.options.options_dashboard import dashboard
    return cached(f"opt::{symbol.upper()}", lambda: _silent(dashboard, symbol.upper()))


@router.get("/book")
def book():
    from pipelines.portfolio_book import build_book
    return cached("book", lambda: _silent(build_book))


@router.get("/screen")
def screen_ep():
    from pipelines.screener import screen
    return cached("screen", lambda: _silent(screen))


@router.get("/wallet")
def wallet_status():
    from aos.sim_wallet import status
    return _silent(status)


@router.post("/wallet/deposit")
def wallet_deposit(body: Deposit):
    from aos.sim_wallet import deposit
    return _silent(deposit, body.amount)


@router.post("/wallet/reset")
def wallet_reset():
    from aos.sim_wallet import reset
    return {"wallet": _silent(reset)}


@router.post("/wallet/trade/start")
def wallet_trade_start():
    from aos.sim_wallet import start_daily_trade
    return _silent(start_daily_trade)


@router.post("/wallet/tick")
def wallet_tick():
    from aos.sim_wallet import tick, status
    _silent(tick)
    return _silent(status)


@router.get("/equity/recommendation")
def equity_recommendation(user: dict = Depends(auth.current_user)):
    from api.market import recommendation
    from pipelines.screener import screen

    balance = _silent(uw.get_wallet, user["id"]).get("balance", 10_000)
    reco = cached("equity_reco", lambda: _silent(recommendation, balance))
    screener = cached("screen", lambda: _silent(screen))
    top_picks = []
    if screener and isinstance(screener, dict):
        picks = screener.get("picks") or screener.get("stocks") or []
        if isinstance(picks, list):
            top_picks = picks[:5]
    return {"recommendation": reco, "screener_picks": top_picks}


@router.get("/candles/{symbol}")
def candles_ep(symbol: str, interval: str = "5m", period: str = "1d"):
    from api.market import candles
    return cached(f"candles::{symbol.upper()}::{interval}::{period}",
                  lambda: _silent(candles, symbol, interval, period))


@router.get("/recommendation")
def recommendation_ep(user: dict = Depends(auth.current_user)):
    from api.market import recommendation
    balance = _silent(uw.get_wallet, user["id"]).get("balance", 10_000)
    return cached("reco", lambda: _silent(recommendation, balance))


# ── Multi-segment recommendations ──────────────────────
@router.get("/recommendations")
def recommendations_ep(user: dict = Depends(auth.current_user)):
    from api.recommendations import segment_recommendations
    balance = _silent(uw.get_wallet, user["id"]).get("balance", 100_000)
    return cached("reco_multi", lambda: _silent(segment_recommendations, balance), ttl=120)


@router.get("/recommendations/allocate")
def allocate_ep(user: dict = Depends(auth.current_user)):
    from api.recommendations import segment_recommendations, allocate_capital
    balance = _silent(uw.get_wallet, user["id"]).get("balance", 100_000)
    data = cached("reco_multi", lambda: _silent(segment_recommendations, balance), ttl=120)
    if not data:
        return {"allocation": {}, "balance": balance}
    return {"allocation": allocate_capital(balance, data), "balance": balance}


@router.get("/forex/pairs")
def forex_pairs():
    from pipelines.forex.data import list_pairs
    return {"pairs": list_pairs()}


@router.get("/forex/candles/{pair:path}")
def forex_candles(pair: str, interval: str = "15m", period: str = "5d"):
    from pipelines.forex.data import candles_to_list, fetch_candles
    df = fetch_candles(pair, interval, period)
    return cached(f"fxc::{pair}::{interval}::{period}",
                  lambda: {"pair": pair, "candles": candles_to_list(df)})


@router.get("/forex/signals/{pair:path}")
def forex_signals(pair: str):
    from pipelines.forex.confluence import score_pair
    return cached(f"fxsig::{pair}", lambda: _silent(score_pair, pair))


@router.get("/forex/scan")
def forex_scan():
    from pipelines.forex.confluence import scan_all_pairs
    return cached("fxscan", lambda: {"pairs": _silent(scan_all_pairs)})


@router.get("/forex/recommendation")
def forex_recommendation():
    from pipelines.forex.confluence import best_trade
    result = cached("fxreco", lambda: _silent(best_trade))
    if not result:
        return {
            "answer": "No forex pair meets the confluence threshold right now - the system is waiting for a high-confidence setup.",
            "trade": None,
        }
    return {
        "answer": f"Best forex setup: {result['direction'].upper()} {result['pair']} "
                  f"(confluence {result['score']:.2f}, {result['confidence']} confidence, "
                  f"{result['agreeing_timeframes']}/{result['total_timeframes']} TFs agree). "
                  f"Entry {result['trade_plan']['entry']}, SL {result['trade_plan']['stop_loss']}, "
                  f"TP {result['trade_plan']['take_profit']} "
                  f"(R:R {result['trade_plan']['risk_reward']}:1).",
        "trade": result,
    }

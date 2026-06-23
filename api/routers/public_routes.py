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
    from api.precompute import get_cached_recommendations
    return get_cached_recommendations()


@router.get("/universe/tiers")
def universe_tiers_ep(user: dict = Depends(auth.current_user)):
    from engines.market_scheduler import get_tier_universe
    return get_tier_universe()


@router.get("/market/breadth")
def market_breadth_ep(user: dict = Depends(auth.current_user)):
    from engines.market_scheduler import _cache_get
    breadth = _cache_get("breadth")
    return breadth or {"score": 0, "regime": "unknown", "note": "computing"}


# ── ML System endpoints ─────────────────────────────────
@router.get("/ml/predictions")
def ml_predictions_ep(user: dict = Depends(auth.current_user)):
    from engines.ml_inference import predict_all
    return _silent(predict_all, top_n=20)


@router.get("/ml/trades")
def ml_trades_ep(user: dict = Depends(auth.current_user)):
    balance = _silent(uw.get_wallet, user["id"]).get("balance", 100_000)
    from engines.ml_inference import get_top_picks_for_trading
    return {"trades": _silent(get_top_picks_for_trading, capital=balance, max_picks=5)}


@router.get("/ml/options")
def ml_options_ep(user: dict = Depends(auth.current_user)):
    balance = _silent(uw.get_wallet, user["id"]).get("balance", 100_000)
    from engines.ml_inference import get_options_trades
    return {"trades": _silent(get_options_trades, capital=balance, max_picks=3)}


@router.get("/ml/status")
def ml_status_ep():
    from engines.ml_inference import model_status
    return model_status()


@router.get("/ml/portfolio")
def ml_portfolio_ep(user: dict = Depends(auth.current_user)):
    balance = _silent(uw.get_wallet, user["id"]).get("balance", 100_000)
    from engines.portfolio_optimizer import get_portfolio_recommendations
    return _silent(get_portfolio_recommendations, capital=balance, uid=user["id"])


@router.get("/ml/performance")
def ml_performance_ep():
    from engines.performance_tracker import compute_performance
    return _silent(compute_performance, lookback_days=30)


@router.get("/ml/accuracy-trend")
def ml_accuracy_trend_ep():
    from engines.performance_tracker import get_accuracy_trend
    return _silent(get_accuracy_trend, days=90)


@router.get("/ml/retrain-check")
def ml_retrain_check_ep():
    from engines.performance_tracker import check_retrain_needed
    return _silent(check_retrain_needed)


@router.get("/ml/feature-store")
def ml_feature_store_ep():
    from engines.feature_store import get_store_stats
    return get_store_stats()


@router.get("/ml/intraday/equity")
def ml_intraday_equity_ep(user: dict = Depends(auth.current_user)):
    balance = _silent(uw.get_wallet, user["id"]).get("balance", 100_000)
    from engines.intraday_inference import get_intraday_equity_trades
    return _silent(get_intraday_equity_trades, capital=balance, max_picks=5)


@router.get("/ml/intraday/options")
def ml_intraday_options_ep(user: dict = Depends(auth.current_user)):
    balance = _silent(uw.get_wallet, user["id"]).get("balance", 100_000)
    from engines.intraday_inference import get_intraday_options_trades
    return _silent(get_intraday_options_trades, capital=balance, max_picks=3)


@router.get("/ml/intraday/status")
def ml_intraday_status_ep():
    from engines.intraday_inference import model_status
    return model_status()


@router.get("/market-intel")
def market_intel_ep():
    from pipelines.market_intel import market_context
    return cached("market_intel", lambda: _silent(market_context, "NIFTY"), ttl=120)


@router.get("/market-intel/{symbol}")
def stock_intel_ep(symbol: str):
    from pipelines.market_intel import stock_context
    return cached(f"stock_intel_{symbol.upper()}",
                  lambda: _silent(stock_context, symbol.upper()), ttl=120)


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

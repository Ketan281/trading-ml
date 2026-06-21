from fastapi import APIRouter, Depends

from api import auth
from api.ml_runtime import apply_mode_change
from api.router import _silent
from api.schemas import DepositAmt, ModeChange, TradeSpec, TradingModeChange, BrokerConfig
from aos import user_wallet as uw
from aos import user_wallet_views as uw_views

router = APIRouter(tags=["user"])


@router.get("/me/wallet")
def me_wallet(user: dict = Depends(auth.current_user)):
    return _silent(uw.status, user["id"])


@router.post("/me/wallet/deposit")
def me_deposit(body: DepositAmt, user: dict = Depends(auth.current_user)):
    return _silent(uw.deposit, user["id"], body.amount)


@router.post("/me/trade")
def me_trade(body: TradeSpec, user: dict = Depends(auth.current_user)):
    spec = {k: v for k, v in body.model_dump().items() if v is not None}
    return _silent(uw.open_trade, user["id"], spec)


@router.post("/me/trade/{trade_id}/close")
def me_trade_close(trade_id: str, user: dict = Depends(auth.current_user)):
    return _silent(uw.close_trade, user["id"], trade_id)


@router.get("/me/history")
def me_history(user: dict = Depends(auth.current_user)):
    return {"trades": _silent(uw.history_full, user["id"])}


@router.post("/me/trade/{trade_id}/explain")
def me_trade_explain(trade_id: str, user: dict = Depends(auth.current_user)):
    return _silent(uw.explain_trade, user["id"], trade_id)


@router.get("/me/forex-wallet")
def me_forex_wallet(user: dict = Depends(auth.current_user)):
    return _silent(uw_views.forex_wallet_snapshot, user["id"])


@router.post("/me/forex-wallet/deposit")
def me_forex_deposit(body: DepositAmt, user: dict = Depends(auth.current_user)):
    return _silent(uw.deposit_forex, user["id"], body.amount)


@router.post("/me/forex-wallet/reset")
def me_forex_reset(user: dict = Depends(auth.current_user)):
    return {"wallet": _silent(uw.reset_forex, user["id"])}


@router.get("/me/mode")
def me_mode(user: dict = Depends(auth.current_user)):
    return uw.get_mode(user["id"])


@router.post("/me/mode")
def me_set_mode(body: ModeChange, user: dict = Depends(auth.current_user)):
    return apply_mode_change(user["id"], body.mode, body.market)


@router.post("/me/mode/indian")
def me_set_indian_mode(body: ModeChange, user: dict = Depends(auth.current_user)):
    return apply_mode_change(user["id"], body.mode, "indian")


@router.post("/me/mode/forex")
def me_set_forex_mode(body: ModeChange, user: dict = Depends(auth.current_user)):
    return apply_mode_change(user["id"], body.mode, "forex")


# ── Trading mode (paper / live) ──────────────────────
@router.get("/me/trading-mode")
def me_trading_mode(user: dict = Depends(auth.current_user)):
    return {"trading_mode": uw.get_trading_mode(user["id"])}


@router.post("/me/trading-mode")
def me_set_trading_mode(body: TradingModeChange, user: dict = Depends(auth.current_user)):
    result = uw.set_trading_mode(user["id"], body.mode)
    if result.get("error"):
        from fastapi import HTTPException
        raise HTTPException(400, result["error"])
    return result


# ── Broker configuration ─────────────────────────────
@router.get("/me/broker-config")
def me_broker_config(user: dict = Depends(auth.current_user)):
    return uw.get_broker_config(user["id"])


@router.post("/me/broker-config")
def me_save_broker_config(body: BrokerConfig, user: dict = Depends(auth.current_user)):
    return uw.save_broker_config(
        user["id"], body.api_key, body.client_id, body.password, body.totp_secret)


@router.delete("/me/broker-config")
def me_delete_broker_config(user: dict = Depends(auth.current_user)):
    return uw.delete_broker_config(user["id"])

"""Shared FastAPI request models."""

from pydantic import BaseModel


class Query(BaseModel):
    q: str
    polish: bool = True


class Deposit(BaseModel):
    amount: float


class Credentials(BaseModel):
    email: str
    password: str


class GoogleToken(BaseModel):
    id_token: str


class PasswordChange(BaseModel):
    old_password: str
    new_password: str


class TradeSpec(BaseModel):
    segment: str
    underlying: str | None = None
    symbol: str | None = None
    pair: str | None = None
    leg: str | None = None
    strike: int | None = None
    side: str | None = None
    lots: int | None = None
    qty: int | None = None
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    reason: str | None = None


class DepositAmt(BaseModel):
    amount: float


class ModeChange(BaseModel):
    mode: str
    market: str = "indian"


class RoleChange(BaseModel):
    role: str


class TradingModeChange(BaseModel):
    mode: str


class BrokerConfig(BaseModel):
    api_key: str
    client_id: str
    password: str
    totp_secret: str

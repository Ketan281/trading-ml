"""Abstract broker adapter — every broker implements this interface.

The executor calls these methods; the adapter translates to the broker's SDK.
Return dicts follow a fixed schema so the executor never cares which broker
is behind the call.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class OrderResult:
    order_id: str = ""
    status: str = "rejected"       # pending | filled | partial | rejected
    filled_price: float = 0.0
    filled_qty: int = 0
    broker_order_id: str = ""
    error: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self):
        return asdict(self)


@dataclass
class PositionInfo:
    status: str = "unknown"        # open | closed | partial
    qty: int = 0
    avg_price: float = 0.0
    pnl: float = 0.0
    error: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class FundsInfo:
    available_cash: float = 0.0
    used_margin: float = 0.0
    net_value: float = 0.0
    error: str = ""

    def to_dict(self):
        return asdict(self)


class BrokerAdapter(ABC):
    name: str = "base"

    @abstractmethod
    def connect(self) -> bool:
        ...

    @abstractmethod
    def place_order(self, symbol: str, segment: str, side: str,
                    qty: int, price: float, stop_loss: float = 0,
                    target: float = 0, order_type: str = "market") -> OrderResult:
        ...

    @abstractmethod
    def modify_order(self, broker_order_id: str, price: float = None,
                     stop_loss: float = None, target: float = None) -> OrderResult:
        ...

    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> OrderResult:
        ...

    @abstractmethod
    def close_position(self, broker_order_id: str, qty: int,
                       price: float, reason: str = "") -> OrderResult:
        ...

    @abstractmethod
    def get_position(self, broker_order_id: str) -> PositionInfo:
        ...

    @abstractmethod
    def get_funds(self) -> FundsInfo:
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        ...

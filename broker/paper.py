"""Paper broker adapter — instant fills at market price, no real orders.

This is the default adapter and preserves the system's current behavior
exactly. Every trade is accepted and filled immediately at the requested
price. No external calls are made.
"""

import uuid
from datetime import datetime

from broker.base import BrokerAdapter, OrderResult, PositionInfo, FundsInfo


class PaperBroker(BrokerAdapter):
    name = "paper"

    def __init__(self):
        self._positions = {}
        self._connected = True

    def connect(self) -> bool:
        self._connected = True
        return True

    def place_order(self, symbol, segment, side, qty, price,
                    stop_loss=0, target=0, order_type="market") -> OrderResult:
        oid = f"PAPER-{uuid.uuid4().hex[:8]}"
        self._positions[oid] = {
            "symbol": symbol, "segment": segment, "side": side,
            "qty": qty, "entry": price, "status": "open",
        }
        return OrderResult(
            order_id=oid, status="filled", filled_price=price,
            filled_qty=qty, broker_order_id=oid)

    def modify_order(self, broker_order_id, price=None,
                     stop_loss=None, target=None) -> OrderResult:
        return OrderResult(order_id=broker_order_id, status="filled",
                           broker_order_id=broker_order_id)

    def cancel_order(self, broker_order_id) -> OrderResult:
        self._positions.pop(broker_order_id, None)
        return OrderResult(order_id=broker_order_id, status="cancelled",
                           broker_order_id=broker_order_id)

    def close_position(self, broker_order_id, qty, price,
                       reason="") -> OrderResult:
        pos = self._positions.pop(broker_order_id, None)
        return OrderResult(
            order_id=f"EXIT-{broker_order_id}",
            status="filled", filled_price=price, filled_qty=qty,
            broker_order_id=broker_order_id)

    def get_position(self, broker_order_id) -> PositionInfo:
        pos = self._positions.get(broker_order_id)
        if not pos:
            return PositionInfo(status="closed")
        return PositionInfo(status="open", qty=pos["qty"],
                            avg_price=pos["entry"])

    def get_funds(self) -> FundsInfo:
        return FundsInfo(available_cash=float("inf"))

    def is_connected(self) -> bool:
        return self._connected

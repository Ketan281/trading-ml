"""Angel One broker adapter — wraps the existing SmartAPI connection.

Requires ANGELONE_* env vars in .env. Call connect() before use.
Falls back to paper if credentials are missing or login fails.
"""

import logging

from broker.base import BrokerAdapter, OrderResult, PositionInfo, FundsInfo

log = logging.getLogger("broker.angelone")

# Angel One exchange/product constants
EXCHANGE_MAP = {
    "options": "NFO", "futures": "NFO",
    "equity": "NSE", "equity_intraday": "NSE",
    "forex": "CDS",
}
PRODUCT_MAP = {
    "options": "NRML", "futures": "NRML",
    "equity": "MIS", "equity_intraday": "MIS",
    "forex": "NRML",
}


class AngelOneBroker(BrokerAdapter):
    name = "angelone"

    def __init__(self):
        self._conn = None

    def connect(self) -> bool:
        try:
            from pipelines.angelone_connect import AngelOneConnection
            self._conn = AngelOneConnection()
            ok = self._conn.login()
            if ok:
                log.info("Angel One broker connected")
            else:
                log.warning("Angel One login failed — check credentials")
            return ok
        except Exception as e:
            log.error("Angel One connection error: %s", e)
            return False

    def is_connected(self) -> bool:
        return self._conn is not None and self._conn.connected

    def place_order(self, symbol, segment, side, qty, price,
                    stop_loss=0, target=0, order_type="market") -> OrderResult:
        if not self.is_connected():
            return OrderResult(error="broker not connected")
        try:
            exchange = EXCHANGE_MAP.get(segment, "NSE")
            product = PRODUCT_MAP.get(segment, "MIS")
            tx = "BUY" if side in ("long", "buy") else "SELL"
            params = {
                "variety": "NORMAL",
                "tradingsymbol": symbol,
                "symboltoken": self._resolve_token(symbol, exchange),
                "transactiontype": tx,
                "exchange": exchange,
                "ordertype": "MARKET" if order_type == "market" else "LIMIT",
                "producttype": product,
                "duration": "DAY",
                "quantity": str(qty),
            }
            if order_type == "limit":
                params["price"] = str(round(price, 2))

            resp = self._conn.smart_api.placeOrder(params)
            if resp and resp.get("status"):
                oid = resp["data"]["orderid"]
                log.info("Order placed: %s %s %s qty=%d => %s", tx, symbol, segment, qty, oid)
                return OrderResult(
                    order_id=oid, status="pending",
                    filled_price=price, filled_qty=qty,
                    broker_order_id=oid)
            return OrderResult(error=resp.get("message", "order rejected"))
        except Exception as e:
            log.error("place_order failed: %s", e)
            return OrderResult(error=str(e))

    def modify_order(self, broker_order_id, price=None,
                     stop_loss=None, target=None) -> OrderResult:
        if not self.is_connected():
            return OrderResult(error="broker not connected")
        try:
            params = {"variety": "NORMAL", "orderid": broker_order_id}
            if price is not None:
                params["price"] = str(round(price, 2))
            resp = self._conn.smart_api.modifyOrder(params)
            if resp and resp.get("status"):
                return OrderResult(order_id=broker_order_id, status="modified",
                                   broker_order_id=broker_order_id)
            return OrderResult(error=resp.get("message", "modify rejected"))
        except Exception as e:
            return OrderResult(error=str(e))

    def cancel_order(self, broker_order_id) -> OrderResult:
        if not self.is_connected():
            return OrderResult(error="broker not connected")
        try:
            resp = self._conn.smart_api.cancelOrder(broker_order_id, "NORMAL")
            if resp and resp.get("status"):
                return OrderResult(order_id=broker_order_id, status="cancelled",
                                   broker_order_id=broker_order_id)
            return OrderResult(error=resp.get("message", "cancel failed"))
        except Exception as e:
            return OrderResult(error=str(e))

    def close_position(self, broker_order_id, qty, price,
                       reason="") -> OrderResult:
        if not self.is_connected():
            return OrderResult(error="broker not connected")
        pos = self.get_position(broker_order_id)
        if pos.status != "open":
            return OrderResult(status="filled", filled_price=price,
                               broker_order_id=broker_order_id)
        opposite = "SELL" if pos.qty > 0 else "BUY"
        try:
            params = {
                "variety": "NORMAL",
                "tradingsymbol": broker_order_id,
                "transactiontype": opposite,
                "exchange": "NSE",
                "ordertype": "MARKET",
                "producttype": "MIS",
                "duration": "DAY",
                "quantity": str(abs(qty)),
            }
            resp = self._conn.smart_api.placeOrder(params)
            if resp and resp.get("status"):
                oid = resp["data"]["orderid"]
                return OrderResult(order_id=oid, status="filled",
                                   filled_price=price, filled_qty=qty,
                                   broker_order_id=oid)
            return OrderResult(error=resp.get("message", "exit order rejected"))
        except Exception as e:
            return OrderResult(error=str(e))

    def get_position(self, broker_order_id) -> PositionInfo:
        if not self.is_connected():
            return PositionInfo(error="broker not connected")
        try:
            resp = self._conn.smart_api.position()
            if resp and resp.get("status"):
                for pos in (resp.get("data") or []):
                    if pos.get("orderid") == broker_order_id:
                        net_qty = int(pos.get("netqty", 0))
                        return PositionInfo(
                            status="open" if net_qty != 0 else "closed",
                            qty=net_qty,
                            avg_price=float(pos.get("averageprice", 0)),
                            pnl=float(pos.get("pnl", 0)))
            return PositionInfo(status="unknown")
        except Exception as e:
            return PositionInfo(error=str(e))

    def get_funds(self) -> FundsInfo:
        if not self.is_connected():
            return FundsInfo(error="broker not connected")
        try:
            resp = self._conn.smart_api.rmsLimit()
            if resp and resp.get("status"):
                d = resp["data"]
                return FundsInfo(
                    available_cash=float(d.get("availablecash", 0)),
                    used_margin=float(d.get("utiliseddebits", 0)),
                    net_value=float(d.get("net", 0)))
            return FundsInfo(error="funds fetch failed")
        except Exception as e:
            return FundsInfo(error=str(e))

    def _resolve_token(self, symbol, exchange):
        """Resolve Angel One symbol token. Returns empty string if not found."""
        try:
            resp = self._conn.smart_api.searchScrip(exchange, symbol)
            if resp and resp.get("status") and resp["data"]:
                return resp["data"][0].get("symboltoken", "")
        except Exception:
            pass
        return ""

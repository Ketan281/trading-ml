"""Execution boundary — the single gate between trading logic and order routing.

Every trade open/close in the system passes through ExecutionGate. It:
  1. Validates pre-trade risk checks (daily loss limit, position count, margin)
  2. Routes to the configured broker adapter (paper or live)
  3. Logs every order attempt, fill, and rejection to an audit trail
  4. Enforces a kill-switch that blocks all live orders when tripped

Configuration is via environment variables:
  TRADING_MODE=paper|live       (default: paper)
  BROKER_NAME=angelone          (only used when TRADING_MODE=live)
  DAILY_LOSS_LIMIT=5000         (max daily loss in base currency, default: 5000)
  MAX_OPEN_POSITIONS=5          (default: 5)
  KILL_SWITCH=0|1               (1 = block all orders, default: 0)
"""

import json
import logging
import os
import threading
from datetime import date, datetime

from broker.base import BrokerAdapter, OrderResult, FundsInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_DIR = os.path.join(ROOT, "data", "broker")
os.makedirs(AUDIT_DIR, exist_ok=True)
AUDIT_LOG = os.path.join(AUDIT_DIR, "audit.jsonl")

log = logging.getLogger("broker.executor")

_lock = threading.Lock()
_daily_pnl = {"date": None, "pnl": 0.0}


def _load_adapter(mode: str, broker_name: str) -> BrokerAdapter:
    if mode == "live":
        if broker_name == "angelone":
            from broker.angelone import AngelOneBroker
            adapter = AngelOneBroker()
            if not adapter.connect():
                log.error("Live broker connection failed — falling back to paper")
                from broker.paper import PaperBroker
                return PaperBroker()
            return adapter
        raise ValueError(f"Unknown broker: {broker_name}")
    from broker.paper import PaperBroker
    return PaperBroker()


class ExecutionGate:
    def __init__(self, adapter: BrokerAdapter = None):
        self.mode = os.getenv("TRADING_MODE", "paper").lower()
        self.broker_name = os.getenv("BROKER_NAME", "angelone").lower()
        self.daily_loss_limit = float(os.getenv("DAILY_LOSS_LIMIT", "5000"))
        self.max_open_positions = int(os.getenv("MAX_OPEN_POSITIONS", "5"))
        self.kill_switch = os.getenv("KILL_SWITCH", "0") == "1"
        self.adapter = adapter or _load_adapter(self.mode, self.broker_name)
        log.info("ExecutionGate initialized: mode=%s, broker=%s, kill_switch=%s",
                 self.mode, self.adapter.name, self.kill_switch)

    def is_live(self) -> bool:
        return self.mode == "live" and self.adapter.name != "paper"

    def open(self, symbol: str, segment: str, side: str, qty: int,
             price: float, stop_loss: float = 0, target: float = 0,
             order_type: str = "market", uid: int = 0,
             reason: str = "") -> OrderResult:
        """Validate, route to broker, audit-log, return result."""
        entry = {
            "action": "open", "symbol": symbol, "segment": segment,
            "side": side, "qty": qty, "price": price,
            "stop_loss": stop_loss, "target": target,
            "uid": uid, "reason": reason, "mode": self.mode,
            "broker": self.adapter.name, "ts": datetime.now().isoformat(),
        }

        if self.kill_switch:
            entry["result"] = "blocked_kill_switch"
            self._audit(entry)
            return OrderResult(error="kill switch is active — all orders blocked")

        breach = self._pre_trade_checks(segment, qty, price)
        if breach:
            entry["result"] = f"blocked_{breach}"
            self._audit(entry)
            return OrderResult(error=breach)

        result = self.adapter.place_order(
            symbol, segment, side, qty, price, stop_loss, target, order_type)

        entry["result"] = result.status
        entry["broker_order_id"] = result.broker_order_id
        entry["filled_price"] = result.filled_price
        entry["error"] = result.error
        self._audit(entry)

        if result.error:
            log.warning("Order rejected: %s %s — %s", symbol, segment, result.error)
        else:
            log.info("Order %s: %s %s %s qty=%d @ %.2f [%s]",
                     result.status, side, symbol, segment, qty,
                     result.filled_price, self.adapter.name)
        return result

    def close(self, broker_order_id: str, qty: int, price: float,
              reason: str = "", uid: int = 0,
              symbol: str = "") -> OrderResult:
        """Close a position through the broker."""
        entry = {
            "action": "close", "broker_order_id": broker_order_id,
            "symbol": symbol, "qty": qty, "price": price,
            "reason": reason, "uid": uid, "mode": self.mode,
            "broker": self.adapter.name, "ts": datetime.now().isoformat(),
        }

        if self.kill_switch and self.is_live():
            entry["result"] = "blocked_kill_switch"
            self._audit(entry)
            return OrderResult(error="kill switch active — close manually in broker terminal")

        result = self.adapter.close_position(broker_order_id, qty, price, reason)
        entry["result"] = result.status
        entry["filled_price"] = result.filled_price
        entry["error"] = result.error
        self._audit(entry)

        if result.status == "filled":
            self._track_daily_pnl(result.filled_price, price, qty)

        return result

    def sync_position(self, broker_order_id: str):
        """Check broker-side position status (for live mode)."""
        return self.adapter.get_position(broker_order_id)

    def funds(self) -> FundsInfo:
        return self.adapter.get_funds()

    def trip_kill_switch(self, reason: str = "manual"):
        """Emergency stop — blocks all new orders."""
        self.kill_switch = True
        self._audit({"action": "kill_switch_tripped", "reason": reason,
                      "ts": datetime.now().isoformat()})
        log.critical("KILL SWITCH TRIPPED: %s", reason)

    def reset_kill_switch(self):
        self.kill_switch = False
        self._audit({"action": "kill_switch_reset",
                      "ts": datetime.now().isoformat()})
        log.info("Kill switch reset")

    def _pre_trade_checks(self, segment, qty, price):
        with _lock:
            today = date.today().isoformat()
            if _daily_pnl["date"] != today:
                _daily_pnl["date"] = today
                _daily_pnl["pnl"] = 0.0

            if _daily_pnl["pnl"] <= -self.daily_loss_limit:
                return f"daily loss limit breached ({_daily_pnl['pnl']:.0f})"

        return None

    def _track_daily_pnl(self, exit_price, entry_price, qty):
        with _lock:
            today = date.today().isoformat()
            if _daily_pnl["date"] != today:
                _daily_pnl["date"] = today
                _daily_pnl["pnl"] = 0.0
            pnl = (exit_price - entry_price) * qty
            _daily_pnl["pnl"] += pnl

            if _daily_pnl["pnl"] <= -self.daily_loss_limit:
                self.trip_kill_switch(
                    f"daily loss limit hit: {_daily_pnl['pnl']:.0f}")

    def _audit(self, entry: dict):
        try:
            with open(AUDIT_LOG, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass


_executor = None


def get_executor() -> ExecutionGate:
    global _executor
    if _executor is None:
        _executor = ExecutionGate()
    return _executor

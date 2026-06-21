"""Broker execution layer — the boundary between paper and live trading."""

from broker.executor import get_executor

__all__ = ["get_executor"]

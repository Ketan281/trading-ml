"""Reactive ML helpers for the API layer.

Automation (ticking, learning, retraining) lives exclusively in
aos/scheduler.py.  This module provides only the *reactive* actions that
API routes need: immediately opening a trade when the user toggles ML mode,
and applying mode changes.
"""

import logging

from api.router import _silent
from aos import user_wallet as uw

_ml_log = logging.getLogger("ml_auto")


def trigger_ml_immediately(uid, market):
    """Open ML-mode trades immediately rather than waiting for the next cycle."""
    from datetime import datetime as dt, time as dtime

    from aos.sim_wallet import SQUARE_OFF

    opened = []
    now_time = dt.now().time()
    if market in ("indian", "both"):
        indian_market_open = dtime(9, 15) <= now_time <= SQUARE_OFF
        if indian_market_open:
            try:
                trades = uw.auto_open_trades(uid)
                if trades:
                    for trade in trades:
                        opened.append({"market": "indian", "symbol": trade["symbol"], "trade": trade})
                else:
                    opened.append({"market": "indian", "symbol": None,
                                   "info": "no actionable Indian trade right now"})
            except Exception as exc:
                _ml_log.warning("ML immediate Indian trade for uid %s failed: %s", uid, exc)
                opened.append({"market": "indian", "error": str(exc)})
        else:
            opened.append({"market": "indian", "symbol": None,
                           "info": "Indian market is closed (09:15-15:15 IST)"})
    if market in ("forex", "both"):
        try:
            trade = uw.auto_open_forex_trade(uid)
            if trade:
                opened.append({"market": "forex", "symbol": trade["symbol"], "trade": trade})
            else:
                opened.append({"market": "forex", "symbol": None,
                               "info": "no actionable forex setup meets confluence threshold"})
        except Exception as exc:
            _ml_log.warning("ML immediate Forex trade for uid %s failed: %s", uid, exc)
            opened.append({"market": "forex", "error": str(exc)})
    return opened


def apply_mode_change(uid, mode, market):
    result = _silent(uw.set_mode, uid, mode, market)
    if mode == "ml":
        result["auto_opened"] = trigger_ml_immediately(uid, market)
    result["current_mode"] = uw.get_mode(uid)
    result["status"] = _silent(uw.status, uid, do_tick=False)
    return result

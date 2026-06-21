"""Wallet view helpers used by the API streaming surface."""

import json

from aos import user_wallet as uw


def live_snapshot(uid):
    from datetime import datetime as dt

    try:
        uw.tick_user(uid)
    except Exception:
        pass

    indian_wallet = uw.get_wallet(uid)
    forex_wallet = uw.get_forex_wallet(uid)
    open_trades = uw._open_trades(uid)
    trades = []
    indian_unrealized = 0.0
    forex_unrealized = 0.0

    for trade in open_trades:
        last_price = trade["pnl_series"][-1][1] if trade["pnl_series"] else trade["entry"]
        last_pnl = trade["pnl_series"][-1][2] if trade["pnl_series"] else 0.0
        pnl_pct = round((last_pnl / trade["cost"]) * 100, 2) if trade["cost"] else 0
        is_forex = trade.get("segment") == "forex"
        if is_forex:
            forex_unrealized += last_pnl
        else:
            indian_unrealized += last_pnl
        trades.append({
            "id": trade["id"],
            "symbol": trade["symbol"],
            "segment": trade["segment"],
            "kind": trade["kind"],
            "side": trade["side"],
            "entry": trade["entry"],
            "current_price": last_price,
            "qty": trade["qty"],
            "lots": trade.get("lots"),
            "stop": trade["stop"],
            "target": trade["target"],
            "gross_pnl": last_pnl,
            "pnl_pct": pnl_pct,
            "currency": "$" if is_forex else "₹",
            "pnl_series": trade["pnl_series"][-60:],
            "opened_at": trade["opened_at"],
        })

    return {
        "indian_wallet": {"balance": indian_wallet["balance"], "currency": "INR"},
        "forex_wallet": {"balance": forex_wallet["balance"], "currency": "USD"},
        "indian_equity": round(indian_wallet["balance"] + indian_unrealized, 2),
        "forex_equity": round(forex_wallet["balance"] + forex_unrealized, 2),
        "trades": trades,
        "timestamp": dt.now().isoformat(),
    }


def trade_snapshot(uid, trade_id):
    from datetime import datetime as dt

    with uw._conn() as con:
        row = con.execute("SELECT * FROM trades WHERE id=? AND user_id=?", (trade_id, uid)).fetchone()
    if not row:
        return {"error": "trade not found"}

    trade = uw._row_to_trade(row)
    if trade["status"] != "open":
        return {"error": f"trade is {trade['status']}", "trade": trade}

    price = uw._live_price(trade)
    if price is not None:
        gross = uw._signed_gross(trade, price)
        trade["pnl_series"].append([dt.now().strftime("%H:%M:%S"), round(price, 2), round(gross, 1)])
        with uw._conn() as con:
            con.execute("UPDATE trades SET pnl_series=? WHERE id=?",
                        (json.dumps(trade["pnl_series"]), trade["id"]))

    last_price = trade["pnl_series"][-1][1] if trade["pnl_series"] else trade["entry"]
    last_pnl = trade["pnl_series"][-1][2] if trade["pnl_series"] else 0
    is_forex = trade.get("segment") == "forex"
    return {
        "id": trade["id"],
        "symbol": trade["symbol"],
        "segment": trade["segment"],
        "side": trade["side"],
        "entry": trade["entry"],
        "current_price": last_price,
        "qty": trade["qty"],
        "stop": trade["stop"],
        "target": trade["target"],
        "gross_pnl": last_pnl,
        "pnl_pct": round((last_pnl / trade["cost"]) * 100, 2) if trade["cost"] else 0,
        "currency": "$" if is_forex else "₹",
        "pnl_series": trade["pnl_series"],
        "opened_at": trade["opened_at"],
        "timestamp": dt.now().isoformat(),
    }


def forex_wallet_snapshot(uid):
    wallet = uw.get_forex_wallet(uid)
    forex_trades = [trade for trade in uw.status(uid).get("forex_open_trades", [])
                    if trade.get("segment") == "forex"]
    unrealized = sum(trade["pnl_series"][-1][2] for trade in forex_trades if trade.get("pnl_series"))
    return {
        "wallet": wallet,
        "live_equity": round(wallet["balance"] + unrealized, 2),
        "unrealized": round(unrealized, 1),
        "open_trades": forex_trades,
    }

"""
Brokerage / cost agent — accurate Indian (NSE) trading charges.

Every entry and exit costs money, on BOTH sides, and the charges differ by
segment. Getting this exactly right is what separates a real P&L from a
fantasy one. Defaults model a DISCOUNT broker (Zerodha-style); all rates are
configurable.

Segments:
  • equity_delivery (CNC swing) — STT 0.1% both sides, ₹0 brokerage, stamp 0.015% buy
  • equity_intraday (MIS)       — brokerage 0.03%/₹20, STT 0.025% sell only
  • options                     — flat ₹20/leg, STT 0.1% sell premium, higher txn
  • futures                      — brokerage 0.03%/₹20, STT 0.02% sell, low exch txn

charges(segment, side, price, qty) → full breakdown + total, so the trade
manager can compute net P&L to the rupee.

NOTE: regulatory rates change (e.g., the Oct-2024 options STT hike). Verify
against your broker's live contract note; rates here are kept in RATES for easy
update.
"""

RATES = {
    "equity_delivery": {
        "brokerage_pct": 0.0, "brokerage_max": 0.0,
        "stt_buy": 0.001, "stt_sell": 0.001,          # 0.1% both sides
        "exch_txn": 0.0000297,                          # NSE 0.00297%
        "sebi": 0.000001, "stamp_buy": 0.00015,         # SEBI 10/cr, stamp 0.015%
    },
    "equity_intraday": {
        "brokerage_pct": 0.0003, "brokerage_max": 20.0,
        "stt_buy": 0.0, "stt_sell": 0.00025,            # 0.025% sell only
        "exch_txn": 0.0000297,
        "sebi": 0.000001, "stamp_buy": 0.00003,         # stamp 0.003% buy
    },
    "options": {
        "brokerage_flat": 20.0,
        "stt_buy": 0.0, "stt_sell": 0.001,              # 0.1% sell premium (post Oct-24)
        "exch_txn": 0.0003503,                          # NSE options 0.03503%
        "sebi": 0.000001, "stamp_buy": 0.00003,         # stamp 0.003% buy
    },
    "futures": {
        "brokerage_pct": 0.0003, "brokerage_max": 20.0,  # 0.03% capped at ₹20
        "stt_buy": 0.0, "stt_sell": 0.0002,              # STT 0.02% sell side only
        "exch_txn": 0.0000173,                           # NSE futures 0.00173%
        "sebi": 0.000001, "stamp_buy": 0.00002,          # stamp 0.002% buy
    },
    "forex": {
        "spread_based": True,
    },
}
GST = 0.18


def charges(segment, side, price, qty):
    """side: 'buy' or 'sell'. price = per-share / per-premium-unit. qty = shares
    (options: lots × lot_size). Returns a breakdown dict including 'total'."""
    r = RATES[segment]
    side = side.lower()
    turnover = price * qty

    if r.get("spread_based"):
        return {"segment": segment, "side": side, "turnover": round(turnover, 2),
                "brokerage": 0, "stt": 0, "exchange": 0, "sebi": 0, "stamp": 0,
                "gst": 0, "total": 0, "note": "forex cost is spread-based (built into entry)"}

    if segment == "options":
        brokerage = r["brokerage_flat"]
    else:
        brokerage = min(r["brokerage_pct"] * turnover, r["brokerage_max"]) \
            if r["brokerage_max"] else r["brokerage_pct"] * turnover

    stt = (r["stt_buy"] if side == "buy" else r["stt_sell"]) * turnover
    exch = r["exch_txn"] * turnover
    sebi = r["sebi"] * turnover
    stamp = r["stamp_buy"] * turnover if side == "buy" else 0.0
    gst = GST * (brokerage + exch + sebi)
    total = brokerage + stt + exch + sebi + stamp + gst
    return {
        "segment": segment, "side": side, "turnover": round(turnover, 2),
        "brokerage": round(brokerage, 2), "stt": round(stt, 2),
        "exchange": round(exch, 2), "sebi": round(sebi, 4),
        "stamp": round(stamp, 2), "gst": round(gst, 2),
        "total": round(total, 2),
    }


def round_trip(segment, entry, exit_price, qty):
    """Total cost of a full in-and-out trade (buy then sell)."""
    buy = charges(segment, "buy", entry, qty)
    sell = charges(segment, "sell", exit_price, qty)
    return {"entry_cost": buy["total"], "exit_cost": sell["total"],
            "total_cost": round(buy["total"] + sell["total"], 2),
            "buy": buy, "sell": sell}


if __name__ == "__main__":
    print("=" * 60)
    print("  BROKERAGE AGENT — sample charges (Zerodha-style)")
    print("=" * 60)
    # Swing delivery: buy 100 @ 1000, sell @ 1100
    rt = round_trip("equity_delivery", 1000, 1100, 100)
    print(f"  Delivery 100 @1000→1100: round-trip cost ₹{rt['total_cost']} "
          f"(entry ₹{rt['entry_cost']} + exit ₹{rt['exit_cost']})")
    # Intraday: buy 100 @ 1000, sell @ 1010
    rt = round_trip("equity_intraday", 1000, 1010, 100)
    print(f"  Intraday 100 @1000→1010: round-trip cost ₹{rt['total_cost']}")
    # Options: buy 1 BANKNIFTY lot (35) @ 300, sell @ 360
    rt = round_trip("options", 300, 360, 35)
    print(f"  Options 35 @300→360 (buy): round-trip cost ₹{rt['total_cost']}")
    # Options selling 35 @ 300 → buy back @ 240
    s = charges("options", "sell", 300, 35); b = charges("options", "buy", 240, 35)
    print(f"  Options SELL 35 @300, buyback @240: cost ₹{round(s['total']+b['total'],2)} "
          f"(sell-side STT ₹{s['stt']})")

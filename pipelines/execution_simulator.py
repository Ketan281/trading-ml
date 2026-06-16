import os
import sys
import json
import random
import numpy as np
from datetime import datetime, time

ROOT = os.path.dirname(os.path.dirname(
       os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUTPUT_DIR = os.path.join(ROOT, "outputs",
                          "executions")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Market Config ─────────────────────────────────────
MARKET_CONFIG = {
    "nifty_lot_size"    : 75,
    "banknifty_lot_size": 35,
    "default_lot_size"  : 1,
    "market_open"       : time(9, 15),
    "market_close"      : time(15, 30),
    "brokerage_flat"    : 20.0,    # ₹20 flat
    "stt_rate"          : 0.0005,  # 0.05%
    "exchange_charge"   : 0.0000325,
    "gst_rate"          : 0.18,
    "sebi_rate"         : 0.000001,
    "stamp_duty"        : 0.00003
}

LOT_SIZES = {
    "NIFTY"    : 75,
    "BANKNIFTY": 35,
    "RELIANCE" : 250,
    "TCS"      : 150,
    "HDFCBANK" : 550,
    "ICICIBANK": 700,
    "INFY"     : 300,
    "DEFAULT"  : 1
}

# ── Slippage Model ────────────────────────────────────
class SlippageModel:
    """
    Realistic slippage based on:
    - Market conditions
    - Order size
    - Time of day
    - Volatility
    """

    def __init__(self):
        self.base_slippage = {
            "very_low" : 0.0002,
            "low"      : 0.0005,
            "normal"   : 0.001,
            "high"     : 0.002,
            "extreme"  : 0.004
        }

    def calculate_slippage(self, price, quantity,
                            vol_regime="normal",
                            order_type="market",
                            time_of_day=None):
        base = self.base_slippage.get(
            vol_regime, 0.001
        )

        # Order type multiplier
        type_mult = {
            "market" : 1.0,
            "limit"  : 0.3,
            "stop"   : 1.5
        }.get(order_type, 1.0)

        # Time of day multiplier
        if time_of_day:
            hour = time_of_day.hour
            min_ = time_of_day.minute
            t    = hour * 100 + min_

            if t <= 930:        # First 15 min
                time_mult = 2.0
            elif t >= 1515:     # Last 15 min
                time_mult = 1.8
            elif 1200 <= t <= 1330:  # Lunch
                time_mult = 1.3
            else:
                time_mult = 1.0
        else:
            time_mult = 1.0

        # Size impact
        value      = price * quantity
        if value > 1000000:      # >10 Lakhs
            size_mult = 1.5
        elif value > 500000:     # >5 Lakhs
            size_mult = 1.2
        else:
            size_mult = 1.0

        # Random component (market noise)
        noise = random.uniform(0.8, 1.2)

        total_slip = (
            base * type_mult *
            time_mult * size_mult * noise
        )

        slip_points = price * total_slip

        return {
            "slip_pct"   : round(total_slip * 100, 4),
            "slip_points": round(slip_points, 2),
            "multipliers": {
                "base"      : base,
                "order_type": type_mult,
                "time"      : time_mult,
                "size"      : size_mult,
                "noise"     : round(noise, 3)
            }
        }

    def apply_slippage(self, price, action,
                        quantity,
                        vol_regime="normal",
                        order_type="market",
                        time_of_day=None):
        slip = self.calculate_slippage(
            price, quantity,
            vol_regime, order_type, time_of_day
        )

        if action == "buy":
            filled_price = price + slip["slip_points"]
        else:
            filled_price = price - slip["slip_points"]

        return round(filled_price, 2), slip

# ── Partial Fill Model ────────────────────────────────
class PartialFillModel:
    """
    Simulates partial order fills based on
    liquidity and market conditions.
    """

    def simulate_fill(self, quantity, price,
                       vol_regime="normal",
                       liquidity="high"):
        fill_rate_map = {
            ("low",    "extreme"): 0.50,
            ("low",    "high")   : 0.65,
            ("low",    "normal") : 0.80,
            ("medium", "extreme"): 0.70,
            ("medium", "high")   : 0.80,
            ("medium", "normal") : 0.90,
            ("high",   "extreme"): 0.85,
            ("high",   "high")   : 0.90,
            ("high",   "normal") : 0.98,
        }

        base_fill = fill_rate_map.get(
            (liquidity, vol_regime), 0.95
        )

        # Random variation
        fill_rate = base_fill * random.uniform(
            0.95, 1.0
        )
        fill_rate = min(fill_rate, 1.0)

        filled_qty = max(
            1, int(quantity * fill_rate)
        )

        return {
            "requested_qty": quantity,
            "filled_qty"   : filled_qty,
            "fill_rate"    : round(fill_rate, 3),
            "partial_fill" : filled_qty < quantity,
            "unfilled_qty" : quantity - filled_qty
        }

# ── Spread Model ──────────────────────────────────────
class SpreadModel:
    """
    Models bid-ask spread based on
    volatility and liquidity.
    """

    def get_spread(self, price, symbol,
                    vol_regime="normal"):
        # Base spread as % of price
        base_spread_map = {
            "NIFTY"    : 0.0001,
            "BANKNIFTY": 0.0001,
            "RELIANCE" : 0.0005,
            "TCS"      : 0.0005,
            "DEFAULT"  : 0.001
        }

        base = base_spread_map.get(
            symbol, base_spread_map["DEFAULT"]
        )

        vol_mult = {
            "very_low": 0.5,
            "low"     : 0.8,
            "normal"  : 1.0,
            "high"    : 2.0,
            "extreme" : 4.0
        }.get(vol_regime, 1.0)

        spread_pct    = base * vol_mult
        spread_points = price * spread_pct

        bid = price - spread_points / 2
        ask = price + spread_points / 2

        return {
            "spread_pct"   : round(spread_pct * 100, 4),
            "spread_points": round(spread_points, 2),
            "bid"          : round(bid, 2),
            "ask"          : round(ask, 2),
            "mid"          : price
        }

# ── Brokerage Calculator ──────────────────────────────
def calculate_total_charges(price, quantity,
                              action,
                              instrument="equity"):
    cfg   = MARKET_CONFIG
    value = price * quantity

    # Brokerage
    brokerage = cfg["brokerage_flat"]

    # STT
    if instrument == "options":
        stt = value * cfg["stt_rate"] \
              if action == "sell" else 0
    elif instrument == "futures":
        stt = value * 0.0001
    else:
        stt = value * cfg["stt_rate"]

    # Other charges
    exchange = value * cfg["exchange_charge"]
    sebi     = value * cfg["sebi_rate"]
    stamp    = value * cfg["stamp_duty"]
    gst      = (brokerage + exchange) * cfg["gst_rate"]

    total    = (brokerage + stt + exchange +
                sebi + stamp + gst)

    return {
        "brokerage"     : round(brokerage, 2),
        "stt"           : round(stt,       2),
        "exchange"      : round(exchange,  2),
        "sebi"          : round(sebi,      4),
        "stamp_duty"    : round(stamp,     4),
        "gst"           : round(gst,       2),
        "total_charges" : round(total,     2),
        "charges_pct"   : round(
            total / value * 100, 4
        )
    }

# ── Order Class ───────────────────────────────────────
class Order:
    def __init__(self, symbol, action, quantity,
                 order_type="market",
                 limit_price=None,
                 stop_price=None,
                 instrument="equity"):
        self.order_id    = (
            f"ORD_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            f"_{random.randint(1000,9999)}"
        )
        self.symbol      = symbol
        self.action      = action
        self.quantity    = quantity
        self.order_type  = order_type
        self.limit_price = limit_price
        self.stop_price  = stop_price
        self.instrument  = instrument
        self.status      = "pending"
        self.filled_qty  = 0
        self.filled_price = None
        self.charges     = None
        self.slippage    = None
        self.timestamp   = datetime.now()
        self.fill_time   = None
        self.rejection_reason = None

    def to_dict(self):
        return {
            "order_id"        : self.order_id,
            "symbol"          : self.symbol,
            "action"          : self.action,
            "quantity"        : self.quantity,
            "order_type"      : self.order_type,
            "limit_price"     : self.limit_price,
            "stop_price"      : self.stop_price,
            "instrument"      : self.instrument,
            "status"          : self.status,
            "filled_qty"      : self.filled_qty,
            "filled_price"    : self.filled_price,
            "charges"         : self.charges,
            "slippage"        : self.slippage,
            "timestamp"       : str(self.timestamp),
            "fill_time"       : str(
                self.fill_time
            ) if self.fill_time else None,
            "rejection_reason": self.rejection_reason
        }

# ── Execution Engine ──────────────────────────────────
class ExecutionEngine:
    def __init__(self):
        self.slippage_model    = SlippageModel()
        self.partial_fill      = PartialFillModel()
        self.spread_model      = SpreadModel()
        self.execution_log     = []
        self.total_charges_paid = 0.0
        self.total_slippage_cost = 0.0

    def execute_order(self, order, market_price,
                       vol_regime="normal",
                       liquidity="high"):
        print(f"\n  📋 Executing Order: "
              f"{order.order_id}")
        print(f"     {order.action.upper()} "
              f"{order.quantity} {order.symbol} "
              f"@ ₹{market_price} "
              f"[{order.order_type}]")

        now = datetime.now()

        # Check market hours
        market_open  = MARKET_CONFIG["market_open"]
        market_close = MARKET_CONFIG["market_close"]

        if not (market_open <=
                now.time() <= market_close):
            order.status           = "rejected"
            order.rejection_reason = "Market closed"
            print(f"     ❌ Rejected: Market closed")
            return order

        # Get spread
        spread = self.spread_model.get_spread(
            market_price, order.symbol, vol_regime
        )

        # Determine execution price
        if order.order_type == "market":
            base_price = (
                spread["ask"]
                if order.action == "buy"
                else spread["bid"]
            )
        elif order.order_type == "limit":
            if order.limit_price:
                if (order.action == "buy" and
                        market_price <=
                        order.limit_price):
                    base_price = order.limit_price
                elif (order.action == "sell" and
                        market_price >=
                        order.limit_price):
                    base_price = order.limit_price
                else:
                    order.status           = "pending"
                    order.rejection_reason = (
                        "Limit not reached"
                    )
                    print(
                        f"     ⏳ Limit not reached — "
                        f"order pending"
                    )
                    return order
            else:
                base_price = market_price
        else:
            base_price = market_price

        # Apply slippage
        filled_price, slip_info = (
            self.slippage_model.apply_slippage(
                base_price,
                order.action,
                order.quantity,
                vol_regime,
                order.order_type,
                now.time()
            )
        )

        # Simulate partial fill
        fill_info = self.partial_fill.simulate_fill(
            order.quantity, filled_price,
            vol_regime, liquidity
        )

        # Calculate charges
        charges = calculate_total_charges(
            filled_price,
            fill_info["filled_qty"],
            order.action,
            order.instrument
        )

        # Update order
        order.status       = (
            "partial" if fill_info["partial_fill"]
            else "filled"
        )
        order.filled_qty   = fill_info["filled_qty"]
        order.filled_price = filled_price
        order.charges      = charges
        order.slippage     = slip_info
        order.fill_time    = now

        # Track costs
        self.total_charges_paid  += (
            charges["total_charges"]
        )
        self.total_slippage_cost += (
            abs(filled_price - market_price) *
            fill_info["filled_qty"]
        )

        # Log execution
        self.execution_log.append(order.to_dict())

        # Print result
        status_icon = (
            "✅" if order.status == "filled"
            else "⚠️"
        )
        print(
            f"     {status_icon} "
            f"Status        : {order.status.upper()}"
        )
        print(
            f"     📍 Filled Price  : "
            f"₹{filled_price}"
        )
        print(
            f"     📦 Filled Qty    : "
            f"{fill_info['filled_qty']}/"
            f"{order.quantity}"
        )
        print(
            f"     📉 Slippage      : "
            f"{slip_info['slip_pct']}% "
            f"(₹{slip_info['slip_points']})"
        )
        print(
            f"     💸 Total Charges : "
            f"₹{charges['total_charges']}"
        )
        print(
            f"     📊 Spread        : "
            f"{spread['spread_pct']}% "
            f"(₹{spread['spread_points']})"
        )

        if fill_info["partial_fill"]:
            print(
                f"     ⚠ Partial fill — "
                f"{fill_info['unfilled_qty']} "
                f"units unfilled"
            )

        return order

    def simulate_trade(self, signal,
                        market_price,
                        vol_regime="normal",
                        liquidity="high"):
        symbol   = signal.get("symbol", "")
        action   = signal.get("action",  "hold")
        lot_size = LOT_SIZES.get(
            symbol, LOT_SIZES["DEFAULT"]
        )

        # Calculate quantity from signal
        rec_size = float(
            signal.get("recommended_size", 0)
        )
        if rec_size > 0 and market_price > 0:
            raw_qty  = int(rec_size / market_price)
            # Round to lot size
            lots     = max(1, raw_qty // lot_size)
            quantity = lots * lot_size
        else:
            quantity = lot_size

        print(f"\n  {'=' * 55}")
        print(f"  🚀 Simulating Trade: {symbol}")
        print(f"  {'=' * 55}")
        print(f"  Signal       : {action.upper()}")
        print(f"  Market Price : ₹{market_price}")
        print(f"  Quantity     : {quantity} "
              f"({quantity//lot_size} lots)")
        print(f"  Vol Regime   : {vol_regime}")
        print(f"  Liquidity    : {liquidity}")

        if action not in ["buy", "sell"]:
            print(f"  ⏭ No trade — action is {action}")
            return None

        # Entry order
        entry_order = Order(
            symbol     = symbol,
            action     = action,
            quantity   = quantity,
            order_type = "market"
        )

        entry_order = self.execute_order(
            entry_order, market_price,
            vol_regime, liquidity
        )

        # Calculate stop loss order
        sl_str   = signal.get("stop_loss", "N/A")
        sl_price = None

        try:
            sl_price = float(
                str(sl_str).replace("₹", "").strip()
            )
        except Exception:
            pass

        # Simulate exit at stop loss or target
        target_str   = signal.get("target", "N/A")
        target_price = None

        try:
            target_price = float(
                str(target_str).replace(
                    "₹", ""
                ).strip()
            )
        except Exception:
            pass

        # Build execution summary
        filled_price = entry_order.filled_price or \
                       market_price
        filled_qty   = entry_order.filled_qty or \
                       quantity
        charges      = entry_order.charges or {}
        slip         = entry_order.slippage or {}

        trade_value  = filled_price * filled_qty
        total_cost   = charges.get(
            "total_charges", 0
        )

        # Projected PnL scenarios
        scenarios = {}

        if sl_price and target_price:
            if action == "buy":
                sl_pnl  = (
                    (sl_price - filled_price) *
                    filled_qty - total_cost * 2
                )
                tgt_pnl = (
                    (target_price - filled_price) *
                    filled_qty - total_cost * 2
                )
            else:
                sl_pnl  = (
                    (filled_price - sl_price) *
                    filled_qty - total_cost * 2
                )
                tgt_pnl = (
                    (filled_price - target_price) *
                    filled_qty - total_cost * 2
                )

            rr_ratio = (
                abs(tgt_pnl) / abs(sl_pnl)
                if sl_pnl != 0 else 0
            )

            scenarios = {
                "stop_loss_hit" : round(sl_pnl,  2),
                "target_hit"    : round(tgt_pnl, 2),
                "rr_ratio"      : round(rr_ratio, 2),
                "breakeven"     : round(
                    filled_price + (
                        total_cost * 2 / filled_qty
                    ) * (1 if action == "buy" else -1),
                    2
                )
            }

        result = {
            "symbol"        : symbol,
            "action"        : action,
            "timestamp"     : datetime.now().isoformat(),
            "market_price"  : market_price,
            "filled_price"  : filled_price,
            "quantity"      : filled_qty,
            "lots"          : filled_qty // lot_size,
            "trade_value"   : round(trade_value, 2),
            "charges"       : charges,
            "slippage"      : slip,
            "total_cost"    : round(total_cost, 2),
            "stop_loss"     : sl_price,
            "target"        : target_price,
            "scenarios"     : scenarios,
            "order_status"  : entry_order.status,
            "vol_regime"    : vol_regime,
            "liquidity"     : liquidity
        }

        # Print summary
        print(f"\n  {'─' * 55}")
        print(f"  EXECUTION SUMMARY")
        print(f"  {'─' * 55}")
        print(f"  Trade Value    : ₹{trade_value:,.2f}")
        print(f"  Total Charges  : ₹{total_cost:,.2f}")
        print(
            f"  Charges %      : "
            f"{charges.get('charges_pct', 0)}%"
        )

        if scenarios:
            print(f"\n  Projected Scenarios:")
            print(
                f"  ✅ If Target Hit  : "
                f"₹{scenarios['target_hit']:,.2f}"
            )
            print(
                f"  ❌ If Stop Hit    : "
                f"₹{scenarios['stop_loss_hit']:,.2f}"
            )
            print(
                f"  ⚖️  Risk/Reward    : "
                f"{scenarios['rr_ratio']}x"
            )
            print(
                f"  🎯 Breakeven      : "
                f"₹{scenarios['breakeven']}"
            )

        return result

    def get_execution_stats(self):
        if not self.execution_log:
            return {}

        filled  = [
            o for o in self.execution_log
            if o["status"] in ["filled", "partial"]
        ]
        partial = [
            o for o in self.execution_log
            if o["status"] == "partial"
        ]
        rejected = [
            o for o in self.execution_log
            if o["status"] == "rejected"
        ]

        return {
            "total_orders"      : len(
                self.execution_log
            ),
            "filled_orders"     : len(filled),
            "partial_fills"     : len(partial),
            "rejected_orders"   : len(rejected),
            "fill_rate"         : round(
                len(filled) /
                len(self.execution_log) * 100, 1
            ) if self.execution_log else 0,
            "total_charges_paid": round(
                self.total_charges_paid, 2
            ),
            "total_slippage_cost": round(
                self.total_slippage_cost, 2
            ),
            "avg_slippage_pct"  : round(
                sum(
                    o.get("slippage", {}).get(
                        "slip_pct", 0
                    )
                    for o in filled
                ) / len(filled), 4
            ) if filled else 0
        }

# ── Simulate Full Day Execution ───────────────────────
def simulate_day_execution(signals,
                             market_prices,
                             vol_regime="normal"):
    print("\n" + "🔥" * 27)
    print("  EXECUTION SIMULATOR — Daily Run")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔥" * 27)

    engine  = ExecutionEngine()
    results = []

    for signal in signals:
        symbol = signal.get("symbol", "")
        action = signal.get("action",  "hold")

        if action not in ["buy", "sell"]:
            continue

        price = market_prices.get(symbol, 0)
        if not price:
            print(f"\n  ⚠ No price for {symbol}")
            continue

        result = engine.simulate_trade(
            signal, price, vol_regime
        )

        if result:
            results.append(result)

    # Execution stats
    stats = engine.get_execution_stats()

    print(f"\n{'=' * 55}")
    print(f"  EXECUTION STATISTICS")
    print(f"{'=' * 55}")
    print(f"  Total Orders    : "
          f"{stats.get('total_orders', 0)}")
    print(f"  Fill Rate       : "
          f"{stats.get('fill_rate', 0)}%")
    print(f"  Partial Fills   : "
          f"{stats.get('partial_fills', 0)}")
    print(f"  Total Charges   : "
          f"₹{stats.get('total_charges_paid', 0):,.2f}")
    print(f"  Total Slippage  : "
          f"₹{stats.get('total_slippage_cost', 0):,.2f}")
    print(f"  Avg Slippage %  : "
          f"{stats.get('avg_slippage_pct', 0)}%")

    # Save results
    path = os.path.join(
        OUTPUT_DIR,
        f"executions_"
        f"{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    )
    with open(path, "w") as f:
        json.dump({
            "timestamp" : datetime.now().isoformat(),
            "results"   : results,
            "stats"     : stats,
            "log"       : engine.execution_log
        }, f, indent=2, default=str)

    print(f"\n  ✅ Execution log → {path}")
    return results, stats

# ── Main Test ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Trading AI — Execution Simulator")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    # Test signals
    test_signals = [
        {
            "symbol"          : "NIFTY",
            "action"          : "buy",
            "confidence"      : 0.72,
            "risk_level"      : "medium",
            "position_size"   : "half",
            "recommended_size": 50000,
            "entry_zone"      : "23700-23750",
            "stop_loss"       : "23500",
            "target"          : "24000"
        },
        {
            "symbol"          : "BANKNIFTY",
            "action"          : "sell",
            "confidence"      : 0.68,
            "risk_level"      : "medium",
            "position_size"   : "half",
            "recommended_size": 45000,
            "entry_zone"      : "51500-51600",
            "stop_loss"       : "52000",
            "target"          : "50500"
        },
        {
            "symbol"          : "RELIANCE",
            "action"          : "buy",
            "confidence"      : 0.75,
            "risk_level"      : "low",
            "position_size"   : "full",
            "recommended_size": 30000,
            "entry_zone"      : "1380-1400",
            "stop_loss"       : "1350",
            "target"          : "1450"
        }
    ]

    # Market prices
    market_prices = {
        "NIFTY"    : 23719.3,
        "BANKNIFTY": 51650.0,
        "RELIANCE" : 1392.5
    }

    # Run simulation
    results, stats = simulate_day_execution(
        test_signals,
        market_prices,
        vol_regime="normal"
    )

    print(f"\n  Simulated {len(results)} trades")
    print(f"  ✅ Execution Simulator complete!")
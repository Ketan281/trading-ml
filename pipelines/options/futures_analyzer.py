import os
import sys
import json
import pandas as pd
import numpy as np
from datetime        import datetime, timedelta
from jugaad_data.nse import NSELive

ROOT = os.path.dirname(os.path.dirname(
       os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

OUTPUT_DIR = os.path.join(ROOT, "data", "options")
os.makedirs(OUTPUT_DIR, exist_ok=True)

nse = NSELive()

# ── Futures Symbol Map ────────────────────────────────
FUTURES_MAP = {
    "NIFTY"    : "NIFTY",
    "BANKNIFTY": "BANKNIFTY"
}

INDEX_MAP = {
    "NIFTY"    : "NIFTY 50",
    "BANKNIFTY": "NIFTY BANK"
}

# ── Get Spot Price ────────────────────────────────────
def get_spot(symbol):
    try:
        data  = nse.live_index(INDEX_MAP[symbol])
        price = data["data"][0]["lastPrice"]
        return float(price)
    except Exception as e:
        print(f"  ❌ Spot fetch failed: {e}")
        return None

# ── Get Futures Data ──────────────────────────────────
def get_futures_data(symbol):
    try:
        print(f"  📊 Fetching {symbol} futures data...")

        # Use stock_quote_fno for futures data
        fno_data = nse.stock_quote_fno(symbol)

        futures  = []

        # Extract futures contracts
        if "stocks" in fno_data:
            for item in fno_data["stocks"]:
                metadata = item.get("metadata", {})
                if metadata.get(
                    "instrumentType"
                ) == "Index Futures":
                    futures.append({
                        "symbol"     : symbol,
                        "expiry"     : metadata.get(
                                           "expiryDate"),
                        "ltp"        : metadata.get(
                                           "lastPrice", 0),
                        "open"       : metadata.get(
                                           "openPrice", 0),
                        "high"       : metadata.get(
                                           "highPrice", 0),
                        "low"        : metadata.get(
                                           "lowPrice",  0),
                        "prev_close" : metadata.get(
                                           "prevClose", 0),
                        "change"     : metadata.get(
                                           "change",    0),
                        "pchange"    : metadata.get(
                                           "pChange",   0),
                        "oi"         : metadata.get(
                                           "openInterest", 0),
                        "chng_oi"    : metadata.get(
                                           "changeinOpenInterest", 0),
                        "volume"     : metadata.get(
                                           "numberOfContractsTraded", 0),
                        "value"      : metadata.get(
                                           "totalTurnover", 0)
                    })

        if futures:
            print(f"  ✅ Found {len(futures)} "
                  f"futures contracts")
        else:
            print(f"  ⚠ No futures found via FNO quote")
            # Fallback — try live_fno
            futures = get_futures_live_fno(symbol)

        return futures

    except Exception as e:
        print(f"  ❌ Futures fetch failed: {e}")
        import traceback
        traceback.print_exc()
        return get_futures_live_fno(symbol)

# ── Fallback via live_fno ─────────────────────────────
def get_futures_live_fno(symbol):
    try:
        print(f"  🔄 Trying live_fno fallback...")
        data     = nse.live_fno()
        futures  = []

        for item in data.get("data", []):
            if (symbol in item.get("symbol", "") and
                    "FUT" in item.get(
                        "instrumentType", "")):
                futures.append({
                    "symbol"   : symbol,
                    "expiry"   : item.get("expiryDate"),
                    "ltp"      : item.get("lastPrice",  0),
                    "oi"       : item.get(
                                     "openInterest",    0),
                    "chng_oi"  : item.get(
                                     "changeinOpenInterest", 0),
                    "volume"   : item.get(
                                     "totalTradedVolume", 0),
                    "change"   : item.get("change",     0),
                    "pchange"  : item.get("pChange",    0)
                })

        if futures:
            print(f"  ✅ Found {len(futures)} contracts "
                  f"via live_fno")
        return futures

    except Exception as e:
        print(f"  ❌ live_fno also failed: {e}")
        return []

# ── Calculate Basis ───────────────────────────────────
def calculate_basis(spot, futures):
    if not futures:
        return []

    basis_data = []

    for fut in futures:
        fut_price = float(fut.get("ltp", 0))
        if fut_price == 0:
            continue

        basis        = round(fut_price - spot, 2)
        basis_pct    = round(basis / spot * 100, 3)

        # Annualize basis if expiry available
        annualized   = None
        try:
            exp_str  = fut.get("expiry", "")
            exp_date = datetime.strptime(
                exp_str, "%d-%b-%Y"
            )
            dte      = max(
                (exp_date - datetime.now()).days, 1
            )
            annualized = round(
                basis_pct / dte * 365, 2
            )
        except Exception:
            dte = 30

        # Basis interpretation
        if basis > 0:
            basis_type = "contango"
            signal     = "bullish"
        elif basis < -20:
            basis_type = "deep_backwardation"
            signal     = "very_bearish"
        elif basis < 0:
            basis_type = "backwardation"
            signal     = "bearish"
        else:
            basis_type = "at_par"
            signal     = "neutral"

        basis_data.append({
            **fut,
            "spot"           : spot,
            "basis"          : basis,
            "basis_pct"      : basis_pct,
            "annualized_basis": annualized,
            "basis_type"     : basis_type,
            "signal"         : signal,
            "dte"            : dte
        })

    return basis_data

# ── Rollover Analysis ─────────────────────────────────
def analyze_rollover(futures):
    if len(futures) < 2:
        return None

    near_month = futures[0]
    next_month = futures[1] if len(futures) > 1 else None

    if not next_month:
        return None

    near_oi   = float(near_month.get("oi",      0))
    next_oi   = float(next_month.get("oi",      0))
    total_oi  = near_oi + next_oi

    # Rollover percentage
    rollover_pct = round(
        next_oi / total_oi * 100, 2
    ) if total_oi > 0 else 0

    # Near month OI change
    near_oi_chng = float(
        near_month.get("chng_oi", 0)
    )
    next_oi_chng = float(
        next_month.get("chng_oi", 0)
    )

    # Rollover interpretation
    if rollover_pct > 70:
        rollover_signal = "high_rollover_bullish"
        rollover_bias   = "positions_rolling_forward"
    elif rollover_pct > 50:
        rollover_signal = "normal_rollover"
        rollover_bias   = "neutral"
    elif rollover_pct > 30:
        rollover_signal = "low_rollover"
        rollover_bias   = "unwinding_positions"
    else:
        rollover_signal = "very_low_rollover_bearish"
        rollover_bias   = "closing_longs"

    # Cost of carry
    near_price = float(near_month.get("ltp", 0))
    next_price = float(next_month.get("ltp", 0))
    carry      = round(next_price - near_price, 2)

    if carry > 0:
        carry_signal = "positive_carry_bullish"
    else:
        carry_signal = "negative_carry_bearish"

    return {
        "near_month_expiry" : near_month.get("expiry"),
        "next_month_expiry" : next_month.get("expiry"),
        "near_month_oi"     : int(near_oi),
        "next_month_oi"     : int(next_oi),
        "total_oi"          : int(total_oi),
        "rollover_pct"      : rollover_pct,
        "rollover_signal"   : rollover_signal,
        "rollover_bias"     : rollover_bias,
        "near_oi_change"    : int(near_oi_chng),
        "next_oi_change"    : int(next_oi_chng),
        "cost_of_carry"     : carry,
        "carry_signal"      : carry_signal
    }

# ── OI Buildup in Futures ─────────────────────────────
def futures_oi_analysis(futures, spot):
    if not futures:
        return None

    near  = futures[0]
    oi    = float(near.get("oi",       0))
    chng  = float(near.get("chng_oi",  0))
    price = float(near.get("ltp",      0))
    pchng = float(near.get("pchange",  0))

    # Long/Short buildup detection
    if chng > 0 and pchng > 0:
        activity = "long_buildup"
        signal   = "bullish"
    elif chng > 0 and pchng < 0:
        activity = "short_buildup"
        signal   = "bearish"
    elif chng < 0 and pchng > 0:
        activity = "short_covering"
        signal   = "bullish"
    elif chng < 0 and pchng < 0:
        activity = "long_unwinding"
        signal   = "bearish"
    else:
        activity = "neutral"
        signal   = "neutral"

    # Premium/Discount to spot
    premium  = round(price - spot, 2)
    prem_pct = round(premium / spot * 100, 3)

    return {
        "futures_price"  : price,
        "spot_price"     : spot,
        "premium"        : premium,
        "premium_pct"    : prem_pct,
        "oi"             : int(oi),
        "oi_change"      : int(chng),
        "price_change_pct": pchng,
        "activity"       : activity,
        "signal"         : signal
    }

# ── Full Futures Analysis ─────────────────────────────
def full_futures_analysis(symbol):
    print(f"\n{'=' * 55}")
    print(f"  Futures Analysis — {symbol}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 55}")

    # Get spot
    spot = get_spot(symbol)
    if not spot:
        print(f"  ❌ Could not get spot price")
        return None

    print(f"  Spot Price : ₹{spot}")

    # Get futures data
    futures = get_futures_data(symbol)

    if not futures:
        print(f"  ⚠ No futures data available")
        print(f"  Building synthetic analysis from spot...")

        # Build minimal report from spot
        report = {
            "symbol"    : symbol,
            "timestamp" : datetime.now().isoformat(),
            "spot"      : spot,
            "note"      : "Futures data unavailable — "
                          "market may be closed",
            "basis"     : [],
            "rollover"  : None,
            "oi_analysis": None
        }

        path = os.path.join(
            OUTPUT_DIR,
            f"{symbol}_futures_"
            f"{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        )
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"  ✅ Minimal report saved → {path}")
        return report

    # Calculate basis
    basis_data  = calculate_basis(spot, futures)

    # Rollover analysis
    rollover    = analyze_rollover(futures)

    # OI analysis
    oi_analysis = futures_oi_analysis(futures, spot)

    # Print Futures Contracts
    print(f"\n  📋 Futures Contracts:")
    print(f"  {'EXPIRY':<15} {'LTP':<12} "
          f"{'BASIS':<10} {'OI':<12} {'OI CHG'}")
    print("  " + "─" * 58)

    for b in basis_data:
        print(
            f"  {str(b.get('expiry','')):<15} "
            f"₹{b.get('ltp', 0):<11} "
            f"{b.get('basis', 0):<10} "
            f"{int(b.get('oi', 0)):<12} "
            f"{int(b.get('chng_oi', 0))}"
        )

    # Print Basis Analysis
    if basis_data:
        near = basis_data[0]
        print(f"\n  📐 Basis Analysis (Near Month):")
        print(f"     Spot Price     : ₹{spot}")
        print(f"     Futures Price  : ₹{near.get('ltp')}")
        print(f"     Basis          : "
              f"₹{near.get('basis')} "
              f"({near.get('basis_pct')}%)")
        print(f"     Annualized     : "
              f"{near.get('annualized_basis')}%")
        print(f"     Basis Type     : "
              f"{near.get('basis_type').upper()}")
        print(f"     Signal         : "
              f"{near.get('signal').upper()}")

    # Print Rollover
    if rollover:
        print(f"\n  🔄 Rollover Analysis:")
        print(f"     Near Month OI  : "
              f"{rollover['near_month_oi']:,}")
        print(f"     Next Month OI  : "
              f"{rollover['next_month_oi']:,}")
        print(f"     Rollover %     : "
              f"{rollover['rollover_pct']}%")
        print(f"     Signal         : "
              f"{rollover['rollover_signal'].upper()}")
        print(f"     Bias           : "
              f"{rollover['rollover_bias'].upper()}")
        print(f"     Cost of Carry  : "
              f"₹{rollover['cost_of_carry']}")
        print(f"     Carry Signal   : "
              f"{rollover['carry_signal'].upper()}")

    # Print OI Analysis
    if oi_analysis:
        print(f"\n  📊 Futures OI Activity:")
        print(f"     Activity       : "
              f"{oi_analysis['activity'].upper()}")
        print(f"     Signal         : "
              f"{oi_analysis['signal'].upper()}")
        print(f"     OI             : "
              f"{oi_analysis['oi']:,}")
        print(f"     OI Change      : "
              f"{oi_analysis['oi_change']:,}")
        print(f"     Premium        : "
              f"₹{oi_analysis['premium']} "
              f"({oi_analysis['premium_pct']}%)")

    # Overall signal
    signals = []
    if basis_data:
        signals.append(basis_data[0].get("signal"))
    if oi_analysis:
        signals.append(oi_analysis.get("signal"))
    if rollover:
        if "bullish" in rollover.get(
            "rollover_signal", ""
        ):
            signals.append("bullish")
        elif "bearish" in rollover.get(
            "rollover_signal", ""
        ):
            signals.append("bearish")

    bull = signals.count("bullish")
    bear = signals.count("bearish")

    if bull > bear:
        overall = "🐂 BULLISH"
    elif bear > bull:
        overall = "🐻 BEARISH"
    else:
        overall = "😐 NEUTRAL"

    print(f"\n  {'─' * 45}")
    print(f"  Overall Futures Signal : {overall}")
    print(f"  {'─' * 45}")

    # Build report
    report = {
        "symbol"     : symbol,
        "timestamp"  : datetime.now().isoformat(),
        "spot"       : spot,
        "futures"    : futures,
        "basis"      : basis_data,
        "rollover"   : rollover,
        "oi_analysis": oi_analysis,
        "overall"    : overall.split()[1].lower()
    }

    # Save
    path = os.path.join(
        OUTPUT_DIR,
        f"{symbol}_futures_"
        f"{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    )
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  ✅ Futures analysis saved → {path}")
    return report

# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    for symbol in ["NIFTY", "BANKNIFTY"]:
        full_futures_analysis(symbol)
    print("\n  ✅ Futures Analysis complete!")
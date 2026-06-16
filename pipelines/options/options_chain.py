import os
import sys
import json
import pandas as pd
from datetime        import datetime
from jugaad_data.nse import NSELive

ROOT = os.path.dirname(os.path.dirname(
       os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

OUTPUT_DIR = os.path.join(ROOT, "data", "options")
os.makedirs(OUTPUT_DIR, exist_ok=True)

nse = NSELive()

# ── Index Name Mapping ────────────────────────────────
INDEX_MAP = {
    "NIFTY"    : "NIFTY 50",
    "BANKNIFTY": "NIFTY BANK"
}

# ── Strike Step ───────────────────────────────────────
def get_strike_step(symbol):
    return 50 if symbol == "NIFTY" else 100

def round_to_strike(price, step):
    return round(round(price / step) * step)

# ── Get Spot Price ────────────────────────────────────
def get_spot_price(symbol):
    try:
        data  = nse.live_index(INDEX_MAP[symbol])
        price = data["data"][0]["lastPrice"]
        print(f"  {symbol} Spot Price : ₹{price}")
        return price
    except Exception as e:
        print(f"  ❌ Spot price failed: {e}")
        return None

# ── Fetch Options Chain ───────────────────────────────
def fetch_options_chain(symbol):
    print(f"\n  📊 Fetching {symbol} options chain...")

    try:
        # Get spot price
        spot = get_spot_price(symbol)
        if not spot:
            return None

        # Get raw options chain
        raw     = nse.index_option_chain(symbol)
        records = raw["records"]

        # Extract expiry dates
        expiries = records["expiryDates"]
        print(f"  Available Expiries : {expiries[:4]}")

        # Use nearest expiry
        nearest  = expiries[0]
        print(f"  Using Expiry       : {nearest}")

        # ATM strike
        step = get_strike_step(symbol)
        atm  = round_to_strike(spot, step)
        print(f"  ATM Strike         : {atm}")

        # Debug first item structure
        if records["data"]:
            first     = records["data"][0]
            first_exp = first.get("expiryDates", [])
            print(f"  Item expiry field  : {first_exp}")

        # Parse chain data
        chain = []
        for item in records["data"]:

            # Fix — use expiryDates list not expiryDate
            item_expiries = item.get("expiryDates", [])
            if nearest not in item_expiries:
                continue

            strike = item.get("strikePrice", 0)
            if strike == 0:
                continue

            row = {
                "strike"   : strike,
                "expiry"   : nearest,
                "spot"     : spot,
                "atm"      : atm,
                "moneyness": (
                    "ATM" if strike == atm
                    else "OTM" if strike > atm
                    else "ITM"
                )
            }

            # Call side
            if "CE" in item:
                ce = item["CE"]
                row.update({
                    "ce_ltp"    : ce.get("lastPrice",            0),
                    "ce_oi"     : ce.get("openInterest",         0),
                    "ce_chng_oi": ce.get("changeinOpenInterest", 0),
                    "ce_volume" : ce.get("totalTradedVolume",    0),
                    "ce_iv"     : ce.get("impliedVolatility",    0),
                    "ce_bid"    : ce.get("bidprice",             0),
                    "ce_ask"    : ce.get("askPrice",             0),
                    "ce_delta"  : ce.get("delta",                0),
                    "ce_theta"  : ce.get("theta",                0),
                    "ce_gamma"  : ce.get("gamma",                0),
                    "ce_vega"   : ce.get("vega",                 0)
                })
            else:
                row.update({
                    "ce_ltp"    : 0, "ce_oi"     : 0,
                    "ce_chng_oi": 0, "ce_volume" : 0,
                    "ce_iv"     : 0, "ce_bid"    : 0,
                    "ce_ask"    : 0, "ce_delta"  : 0,
                    "ce_theta"  : 0, "ce_gamma"  : 0,
                    "ce_vega"   : 0
                })

            # Put side
            if "PE" in item:
                pe = item["PE"]
                row.update({
                    "pe_ltp"    : pe.get("lastPrice",            0),
                    "pe_oi"     : pe.get("openInterest",         0),
                    "pe_chng_oi": pe.get("changeinOpenInterest", 0),
                    "pe_volume" : pe.get("totalTradedVolume",    0),
                    "pe_iv"     : pe.get("impliedVolatility",    0),
                    "pe_bid"    : pe.get("bidprice",             0),
                    "pe_ask"    : pe.get("askPrice",             0),
                    "pe_delta"  : pe.get("delta",                0),
                    "pe_theta"  : pe.get("theta",                0),
                    "pe_gamma"  : pe.get("gamma",                0),
                    "pe_vega"   : pe.get("vega",                 0)
                })
            else:
                row.update({
                    "pe_ltp"    : 0, "pe_oi"     : 0,
                    "pe_chng_oi": 0, "pe_volume" : 0,
                    "pe_iv"     : 0, "pe_bid"    : 0,
                    "pe_ask"    : 0, "pe_delta"  : 0,
                    "pe_theta"  : 0, "pe_gamma"  : 0,
                    "pe_vega"   : 0
                })

            chain.append(row)

        print(f"  Matched strikes    : {len(chain)}")

        if len(chain) == 0:
            print("  ⚠ No strikes matched. Dumping sample:")
            for i, item in enumerate(records["data"][:3]):
                print(f"    Item {i}: "
                      f"expiryDates={item.get('expiryDates')} "
                      f"strike={item.get('strikePrice')}")
            return None

        # Build DataFrame
        df   = pd.DataFrame(chain)
        df   = df.sort_values("strike").reset_index(drop=True)

        # Save CSV
        path = os.path.join(
            OUTPUT_DIR,
            f"{symbol}_chain_"
            f"{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        )
        df.to_csv(path, index=False)

        print(f"  ✅ {len(chain)} strikes fetched")
        print(f"  ✅ Saved → {path}")

        return {
            "symbol"  : symbol,
            "spot"    : spot,
            "atm"     : atm,
            "expiry"  : nearest,
            "expiries": expiries[:4],
            "strikes" : len(chain),
            "df"      : df,
            "chain"   : chain,
            "path"    : path
        }

    except Exception as e:
        print(f"  ❌ Options chain fetch failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# ── Print Chain Preview ───────────────────────────────
def print_chain_preview(result):
    if not result:
        print("  ⚠ No result to preview")
        return

    symbol = result["symbol"]
    spot   = result["spot"]
    atm    = result["atm"]
    step   = get_strike_step(symbol)
    df     = result["df"]

    if df.empty:
        print("  ⚠ Empty dataframe")
        return

    print(f"\n  📊 {symbol} Chain Preview "
          f"(ATM ± 5 strikes)")
    print(f"  Spot: ₹{spot} | ATM: {atm} | "
          f"Expiry: {result['expiry']}")
    print()
    print(f"  {'STRIKE':<10} {'CE OI':<12} {'CE IV':<8} "
          f"{'CE LTP':<10} {'PE LTP':<10} "
          f"{'PE OI':<12} {'PE IV':<8}")
    print("  " + "─" * 72)

    nearby = df[
        (df["strike"] >= atm - 5 * step) &
        (df["strike"] <= atm + 5 * step)
    ]

    for _, row in nearby.iterrows():
        marker = " ◄ ATM" if row["strike"] == atm else ""
        print(
            f"  {int(row['strike']):<10} "
            f"{int(row.get('ce_oi', 0)):<12} "
            f"{row.get('ce_iv', 0):<8} "
            f"{row.get('ce_ltp', 0):<10} "
            f"{row.get('pe_ltp', 0):<10} "
            f"{int(row.get('pe_oi', 0)):<12} "
            f"{row.get('pe_iv', 0):<8}"
            f"{marker}"
        )

    # OI Summary
    total_ce_oi = df["ce_oi"].sum()
    total_pe_oi = df["pe_oi"].sum()
    pcr         = round(total_pe_oi / total_ce_oi, 3) \
                  if total_ce_oi > 0 else 0

    print()
    print(f"  {'─' * 50}")
    print(f"  Total CE OI  : {int(total_ce_oi):,}")
    print(f"  Total PE OI  : {int(total_pe_oi):,}")
    print(f"  PCR (PE/CE)  : {pcr}")
    print(f"  Sentiment    : "
          f"{'🐂 Bullish' if pcr > 1.2 else '🐻 Bearish' if pcr < 0.8 else '😐 Neutral'}"
    )

    # Max Pain
    max_pain = calculate_max_pain(df)
    print(f"  Max Pain     : {max_pain}")
    print(f"  {'─' * 50}")

# ── Calculate Max Pain ────────────────────────────────
def calculate_max_pain(df):
    try:
        max_pain_strike = None
        min_pain        = float("inf")

        for target in df["strike"]:
            # Loss to CE writers
            ce_loss = df[df["strike"] < target]["ce_oi"].sum()
            # Loss to PE writers
            pe_loss = df[df["strike"] > target]["pe_oi"].sum()
            total   = ce_loss + pe_loss

            if total < min_pain:
                min_pain        = total
                max_pain_strike = target

        return max_pain_strike
    except Exception:
        return "N/A"

# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Trading AI — Options Chain Fetcher")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    for symbol in ["NIFTY", "BANKNIFTY"]:
        result = fetch_options_chain(symbol)
        print_chain_preview(result)

    print("\n  ✅ Options chain fetch complete!")
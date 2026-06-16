import os
import sys
import json
import pandas as pd
import numpy as np
from datetime        import datetime
from jugaad_data.nse import NSELive

ROOT = os.path.dirname(os.path.dirname(
       os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

sys.path.insert(0, os.path.join(ROOT, "pipelines", "options"))
from options_chain import (fetch_options_chain,
                            get_strike_step,
                            round_to_strike)

OUTPUT_DIR = os.path.join(ROOT, "data", "options")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── IV Surface Analysis ───────────────────────────────
def analyze_iv_surface(df, symbol):
    spot = df["spot"].iloc[0]
    step = get_strike_step(symbol)
    atm  = round_to_strike(spot, step)

    # ATM IV
    atm_row    = df[df["strike"] == atm]
    atm_ce_iv  = float(atm_row["ce_iv"].values[0]) \
                 if not atm_row.empty else 0
    atm_pe_iv  = float(atm_row["pe_iv"].values[0]) \
                 if not atm_row.empty else 0
    atm_iv     = round((atm_ce_iv + atm_pe_iv) / 2, 2)

    # OTM IV (5 strikes away)
    otm_ce_strike = atm + 5 * step
    otm_pe_strike = atm - 5 * step

    otm_ce_row = df[df["strike"] == otm_ce_strike]
    otm_pe_row = df[df["strike"] == otm_pe_strike]

    otm_ce_iv  = float(otm_ce_row["ce_iv"].values[0]) \
                 if not otm_ce_row.empty else 0
    otm_pe_iv  = float(otm_pe_row["pe_iv"].values[0]) \
                 if not otm_pe_row.empty else 0

    # IV Skew — PE IV vs CE IV
    # Positive skew = puts more expensive = fear
    # Negative skew = calls more expensive = greed
    iv_skew = round(atm_pe_iv - atm_ce_iv, 2)

    # Average IV across all strikes
    all_iv    = pd.concat([
        df["ce_iv"][df["ce_iv"] > 0],
        df["pe_iv"][df["pe_iv"] > 0]
    ])
    avg_iv    = round(float(all_iv.mean()), 2) \
                if not all_iv.empty else 0
    max_iv    = round(float(all_iv.max()),  2) \
                if not all_iv.empty else 0
    min_iv    = round(float(all_iv.min()),  2) \
                if not all_iv.empty else 0

    # IV Rank (0-100) within today's range
    iv_rank   = round(
        (atm_iv - min_iv) / (max_iv - min_iv) * 100, 1
    ) if max_iv > min_iv else 50

    # IV interpretation
    if atm_iv > 20:
        iv_regime = "very_high"
        iv_signal = "sell_options"
    elif atm_iv > 15:
        iv_regime = "high"
        iv_signal = "prefer_selling"
    elif atm_iv > 10:
        iv_regime = "normal"
        iv_signal = "neutral"
    elif atm_iv > 7:
        iv_regime = "low"
        iv_signal = "prefer_buying"
    else:
        iv_regime = "very_low"
        iv_signal = "buy_options"

    # Skew interpretation
    if iv_skew > 2:
        skew_signal = "fear_in_market"
    elif iv_skew > 0:
        skew_signal = "slight_caution"
    elif iv_skew > -2:
        skew_signal = "slight_greed"
    else:
        skew_signal = "extreme_greed"

    return {
        "atm_iv"      : atm_iv,
        "atm_ce_iv"   : atm_ce_iv,
        "atm_pe_iv"   : atm_pe_iv,
        "otm_ce_iv"   : otm_ce_iv,
        "otm_pe_iv"   : otm_pe_iv,
        "avg_iv"      : avg_iv,
        "max_iv"      : max_iv,
        "min_iv"      : min_iv,
        "iv_rank"     : iv_rank,
        "iv_skew"     : iv_skew,
        "iv_regime"   : iv_regime,
        "iv_signal"   : iv_signal,
        "skew_signal" : skew_signal
    }

# ── IV Smile / Skew Curve ─────────────────────────────
def build_iv_curve(df, symbol, depth=8):
    spot = df["spot"].iloc[0]
    step = get_strike_step(symbol)
    atm  = round_to_strike(spot, step)

    strikes = [
        atm + (i * step)
        for i in range(-depth, depth + 1)
    ]

    curve = []
    for strike in strikes:
        row = df[df["strike"] == strike]
        if row.empty:
            continue

        ce_iv  = float(row["ce_iv"].values[0])
        pe_iv  = float(row["pe_iv"].values[0])
        avg_iv = round((ce_iv + pe_iv) / 2, 2) \
                 if ce_iv > 0 and pe_iv > 0 \
                 else max(ce_iv, pe_iv)

        distance = round((strike - atm) / step)

        curve.append({
            "strike"  : strike,
            "distance": distance,
            "ce_iv"   : ce_iv,
            "pe_iv"   : pe_iv,
            "avg_iv"  : avg_iv,
            "label"   : (
                f"ATM" if distance == 0
                else f"ATM+{distance}" if distance > 0
                else f"ATM{distance}"
            )
        })

    return curve

# ── Precise Max Pain ──────────────────────────────────
def calculate_max_pain_detailed(df):
    try:
        results = []
        strikes = df["strike"].unique()

        for target in strikes:
            # CE writers lose when price > strike
            ce_pain = 0
            for _, row in df[
                df["strike"] < target
            ].iterrows():
                ce_pain += (
                    (target - row["strike"]) * row["ce_oi"]
                )

            # PE writers lose when price < strike
            pe_pain = 0
            for _, row in df[
                df["strike"] > target
            ].iterrows():
                pe_pain += (
                    (row["strike"] - target) * row["pe_oi"]
                )

            results.append({
                "strike"    : target,
                "ce_pain"   : ce_pain,
                "pe_pain"   : pe_pain,
                "total_pain": ce_pain + pe_pain
            })

        pain_df  = pd.DataFrame(results).sort_values(
            "total_pain"
        )

        max_pain         = int(pain_df.iloc[0]["strike"])
        second_max_pain  = int(pain_df.iloc[1]["strike"]) \
                           if len(pain_df) > 1 else max_pain

        spot = df["spot"].iloc[0]
        dist = round(
            abs(spot - max_pain) / spot * 100, 2
        )

        # Max pain signal
        if max_pain > spot * 1.01:
            mp_signal = "price_likely_rise"
        elif max_pain < spot * 0.99:
            mp_signal = "price_likely_fall"
        else:
            mp_signal = "price_near_max_pain"

        return {
            "max_pain"        : max_pain,
            "second_max_pain" : second_max_pain,
            "spot"            : spot,
            "distance_pct"    : dist,
            "signal"          : mp_signal,
            "pain_table"      : pain_df.head(5).to_dict(
                                    "records"
                                )
        }

    except Exception as e:
        print(f"  ⚠ Max pain failed: {e}")
        return None

# ── Expected Move ─────────────────────────────────────
def calculate_expected_move(df, spot, atm_iv,
                             days_to_expiry=7):
    try:
        # Expected move = Spot × ATM IV × √(DTE/365)
        daily_move = spot * (atm_iv / 100) * \
                     np.sqrt(days_to_expiry / 365)
        daily_move = round(daily_move, 2)

        upper = round(spot + daily_move, 2)
        lower = round(spot - daily_move, 2)

        return {
            "expected_move"  : daily_move,
            "upper_range"    : upper,
            "lower_range"    : lower,
            "days_to_expiry" : days_to_expiry,
            "interpretation" : (
                f"Market expects ±{daily_move} points "
                f"move in {days_to_expiry} days"
            )
        }
    except Exception as e:
        return None

# ── Full IV Analysis ──────────────────────────────────
def full_iv_analysis(symbol):
    print(f"\n{'=' * 55}")
    print(f"  IV + Max Pain Analysis — {symbol}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 55}")

    # Fetch chain
    result = fetch_options_chain(symbol)
    if not result or result["df"].empty:
        print(f"  ❌ Could not fetch chain for {symbol}")
        return None

    df      = result["df"]
    spot    = result["spot"]
    atm     = result["atm"]
    expiry  = result["expiry"]

    # Calculate days to expiry
    try:
        exp_date = datetime.strptime(expiry, "%d-%b-%Y")
        dte      = max((exp_date - datetime.now()).days, 1)
    except Exception:
        dte      = 7

    print(f"  Days to Expiry   : {dte}")

    # Run analyses
    iv_surface = analyze_iv_surface(df, symbol)
    iv_curve   = build_iv_curve(df, symbol)
    max_pain   = calculate_max_pain_detailed(df)
    exp_move   = calculate_expected_move(
                     df, spot,
                     iv_surface["atm_iv"], dte
                 )

    # Print IV Surface
    print(f"\n  📊 IV Surface:")
    print(f"     ATM IV         : {iv_surface['atm_iv']}%")
    print(f"     ATM CE IV      : {iv_surface['atm_ce_iv']}%")
    print(f"     ATM PE IV      : {iv_surface['atm_pe_iv']}%")
    print(f"     Average IV     : {iv_surface['avg_iv']}%")
    print(f"     IV Rank        : {iv_surface['iv_rank']}/100")
    print(f"     IV Regime      : "
          f"{iv_surface['iv_regime'].upper()}")
    print(f"     IV Signal      : "
          f"{iv_surface['iv_signal'].upper()}")

    # Print IV Skew
    print(f"\n  📐 IV Skew:")
    print(f"     Skew (PE-CE)   : {iv_surface['iv_skew']}%")
    print(f"     Skew Signal    : "
          f"{iv_surface['skew_signal'].upper()}")
    print(f"     OTM CE IV      : {iv_surface['otm_ce_iv']}%")
    print(f"     OTM PE IV      : {iv_surface['otm_pe_iv']}%")

    # Print IV Smile Curve
    print(f"\n  📈 IV Smile Curve:")
    print(f"  {'LABEL':<12} {'STRIKE':<10} "
          f"{'CE IV':<10} {'PE IV':<10} {'AVG IV'}")
    print("  " + "─" * 50)
    for point in iv_curve:
        marker = " ◄" if point["distance"] == 0 else ""
        print(
            f"  {point['label']:<12} "
            f"{point['strike']:<10} "
            f"{point['ce_iv']:<10} "
            f"{point['pe_iv']:<10} "
            f"{point['avg_iv']}"
            f"{marker}"
        )

    # Print Max Pain
    if max_pain:
        print(f"\n  😣 Max Pain Analysis:")
        print(f"     Max Pain       : {max_pain['max_pain']}")
        print(f"     2nd Max Pain   : "
              f"{max_pain['second_max_pain']}")
        print(f"     Distance       : "
              f"{max_pain['distance_pct']}% from spot")
        print(f"     Signal         : "
              f"{max_pain['signal'].upper()}")

    # Print Expected Move
    if exp_move:
        print(f"\n  📏 Expected Move ({dte} days):")
        print(f"     Expected Move  : "
              f"±₹{exp_move['expected_move']}")
        print(f"     Upper Range    : ₹{exp_move['upper_range']}")
        print(f"     Lower Range    : ₹{exp_move['lower_range']}")
        print(f"     Interpretation : "
              f"{exp_move['interpretation']}")

    # Overall IV signal
    print(f"\n  {'─' * 45}")
    print(f"  IV Regime      : "
          f"{iv_surface['iv_regime'].upper()}")
    print(f"  Trading Edge   : "
          f"{iv_surface['iv_signal'].upper()}")
    print(f"  Market Mood    : "
          f"{iv_surface['skew_signal'].upper()}")
    print(f"  {'─' * 45}")

    # Build report
    report = {
        "symbol"    : symbol,
        "timestamp" : datetime.now().isoformat(),
        "spot"      : spot,
        "atm"       : atm,
        "expiry"    : expiry,
        "dte"       : dte,
        "iv_surface": iv_surface,
        "iv_curve"  : iv_curve,
        "max_pain"  : max_pain,
        "exp_move"  : exp_move
    }

    # Save
    path = os.path.join(
        OUTPUT_DIR,
        f"{symbol}_iv_analysis_"
        f"{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    )
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  ✅ IV analysis saved → {path}")
    return report

# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    for symbol in ["NIFTY", "BANKNIFTY"]:
        full_iv_analysis(symbol)
    print("\n  ✅ IV + Max Pain Analysis complete!")
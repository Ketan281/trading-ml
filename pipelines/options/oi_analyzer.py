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

# ── PCR Analysis ──────────────────────────────────────
def analyze_pcr(df, symbol):
    total_ce_oi  = df["ce_oi"].sum()
    total_pe_oi  = df["pe_oi"].sum()
    total_ce_vol = df["ce_volume"].sum()
    total_pe_vol = df["pe_volume"].sum()

    # PCR by OI
    pcr_oi  = round(total_pe_oi  / total_ce_oi,  3) \
              if total_ce_oi  > 0 else 0

    # PCR by Volume
    pcr_vol = round(total_pe_vol / total_ce_vol, 3) \
              if total_ce_vol > 0 else 0

    # PCR Interpretation
    def interpret_pcr(pcr):
        if pcr > 1.5:
            return "extremely_bullish"
        elif pcr > 1.2:
            return "bullish"
        elif pcr > 0.8:
            return "neutral"
        elif pcr > 0.5:
            return "bearish"
        else:
            return "extremely_bearish"

    return {
        "symbol"         : symbol,
        "total_ce_oi"    : int(total_ce_oi),
        "total_pe_oi"    : int(total_pe_oi),
        "total_ce_volume": int(total_ce_vol),
        "total_pe_volume": int(total_pe_vol),
        "pcr_oi"         : pcr_oi,
        "pcr_volume"     : pcr_vol,
        "pcr_oi_signal"  : interpret_pcr(pcr_oi),
        "pcr_vol_signal" : interpret_pcr(pcr_vol)
    }

# ── OI Wall Detection ─────────────────────────────────
def find_oi_walls(df, symbol, top_n=5):
    step = get_strike_step(symbol)
    spot = df["spot"].iloc[0]
    atm  = round_to_strike(spot, step)

    # Top CE OI strikes = Resistance walls
    ce_walls = (
        df.nlargest(top_n, "ce_oi")[["strike", "ce_oi"]]
        .rename(columns={"ce_oi": "oi"})
        .assign(type="resistance")
    )

    # Top PE OI strikes = Support walls
    pe_walls = (
        df.nlargest(top_n, "pe_oi")[["strike", "pe_oi"]]
        .rename(columns={"pe_oi": "oi"})
        .assign(type="support")
    )

    # Nearest resistance above spot
    resistance = ce_walls[
        ce_walls["strike"] > spot
    ].sort_values("strike")

    # Nearest support below spot
    support = pe_walls[
        pe_walls["strike"] < spot
    ].sort_values("strike", ascending=False)

    nearest_resistance = (
        int(resistance.iloc[0]["strike"])
        if not resistance.empty else None
    )
    nearest_support = (
        int(support.iloc[0]["strike"])
        if not support.empty else None
    )

    return {
        "ce_walls"           : ce_walls["strike"].tolist(),
        "pe_walls"           : pe_walls["strike"].tolist(),
        "nearest_resistance" : nearest_resistance,
        "nearest_support"    : nearest_support,
        "atm"                : atm,
        "spot"               : spot
    }

# ── OI Change Analysis ────────────────────────────────
def analyze_oi_change(df, symbol):
    # Fresh long buildup  → CE OI up + price up
    # Long unwinding      → CE OI down + price down
    # Short buildup       → PE OI up + price down
    # Short covering      → PE OI down + price up

    # Strikes with highest OI addition
    ce_oi_added = df.nlargest(3, "ce_chng_oi")[
        ["strike", "ce_chng_oi", "ce_ltp"]
    ]
    pe_oi_added = df.nlargest(3, "pe_chng_oi")[
        ["strike", "pe_chng_oi", "pe_ltp"]
    ]

    # Strikes with highest OI reduction
    ce_oi_shed  = df.nsmallest(3, "ce_chng_oi")[
        ["strike", "ce_chng_oi", "ce_ltp"]
    ]
    pe_oi_shed  = df.nsmallest(3, "pe_chng_oi")[
        ["strike", "pe_chng_oi", "pe_ltp"]
    ]

    total_ce_added = df[df["ce_chng_oi"] > 0]["ce_chng_oi"].sum()
    total_pe_added = df[df["pe_chng_oi"] > 0]["pe_chng_oi"].sum()
    total_ce_shed  = df[df["ce_chng_oi"] < 0]["ce_chng_oi"].sum()
    total_pe_shed  = df[df["pe_chng_oi"] < 0]["pe_chng_oi"].sum()

    # Determine market activity
    if total_ce_added > total_pe_added:
        activity = "call_writing"
        activity_signal = "bearish_pressure"
    else:
        activity = "put_writing"
        activity_signal = "bullish_pressure"

    return {
        "ce_oi_added_strikes" : ce_oi_added["strike"].tolist(),
        "pe_oi_added_strikes" : pe_oi_added["strike"].tolist(),
        "ce_oi_shed_strikes"  : ce_oi_shed["strike"].tolist(),
        "pe_oi_shed_strikes"  : pe_oi_shed["strike"].tolist(),
        "total_ce_oi_added"   : int(total_ce_added),
        "total_pe_oi_added"   : int(total_pe_added),
        "total_ce_oi_shed"    : int(abs(total_ce_shed)),
        "total_pe_oi_shed"    : int(abs(total_pe_shed)),
        "dominant_activity"   : activity,
        "activity_signal"     : activity_signal
    }

# ── Max Pain Calculation ──────────────────────────────
def calculate_max_pain(df):
    try:
        results = []
        for target in df["strike"].unique():
            ce_loss = (
                (target - df[df["strike"] < target]["strike"])
                * df[df["strike"] < target]["ce_oi"]
            ).sum()
            pe_loss = (
                (df[df["strike"] > target]["strike"] - target)
                * df[df["strike"] > target]["pe_oi"]
            ).sum()
            results.append({
                "strike"    : target,
                "total_pain": ce_loss + pe_loss
            })

        pain_df    = pd.DataFrame(results)
        max_pain   = pain_df.loc[
            pain_df["total_pain"].idxmin(), "strike"
        ]
        return int(max_pain)
    except Exception as e:
        print(f"  ⚠ Max pain calculation failed: {e}")
        return None

# ── OI Concentration ──────────────────────────────────
def oi_concentration(df, symbol):
    spot = df["spot"].iloc[0]
    step = get_strike_step(symbol)
    atm  = round_to_strike(spot, step)

    # OI within 2% of spot
    band   = spot * 0.02
    nearby = df[
        (df["strike"] >= spot - band) &
        (df["strike"] <= spot + band)
    ]

    nearby_ce = nearby["ce_oi"].sum()
    nearby_pe = nearby["pe_oi"].sum()
    total_ce  = df["ce_oi"].sum()
    total_pe  = df["pe_oi"].sum()

    ce_concentration = round(
        nearby_ce / total_ce * 100, 1
    ) if total_ce > 0 else 0
    pe_concentration = round(
        nearby_pe / total_pe * 100, 1
    ) if total_pe > 0 else 0

    return {
        "ce_concentration_pct": ce_concentration,
        "pe_concentration_pct": pe_concentration,
        "interpretation"      : (
            "high_activity_near_spot"
            if ce_concentration > 30
            or pe_concentration > 30
            else "spread_activity"
        )
    }

# ── Full OI Analysis ──────────────────────────────────
def full_oi_analysis(symbol):
    print(f"\n{'=' * 55}")
    print(f"  OI + PCR Analysis — {symbol}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 55}")

    # Fetch options chain
    result = fetch_options_chain(symbol)
    if not result or result["df"].empty:
        print(f"  ❌ Could not fetch chain for {symbol}")
        return None

    df   = result["df"]
    spot = result["spot"]
    atm  = result["atm"]

    # Run all analyses
    pcr         = analyze_pcr(df, symbol)
    walls       = find_oi_walls(df, symbol)
    oi_change   = analyze_oi_change(df, symbol)
    max_pain    = calculate_max_pain(df)
    conc        = oi_concentration(df, symbol)

    # Build final report
    report = {
        "symbol"        : symbol,
        "timestamp"     : datetime.now().isoformat(),
        "spot"          : spot,
        "atm"           : atm,
        "max_pain"      : max_pain,
        "pcr"           : pcr,
        "oi_walls"      : walls,
        "oi_change"     : oi_change,
        "concentration" : conc
    }

    # Print report
    print(f"\n  📍 Spot Price    : ₹{spot}")
    print(f"  🎯 ATM Strike    : {atm}")
    print(f"  😣 Max Pain      : {max_pain}")
    print(f"  📏 Distance      : "
          f"{round(abs(spot - max_pain) / spot * 100, 2)}% "
          f"from spot")

    print(f"\n  📊 PCR Analysis:")
    print(f"     PCR (OI)     : {pcr['pcr_oi']} "
          f"→ {pcr['pcr_oi_signal'].upper()}")
    print(f"     PCR (Volume) : {pcr['pcr_volume']} "
          f"→ {pcr['pcr_vol_signal'].upper()}")
    print(f"     Total CE OI  : "
          f"{pcr['total_ce_oi']:,}")
    print(f"     Total PE OI  : "
          f"{pcr['total_pe_oi']:,}")

    print(f"\n  🧱 OI Walls:")
    print(f"     Resistance   : {walls['nearest_resistance']}"
          f" (CE wall)")
    print(f"     Support      : {walls['nearest_support']}"
          f" (PE wall)")
    print(f"     Top CE Walls : {walls['ce_walls'][:3]}")
    print(f"     Top PE Walls : {walls['pe_walls'][:3]}")

    print(f"\n  📈 OI Change:")
    print(f"     Activity     : "
          f"{oi_change['dominant_activity'].upper()}")
    print(f"     Signal       : "
          f"{oi_change['activity_signal'].upper()}")
    print(f"     CE OI Added  : "
          f"{oi_change['total_ce_oi_added']:,}")
    print(f"     PE OI Added  : "
          f"{oi_change['total_pe_oi_added']:,}")

    print(f"\n  🎯 OI Concentration:")
    print(f"     CE near spot : "
          f"{conc['ce_concentration_pct']}%")
    print(f"     PE near spot : "
          f"{conc['pe_concentration_pct']}%")
    print(f"     Activity     : "
          f"{conc['interpretation'].upper()}")

    # Overall signal
    signals = []
    if pcr["pcr_oi_signal"] in ["bullish", "extremely_bullish"]:
        signals.append("bullish")
    elif pcr["pcr_oi_signal"] in ["bearish", "extremely_bearish"]:
        signals.append("bearish")

    if oi_change["activity_signal"] == "bullish_pressure":
        signals.append("bullish")
    else:
        signals.append("bearish")

    bull_count = signals.count("bullish")
    bear_count = signals.count("bearish")

    if bull_count > bear_count:
        overall = "🐂 BULLISH"
    elif bear_count > bull_count:
        overall = "🐻 BEARISH"
    else:
        overall = "😐 NEUTRAL"

    print(f"\n  {'─' * 45}")
    print(f"  Overall OI Signal : {overall}")
    print(f"  {'─' * 45}")

    report["overall_signal"] = overall.split()[1].lower()

    # Save report
    path = os.path.join(
        OUTPUT_DIR,
        f"{symbol}_oi_analysis_"
        f"{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    )
    with open(path, "w") as f:
        # Remove df from report before saving
        save_report = {
            k: v for k, v in report.items()
        }
        json.dump(save_report, f, indent=2)

    print(f"\n  ✅ OI analysis saved → {path}")
    return report

# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    for symbol in ["NIFTY", "BANKNIFTY"]:
        full_oi_analysis(symbol)
    print("\n  ✅ OI + PCR Analysis complete!")
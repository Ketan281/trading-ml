import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(
       os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FEATURE_DIR = os.path.join(ROOT, "data",
                            "features")
LABEL_DIR   = os.path.join(ROOT, "data",
                            "labels")
os.makedirs(LABEL_DIR, exist_ok=True)

# ── Label Config ──────────────────────────────────────
LABEL_CONFIG = {
    # Direction label
    "direction_window"   : 5,    # 5 day forward
    "direction_threshold": 0.01, # 1% move

    # Entry timing label
    "entry_window"       : 3,    # 3 day forward
    "entry_threshold"    : 0.008,# 0.8% move

    # Risk label
    "risk_atr_mult"      : 1.5,  # 1.5x ATR = high

    # Strategy label
    "trend_adx_min"      : 25,   # ADX > 25 = trending
    "vol_high_threshold" : 18,   # HV > 18 = high vol
}

# ── Label 1 — Market Direction ────────────────────────
def build_direction_label(df, window=5,
                           threshold=0.01):
    """
    What will price do in next N days?
    0 = sideways
    1 = uptrend
    2 = downtrend
    """
    close        = df["Close"]
    future_ret   = close.shift(-window) / close - 1

    direction    = pd.Series(
        "sideways", index=df.index
    )
    direction[future_ret >  threshold] = "uptrend"
    direction[future_ret < -threshold] = "downtrend"

    # Confidence — how strong is the move
    direction_conf = future_ret.abs().clip(
        0, threshold * 5
    ) / (threshold * 5)

    return direction, direction_conf, future_ret

# ── Label 2 — Best Strategy ───────────────────────────
def build_strategy_label(df):
    """
    What is the best strategy for current conditions?
    Based on regime + volatility + trend strength
    """
    adx    = df.get("adx",    pd.Series(20, index=df.index))
    hv20   = df.get("hv_20",  pd.Series(15, index=df.index))
    rsi14  = df.get("rsi_14", pd.Series(50, index=df.index))
    trend  = df.get("trend_regime",
                    pd.Series(0, index=df.index))
    bb_pos = df.get("bb_position_20",
                    pd.Series(0, index=df.index))

    strategy = pd.Series(
        "neutral", index=df.index
    )

    # Strong uptrend → buy calls / bull spread
    mask = (
        (trend >= 2) &
        (adx > LABEL_CONFIG["trend_adx_min"]) &
        (rsi14 < 70)
    )
    strategy[mask] = "bull_trend"

    # Strong downtrend → buy puts / bear spread
    mask = (
        (trend <= -2) &
        (adx > LABEL_CONFIG["trend_adx_min"]) &
        (rsi14 > 30)
    )
    strategy[mask] = "bear_trend"

    # High volatility sideways → sell premium
    mask = (
        (adx < 20) &
        (hv20 > LABEL_CONFIG["vol_high_threshold"])
    )
    strategy[mask] = "sell_premium"

    # Low volatility sideways → buy straddle
    mask = (
        (adx < 20) &
        (hv20 < 10)
    )
    strategy[mask] = "buy_volatility"

    # Overbought → reduce / avoid longs
    mask = (
        (rsi14 > 75) |
        (bb_pos == 1)
    )
    strategy[mask] = "overbought_avoid"

    # Oversold → watch for reversal
    mask = (
        (rsi14 < 25) |
        (bb_pos == -1)
    )
    strategy[mask] = "oversold_watch"

    # Mean reversion
    mask = (
        (adx < 20) &
        (hv20.between(10, 18)) &
        (rsi14.between(35, 65))
    )
    strategy[mask] = "mean_reversion"

    return strategy

# ── Label 3 — Entry Timing ────────────────────────────
def build_entry_label(df, window=3,
                       threshold=0.008):
    """
    Is NOW a good entry point?
    good_entry    = price moves in signal direction
    bad_entry     = price moves against signal
    neutral_entry = price stays flat
    """
    close      = df["Close"]
    future_ret = close.shift(-window) / close - 1
    trend      = df.get(
        "trend_regime",
        pd.Series(0, index=df.index)
    )

    entry = pd.Series(
        "neutral_entry", index=df.index
    )

    # Good entry — trending up and price goes up
    mask = (
        (trend > 0) &
        (future_ret > threshold)
    )
    entry[mask] = "good_entry"

    # Good entry — trending down and price goes down
    mask = (
        (trend < 0) &
        (future_ret < -threshold)
    )
    entry[mask] = "good_entry"

    # Bad entry — wrong direction
    mask = (
        (trend > 0) &
        (future_ret < -threshold)
    )
    entry[mask] = "bad_entry"

    mask = (
        (trend < 0) &
        (future_ret > threshold)
    )
    entry[mask] = "bad_entry"

    return entry

# ── Label 4 — Risk Level ──────────────────────────────
def build_risk_label(df):
    """
    What is the risk level of trading today?
    low / medium / high / extreme
    """
    atr14  = df.get(
        "atr_14", pd.Series(0, index=df.index)
    )
    hv20   = df.get(
        "hv_20",  pd.Series(15, index=df.index)
    )
    close  = df["Close"]
    rsi14  = df.get(
        "rsi_14", pd.Series(50, index=df.index)
    )
    adx    = df.get(
        "adx",    pd.Series(20, index=df.index)
    )
    bb_pos = df.get(
        "bb_position_20",
        pd.Series(0, index=df.index)
    )

    atr_pct = atr14 / close * 100
    risk    = pd.Series(
        "medium", index=df.index
    )

    # Low risk
    mask = (
        (atr_pct < 0.8) &
        (hv20 < 10) &
        (rsi14.between(40, 60))
    )
    risk[mask] = "low"

    # High risk
    mask = (
        (atr_pct > 1.8) |
        (hv20 > 20) |
        (rsi14 > 75) |
        (rsi14 < 25)
    )
    risk[mask] = "high"

    # Extreme risk
    mask = (
        (atr_pct > 2.5) |
        (hv20 > 28) |
        (bb_pos != 0)
    )
    risk[mask] = "extreme"

    return risk

# ── Action Label (Combined) ───────────────────────────
def build_action_label(direction, strategy,
                        risk):
    """
    Final recommended action combining all labels
    buy / sell / hold / avoid
    """
    action = pd.Series("hold", index=direction.index)

    # Buy conditions
    mask = (
        (direction == "uptrend") &
        (strategy.isin(
            ["bull_trend", "mean_reversion",
             "buy_volatility"]
        )) &
        (risk.isin(["low", "medium"]))
    )
    action[mask] = "buy"

    # Sell conditions
    mask = (
        (direction == "downtrend") &
        (strategy.isin(
            ["bear_trend", "sell_premium"]
        )) &
        (risk.isin(["low", "medium"]))
    )
    action[mask] = "sell"

    # Avoid conditions
    mask = (
        (risk == "extreme") |
        (strategy == "overbought_avoid")
    )
    action[mask] = "avoid"

    # Reduce exposure
    mask = (
        (risk == "high") &
        (direction != "sideways")
    )
    action[mask] = "reduce_exposure"

    return action

# ── Confidence Score ──────────────────────────────────
def build_confidence_score(df, direction_conf):
    """
    How confident should the AI be in this signal?
    Based on signal agreement across indicators
    """
    adx   = df.get(
        "adx",    pd.Series(20, index=df.index)
    )
    rsi14 = df.get(
        "rsi_14", pd.Series(50, index=df.index)
    )
    macd_hist = df.get(
        "macd_hist",
        pd.Series(0, index=df.index)
    )
    trend = df.get(
        "trend_regime",
        pd.Series(0, index=df.index)
    )

    # Start with direction confidence
    conf = direction_conf.copy()

    # Boost if ADX strong
    adx_boost = ((adx - 20) / 40).clip(0, 0.2)
    conf      = conf + adx_boost

    # Boost if RSI not extreme
    rsi_ok    = (
        (rsi14 > 30) & (rsi14 < 70)
    ).astype(float) * 0.1
    conf      = conf + rsi_ok

    # Boost if MACD agrees with trend
    macd_agree = (
        (trend > 0) & (macd_hist > 0) |
        (trend < 0) & (macd_hist < 0)
    ).astype(float) * 0.1
    conf       = conf + macd_agree

    return conf.clip(0.1, 0.95).round(3)

# ── Build Labels for One Symbol ───────────────────────
def build_labels(name):
    print(f"  Building labels: {name}... ", end="")

    # Load features
    path = os.path.join(
        FEATURE_DIR, f"{name}_features.csv"
    )
    if not os.path.exists(path):
        print(f"❌ No features found")
        return None

    try:
        df = pd.read_csv(
            path,
            index_col="Date",
            parse_dates=True
        )

        if len(df) < 200:
            print(f"❌ Not enough data")
            return None

        # Build all labels
        direction, dir_conf, future_ret = \
            build_direction_label(
                df,
                LABEL_CONFIG["direction_window"],
                LABEL_CONFIG["direction_threshold"]
            )

        strategy   = build_strategy_label(df)
        entry      = build_entry_label(
            df,
            LABEL_CONFIG["entry_window"],
            LABEL_CONFIG["entry_threshold"]
        )
        risk       = build_risk_label(df)
        action     = build_action_label(
            direction, strategy, risk
        )
        confidence = build_confidence_score(
            df, dir_conf
        )

        # Combine into label dataframe
        labels = pd.DataFrame({
            "direction"  : direction,
            "strategy"   : strategy,
            "entry"      : entry,
            "risk"       : risk,
            "action"     : action,
            "confidence" : confidence,
            "future_ret" : future_ret.round(4)
        }, index=df.index)

        # Remove last N rows (no future data)
        labels = labels.iloc[
            :-LABEL_CONFIG["direction_window"]
        ]

        # Save
        label_path = os.path.join(
            LABEL_DIR, f"{name}_labels.csv"
        )
        labels.to_csv(label_path)

        # Stats
        dir_dist  = direction.value_counts()
        act_dist  = action.value_counts()

        print(
            f"✅ {len(labels)} rows | "
            f"Up:{dir_dist.get('uptrend',0)} "
            f"Dn:{dir_dist.get('downtrend',0)} "
            f"Sw:{dir_dist.get('sideways',0)}"
        )

        return labels

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

# ── Build Labels for All Symbols ──────────────────────
def build_all_labels():
    print("=" * 60)
    print("  Trading AI — Label Generation")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Get symbols from features dir
    symbols = [
        f.replace("_features.csv", "")
        for f in os.listdir(FEATURE_DIR)
        if f.endswith("_features.csv")
    ]

    print(f"\n  Processing {len(symbols)} symbols...")
    print()

    success     = []
    failed      = []
    all_labels  = []

    for i, name in enumerate(
        sorted(symbols), 1
    ):
        print(f"  [{i:02d}/{len(symbols)}] ",
              end="")
        labels = build_labels(name)

        if labels is not None:
            success.append(name)
            labels["symbol"] = name
            all_labels.append(labels)
        else:
            failed.append(name)

    # Aggregate statistics
    if all_labels:
        combined = pd.concat(all_labels)

        print(f"\n{'=' * 60}")
        print(f"  LABEL STATISTICS")
        print(f"{'=' * 60}")

        # Direction distribution
        print(f"\n  Direction Distribution:")
        dir_counts = combined[
            "direction"
        ].value_counts()
        total = len(combined)
        for label, count in dir_counts.items():
            pct = round(count/total*100, 1)
            bar = "█" * int(pct / 2)
            print(
                f"     {label:<12}: "
                f"{bar:<25} {pct}%"
            )

        # Action distribution
        print(f"\n  Action Distribution:")
        act_counts = combined[
            "action"
        ].value_counts()
        for label, count in act_counts.items():
            pct = round(count/total*100, 1)
            bar = "█" * int(pct / 2)
            print(
                f"     {label:<18}: "
                f"{bar:<25} {pct}%"
            )

        # Strategy distribution
        print(f"\n  Strategy Distribution:")
        str_counts = combined[
            "strategy"
        ].value_counts()
        for label, count in str_counts.items():
            pct = round(count/total*100, 1)
            print(
                f"     {label:<20}: "
                f"{count:>7,} ({pct}%)"
            )

        # Risk distribution
        print(f"\n  Risk Distribution:")
        risk_counts = combined[
            "risk"
        ].value_counts()
        for label, count in risk_counts.items():
            pct = round(count/total*100, 1)
            print(
                f"     {label:<10}: "
                f"{count:>7,} ({pct}%)"
            )

        # Average confidence
        avg_conf = combined["confidence"].mean()
        print(
            f"\n  Avg Confidence  : "
            f"{avg_conf:.3f}"
        )
        print(
            f"  Total Samples   : "
            f"{len(combined):,}"
        )

        # Save combined stats
        stats = {
            "total_samples"   : len(combined),
            "direction"       : dir_counts.to_dict(),
            "action"          : act_counts.to_dict(),
            "strategy"        : str_counts.to_dict(),
            "risk"            : risk_counts.to_dict(),
            "avg_confidence"  : round(avg_conf, 3),
            "success_symbols" : success,
            "failed_symbols"  : failed
        }

        stats_path = os.path.join(
            LABEL_DIR, "label_stats.json"
        )
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(
            f"\n  ✅ Labels saved → {LABEL_DIR}"
        )
        print(
            f"  ✅ Stats saved  → {stats_path}"
        )

    return success

# ── Verify Label Quality ──────────────────────────────
def verify_label_quality():
    print(f"\n{'=' * 60}")
    print(f"  LABEL QUALITY CHECK")
    print(f"{'=' * 60}")

    # Check NIFTY labels in detail
    path = os.path.join(
        LABEL_DIR, "NIFTY_labels.csv"
    )
    if not os.path.exists(path):
        print("  ⚠ NIFTY labels not found")
        return

    df = pd.read_csv(
        path,
        index_col="Date",
        parse_dates=True
    )

    print(f"\n  NIFTY Label Sample (last 10 rows):")
    print(f"\n  {'DATE':<12} {'DIR':<12} "
          f"{'ACTION':<20} {'RISK':<10} "
          f"{'CONF':<8} {'RET%'}")
    print("  " + "─" * 72)

    for date, row in df.tail(10).iterrows():
        print(
            f"  {str(date.date()):<12} "
            f"{row['direction']:<12} "
            f"{row['action']:<20} "
            f"{row['risk']:<10} "
            f"{row['confidence']:<8} "
            f"{row['future_ret']*100:.2f}%"
        )

    # Check year-by-year distribution
    print(f"\n  Year-by-Year Direction (NIFTY):")
    print(f"\n  {'YEAR':<6} {'UP':<10} "
          f"{'DOWN':<10} {'SIDE':<10} "
          f"{'AVG RET%'}")
    print("  " + "─" * 45)

    for year in range(2014, 2025):
        year_df = df[df.index.year == year]
        if year_df.empty:
            continue

        up   = (
            year_df["direction"] == "uptrend"
        ).sum()
        dn   = (
            year_df["direction"] == "downtrend"
        ).sum()
        sw   = (
            year_df["direction"] == "sideways"
        ).sum()
        ret  = round(
            year_df["future_ret"].mean() * 100, 2
        )

        # Year assessment
        if ret > 0.05:
            icon = "🐂"
        elif ret < -0.05:
            icon = "🐻"
        else:
            icon = "😐"

        print(
            f"  {year:<6} {up:<10} "
            f"{dn:<10} {sw:<10} "
            f"{ret:>+.2f}% {icon}"
        )

# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    # Build all labels
    build_all_labels()

    # Verify quality
    verify_label_quality()

    print(f"\n  🎯 Next step: Build walk forward dataset")
    print(
        f"  Command: "
        f"python training/build_walk_forward_dataset.py"
    )
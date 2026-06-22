import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(
       os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FEATURE_DIR  = os.path.join(ROOT, "data",
                              "features")
LABEL_DIR    = os.path.join(ROOT, "data",
                              "labels")
DATASET_DIR  = os.path.join(ROOT, "data",
                              "datasets")
os.makedirs(DATASET_DIR, exist_ok=True)

# ── Walk Forward Windows ──────────────────────────────
# 20-year walk-forward: train on multi-year blocks, test on unseen future.
# Each window validates that the model generalises across different market
# regimes (2008 crash, 2011 correction, 2016 demonetisation, 2020 COVID,
# 2022 rate hikes, 2025 current market).
WINDOWS = [
    # Epoch 1: Pre-GFC through GFC recovery → test post-recovery
    {"train_start": 2006, "train_end": 2010, "test_year": 2011},
    # Epoch 2: Post-GFC bull run → test demonetisation year
    {"train_start": 2012, "train_end": 2015, "test_year": 2016},
    # Epoch 3: Post-demo through COVID → test COVID recovery
    {"train_start": 2016, "train_end": 2019, "test_year": 2020},
    # Epoch 4: COVID + bull run + rate hikes → test current market
    {"train_start": 2020, "train_end": 2025, "test_year": 2026},
    # Expanding windows (cumulative knowledge, newest test)
    {"train_start": 2006, "train_end": 2020, "test_year": 2021},
    {"train_start": 2006, "train_end": 2022, "test_year": 2023},
    {"train_start": 2006, "train_end": 2024, "test_year": 2025},
]

TRAIN_START = 2006

# ── Symbols ───────────────────────────────────────────
INDEX_SYMBOLS = ["NIFTY", "BANKNIFTY"]
ALL_SYMBOLS   = [
    f.replace("_features.csv", "")
    for f in os.listdir(FEATURE_DIR)
    if f.endswith("_features.csv")
] if os.path.exists(FEATURE_DIR) else []

# ── Load Features + Labels ────────────────────────────
def load_symbol_data(name):
    feat_path  = os.path.join(
        FEATURE_DIR, f"{name}_features.csv"
    )
    label_path = os.path.join(
        LABEL_DIR, f"{name}_labels.csv"
    )

    if not os.path.exists(feat_path) or \
       not os.path.exists(label_path):
        return None, None

    features = pd.read_csv(
        feat_path,
        index_col="Date",
        parse_dates=True
    )
    labels   = pd.read_csv(
        label_path,
        index_col="Date",
        parse_dates=True
    )

    # Align on common index
    common = features.index.intersection(
        labels.index
    )
    return features.loc[common], \
           labels.loc[common]

# ── Build AI Prompt from Features + Labels ────────────
def build_training_prompt(name, date,
                            features, label_row):
    # Extract key features
    price    = features.get("Close", 0)
    rsi14    = round(
        float(features.get("rsi_14", 50)), 1
    )
    macd_h   = round(
        float(features.get("macd_hist", 0)), 2
    )
    adx      = round(
        float(features.get("adx", 20)), 1
    )
    hv20     = round(
        float(features.get("hv_20", 15)), 1
    )
    atr_pct  = round(
        float(features.get("atr_pct_14", 1)), 2
    )
    bb_pct   = round(
        float(features.get("bb_pct_20", 0)), 2
    )
    ema9_dev = round(
        float(features.get("price_vs_ema9", 0)), 2
    )
    ema20_dev = round(
        float(features.get("price_vs_ema20", 0)), 2
    )
    ema50_dev = round(
        float(features.get("price_vs_ema50", 0)), 2
    )
    vol_regime = int(
        features.get("vol_regime", 2)
    )
    trend_reg  = int(
        features.get("trend_regime", 0)
    )
    iv_pct     = round(
        float(features.get(
            "iv_percentile", 50
        )), 1
    )
    roc20      = round(
        float(features.get("roc_20", 0)), 2
    )
    day_of_week = int(
        features.get("day_of_week", 2)
    )
    month      = int(
        features.get("month", 6)
    )
    results_s  = int(
        features.get("results_season", 0)
    )

    # Encode categorical features
    vol_map    = {
        0: "very_low", 1: "low",
        2: "normal",   3: "high", 4: "extreme"
    }
    trend_map  = {
        3: "strong_uptrend", 2: "uptrend",
        1: "weak_uptrend",   0: "sideways",
        -1: "weak_downtrend",-2: "downtrend",
        -3: "strong_downtrend"
    }
    day_map    = {
        0: "Monday", 1: "Tuesday",
        2: "Wednesday", 3: "Thursday",
        4: "Friday"
    }
    month_map  = {
        1: "January",   2: "February",
        3: "March",     4: "April",
        5: "May",       6: "June",
        7: "July",      8: "August",
        9: "September", 10: "October",
        11: "November", 12: "December"
    }

    vol_str   = vol_map.get(vol_regime, "normal")
    trend_str = trend_map.get(trend_reg, "sideways")
    day_str   = day_map.get(day_of_week, "Wednesday")
    month_str = month_map.get(month, "June")
    macd_str  = "bullish" if macd_h > 0 \
                else "bearish"
    results_str = "YES" if results_s else "NO"

    # MACD signal
    macd_cross = int(
        features.get("macd_cross", 0)
    )
    macd_cross_str = " [FRESH CROSSOVER]" \
                     if macd_cross else ""

    # RSI zone
    if rsi14 > 70:
        rsi_zone = "overbought"
    elif rsi14 < 30:
        rsi_zone = "oversold"
    elif rsi14 > 60:
        rsi_zone = "bullish_zone"
    elif rsi14 < 40:
        rsi_zone = "bearish_zone"
    else:
        rsi_zone = "neutral_zone"

    # Bollinger position
    if bb_pct > 0.8:
        bb_str = "near_upper_band"
    elif bb_pct < -0.8:
        bb_str = "near_lower_band"
    else:
        bb_str = "inside_bands"

    # ADX strength
    if adx > 40:
        adx_str = "very_strong_trend"
    elif adx > 25:
        adx_str = "strong_trend"
    elif adx > 20:
        adx_str = "moderate_trend"
    else:
        adx_str = "weak_or_no_trend"

    prompt = f"""You are a professional trading intelligence system analyzing Indian markets.

Analyze the following market data and provide a structured trading decision.

═══════════════════════════════════════════
SYMBOL    : {name}
DATE      : {str(date)[:10]}
DAY       : {day_str}
MONTH     : {month_str}
═══════════════════════════════════════════

── TREND ANALYSIS ─────────────────────────
Trend Regime   : {trend_str}
ADX            : {adx} ({adx_str})
Price vs EMA9  : {ema9_dev:+.2f}%
Price vs EMA20 : {ema20_dev:+.2f}%
Price vs EMA50 : {ema50_dev:+.2f}%
20D Return     : {roc20:+.2f}%

── MOMENTUM ───────────────────────────────
RSI 14         : {rsi14} ({rsi_zone})
MACD           : {macd_str}{macd_cross_str}
MACD Histogram : {macd_h:+.3f}
Bollinger      : {bb_str} ({bb_pct:+.2f})

── VOLATILITY ─────────────────────────────
Vol Regime     : {vol_str}
HV 20D         : {hv20:.1f}%
ATR %          : {atr_pct:.2f}%
IV Percentile  : {iv_pct:.0f}th percentile

── MARKET CONTEXT ─────────────────────────
Results Season : {results_str}
═══════════════════════════════════════════

Based on the above data provide a complete trading analysis.

Return ONLY valid JSON:
{{
  "symbol"           : "{name}",
  "date"             : "{str(date)[:10]}",
  "market_condition" : "<strong_uptrend/uptrend/weak_uptrend/sideways/weak_downtrend/downtrend/strong_downtrend>",
  "action"           : "<buy/sell/hold/avoid/reduce_exposure>",
  "strategy"         : "<specific strategy name>",
  "confidence"       : <0.0-1.0>,
  "risk_level"       : "<low/medium/high/extreme>",
  "entry_condition"  : "<when to enter>",
  "stop_loss_pct"    : <stop loss as % from entry>,
  "target_pct"       : <target as % from entry>,
  "reasoning"        : [
    "<reason 1>",
    "<reason 2>",
    "<reason 3>"
  ]
}}"""

    return prompt

# ── Build Ideal Response from Labels ──────────────────
def build_ideal_response(name, date,
                          label_row,
                          features):
    direction  = label_row.get(
        "direction", "sideways"
    )
    strategy   = label_row.get(
        "strategy", "neutral"
    )
    action     = label_row.get("action", "hold")
    risk       = label_row.get("risk", "medium")
    confidence = float(
        label_row.get("confidence", 0.5)
    )
    future_ret = float(
        label_row.get("future_ret", 0)
    )

    # Map direction to market condition
    adx        = float(
        features.get("adx", 20)
    )
    trend_reg  = int(
        features.get("trend_regime", 0)
    )

    condition_map = {
        3 : "strong_uptrend",
        2 : "uptrend",
        1 : "weak_uptrend",
        0 : "sideways",
        -1: "weak_downtrend",
        -2: "downtrend",
        -3: "strong_downtrend"
    }
    market_condition = condition_map.get(
        trend_reg, "sideways"
    )

    # Strategy name mapping
    strategy_names = {
        "bull_trend"     : "bull_put_spread",
        "bear_trend"     : "bear_call_spread",
        "sell_premium"   : "short_straddle",
        "buy_volatility" : "long_straddle",
        "mean_reversion" : "iron_condor",
        "overbought_avoid": "avoid",
        "oversold_watch" : "wait_and_watch",
        "neutral"        : "hold"
    }
    strategy_name = strategy_names.get(
        strategy, "hold"
    )

    # Build reasoning based on labels
    reasoning = []

    rsi14  = float(features.get("rsi_14", 50))
    hv20   = float(features.get("hv_20",  15))
    macd_h = float(features.get(
        "macd_hist", 0
    ))

    if direction == "uptrend":
        reasoning.append(
            f"Price expected to rise "
            f"{future_ret*100:+.1f}% "
            f"in next 5 days"
        )
    elif direction == "downtrend":
        reasoning.append(
            f"Price expected to fall "
            f"{future_ret*100:+.1f}% "
            f"in next 5 days"
        )
    else:
        reasoning.append(
            "Price expected to remain sideways"
        )

    if adx > 25:
        reasoning.append(
            f"Strong trend confirmed — "
            f"ADX at {adx:.0f}"
        )
    else:
        reasoning.append(
            f"Weak trend — ADX at {adx:.0f}, "
            f"range-bound conditions"
        )

    if rsi14 > 70:
        reasoning.append(
            f"RSI {rsi14:.0f} — overbought, "
            f"momentum may reverse"
        )
    elif rsi14 < 30:
        reasoning.append(
            f"RSI {rsi14:.0f} — oversold, "
            f"bounce possible"
        )
    else:
        reasoning.append(
            f"RSI {rsi14:.0f} — healthy momentum"
        )

    if hv20 > 20:
        reasoning.append(
            f"High volatility {hv20:.0f}% — "
            f"use spreads not naked options"
        )
    elif hv20 < 10:
        reasoning.append(
            f"Low volatility {hv20:.0f}% — "
            f"options cheap, favor buying"
        )

    if macd_h > 0:
        reasoning.append(
            "MACD histogram positive — "
            "bullish momentum"
        )
    else:
        reasoning.append(
            "MACD histogram negative — "
            "bearish momentum"
        )

    # Stop loss and target
    atr_pct  = float(
        features.get("atr_pct_14", 1.5)
    )

    if action == "buy":
        sl_pct  = round(atr_pct * 1.5, 2)
        tgt_pct = round(atr_pct * 3.0, 2)
    elif action == "sell":
        sl_pct  = round(atr_pct * 1.5, 2)
        tgt_pct = round(atr_pct * 3.0, 2)
    else:
        sl_pct  = 0.0
        tgt_pct = 0.0

    response = {
        "symbol"          : name,
        "date"            : str(date)[:10],
        "market_condition": market_condition,
        "action"          : action,
        "strategy"        : strategy_name,
        "confidence"      : confidence,
        "risk_level"      : risk,
        "entry_condition" : (
            f"Enter on pullback to EMA9 "
            f"with RSI below 65"
            if action == "buy" else
            f"Enter on bounce to EMA9 "
            f"with RSI above 35"
            if action == "sell" else
            "Wait for better setup"
        ),
        "stop_loss_pct"   : sl_pct,
        "target_pct"      : tgt_pct,
        "reasoning"       : reasoning[:4]
    }

    return json.dumps(response, indent=2)

# ── Build Dataset for One Window ──────────────────────
def build_window_dataset(window,
                          symbols=None,
                          max_samples=5000):
    train_start = window.get("train_start", TRAIN_START)
    train_end  = window["train_end"]
    test_year  = window["test_year"]
    symbols    = symbols or ALL_SYMBOLS

    print(f"\n  {'─' * 55}")
    print(
        f"  Window: Train {train_start}-{train_end}"
        f" | Test {test_year}"
    )
    print(f"  {'─' * 55}")

    train_samples = []
    test_samples  = []

    for name in symbols:
        features, labels = load_symbol_data(name)

        if features is None or labels is None:
            continue

        # Training period
        train_feat = features[
            (features.index.year >= train_start) &
            (features.index.year <= train_end)
        ]
        train_lbl  = labels[
            (labels.index.year >= train_start) &
            (labels.index.year <= train_end)
        ]

        # Test period
        test_feat  = features[
            features.index.year == test_year
        ]
        test_lbl   = labels[
            labels.index.year == test_year
        ]

        # Align
        train_common = train_feat.index\
            .intersection(train_lbl.index)
        test_common  = test_feat.index\
            .intersection(test_lbl.index)

        train_feat = train_feat.loc[train_common]
        train_lbl  = train_lbl.loc[train_common]
        test_feat  = test_feat.loc[test_common]
        test_lbl   = test_lbl.loc[test_common]

        # Build samples
        for idx in train_common:
            feat_row  = train_feat.loc[idx]
            label_row = train_lbl.loc[idx]

            # Skip neutral/boring samples
            if (label_row.get("action") == "hold"
                    and np.random.random() > 0.3):
                continue

            prompt   = build_training_prompt(
                name, idx,
                feat_row, label_row
            )
            response = build_ideal_response(
                name, idx,
                label_row, feat_row
            )

            train_samples.append({
                "instruction": prompt,
                "output"     : response,
                "symbol"     : name,
                "date"       : str(idx)[:10],
                "year"       : idx.year,
                "action"     : label_row.get(
                    "action", "hold"
                ),
                "direction"  : label_row.get(
                    "direction", "sideways"
                ),
                "risk"       : label_row.get(
                    "risk", "medium"
                )
            })

        for idx in test_common:
            feat_row  = test_feat.loc[idx]
            label_row = test_lbl.loc[idx]

            prompt   = build_training_prompt(
                name, idx,
                feat_row, label_row
            )
            response = build_ideal_response(
                name, idx,
                label_row, feat_row
            )

            test_samples.append({
                "instruction": prompt,
                "output"     : response,
                "symbol"     : name,
                "date"       : str(idx)[:10],
                "year"       : idx.year,
                "action"     : label_row.get(
                    "action", "hold"
                ),
                "direction"  : label_row.get(
                    "direction", "sideways"
                ),
                "risk"       : label_row.get(
                    "risk", "medium"
                )
            })

    # Shuffle and limit training samples
    np.random.shuffle(train_samples)
    if len(train_samples) > max_samples:
        train_samples = train_samples[:max_samples]

    # Balance classes in training
    train_samples = balance_classes(train_samples)

    print(
        f"  Train samples : {len(train_samples):,}"
    )
    print(
        f"  Test samples  : {len(test_samples):,}"
    )

    # Save window dataset
    window_dir = os.path.join(
        DATASET_DIR,
        f"window_{train_start}_{train_end}"
        f"_test_{test_year}"
    )
    os.makedirs(window_dir, exist_ok=True)

    # Save JSONL for fine-tuning
    train_jsonl = os.path.join(
        window_dir, "train.jsonl"
    )
    with open(train_jsonl, "w") as f:
        for sample in train_samples:
            f.write(json.dumps({
                "instruction": sample["instruction"],
                "output"     : sample["output"]
            }) + "\n")

    test_jsonl = os.path.join(
        window_dir, "test.jsonl"
    )
    with open(test_jsonl, "w") as f:
        for sample in test_samples:
            f.write(json.dumps({
                "instruction": sample["instruction"],
                "output"     : sample["output"]
            }) + "\n")

    # Save full JSON with metadata
    train_json = os.path.join(
        window_dir, "train_full.json"
    )
    with open(train_json, "w") as f:
        json.dump(train_samples, f, indent=2)

    test_json = os.path.join(
        window_dir, "test_full.json"
    )
    with open(test_json, "w") as f:
        json.dump(test_samples, f, indent=2)

    # Window stats
    stats = build_window_stats(
        train_samples, test_samples,
        train_end, test_year
    )
    stats_path = os.path.join(
        window_dir, "stats.json"
    )
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(
        f"  Saved → {window_dir}"
    )
    return {
        "window_dir"    : window_dir,
        "train_samples" : len(train_samples),
        "test_samples"  : len(test_samples),
        "train_start"   : train_start,
        "train_end"     : train_end,
        "test_year"     : test_year,
        "stats"         : stats
    }

# ── Balance Classes ───────────────────────────────────
def balance_classes(samples, target_per_class=None):
    """Ensure no action class dominates dataset"""
    from collections import Counter

    counts = Counter(
        s["action"] for s in samples
    )

    if not target_per_class:
        # Use median count as target
        sorted_counts = sorted(counts.values())
        target = sorted_counts[
            len(sorted_counts) // 2
        ]
        target_per_class = min(
            target * 2, max(sorted_counts)
        )

    balanced    = []
    class_count = Counter()

    # Shuffle first
    np.random.shuffle(samples)

    for sample in samples:
        action = sample["action"]
        if class_count[action] < target_per_class:
            balanced.append(sample)
            class_count[action] += 1

    np.random.shuffle(balanced)
    return balanced

# ── Window Stats ──────────────────────────────────────
def build_window_stats(train_samples,
                        test_samples,
                        train_end, test_year):
    from collections import Counter

    train_actions = Counter(
        s["action"] for s in train_samples
    )
    test_actions  = Counter(
        s["action"] for s in test_samples
    )
    train_dirs    = Counter(
        s["direction"] for s in train_samples
    )
    test_dirs     = Counter(
        s["direction"] for s in test_samples
    )
    train_risks   = Counter(
        s["risk"] for s in train_samples
    )

    return {
        "train_end"      : train_end,
        "test_year"      : test_year,
        "train_count"    : len(train_samples),
        "test_count"     : len(test_samples),
        "train_actions"  : dict(train_actions),
        "test_actions"   : dict(test_actions),
        "train_direction": dict(train_dirs),
        "test_direction" : dict(test_dirs),
        "train_risk"     : dict(train_risks)
    }

# ── Build All Windows ─────────────────────────────────
def build_all_windows():
    print("=" * 60)
    print("  Trading AI — Walk Forward Dataset Builder")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Windows : {len(WINDOWS)}")
    print(f"  Symbols : {len(ALL_SYMBOLS)}")
    print("=" * 60)

    all_results = []

    for window in WINDOWS:
        result = build_window_dataset(
            window,
            symbols=ALL_SYMBOLS,
            max_samples=5000
        )
        all_results.append(result)

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"  WALK FORWARD DATASET SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"  {'WINDOW':<25} {'TRAIN':<10} {'TEST'}"
    )
    print("  " + "─" * 45)

    total_train = 0
    total_test  = 0

    for r in all_results:
        label = (
            f"{r.get('train_start', TRAIN_START)}-{r['train_end']}"
            f" → Test {r['test_year']}"
        )
        print(
            f"  {label:<25} "
            f"{r['train_samples']:<10} "
            f"{r['test_samples']}"
        )
        total_train += r["train_samples"]
        total_test  += r["test_samples"]

    print("  " + "─" * 45)
    print(
        f"  {'TOTAL':<25} "
        f"{total_train:<10} "
        f"{total_test}"
    )

    print(f"\n  ✅ All datasets saved → {DATASET_DIR}")
    print(f"\n  Dataset structure:")
    print(f"  {DATASET_DIR}/")

    for r in all_results:
        ts = r.get('train_start', TRAIN_START)
        folder = (
            f"window_{ts}"
            f"_{r['train_end']}"
            f"_test_{r['test_year']}/"
        )
        print(f"    ├── {folder}")
        print(f"    │   ├── train.jsonl")
        print(f"    │   ├── test.jsonl")
        print(f"    │   ├── train_full.json")
        print(f"    │   ├── test_full.json")
        print(f"    │   └── stats.json")

    return all_results

# ── Preview Sample ────────────────────────────────────
def preview_sample():
    print(f"\n{'=' * 60}")
    print(f"  SAMPLE TRAINING ENTRY PREVIEW")
    print(f"{'=' * 60}")

    # Load first window
    window_dir = os.path.join(
        DATASET_DIR,
        f"window_2006_2010_test_2011"
    )
    train_path = os.path.join(
        window_dir, "train.jsonl"
    )

    if not os.path.exists(train_path):
        print("  ⚠ No dataset found yet")
        return

    with open(train_path) as f:
        first_line = f.readline()

    sample = json.loads(first_line)

    print(f"\n  ── INSTRUCTION (Prompt) ──")
    # Show first 30 lines of prompt
    lines = sample["instruction"].split("\n")
    for line in lines[:30]:
        print(f"  {line}")
    print(f"  ... (truncated)")

    print(f"\n  ── OUTPUT (Ideal Response) ──")
    output = json.loads(sample["output"])
    print(
        json.dumps(output, indent=4)
    )

# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    # Build all windows
    results = build_all_windows()

    # Preview a sample
    preview_sample()

    print(f"\n  🎯 Next step: Validate dataset quality")
    print(
        f"  Command: "
        f"python training/"
        f"validate_dataset.py"
    )
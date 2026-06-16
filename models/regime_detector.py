import os
import sys
import json
import numpy as np
import pandas as pd
from datetime        import datetime, date
from jugaad_data.nse import NSELive

ROOT = os.path.dirname(os.path.dirname(
       os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA_DIR   = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

nse = NSELive()

# ── NSE Expiry Calendar ───────────────────────────────
# NIFTY  → Thursday weekly expiry
# BANK   → Wednesday weekly expiry
WEEKLY_EXPIRY_DAY = {
    "NIFTY"    : 3,   # Thursday = 3
    "BANKNIFTY": 2    # Wednesday = 2
}

# ── Trend Regime ──────────────────────────────────────
def detect_trend_regime(df):
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]

    # EMAs
    ema9   = close.ewm(span=9).mean()
    ema21  = close.ewm(span=21).mean()
    ema50  = close.ewm(span=50).mean()
    ema200 = close.ewm(span=200).mean()

    price  = close.iloc[-1]
    e9     = ema9.iloc[-1]
    e21    = ema21.iloc[-1]
    e50    = ema50.iloc[-1]
    e200   = ema200.iloc[-1]

    # ADX for trend strength
    adx    = compute_adx(df)

    # Higher highs / higher lows
    recent_highs = high.iloc[-10:]
    recent_lows  = low.iloc[-10:]
    hh = recent_highs.iloc[-1] > recent_highs.iloc[-5]
    hl = recent_lows.iloc[-1]  > recent_lows.iloc[-5]
    lh = recent_highs.iloc[-1] < recent_highs.iloc[-5]
    ll = recent_lows.iloc[-1]  < recent_lows.iloc[-5]

    # Score trend
    score = 0

    # EMA alignment
    if price > e9 > e21 > e50:
        score += 3
    elif price > e21 > e50:
        score += 2
    elif price > e50:
        score += 1
    elif price < e9 < e21 < e50:
        score -= 3
    elif price < e21 < e50:
        score -= 2
    elif price < e50:
        score -= 1

    # Price vs EMA200
    if price > e200:
        score += 1
    else:
        score -= 1

    # HH/HL or LH/LL
    if hh and hl:
        score += 2
    elif lh and ll:
        score -= 2

    # ADX strength
    if adx > 30:
        trend_strength = "strong"
    elif adx > 20:
        trend_strength = "moderate"
    else:
        trend_strength = "weak"
        score = int(score * 0.5)  # Dampen in weak trend

    # Classify
    if score >= 4:
        regime = "strong_uptrend"
    elif score >= 2:
        regime = "uptrend"
    elif score >= 1:
        regime = "weak_uptrend"
    elif score <= -4:
        regime = "strong_downtrend"
    elif score <= -2:
        regime = "downtrend"
    elif score <= -1:
        regime = "weak_downtrend"
    else:
        regime = "sideways"

    return {
        "trend_regime"  : regime,
        "trend_strength": trend_strength,
        "trend_score"   : score,
        "adx"           : round(adx, 2),
        "price_vs_ema50": round(
            (price - e50) / e50 * 100, 2
        ),
        "price_vs_ema200": round(
            (price - e200) / e200 * 100, 2
        ),
        "higher_highs"  : bool(hh),
        "higher_lows"   : bool(hl)
    }

# ── ADX Calculation ───────────────────────────────────
def compute_adx(df, period=14):
    try:
        high  = df["High"]
        low   = df["Low"]
        close = df["Close"]

        plus_dm  = high.diff()
        minus_dm = low.diff().abs()

        plus_dm[plus_dm  < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs()
        ], axis=1).max(axis=1)

        atr      = tr.rolling(period).mean()
        plus_di  = 100 * (
            plus_dm.rolling(period).mean() / atr
        )
        minus_di = 100 * (
            minus_dm.rolling(period).mean() / atr
        )
        dx       = (
            100 * (plus_di - minus_di).abs()
            / (plus_di + minus_di)
        )
        adx      = dx.rolling(period).mean()

        return float(adx.iloc[-1])
    except Exception:
        return 20.0

# ── Volatility Regime ─────────────────────────────────
def detect_volatility_regime(df):
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]

    # Historical Volatility (20 day)
    returns  = close.pct_change()
    hv20     = returns.rolling(20).std() * np.sqrt(252) * 100
    hv_curr  = float(hv20.iloc[-1])

    # HV percentile over last year
    hv_1y    = hv20.iloc[-252:] if len(hv20) > 252 \
               else hv20
    hv_pct   = float(
        (hv_1y < hv_curr).sum() / len(hv_1y) * 100
    )

    # ATR as % of price
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr      = tr.rolling(14).mean()
    atr_pct  = float(
        atr.iloc[-1] / close.iloc[-1] * 100
    )

    # Bollinger Band Width
    ma20     = close.rolling(20).mean()
    std20    = close.rolling(20).std()
    bb_width = float(
        (4 * std20.iloc[-1]) / ma20.iloc[-1] * 100
    )

    # Recent range expansion
    range_5d = float(
        (high.iloc[-5:].max() - low.iloc[-5:].min())
        / close.iloc[-1] * 100
    )

    # Classify volatility
    if hv_curr > 25 or atr_pct > 2.5:
        vol_regime = "extreme"
        vol_score  = 4
    elif hv_curr > 18 or atr_pct > 1.8:
        vol_regime = "high"
        vol_score  = 3
    elif hv_curr > 12 or atr_pct > 1.2:
        vol_regime = "normal"
        vol_score  = 2
    elif hv_curr > 8 or atr_pct > 0.8:
        vol_regime = "low"
        vol_score  = 1
    else:
        vol_regime = "very_low"
        vol_score  = 0

    # Strategy implications
    strategy_map = {
        "extreme" : "avoid_naked_options_use_spreads",
        "high"    : "prefer_defined_risk_strategies",
        "normal"  : "all_strategies_valid",
        "low"     : "prefer_buying_options",
        "very_low": "buy_straddles_vol_likely_expand"
    }

    return {
        "vol_regime"      : vol_regime,
        "vol_score"       : vol_score,
        "hv20"            : round(hv_curr, 2),
        "hv_percentile"   : round(hv_pct,  2),
        "atr_pct"         : round(atr_pct, 3),
        "bb_width"        : round(bb_width, 2),
        "range_5d_pct"    : round(range_5d, 2),
        "strategy_hint"   : strategy_map[vol_regime]
    }

# ── Expiry Day Detection ──────────────────────────────
def detect_expiry_day(symbol):
    today       = date.today()
    weekday     = today.weekday()
    expiry_day  = WEEKLY_EXPIRY_DAY.get(symbol, 3)

    is_weekly_expiry  = weekday == expiry_day
    days_to_expiry    = (expiry_day - weekday) % 7
    if days_to_expiry == 0:
        days_to_expiry = 0

    # Monthly expiry — last Thursday of month
    # for NIFTY
    is_monthly_expiry = False
    if is_weekly_expiry:
        # Check if last Thursday of month
        next_week = today.day + 7
        if next_week > 31:
            is_monthly_expiry = True

    # Expiry week
    is_expiry_week = days_to_expiry <= 2

    # Classify expiry context
    if is_weekly_expiry and is_monthly_expiry:
        expiry_type = "monthly_expiry"
        expiry_risk = "extreme"
    elif is_weekly_expiry:
        expiry_type = "weekly_expiry"
        expiry_risk = "high"
    elif is_expiry_week:
        expiry_type = "expiry_week"
        expiry_risk = "elevated"
    else:
        expiry_type = "normal_day"
        expiry_risk = "normal"

    # Strategy implications
    strategy_map = {
        "monthly_expiry": "avoid_long_options_sell_premium",
        "weekly_expiry" : "scalp_only_avoid_overnight",
        "expiry_week"   : "reduce_position_size",
        "normal_day"    : "all_strategies_valid"
    }

    return {
        "today"              : str(today),
        "weekday"            : today.strftime("%A"),
        "is_weekly_expiry"   : is_weekly_expiry,
        "is_monthly_expiry"  : is_monthly_expiry,
        "is_expiry_week"     : is_expiry_week,
        "days_to_expiry"     : days_to_expiry,
        "expiry_type"        : expiry_type,
        "expiry_risk"        : expiry_risk,
        "strategy_hint"      : strategy_map[expiry_type]
    }

# ── Intraday Phase Detection ──────────────────────────
def detect_intraday_phase():
    now    = datetime.now()
    hour   = now.hour
    minute = now.minute
    time   = hour * 100 + minute

    # Indian market hours 9:15 AM to 3:30 PM
    if time < 915:
        phase   = "pre_market"
        caution = "wait_for_open"
    elif time <= 945:
        phase   = "opening_volatility"
        caution = "avoid_first_15min"
    elif time <= 1130:
        phase   = "morning_momentum"
        caution = "trend_following_ok"
    elif time <= 1300:
        phase   = "midday_consolidation"
        caution = "low_volume_be_careful"
    elif time <= 1430:
        phase   = "afternoon_trend"
        caution = "watch_for_reversal"
    elif time <= 1515:
        phase   = "closing_volatility"
        caution = "avoid_new_positions"
    elif time <= 1530:
        phase   = "market_close"
        caution = "closing_moves_unreliable"
    else:
        phase   = "after_market"
        caution = "plan_for_tomorrow"

    return {
        "current_time"  : now.strftime("%H:%M"),
        "intraday_phase": phase,
        "caution"       : caution,
        "market_open"   : 915 <= time <= 1530
    }

# ── Market Breadth ────────────────────────────────────
def detect_market_breadth():
    try:
        data     = nse.live_index("NIFTY 50")
        advance  = data.get("advance", {})

        advances  = int(advance.get("advances",  0))
        declines  = int(advance.get("declines",  0))
        unchanged = int(advance.get("unchanged", 0))
        total     = advances + declines + unchanged

        if total == 0:
            return None

        adv_pct  = round(advances  / total * 100, 1)
        dec_pct  = round(declines  / total * 100, 1)
        ad_ratio = round(advances  / declines, 2) \
                   if declines > 0 else 99.0

        # Breadth classification
        if adv_pct > 70:
            breadth = "very_broad_advance"
            signal  = "strong_bullish"
        elif adv_pct > 55:
            breadth = "broad_advance"
            signal  = "bullish"
        elif dec_pct > 70:
            breadth = "very_broad_decline"
            signal  = "strong_bearish"
        elif dec_pct > 55:
            breadth = "broad_decline"
            signal  = "bearish"
        else:
            breadth = "mixed"
            signal  = "neutral"

        return {
            "advances"   : advances,
            "declines"   : declines,
            "unchanged"  : unchanged,
            "adv_pct"    : adv_pct,
            "dec_pct"    : dec_pct,
            "ad_ratio"   : ad_ratio,
            "breadth"    : breadth,
            "signal"     : signal
        }

    except Exception as e:
        print(f"  ⚠ Breadth fetch failed: {e}")
        return None

# ── Regime Fusion ─────────────────────────────────────
def fuse_regimes(trend, volatility,
                 expiry, intraday, breadth):

    # Build overall regime label
    parts = []

    # Trend component
    if "uptrend" in trend["trend_regime"]:
        parts.append("trending_up")
    elif "downtrend" in trend["trend_regime"]:
        parts.append("trending_down")
    else:
        parts.append("sideways")

    # Volatility component
    parts.append(volatility["vol_regime"] + "_vol")

    # Expiry component
    if expiry["is_weekly_expiry"]:
        parts.append("expiry_day")
    elif expiry["is_expiry_week"]:
        parts.append("expiry_week")

    overall_regime = "_".join(parts)

    # Strategy recommendation
    strategies     = []
    avoid          = []

    # Based on trend
    if "strong_uptrend" in trend["trend_regime"]:
        strategies.append("bull_call_spread")
        strategies.append("sell_puts")
    elif "uptrend" in trend["trend_regime"]:
        strategies.append("bull_put_spread")
    elif "strong_downtrend" in trend["trend_regime"]:
        strategies.append("bear_put_spread")
        strategies.append("sell_calls")
    elif "downtrend" in trend["trend_regime"]:
        strategies.append("bear_call_spread")
    else:
        strategies.append("iron_condor")
        strategies.append("short_straddle")

    # Based on volatility
    if volatility["vol_regime"] in [
        "extreme", "high"
    ]:
        avoid.append("naked_options")
        avoid.append("long_straddle")
        strategies = [
            s for s in strategies
            if "spread" in s or "condor" in s
        ]
    elif volatility["vol_regime"] in [
        "low", "very_low"
    ]:
        strategies.append("long_straddle")
        strategies.append("long_strangle")
        avoid.append("short_premium")

    # Based on expiry
    if expiry["is_weekly_expiry"]:
        avoid.append("overnight_positions")
        avoid.append("long_options")
        strategies = ["expiry_scalping",
                      "sell_premium_spreads"]

    # Regime score (0-100)
    score = 50
    score += trend["trend_score"] * 5
    if volatility["vol_regime"] == "normal":
        score += 10
    if expiry["expiry_type"] == "normal_day":
        score += 10
    if breadth and breadth["signal"] in [
        "bullish", "strong_bullish"
    ]:
        score += 10
    score = max(0, min(100, score))

    # Confidence in regime
    confidence = round(
        abs(trend["trend_score"]) / 8, 2
    )
    confidence = max(0.3, min(0.95, confidence))

    return {
        "overall_regime"   : overall_regime,
        "regime_score"     : score,
        "confidence"       : confidence,
        "recommended"      : strategies[:3],
        "avoid"            : avoid,
        "primary_bias"     : (
            "bullish"  if trend["trend_score"] > 1
            else "bearish" if trend["trend_score"] < -1
            else "neutral"
        ),
        "tradeable"        : (
            intraday["market_open"] and
            expiry["expiry_type"] != "monthly_expiry"
        ),
        "caution_level"    : (
            "extreme" if expiry["is_weekly_expiry"]
            and volatility["vol_regime"] == "high"
            else "high" if expiry["is_expiry_week"]
            or volatility["vol_regime"] == "high"
            else "normal"
        )
    }

# ── Full Regime Detection ─────────────────────────────
def detect_full_regime(symbol):
    print(f"\n{'=' * 55}")
    print(f"  Regime Detector — {symbol}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 55}")

    # Load historical data
    path = os.path.join(
        DATA_DIR, f"{symbol}_daily.csv"
    )
    if not os.path.exists(path):
        print(f"  ⚠ No data for {symbol}. "
              f"Run fetch_data.py first.")
        return None

    df = pd.read_csv(
        path, index_col="Date", parse_dates=True
    )

    if len(df) < 50:
        print(f"  ⚠ Not enough data ({len(df)} rows)")
        return None

    # Run all detectors
    print(f"  🔍 Detecting trend regime...")
    trend = detect_trend_regime(df)

    print(f"  📊 Detecting volatility regime...")
    volatility = detect_volatility_regime(df)

    print(f"  📅 Detecting expiry context...")
    expiry = detect_expiry_day(symbol)

    print(f"  ⏰ Detecting intraday phase...")
    intraday = detect_intraday_phase()

    print(f"  🌐 Fetching market breadth...")
    breadth = detect_market_breadth()

    # Fuse all regimes
    print(f"  🔀 Fusing regimes...")
    fusion = fuse_regimes(
        trend, volatility, expiry,
        intraday, breadth or {}
    )

    # Build complete report
    report = {
        "symbol"    : symbol,
        "timestamp" : datetime.now().isoformat(),
        "trend"     : trend,
        "volatility": volatility,
        "expiry"    : expiry,
        "intraday"  : intraday,
        "breadth"   : breadth,
        "fusion"    : fusion
    }

    # Print summary
    print(f"\n  {'─' * 50}")
    print(f"  📊 REGIME SUMMARY — {symbol}")
    print(f"  {'─' * 50}")
    print(f"  Trend Regime    : "
          f"{trend['trend_regime'].upper()}")
    print(f"  Trend Strength  : "
          f"{trend['trend_strength'].upper()}")
    print(f"  ADX             : {trend['adx']}")
    print(f"  Vol Regime      : "
          f"{volatility['vol_regime'].upper()}")
    print(f"  HV20            : {volatility['hv20']}%")
    print(f"  ATR %           : {volatility['atr_pct']}%")
    print(f"  Expiry Type     : "
          f"{expiry['expiry_type'].upper()}")
    print(f"  Days to Expiry  : "
          f"{expiry['days_to_expiry']}")
    print(f"  Market Phase    : "
          f"{intraday['intraday_phase'].upper()}")
    if breadth:
        print(f"  Breadth Signal  : "
              f"{breadth['signal'].upper()}")
        print(f"  Advances/Declines: "
              f"{breadth['advances']}/"
              f"{breadth['declines']}")

    print(f"\n  {'─' * 50}")
    print(f"  🎯 REGIME FUSION")
    print(f"  {'─' * 50}")
    print(f"  Overall Regime  : "
          f"{fusion['overall_regime'].upper()}")
    print(f"  Primary Bias    : "
          f"{fusion['primary_bias'].upper()}")
    print(f"  Regime Score    : "
          f"{fusion['regime_score']}/100")
    print(f"  Confidence      : "
          f"{fusion['confidence']}")
    print(f"  Tradeable       : "
          f"{'✅ YES' if fusion['tradeable'] else '❌ NO'}")
    print(f"  Caution Level   : "
          f"{fusion['caution_level'].upper()}")

    print(f"\n  ✅ Recommended Strategies:")
    for s in fusion["recommended"]:
        print(f"     → {s}")

    if fusion["avoid"]:
        print(f"\n  ❌ Avoid:")
        for a in fusion["avoid"]:
            print(f"     → {a}")

    # Strategy hints
    print(f"\n  💡 Context Hints:")
    print(f"     Vol Hint    : "
          f"{volatility['strategy_hint']}")
    print(f"     Expiry Hint : "
          f"{expiry['strategy_hint']}")
    print(f"     Phase Hint  : "
          f"{intraday['caution']}")

    print(f"  {'─' * 50}")

    # Save report
    path = os.path.join(
        OUTPUT_DIR,
        f"{symbol}_regime_"
        f"{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    )
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  ✅ Regime saved → {path}")
    return report

# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Trading AI — Regime Detector")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    for symbol in ["NIFTY", "BANKNIFTY"]:
        detect_full_regime(symbol)

    print("\n  ✅ Regime Detection complete!")
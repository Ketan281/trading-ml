"""
Multi-indicator technical analysis engine for forex.

Computes 20+ indicators on a candle DataFrame and returns a structured signal
dict with each indicator's value + directional bias (bullish / bearish / neutral).
Designed to run on any timeframe — the confluence layer calls this once per TF.

Every indicator returns a dict: {"value": ..., "signal": "bullish"|"bearish"|"neutral"}
"""

import numpy as np
import pandas as pd


# ── Trend indicators ─────────────────────────────────

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def ema_crossover(df):
    """EMA 20/50/200 alignment — strongest when price > EMA20 > EMA50 > EMA200."""
    c = df["Close"]
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    p = c.iloc[-1]
    v20, v50, v200 = e20.iloc[-1], e50.iloc[-1], e200.iloc[-1]
    if p > v20 > v50 > v200:
        sig = "bullish"
    elif p < v20 < v50 < v200:
        sig = "bearish"
    else:
        sig = "neutral"
    return {"value": {"ema20": round(v20, 5), "ema50": round(v50, 5),
                      "ema200": round(v200, 5), "price": round(p, 5)},
            "signal": sig, "name": "EMA Crossover"}


def adx(df, period=14):
    """Average Directional Index — trend strength. >25 = trending."""
    high, low, close = df["High"], df["Low"], df["Close"]
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    mask = plus_dm < minus_dm
    plus_dm[mask] = 0
    minus_dm[~mask] = 0
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx_val = dx.ewm(span=period, adjust=False).mean().iloc[-1]
    pdi, mdi = plus_di.iloc[-1], minus_di.iloc[-1]
    if adx_val > 25 and pdi > mdi:
        sig = "bullish"
    elif adx_val > 25 and mdi > pdi:
        sig = "bearish"
    else:
        sig = "neutral"
    return {"value": round(float(adx_val), 2), "plus_di": round(float(pdi), 2),
            "minus_di": round(float(mdi), 2), "signal": sig, "name": "ADX"}


def supertrend(df, period=10, multiplier=3.0):
    """Supertrend — trend-following overlay."""
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    st.iloc[0] = upper.iloc[0]
    direction.iloc[0] = -1
    for i in range(1, len(df)):
        if close.iloc[i] > upper.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]
        if direction.iloc[i] == 1:
            st.iloc[i] = max(lower.iloc[i], st.iloc[i - 1]) if direction.iloc[i - 1] == 1 else lower.iloc[i]
        else:
            st.iloc[i] = min(upper.iloc[i], st.iloc[i - 1]) if direction.iloc[i - 1] == -1 else upper.iloc[i]
    d = int(direction.iloc[-1])
    return {"value": round(float(st.iloc[-1]), 5),
            "signal": "bullish" if d == 1 else "bearish", "name": "Supertrend"}


def ichimoku(df):
    """Ichimoku Cloud — conversion/base cross + price vs cloud."""
    high, low, close = df["High"], df["Low"], df["Close"]
    conv = (high.rolling(9).max() + low.rolling(9).min()) / 2
    base = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((conv + base) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    p = close.iloc[-1]
    sa = span_a.iloc[-1] if not np.isnan(span_a.iloc[-1]) else p
    sb = span_b.iloc[-1] if not np.isnan(span_b.iloc[-1]) else p
    cloud_top = max(sa, sb)
    cloud_bot = min(sa, sb)
    cv, bv = conv.iloc[-1], base.iloc[-1]
    if p > cloud_top and cv > bv:
        sig = "bullish"
    elif p < cloud_bot and cv < bv:
        sig = "bearish"
    else:
        sig = "neutral"
    return {"value": {"conversion": round(float(cv), 5), "base": round(float(bv), 5),
                      "cloud_top": round(float(cloud_top), 5), "cloud_bot": round(float(cloud_bot), 5)},
            "signal": sig, "name": "Ichimoku Cloud"}


# ── Momentum indicators ─────────────────────────────

def rsi(df, period=14):
    """RSI — overbought >70, oversold <30."""
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs = gain / (loss + 1e-10)
    val = float(100 - 100 / (1 + rs.iloc[-1]))
    if val < 30:
        sig = "bullish"
    elif val > 70:
        sig = "bearish"
    else:
        sig = "neutral"
    return {"value": round(val, 2), "signal": sig, "name": "RSI"}


def macd(df):
    """MACD — histogram cross."""
    c = df["Close"]
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line
    val = float(hist.iloc[-1])
    prev = float(hist.iloc[-2]) if len(hist) > 1 else 0
    if val > 0 and val > prev:
        sig = "bullish"
    elif val < 0 and val < prev:
        sig = "bearish"
    else:
        sig = "neutral"
    return {"value": {"macd": round(float(macd_line.iloc[-1]), 5),
                      "signal": round(float(signal_line.iloc[-1]), 5),
                      "histogram": round(val, 5)},
            "signal": sig, "name": "MACD"}


def stochastic(df, k_period=14, d_period=3):
    """Stochastic %K/%D — overbought >80, oversold <20."""
    low_min = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    k = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-10)
    d = k.rolling(d_period).mean()
    kv, dv = float(k.iloc[-1]), float(d.iloc[-1])
    if kv < 20 and kv > dv:
        sig = "bullish"
    elif kv > 80 and kv < dv:
        sig = "bearish"
    else:
        sig = "neutral"
    return {"value": {"k": round(kv, 2), "d": round(dv, 2)},
            "signal": sig, "name": "Stochastic"}


def cci(df, period=20):
    """Commodity Channel Index."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    val = float((tp.iloc[-1] - ma.iloc[-1]) / (0.015 * md.iloc[-1] + 1e-10))
    if val < -100:
        sig = "bullish"
    elif val > 100:
        sig = "bearish"
    else:
        sig = "neutral"
    return {"value": round(val, 2), "signal": sig, "name": "CCI"}


def williams_r(df, period=14):
    """Williams %R — overbought > -20, oversold < -80."""
    high_max = df["High"].rolling(period).max()
    low_min = df["Low"].rolling(period).min()
    val = float(-100 * (high_max.iloc[-1] - df["Close"].iloc[-1]) /
                (high_max.iloc[-1] - low_min.iloc[-1] + 1e-10))
    if val < -80:
        sig = "bullish"
    elif val > -20:
        sig = "bearish"
    else:
        sig = "neutral"
    return {"value": round(val, 2), "signal": sig, "name": "Williams %R"}


def momentum(df, period=10):
    """Price momentum — simple rate of change."""
    c = df["Close"]
    val = float((c.iloc[-1] / c.iloc[-period] - 1) * 100) if len(c) > period else 0
    sig = "bullish" if val > 0.1 else ("bearish" if val < -0.1 else "neutral")
    return {"value": round(val, 3), "signal": sig, "name": "Momentum"}


# ── Volatility indicators ────────────────────────────

def atr(df, period=14):
    """Average True Range — volatility measure."""
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    val = float(tr.ewm(span=period, adjust=False).mean().iloc[-1])
    return {"value": round(val, 5), "signal": "neutral", "name": "ATR"}


def bollinger_bands(df, period=20, std_mult=2):
    """Bollinger Bands — price vs upper/lower band."""
    c = df["Close"]
    ma = c.rolling(period).mean()
    std = c.rolling(period).std()
    upper = ma + std_mult * std
    lower = ma - std_mult * std
    p = c.iloc[-1]
    uv, lv, mv = upper.iloc[-1], lower.iloc[-1], ma.iloc[-1]
    width = (uv - lv) / mv if mv else 0
    if p <= lv:
        sig = "bullish"
    elif p >= uv:
        sig = "bearish"
    else:
        sig = "neutral"
    return {"value": {"upper": round(float(uv), 5), "middle": round(float(mv), 5),
                      "lower": round(float(lv), 5), "width": round(float(width), 5)},
            "signal": sig, "name": "Bollinger Bands"}


def keltner_channel(df, period=20, atr_mult=1.5):
    """Keltner Channel — EMA + ATR envelope."""
    c = df["Close"]
    mid = c.ewm(span=period, adjust=False).mean()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - c.shift()).abs(),
                    (df["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    atr_val = tr.ewm(span=period, adjust=False).mean()
    upper = mid + atr_mult * atr_val
    lower = mid - atr_mult * atr_val
    p = c.iloc[-1]
    if p <= lower.iloc[-1]:
        sig = "bullish"
    elif p >= upper.iloc[-1]:
        sig = "bearish"
    else:
        sig = "neutral"
    return {"value": {"upper": round(float(upper.iloc[-1]), 5),
                      "middle": round(float(mid.iloc[-1]), 5),
                      "lower": round(float(lower.iloc[-1]), 5)},
            "signal": sig, "name": "Keltner Channel"}


# ── Volume indicators ────────────────────────────────

def obv(df):
    """On-Balance Volume — cumulative volume pressure."""
    c = df["Close"]
    v = df["Volume"]
    direction = np.sign(c.diff())
    obv_series = (direction * v).cumsum()
    obv_ema = obv_series.ewm(span=20, adjust=False).mean()
    val = float(obv_series.iloc[-1])
    ev = float(obv_ema.iloc[-1])
    sig = "bullish" if val > ev else ("bearish" if val < ev else "neutral")
    return {"value": round(val, 0), "signal": sig, "name": "OBV"}


def vwap(df):
    """Volume-Weighted Average Price — price vs VWAP."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].replace(0, 1)
    vwap_val = (typical * vol).cumsum() / vol.cumsum()
    p = df["Close"].iloc[-1]
    vv = float(vwap_val.iloc[-1])
    sig = "bullish" if p > vv else "bearish"
    return {"value": round(vv, 5), "signal": sig, "name": "VWAP"}


# ── Support / Resistance ─────────────────────────────

def pivot_points(df):
    """Classic pivot points from previous session."""
    prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
    h, l, c = float(prev["High"]), float(prev["Low"]), float(prev["Close"])
    pp = (h + l + c) / 3
    r1 = 2 * pp - l
    s1 = 2 * pp - h
    r2 = pp + (h - l)
    s2 = pp - (h - l)
    price = float(df["Close"].iloc[-1])
    if price > r1:
        sig = "bullish"
    elif price < s1:
        sig = "bearish"
    else:
        sig = "neutral"
    return {"value": {"pp": round(pp, 5), "r1": round(r1, 5), "r2": round(r2, 5),
                      "s1": round(s1, 5), "s2": round(s2, 5)},
            "signal": sig, "name": "Pivot Points"}


def fibonacci_retracement(df, lookback=50):
    """Fibonacci retracement levels from recent swing high/low."""
    window = df.tail(lookback)
    high = float(window["High"].max())
    low = float(window["Low"].min())
    diff = high - low
    levels = {
        "0.0": high, "0.236": high - 0.236 * diff, "0.382": high - 0.382 * diff,
        "0.5": high - 0.5 * diff, "0.618": high - 0.618 * diff, "1.0": low,
    }
    price = float(df["Close"].iloc[-1])
    ratio = (high - price) / diff if diff else 0.5
    if ratio < 0.382:
        sig = "bullish"
    elif ratio > 0.618:
        sig = "bearish"
    else:
        sig = "neutral"
    return {"value": {k: round(v, 5) for k, v in levels.items()},
            "retracement": round(ratio, 3), "signal": sig, "name": "Fibonacci"}


def support_resistance(df, lookback=100):
    """Key support/resistance from recent price action clusters."""
    window = df.tail(lookback)
    highs = window["High"].values
    lows = window["Low"].values
    price = float(df["Close"].iloc[-1])
    all_levels = np.concatenate([highs, lows])
    bins = np.histogram(all_levels, bins=20)
    top_bins = np.argsort(bins[0])[-3:]
    levels = sorted([round(float((bins[1][i] + bins[1][i + 1]) / 2), 5)
                     for i in top_bins])
    nearest_sup = max([l for l in levels if l < price], default=None)
    nearest_res = min([l for l in levels if l > price], default=None)
    return {"value": {"support": nearest_sup, "resistance": nearest_res,
                      "levels": levels},
            "signal": "neutral", "name": "S/R Levels"}


# ── Additional momentum / volatility / structure ─────

def roc(df, period=12):
    """Rate of Change — momentum oscillator."""
    c = df["Close"]
    if len(c) <= period:
        return {"value": 0, "signal": "neutral", "name": "ROC"}
    val = float((c.iloc[-1] / c.iloc[-period] - 1) * 100)
    sig = "bullish" if val > 1.0 else ("bearish" if val < -1.0 else "neutral")
    return {"value": round(val, 3), "signal": sig, "name": "ROC"}


def mfi(df, period=14):
    """Money Flow Index — volume-weighted RSI (overbought >80, oversold <20)."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    mf = tp * df["Volume"].replace(0, 1)
    pos_mf = pd.Series(0.0, index=df.index)
    neg_mf = pd.Series(0.0, index=df.index)
    delta = tp.diff()
    pos_mf[delta > 0] = mf[delta > 0]
    neg_mf[delta < 0] = mf[delta < 0]
    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum()
    ratio = pos_sum / (neg_sum + 1e-10)
    val = float(100 - 100 / (1 + ratio.iloc[-1]))
    if val < 20:
        sig = "bullish"
    elif val > 80:
        sig = "bearish"
    else:
        sig = "neutral"
    return {"value": round(val, 2), "signal": sig, "name": "MFI"}


def donchian_channel(df, period=20):
    """Donchian Channel — breakout bands (price at upper = bullish breakout)."""
    upper = df["High"].rolling(period).max()
    lower = df["Low"].rolling(period).min()
    mid = (upper + lower) / 2
    p = df["Close"].iloc[-1]
    uv, lv, mv = float(upper.iloc[-1]), float(lower.iloc[-1]), float(mid.iloc[-1])
    if p >= uv:
        sig = "bullish"
    elif p <= lv:
        sig = "bearish"
    else:
        sig = "neutral"
    return {"value": {"upper": round(uv, 5), "middle": round(mv, 5),
                      "lower": round(lv, 5)},
            "signal": sig, "name": "Donchian Channel"}


def trix(df, period=15):
    """TRIX — triple-smoothed EMA rate of change, trend confirmation."""
    c = df["Close"]
    e1 = c.ewm(span=period, adjust=False).mean()
    e2 = e1.ewm(span=period, adjust=False).mean()
    e3 = e2.ewm(span=period, adjust=False).mean()
    trix_line = e3.pct_change() * 100
    val = float(trix_line.iloc[-1]) if not np.isnan(trix_line.iloc[-1]) else 0
    prev = float(trix_line.iloc[-2]) if len(trix_line) > 1 and not np.isnan(trix_line.iloc[-2]) else 0
    if val > 0 and val > prev:
        sig = "bullish"
    elif val < 0 and val < prev:
        sig = "bearish"
    else:
        sig = "neutral"
    return {"value": round(val, 5), "signal": sig, "name": "TRIX"}


def dema(df, period=21):
    """Double EMA — faster trend indicator, less lag than single EMA."""
    c = df["Close"]
    e = c.ewm(span=period, adjust=False).mean()
    de = 2 * e - e.ewm(span=period, adjust=False).mean()
    p = c.iloc[-1]
    dv = float(de.iloc[-1])
    sig = "bullish" if p > dv else ("bearish" if p < dv else "neutral")
    return {"value": round(dv, 5), "signal": sig, "name": "DEMA"}


def chaikin_volatility(df, period=10, roc_period=10):
    """Chaikin Volatility — rate of change of high-low spread EMA."""
    hl = df["High"] - df["Low"]
    hl_ema = hl.ewm(span=period, adjust=False).mean()
    if len(hl_ema) <= roc_period:
        return {"value": 0, "signal": "neutral", "name": "Chaikin Volatility"}
    val = float((hl_ema.iloc[-1] / hl_ema.iloc[-roc_period] - 1) * 100)
    if val > 5:
        sig = "bullish"
    elif val < -5:
        sig = "bearish"
    else:
        sig = "neutral"
    return {"value": round(val, 3), "signal": sig, "name": "Chaikin Volatility"}


def ultimate_oscillator(df, p1=7, p2=14, p3=28):
    """Ultimate Oscillator — multi-period weighted momentum."""
    high, low, close = df["High"], df["Low"], df["Close"]
    bp = close - pd.concat([low, close.shift()], axis=1).min(axis=1)
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    a1 = bp.rolling(p1).sum() / (tr.rolling(p1).sum() + 1e-10)
    a2 = bp.rolling(p2).sum() / (tr.rolling(p2).sum() + 1e-10)
    a3 = bp.rolling(p3).sum() / (tr.rolling(p3).sum() + 1e-10)
    uo = float(100 * (4 * a1.iloc[-1] + 2 * a2.iloc[-1] + a3.iloc[-1]) / 7)
    if uo < 30:
        sig = "bullish"
    elif uo > 70:
        sig = "bearish"
    else:
        sig = "neutral"
    return {"value": round(uo, 2), "signal": sig, "name": "Ultimate Oscillator"}


def parabolic_sar(df, af_start=0.02, af_max=0.2):
    """Parabolic SAR — trend-following stop-and-reverse system."""
    high, low, close = df["High"].values, df["Low"].values, df["Close"].values
    n = len(df)
    if n < 3:
        return {"value": 0, "signal": "neutral", "name": "Parabolic SAR"}
    sar = np.zeros(n)
    trend = np.ones(n, dtype=int)
    af = af_start
    ep = high[0]
    sar[0] = low[0]
    for i in range(1, n):
        sar[i] = sar[i-1] + af * (ep - sar[i-1])
        if trend[i-1] == 1:
            if low[i] < sar[i]:
                trend[i] = -1
                sar[i] = ep
                ep = low[i]
                af = af_start
            else:
                trend[i] = 1
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_start, af_max)
        else:
            if high[i] > sar[i]:
                trend[i] = 1
                sar[i] = ep
                ep = high[i]
                af = af_start
            else:
                trend[i] = -1
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_start, af_max)
    sig = "bullish" if trend[-1] == 1 else "bearish"
    return {"value": round(float(sar[-1]), 5), "signal": sig, "name": "Parabolic SAR"}


# ── Candlestick patterns ─────────────────────────────

def candle_patterns(df):
    """Detect common reversal/continuation candlestick patterns."""
    if len(df) < 3:
        return {"value": "insufficient_data", "signal": "neutral", "name": "Candle Patterns"}
    o, h, l, c = [df[col].iloc[-1] for col in ("Open", "High", "Low", "Close")]
    o1, h1, l1, c1 = [df[col].iloc[-2] for col in ("Open", "High", "Low", "Close")]
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    full_range = h - l if h != l else 1e-10

    patterns = []
    # Hammer / Hanging Man
    if lower_wick > 2 * body and upper_wick < body * 0.3:
        patterns.append("hammer" if c > o else "hanging_man")
    # Inverted Hammer / Shooting Star
    if upper_wick > 2 * body and lower_wick < body * 0.3:
        patterns.append("inverted_hammer" if c > o else "shooting_star")
    # Doji
    if body < full_range * 0.1:
        patterns.append("doji")
    # Engulfing
    if c > o and c1 < o1 and c > o1 and o < c1:
        patterns.append("bullish_engulfing")
    if c < o and c1 > o1 and c < o1 and o > c1:
        patterns.append("bearish_engulfing")
    # Marubozu
    if body > full_range * 0.9:
        patterns.append("bullish_marubozu" if c > o else "bearish_marubozu")

    bullish = sum(1 for p in patterns if "bullish" in p or p in ("hammer", "inverted_hammer"))
    bearish = sum(1 for p in patterns if "bearish" in p or p in ("hanging_man", "shooting_star"))
    sig = "bullish" if bullish > bearish else ("bearish" if bearish > bullish else "neutral")
    return {"value": patterns if patterns else ["none"], "signal": sig, "name": "Candle Patterns"}


# ── Master function ──────────────────────────────────

ALL_INDICATORS = [
    # Trend
    ema_crossover, adx, supertrend, ichimoku, dema, parabolic_sar,
    # Momentum
    rsi, macd, stochastic, cci, williams_r, momentum, roc, mfi,
    ultimate_oscillator, trix,
    # Volatility
    atr, bollinger_bands, keltner_channel, donchian_channel, chaikin_volatility,
    # Volume
    obv, vwap,
    # Support / Resistance
    pivot_points, fibonacci_retracement, support_resistance,
    # Patterns
    candle_patterns,
]


def analyse(df: pd.DataFrame) -> dict:
    """Run all indicators on a candle DataFrame. Returns {name: result_dict}."""
    if df is None or len(df) < 30:
        return {}
    results = {}
    for fn in ALL_INDICATORS:
        try:
            r = fn(df)
            results[r["name"]] = r
        except Exception:
            pass
    return results

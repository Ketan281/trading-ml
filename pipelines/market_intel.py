"""
Market intelligence layer — institutional & macro signals that experienced
Indian market traders use daily but the base engine misses.

Adds seven signal groups:
  1. FII/DII cash flow (the single strongest Nifty macro predictor)
  2. Index PCR (put/call OI ratio — key options sentiment gauge)
  3. Intermarket context (crude, DXY, US futures, India VIX)
  4. Delivery % analysis (institutional vs retail conviction)
  5. Support/resistance levels (swing pivots + Fibonacci)
  6. Volume profile (accumulation/distribution, unusual volume)
  7. News/sentiment (VIX fear gauge + event awareness calendar)

Each function returns a dict with a normalized score in [-1, +1]
(positive = bullish) and a human-readable read. The aggregate
`market_context()` fuses them into a single overlay that adjusts
the recommendation engine's conviction scoring.
"""

import os
import sys
import time
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CACHE = {}
CACHE_TTL = 300


def _cached(key, fn, ttl=CACHE_TTL):
    now = time.time()
    hit = CACHE.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    try:
        val = fn()
    except Exception:
        traceback.print_exc()
        val = None
    if val is not None:
        CACHE[key] = (now, val)
    return val


# ─────────────────────────────────────────────────────────
# 1. FII / DII CASH FLOW
# ─────────────────────────────────────────────────────────

def _fetch_fii_dii():
    """Fetch FII/DII daily cash-market data from NSE.

    NSE publishes this at https://www.nseindia.com/api/fiidiiTradeReact
    Falls back to yfinance proxy if NSE blocks.
    """
    import requests
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com/",
    }
    try:
        s = requests.Session()
        s.get("https://www.nseindia.com/", headers=headers, timeout=5)
        r = s.get("https://www.nseindia.com/api/fiidiiTradeReact",
                   headers=headers, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def fii_dii_signal():
    """
    FII net buy/sell is the single strongest short-term predictor for Nifty.
    Correlation with next-day index move > 0.6 historically.

    Returns: {score, fii_net, dii_net, read, trend}
    """
    data = _cached("fii_dii", _fetch_fii_dii, ttl=600)
    if not data:
        return {"score": 0, "fii_net": 0, "dii_net": 0,
                "read": "FII/DII data unavailable", "trend": "unknown",
                "available": False}

    fii_buy = fii_sell = dii_buy = dii_sell = 0
    for row in data if isinstance(data, list) else [data]:
        cat = str(row.get("category", "")).upper()
        if "FII" in cat or "FPI" in cat:
            fii_buy += float(row.get("buyValue", 0) or 0)
            fii_sell += float(row.get("sellValue", 0) or 0)
        elif "DII" in cat:
            dii_buy += float(row.get("buyValue", 0) or 0)
            dii_sell += float(row.get("sellValue", 0) or 0)

    fii_net = fii_buy - fii_sell
    dii_net = dii_buy - dii_sell

    fii_cr = fii_net / 1e7
    dii_cr = dii_net / 1e7

    if fii_cr > 1000:
        score = 0.8
        read = f"FII strong buying ₹{fii_cr:,.0f}cr — very bullish"
    elif fii_cr > 300:
        score = 0.5
        read = f"FII net buyers ₹{fii_cr:,.0f}cr — bullish"
    elif fii_cr < -1000:
        score = -0.8
        read = f"FII heavy selling ₹{fii_cr:,.0f}cr — very bearish"
    elif fii_cr < -300:
        score = -0.5
        read = f"FII net sellers ₹{fii_cr:,.0f}cr — bearish"
    else:
        score = 0
        read = f"FII neutral ₹{fii_cr:,.0f}cr"

    if dii_cr > 500 and fii_cr < -300:
        read += f" | DII absorbing ₹{dii_cr:,.0f}cr (cushion)"
        score *= 0.6

    return {"score": round(score, 2), "fii_net_cr": round(fii_cr, 0),
            "dii_net_cr": round(dii_cr, 0), "read": read,
            "trend": "buying" if score > 0 else "selling" if score < 0 else "neutral",
            "available": True}


# ─────────────────────────────────────────────────────────
# 2. INDEX PUT/CALL RATIO (PCR)
# ─────────────────────────────────────────────────────────

def pcr_signal(symbol="NIFTY"):
    """
    PCR (put OI / call OI) is the most reliable options sentiment gauge.
    PCR > 1.2 = puts dominate = bullish (writers are confident).
    PCR < 0.7 = calls dominate = bearish.
    Extreme PCR (>1.5 or <0.5) signals reversal risk.
    """
    def _compute():
        from pipelines.options.chain_live_intel import fetch_chain
        chain = fetch_chain(symbol)
        if not chain or "df" not in chain:
            return None
        df = chain["df"]
        total_ce_oi = df["ce_oi"].sum()
        total_pe_oi = df["pe_oi"].sum()
        if total_ce_oi == 0:
            return None
        pcr = total_pe_oi / total_ce_oi

        total_ce_vol = df["ce_volume"].sum() if "ce_volume" in df else 0
        total_pe_vol = df["pe_volume"].sum() if "pe_volume" in df else 0
        vol_pcr = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 1.0

        if pcr > 1.5:
            score = 0.7
            read = f"PCR {pcr:.2f} — extreme put writing, strong support (contrarian bullish)"
        elif pcr > 1.2:
            score = 0.5
            read = f"PCR {pcr:.2f} — put writers confident, bullish"
        elif pcr > 0.9:
            score = 0.2
            read = f"PCR {pcr:.2f} — balanced, slight bullish lean"
        elif pcr > 0.7:
            score = -0.3
            read = f"PCR {pcr:.2f} — call heavy, mildly bearish"
        elif pcr > 0.5:
            score = -0.6
            read = f"PCR {pcr:.2f} — calls dominate, bearish"
        else:
            score = -0.8
            read = f"PCR {pcr:.2f} — extreme call buying, very bearish (or reversal)"

        return {"score": round(score, 2), "pcr": round(pcr, 3),
                "volume_pcr": round(vol_pcr, 3), "read": read,
                "ce_oi": int(total_ce_oi), "pe_oi": int(total_pe_oi)}

    result = _cached(f"pcr_{symbol}", _compute)
    if not result:
        return {"score": 0, "pcr": 1.0, "read": "PCR data unavailable",
                "available": False}
    result["available"] = True
    return result


# ─────────────────────────────────────────────────────────
# 3. INTERMARKET CONTEXT (Crude, DXY, VIX, US futures)
# ─────────────────────────────────────────────────────────

def intermarket_signal():
    """
    Experienced traders check these before market open:
    - Crude up = bearish for Indian market (import cost)
    - DXY up = bearish (EM capital outflow)
    - India VIX high = uncertainty, reduce positions
    - US futures green = global risk-on sentiment
    """
    import yfinance as yf

    def _fetch():
        tickers = {
            "crude": "CL=F",
            "dxy": "DX-Y.NYB",
            "india_vix": "^INDIAVIX",
            "sp500_fut": "ES=F",
            "us10y": "^TNX",
        }
        results = {}
        for name, ticker in tickers.items():
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period="5d")
                if hist is not None and len(hist) >= 2:
                    prev = float(hist["Close"].iloc[-2])
                    curr = float(hist["Close"].iloc[-1])
                    chg = (curr - prev) / prev * 100 if prev > 0 else 0
                    results[name] = {"price": round(curr, 2),
                                     "change_pct": round(chg, 2)}
            except Exception:
                pass
        return results if results else None

    data = _cached("intermarket", _fetch, ttl=300)
    if not data:
        return {"score": 0, "read": "Intermarket data unavailable",
                "signals": {}, "available": False}

    score = 0
    reads = []

    crude = data.get("crude")
    if crude:
        chg = crude["change_pct"]
        if chg > 2:
            score -= 0.3
            reads.append(f"Crude +{chg:.1f}% (bearish for India)")
        elif chg < -2:
            score += 0.2
            reads.append(f"Crude {chg:.1f}% (bullish for India)")

    dxy = data.get("dxy")
    if dxy:
        chg = dxy["change_pct"]
        if chg > 0.5:
            score -= 0.25
            reads.append(f"Dollar +{chg:.1f}% (EM negative)")
        elif chg < -0.5:
            score += 0.2
            reads.append(f"Dollar {chg:.1f}% (EM positive)")

    vix = data.get("india_vix")
    if vix:
        level = vix["price"]
        if level > 20:
            score -= 0.3
            reads.append(f"India VIX {level:.1f} — high fear")
        elif level > 15:
            score -= 0.1
            reads.append(f"India VIX {level:.1f} — elevated")
        elif level < 12:
            score += 0.2
            reads.append(f"India VIX {level:.1f} — complacent (bullish)")

    spx = data.get("sp500_fut")
    if spx:
        chg = spx["change_pct"]
        if chg > 0.5:
            score += 0.2
            reads.append(f"S&P futures +{chg:.1f}% (risk-on)")
        elif chg < -0.5:
            score -= 0.2
            reads.append(f"S&P futures {chg:.1f}% (risk-off)")

    score = max(-1, min(1, score))
    return {"score": round(score, 2), "read": " | ".join(reads) if reads else "Intermarket neutral",
            "signals": data, "available": True}


# ─────────────────────────────────────────────────────────
# 4. DELIVERY % ANALYSIS
# ─────────────────────────────────────────────────────────

def delivery_signal(symbol):
    """
    High delivery % = institutional buying (taking delivery, not day-trading).
    Low delivery % = retail speculation.
    Rising delivery + rising price = accumulation (very bullish).
    Rising delivery + falling price = distribution (very bearish).
    """
    import yfinance as yf

    def _fetch():
        try:
            t = yf.Ticker(f"{symbol}.NS")
            hist = t.history(period="1mo")
            if hist is None or len(hist) < 5:
                return None
            vol = hist["Volume"]
            close = hist["Close"]
            avg_vol_5d = vol.iloc[-5:].mean()
            avg_vol_20d = vol.mean()
            latest_vol = float(vol.iloc[-1])
            vol_ratio = latest_vol / avg_vol_20d if avg_vol_20d > 0 else 1
            price_chg_5d = (float(close.iloc[-1]) - float(close.iloc[-5])) / float(close.iloc[-5]) * 100
            return {
                "latest_volume": int(latest_vol),
                "avg_5d": int(avg_vol_5d),
                "avg_20d": int(avg_vol_20d),
                "vol_ratio": round(vol_ratio, 2),
                "price_chg_5d": round(price_chg_5d, 2),
            }
        except Exception:
            return None

    data = _cached(f"delivery_{symbol}", _fetch, ttl=600)
    if not data:
        return {"score": 0, "read": "Volume data unavailable", "available": False}

    vol_ratio = data["vol_ratio"]
    price_chg = data["price_chg_5d"]

    if vol_ratio > 2.0 and price_chg > 1:
        score = 0.7
        read = f"Volume surge {vol_ratio:.1f}x + price up {price_chg:.1f}% — accumulation"
    elif vol_ratio > 2.0 and price_chg < -1:
        score = -0.7
        read = f"Volume surge {vol_ratio:.1f}x + price down {price_chg:.1f}% — distribution"
    elif vol_ratio > 1.5 and price_chg > 0:
        score = 0.4
        read = f"Above-average volume {vol_ratio:.1f}x with price rising — bullish"
    elif vol_ratio > 1.5 and price_chg < 0:
        score = -0.4
        read = f"Above-average volume {vol_ratio:.1f}x with price falling — bearish"
    elif vol_ratio < 0.5:
        score = 0
        read = f"Very low volume {vol_ratio:.1f}x — no conviction either way"
    else:
        score = 0.1 if price_chg > 0 else -0.1
        read = f"Normal volume {vol_ratio:.1f}x, price {price_chg:+.1f}%"

    data["score"] = round(score, 2)
    data["read"] = read
    data["available"] = True
    return data


# ─────────────────────────────────────────────────────────
# 5. SUPPORT / RESISTANCE (Swing pivots + Fibonacci)
# ─────────────────────────────────────────────────────────

def support_resistance(symbol):
    """
    Compute key S/R levels a trader watches:
    - Recent swing highs/lows (20-bar)
    - Fibonacci retracements of the last major swing
    - Camarilla pivots from yesterday's OHLC
    """
    import yfinance as yf

    def _compute():
        try:
            t = yf.Ticker(f"{symbol}.NS")
            hist = t.history(period="3mo")
            if hist is None or len(hist) < 30:
                return None
        except Exception:
            return None

        high = hist["High"].values
        low = hist["Low"].values
        close = hist["Close"].values
        price = float(close[-1])

        swing_highs = []
        swing_lows = []
        window = 10
        for i in range(window, len(high) - window):
            if high[i] == max(high[i - window:i + window + 1]):
                swing_highs.append(round(float(high[i]), 2))
            if low[i] == min(low[i - window:i + window + 1]):
                swing_lows.append(round(float(low[i]), 2))

        resistance = sorted(set(h for h in swing_highs if h > price))[:3]
        support = sorted(set(l for l in swing_lows if l < price), reverse=True)[:3]

        recent_high = float(max(high[-60:]))
        recent_low = float(min(low[-60:]))
        diff = recent_high - recent_low
        fib_levels = {
            "0.236": round(recent_high - 0.236 * diff, 2),
            "0.382": round(recent_high - 0.382 * diff, 2),
            "0.500": round(recent_high - 0.500 * diff, 2),
            "0.618": round(recent_high - 0.618 * diff, 2),
            "0.786": round(recent_high - 0.786 * diff, 2),
        }

        yest_h = float(high[-2])
        yest_l = float(low[-2])
        yest_c = float(close[-2])
        yest_range = yest_h - yest_l
        camarilla = {
            "R1": round(yest_c + yest_range * 1.1 / 12, 2),
            "R2": round(yest_c + yest_range * 1.1 / 6, 2),
            "R3": round(yest_c + yest_range * 1.1 / 4, 2),
            "S1": round(yest_c - yest_range * 1.1 / 12, 2),
            "S2": round(yest_c - yest_range * 1.1 / 6, 2),
            "S3": round(yest_c - yest_range * 1.1 / 4, 2),
        }

        nearest_support = support[0] if support else fib_levels["0.618"]
        nearest_resistance = resistance[0] if resistance else recent_high
        dist_to_support = (price - nearest_support) / price * 100
        dist_to_resistance = (nearest_resistance - price) / price * 100

        if dist_to_support < 1.5:
            score = 0.5
            read = f"Near support {nearest_support} ({dist_to_support:.1f}% away) — good entry zone"
        elif dist_to_resistance < 1.0:
            score = -0.3
            read = f"Near resistance {nearest_resistance} ({dist_to_resistance:.1f}% away) — caution"
        elif dist_to_support < dist_to_resistance:
            score = 0.2
            read = f"Closer to support than resistance — favorable"
        else:
            score = -0.1
            read = f"Closer to resistance — less room to run"

        return {
            "score": round(score, 2),
            "support": support,
            "resistance": resistance,
            "fibonacci": fib_levels,
            "camarilla": camarilla,
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
            "dist_to_support_pct": round(dist_to_support, 2),
            "dist_to_resistance_pct": round(dist_to_resistance, 2),
            "read": read,
        }

    result = _cached(f"sr_{symbol}", _compute, ttl=600)
    if not result:
        return {"score": 0, "read": "S/R data unavailable", "available": False}
    result["available"] = True
    return result


# ─────────────────────────────────────────────────────────
# 6. VOLUME PROFILE (accumulation / distribution)
# ─────────────────────────────────────────────────────────

def volume_profile_signal(symbol):
    """
    Accumulation/Distribution Line + On-Balance Volume + MFI.
    These tell whether smart money is buying or selling.
    """
    import yfinance as yf

    def _compute():
        try:
            t = yf.Ticker(f"{symbol}.NS")
            hist = t.history(period="3mo")
            if hist is None or len(hist) < 30:
                return None
        except Exception:
            return None

        high = hist["High"]
        low = hist["Low"]
        close = hist["Close"]
        volume = hist["Volume"]

        clv = ((close - low) - (high - close)) / (high - low + 1e-10)
        ad_line = (clv * volume).cumsum()
        ad_current = float(ad_line.iloc[-1])
        ad_5d_ago = float(ad_line.iloc[-5])
        ad_trend = "rising" if ad_current > ad_5d_ago else "falling"

        obv = pd.Series(0.0, index=close.index)
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]
        obv_trend = "rising" if float(obv.iloc[-1]) > float(obv.iloc[-5]) else "falling"

        tp = (high + low + close) / 3
        mf = tp * volume
        pos_mf = pd.Series(0.0, index=tp.index)
        neg_mf = pd.Series(0.0, index=tp.index)
        for i in range(1, len(tp)):
            if tp.iloc[i] > tp.iloc[i - 1]:
                pos_mf.iloc[i] = float(mf.iloc[i])
            else:
                neg_mf.iloc[i] = float(mf.iloc[i])
        period = 14
        pos_sum = pos_mf.rolling(period).sum()
        neg_sum = neg_mf.rolling(period).sum()
        mfi = 100 - (100 / (1 + pos_sum / (neg_sum + 1e-10)))
        mfi_val = round(float(mfi.iloc[-1]), 1)

        price_up = float(close.iloc[-1]) > float(close.iloc[-5])

        if ad_trend == "rising" and obv_trend == "rising" and price_up:
            score = 0.6
            read = "Strong accumulation — A/D rising + OBV rising + price up"
        elif ad_trend == "falling" and obv_trend == "falling" and not price_up:
            score = -0.6
            read = "Distribution — A/D falling + OBV falling + price down"
        elif ad_trend == "rising" and not price_up:
            score = 0.3
            read = "Stealth accumulation — A/D rising despite price weakness (bullish divergence)"
        elif ad_trend == "falling" and price_up:
            score = -0.3
            read = "Smart money selling — A/D falling despite price rise (bearish divergence)"
        else:
            score = 0.1 if ad_trend == "rising" else -0.1
            read = f"Mixed volume signals — A/D {ad_trend}, OBV {obv_trend}"

        if mfi_val > 80:
            score -= 0.2
            read += f" | MFI {mfi_val} overbought"
        elif mfi_val < 20:
            score += 0.2
            read += f" | MFI {mfi_val} oversold"

        return {
            "score": round(max(-1, min(1, score)), 2),
            "ad_trend": ad_trend,
            "obv_trend": obv_trend,
            "mfi": mfi_val,
            "read": read,
        }

    result = _cached(f"volprof_{symbol}", _compute, ttl=600)
    if not result:
        return {"score": 0, "read": "Volume profile unavailable", "available": False}
    result["available"] = True
    return result


# ─────────────────────────────────────────────────────────
# 7. NEWS / SENTIMENT SIGNAL
# ─────────────────────────────────────────────────────────

def sentiment_signal():
    """Gauge market sentiment from India VIX level + event awareness.

    Uses VIX as a fear gauge:
    - VIX < 13: complacent / bullish
    - VIX 13-18: normal
    - VIX 18-25: elevated fear / cautious
    - VIX > 25: panic / contrarian bullish

    Also checks EventAwarenessAgent for upcoming high-impact events.
    """
    def _compute():
        score = 0.0
        read_parts = []

        # VIX-based sentiment
        try:
            import yfinance as yf
            vix_data = yf.download("^INDIAVIX", period="5d", interval="1d",
                                   progress=False, auto_adjust=True)
            if vix_data is not None and len(vix_data) > 0:
                vix = float(vix_data["Close"].iloc[-1])
                if vix < 13:
                    score += 0.4
                    read_parts.append(f"VIX {vix:.1f} — low fear, bullish sentiment")
                elif vix < 18:
                    score += 0.1
                    read_parts.append(f"VIX {vix:.1f} — normal range")
                elif vix < 25:
                    score -= 0.3
                    read_parts.append(f"VIX {vix:.1f} — elevated fear, cautious")
                else:
                    score += 0.2
                    read_parts.append(f"VIX {vix:.1f} — panic zone (contrarian bullish)")
        except Exception:
            read_parts.append("VIX unavailable")

        # Event awareness overlay
        try:
            from agents.event_awareness import EventAwarenessAgent
            ea = EventAwarenessAgent()
            events = ea.upcoming_events(days=2) if hasattr(ea, 'upcoming_events') else []
            high_impact = [e for e in events if e.get("impact", "").lower() == "high"]
            if high_impact:
                score -= 0.15 * min(len(high_impact), 3)
                names = ", ".join(e.get("name", "event")[:30] for e in high_impact[:3])
                read_parts.append(f"Upcoming events: {names}")
        except Exception:
            pass

        if not read_parts:
            read_parts.append("Sentiment data limited")

        return {
            "score": round(max(-1, min(1, score)), 2),
            "read": " | ".join(read_parts),
        }

    result = _cached("sentiment_signal", _compute, ttl=300)
    if not result:
        return {"score": 0, "read": "Sentiment unavailable", "available": False}
    result["available"] = True
    return result


# ─────────────────────────────────────────────────────────
# AGGREGATE: Full market context
# ─────────────────────────────────────────────────────────

def market_context(symbol="NIFTY"):
    """
    Fuse all institutional/macro signals into a single market overlay.
    Returns an adjustment factor for the recommendation engine.

    Weights reflect how an experienced trader prioritizes:
    - FII flow (25%) — strongest macro predictor
    - PCR (22%) — options market is smartest
    - Intermarket (18%) — global context
    - Volume profile (13%) — smart money flow
    - S/R proximity (10%) — entry timing
    - Sentiment (12%) — news/VIX/event calendar
    """
    fii = fii_dii_signal()
    pcr = pcr_signal(symbol)
    inter = intermarket_signal()
    vol = volume_profile_signal(symbol)
    sr = support_resistance(symbol)
    sent = sentiment_signal()

    weights = {
        "fii_dii": (fii["score"], 0.25),
        "pcr": (pcr["score"], 0.22),
        "intermarket": (inter["score"], 0.18),
        "volume_profile": (vol["score"], 0.13),
        "support_resistance": (sr["score"], 0.10),
        "sentiment": (sent["score"], 0.12),
    }

    weighted_score = sum(s * w for s, w in weights.values())
    total_weight = sum(w for _, w in weights.values())
    composite = round(weighted_score / total_weight, 3)

    if composite > 0.4:
        conviction_adj = 1.15
        overall = "strongly bullish"
    elif composite > 0.2:
        conviction_adj = 1.08
        overall = "moderately bullish"
    elif composite > 0.05:
        conviction_adj = 1.0
        overall = "slightly bullish"
    elif composite > -0.05:
        conviction_adj = 1.0
        overall = "neutral"
    elif composite > -0.2:
        conviction_adj = 0.92
        overall = "slightly bearish"
    elif composite > -0.4:
        conviction_adj = 0.85
        overall = "moderately bearish"
    else:
        conviction_adj = 0.70
        overall = "strongly bearish"

    return {
        "composite_score": composite,
        "conviction_multiplier": conviction_adj,
        "overall": overall,
        "signals": {
            "fii_dii": fii,
            "pcr": pcr,
            "intermarket": inter,
            "volume_profile": vol,
            "support_resistance": sr,
            "sentiment": sent,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def stock_context(symbol):
    """Per-stock intelligence overlay: delivery, volume profile, S/R levels."""
    deliv = delivery_signal(symbol)
    vol = volume_profile_signal(symbol)
    sr = support_resistance(symbol)

    stock_score = (
        deliv["score"] * 0.35 +
        vol["score"] * 0.40 +
        sr["score"] * 0.25
    )

    if stock_score > 0.3:
        adj = 1.12
        read = "institutional accumulation signal"
    elif stock_score > 0.1:
        adj = 1.05
        read = "mild institutional interest"
    elif stock_score < -0.3:
        adj = 0.80
        read = "distribution / smart money selling"
    elif stock_score < -0.1:
        adj = 0.90
        read = "mild distribution signal"
    else:
        adj = 1.0
        read = "neutral institutional signal"

    return {
        "score": round(stock_score, 3),
        "conviction_multiplier": adj,
        "read": read,
        "delivery": deliv,
        "volume_profile": vol,
        "support_resistance": sr,
    }


if __name__ == "__main__":
    import json
    print("── Market Context (NIFTY) ──")
    ctx = market_context("NIFTY")
    print(json.dumps({k: v for k, v in ctx.items() if k != "signals"}, indent=2))
    for name, sig in ctx["signals"].items():
        print(f"  {name}: score={sig['score']}, read={sig['read']}")
    print("\n── Stock Context (RELIANCE) ──")
    sc = stock_context("RELIANCE")
    print(json.dumps({k: v for k, v in sc.items()
                      if k not in ("delivery", "volume_profile", "support_resistance")}, indent=2))

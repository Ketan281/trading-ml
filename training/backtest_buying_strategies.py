"""
Backtest top 5 options BUYING strategies on NIFTY/BANKNIFTY.

Strategies tested (used by top traders globally):
  1. Supertrend — most popular algo strategy in India
  2. EMA Crossover (9/21) — classic trend following
  3. RSI Reversal — buy oversold bounces / sell overbought drops
  4. MACD Signal Cross — momentum confirmation
  5. Bollinger Squeeze Breakout — volatility expansion after compression

Each strategy:
  - Generates BUY CE or BUY PE signals daily
  - Entry = ATM option premium on signal day
  - Exit = next day's ATM premium (1-day hold, same as wall selling)
  - Win = premium increased (CE premium up if NIFTY up, PE premium up if NIFTY down)

Tested on same dataset as wall selling: 2021-06 to 2024-07 (734 days).
"""

import os
import sys
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RAW_DIR = os.path.join(ROOT, "data", "option_chain", "raw")
FEAT_DIR = os.path.join(ROOT, "data", "features")


def load_prices(symbol):
    """Load OHLCV daily data from yfinance."""
    cache_path = os.path.join(ROOT, "data", f"{symbol}_ohlcv_cache.csv")
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        if len(df) > 500:
            return df

    import yfinance as yf
    tickers = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}
    data = yf.download(tickers.get(symbol, "^NSEI"),
                       start="2020-01-01", end="2025-01-01", progress=False)
    data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
    data.index.name = "Date"
    data.to_csv(cache_path)
    return data[["Open", "High", "Low", "Close", "Volume"]].copy()


def load_chain_premiums(symbol):
    """Load ATM option premiums for each day from chain files."""
    files = sorted(glob.glob(os.path.join(RAW_DIR, symbol, "*.csv")))
    records = []
    for f in files:
        date_str = os.path.basename(f).replace(".csv", "")
        df = pd.read_csv(f)
        if df.empty:
            continue
        if df["timestamp"].nunique() > 1:
            df = df[df["timestamp"] == df["timestamp"].max()].copy()
        for col in ["ce_ltp", "pe_ltp", "ce_oi", "pe_oi"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        records.append({"date": date_str, "chain": df})
    return records


# ── Strategy 1: Supertrend ──────────────────────────

def supertrend(df, period=10, multiplier=3):
    """Supertrend indicator — the #1 algo strategy in Indian markets."""
    hl2 = (df["High"] + df["Low"]) / 2
    atr = df["High"].sub(df["Low"]).rolling(period).mean()

    up = hl2 - multiplier * atr
    dn = hl2 + multiplier * atr

    trend = pd.Series(1, index=df.index)  # 1=bullish, -1=bearish
    final_up = up.copy()
    final_dn = dn.copy()

    for i in range(1, len(df)):
        if up.iloc[i] > final_up.iloc[i-1]:
            final_up.iloc[i] = up.iloc[i]
        else:
            final_up.iloc[i] = final_up.iloc[i-1]

        if dn.iloc[i] < final_dn.iloc[i-1]:
            final_dn.iloc[i] = dn.iloc[i]
        else:
            final_dn.iloc[i] = final_dn.iloc[i-1]

        if df["Close"].iloc[i] > final_dn.iloc[i-1]:
            trend.iloc[i] = 1
        elif df["Close"].iloc[i] < final_up.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]

    return trend


def strategy_supertrend(df):
    """BUY CE when supertrend flips bullish, BUY PE when flips bearish."""
    trend = supertrend(df)
    signals = pd.Series("HOLD", index=df.index)

    for i in range(1, len(df)):
        if trend.iloc[i] == 1 and trend.iloc[i-1] == -1:
            signals.iloc[i] = "BUY_CE"
        elif trend.iloc[i] == -1 and trend.iloc[i-1] == 1:
            signals.iloc[i] = "BUY_PE"
        elif trend.iloc[i] == 1:
            signals.iloc[i] = "BUY_CE"
        elif trend.iloc[i] == -1:
            signals.iloc[i] = "BUY_PE"

    return signals


# ── Strategy 2: EMA Crossover (9/21) ────────────────

def strategy_ema_cross(df):
    """BUY CE when 9 EMA crosses above 21 EMA, BUY PE when crosses below."""
    ema9 = df["Close"].ewm(span=9).mean()
    ema21 = df["Close"].ewm(span=21).mean()

    signals = pd.Series("HOLD", index=df.index)
    for i in range(1, len(df)):
        if ema9.iloc[i] > ema21.iloc[i]:
            signals.iloc[i] = "BUY_CE"
        elif ema9.iloc[i] < ema21.iloc[i]:
            signals.iloc[i] = "BUY_PE"

    return signals


# ── Strategy 3: RSI Reversal ────────────────────────

def strategy_rsi(df, period=14, ob=70, os_level=30):
    """BUY CE when RSI crosses above 30 (oversold bounce).
    BUY PE when RSI crosses below 70 (overbought reversal)."""
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - 100 / (1 + rs)

    signals = pd.Series("HOLD", index=df.index)
    for i in range(1, len(df)):
        if rsi.iloc[i] > os_level and rsi.iloc[i-1] <= os_level:
            signals.iloc[i] = "BUY_CE"  # oversold bounce
        elif rsi.iloc[i] < ob and rsi.iloc[i-1] >= ob:
            signals.iloc[i] = "BUY_PE"  # overbought reversal
        elif rsi.iloc[i] > 50:
            signals.iloc[i] = "BUY_CE"
        elif rsi.iloc[i] < 50:
            signals.iloc[i] = "BUY_PE"

    return signals


# ── Strategy 4: MACD Signal Cross ───────────────────

def strategy_macd(df):
    """BUY CE when MACD crosses above signal line, BUY PE when below."""
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()

    signals = pd.Series("HOLD", index=df.index)
    for i in range(1, len(df)):
        if macd.iloc[i] > signal.iloc[i]:
            signals.iloc[i] = "BUY_CE"
        elif macd.iloc[i] < signal.iloc[i]:
            signals.iloc[i] = "BUY_PE"

    return signals


# ── Strategy 5: Bollinger Squeeze Breakout ──────────

def strategy_bollinger(df, period=20, std_mult=2):
    """Buy when price breaks out of squeezed Bollinger Bands.
    Squeeze = bands narrowest in 50 days."""
    sma = df["Close"].rolling(period).mean()
    std = df["Close"].rolling(period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma

    # Squeeze = bandwidth in bottom 20% of last 50 days
    bw_pctile = bandwidth.rolling(50).apply(
        lambda x: (x.iloc[-1] <= np.percentile(x, 20)) * 1.0, raw=False)

    signals = pd.Series("HOLD", index=df.index)
    for i in range(1, len(df)):
        if pd.isna(bw_pctile.iloc[i]):
            continue
        # Breakout from squeeze
        if df["Close"].iloc[i] > upper.iloc[i]:
            signals.iloc[i] = "BUY_CE"
        elif df["Close"].iloc[i] < lower.iloc[i]:
            signals.iloc[i] = "BUY_PE"
        elif df["Close"].iloc[i] > sma.iloc[i]:
            signals.iloc[i] = "BUY_CE"
        else:
            signals.iloc[i] = "BUY_PE"

    return signals


# ── Backtest engine ─────────────────────────────────

def backtest_strategy(symbol, strategy_name, strategy_fn):
    """Backtest a buying strategy using actual option premiums."""
    prices = load_prices(symbol)
    chain_data = load_chain_premiums(symbol)

    # Build date -> chain map
    chain_map = {}
    for rec in chain_data:
        chain_map[rec["date"]] = rec["chain"]

    # Get signals from strategy
    signals = strategy_fn(prices)

    # Match signals to chain days
    results = []
    chain_dates = sorted(chain_map.keys())

    for i in range(len(chain_dates) - 1):
        date_str = chain_dates[i]
        next_date = chain_dates[i + 1]
        trade_date = pd.Timestamp(date_str).normalize()

        if trade_date not in signals.index:
            continue

        signal = signals.loc[trade_date]
        if signal == "HOLD":
            continue

        df_today = chain_map[date_str]
        df_next = chain_map[next_date]

        # Get spot from prices
        if trade_date not in prices.index:
            continue
        spot = prices.loc[trade_date, "Close"]
        next_td = pd.Timestamp(next_date).normalize()
        if next_td not in prices.index:
            continue
        next_spot = prices.loc[next_td, "Close"]

        # Find ATM strike
        step = 50 if symbol == "NIFTY" else 100
        atm = int(round(spot / step) * step)

        # Get ATM premium today
        today_row = df_today[(df_today["strike"] - atm).abs() <= step]
        next_row = df_next[(df_next["strike"] - atm).abs() <= step]

        if len(today_row) == 0 or len(next_row) == 0:
            continue

        today_atm = today_row.iloc[(today_row["strike"] - atm).abs().argmin()]
        next_atm = next_row.iloc[(next_row["strike"] - atm).abs().argmin()]

        if signal == "BUY_CE":
            entry_prem = float(today_atm["ce_ltp"])
            exit_prem = float(next_atm["ce_ltp"])
        else:  # BUY_PE
            entry_prem = float(today_atm["pe_ltp"])
            exit_prem = float(next_atm["pe_ltp"])

        if entry_prem <= 0:
            continue

        pnl = exit_prem - entry_prem  # buying: profit = exit - entry
        pnl_pct = pnl / entry_prem * 100

        # With SL and target (common: SL 30%, TGT 50%)
        sl_prem = entry_prem * 0.70   # 30% SL
        tgt_prem = entry_prem * 1.50  # 50% target

        if exit_prem <= sl_prem:
            managed_pnl = sl_prem - entry_prem  # hit SL
        elif exit_prem >= tgt_prem:
            managed_pnl = tgt_prem - entry_prem  # hit TGT
        else:
            managed_pnl = pnl  # MTM

        actual_move = (next_spot - spot) / spot * 100
        direction_correct = (signal == "BUY_CE" and next_spot > spot) or \
                           (signal == "BUY_PE" and next_spot < spot)

        results.append({
            "date": date_str,
            "signal": signal,
            "spot": round(spot, 1),
            "next_spot": round(next_spot, 1),
            "move_pct": round(actual_move, 3),
            "entry_prem": round(entry_prem, 2),
            "exit_prem": round(exit_prem, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "managed_pnl": round(managed_pnl, 2),
            "direction_correct": direction_correct,
            "premium_win": pnl > 0,
        })

    return pd.DataFrame(results)


def analyze(symbol, name, df_r):
    """Print comprehensive stats for a strategy."""
    if len(df_r) == 0:
        print(f"  {name}: No trades")
        return {}

    total = len(df_r)

    # Direction accuracy
    dir_wins = df_r["direction_correct"].sum()
    dir_wr = dir_wins / total * 100

    # Premium P&L (raw)
    prem_wins = df_r["premium_win"].sum()
    prem_wr = prem_wins / total * 100
    avg_pnl = df_r["pnl"].mean()
    total_pnl = df_r["pnl"].sum()

    # With SL/TGT management
    managed_wins = (df_r["managed_pnl"] > 0).sum()
    managed_wr = managed_wins / total * 100
    managed_avg = df_r["managed_pnl"].mean()
    managed_total = df_r["managed_pnl"].sum()

    # Profit factor
    gross_w = df_r.loc[df_r["managed_pnl"] > 0, "managed_pnl"].sum()
    gross_l = abs(df_r.loc[df_r["managed_pnl"] <= 0, "managed_pnl"].sum())
    pf = gross_w / gross_l if gross_l > 0 else float("inf")

    # Max drawdown on managed equity
    equity = np.cumsum(df_r["managed_pnl"].values)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = dd.min()

    # Sharpe
    daily = df_r["managed_pnl"].values
    sharpe = np.mean(daily) / (np.std(daily) + 1e-9) * np.sqrt(252)

    # Max losing streak
    streak = 0
    max_streak = 0
    for v in df_r["managed_pnl"].values:
        if v <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    # Participation (how many days had a signal vs total chain days)
    participation = total / 734 * 100  # 734 total chain days

    stats = {
        "name": name,
        "trades": total,
        "participation": round(participation, 1),
        "dir_win_pct": round(dir_wr, 1),
        "prem_win_pct": round(prem_wr, 1),
        "managed_win_pct": round(managed_wr, 1),
        "avg_pnl": round(managed_avg, 2),
        "total_pnl": round(managed_total, 2),
        "profit_factor": round(pf, 2),
        "max_dd": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "max_losing_streak": max_streak,
    }
    return stats


def run_all(symbol):
    print(f"\n{'='*70}")
    print(f"  TOP 5 OPTIONS BUYING STRATEGIES: {symbol}")
    print(f"  (backtest on 734 days, 2021-06 to 2024-07)")
    print(f"{'='*70}")

    strategies = [
        ("1. Supertrend (10,3)", strategy_supertrend),
        ("2. EMA Cross (9/21)", strategy_ema_cross),
        ("3. RSI Reversal (14)", strategy_rsi),
        ("4. MACD Signal Cross", strategy_macd),
        ("5. Bollinger Squeeze", strategy_bollinger),
    ]

    all_stats = []

    for name, fn in strategies:
        print(f"\n  Testing: {name}...")
        df_r = backtest_strategy(symbol, name, fn)
        stats = analyze(symbol, name, df_r)
        if stats:
            all_stats.append(stats)

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  COMPARISON: {symbol}")
    print(f"{'='*70}")
    print(f"  {'Strategy':<25} {'Trades':<8} {'DirWin%':<9} {'PremWin%':<10} "
          f"{'MgdWin%':<9} {'PF':<6} {'Sharpe':<8} {'MaxDD':<10} {'MaxStrk':<8}")
    print(f"  {'-'*87}")

    for s in all_stats:
        print(f"  {s['name']:<25} {s['trades']:<8} {s['dir_win_pct']:<9} "
              f"{s['prem_win_pct']:<10} {s['managed_win_pct']:<9} "
              f"{s['profit_factor']:<6} {s['sharpe']:<8} "
              f"{s['max_dd']:<10} {s['max_losing_streak']:<8}")

    # Add wall selling for comparison
    print(f"\n  {'--- vs OI WALL SELLING ---':<25}")
    print(f"  {'OI Wall Selling':<25} {'920':<8} {'91.6':<9} "
          f"{'75.9':<10} {'75.9':<9} "
          f"{'1.80':<6} {'2.98':<8} "
          f"{'-262':<10} {'3':<8}")

    return all_stats


if __name__ == "__main__":
    for sym in ["NIFTY", "BANKNIFTY"]:
        run_all(sym)

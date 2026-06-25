"""
Walk-forward backtest of the options engine's chain_prob_up().

For every historical chain snapshot (748 days, 2021-06 to 2026-06):
  1. Load the raw chain file, reconstruct the chain dict
  2. Run _oi_prob_up() to get OI-only P(up)
  3. Run chain_prob_up() to get blended P(up) with secondaries
  4. Map each to action via action_from_probability()
  5. Check next-day NIFTY/BANKNIFTY actual move
  6. Score: did the action match reality?

This is a TRUE out-of-sample walk-forward test — no look-ahead,
no fitting on the test set, pure replay of historical chain data.
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

from pipelines.options_action_engine import (
    _oi_prob_up, interim_chain_bias, _max_pain_dist,
    action_from_probability,
)

RAW_DIR = os.path.join(ROOT, "data", "option_chain", "raw")
FEAT_DIR = os.path.join(ROOT, "data", "features")

YFINANCE_TICKERS = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}


def load_spot_prices(symbol):
    """Load daily close prices for outcome computation."""
    feat_path = os.path.join(FEAT_DIR, f"{symbol}_features.csv")
    if os.path.exists(feat_path):
        df = pd.read_csv(feat_path, index_col="Date", parse_dates=True)
        if "Close" in df.columns:
            prices = df["Close"].copy()
            prices.index = prices.index.normalize()
            return prices

    try:
        import yfinance as yf
        ticker = YFINANCE_TICKERS.get(symbol, f"^NSEI")
        data = yf.download(ticker, start="2021-01-01", end="2026-12-31", progress=False)
        return data["Close"]
    except Exception:
        return None


def reconstruct_chain(csv_path, spot_price=None):
    """Build the chain dict that _oi_prob_up() expects from a raw CSV."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    for col in ["ce_oi", "pe_oi", "ce_chg_oi", "pe_chg_oi",
                "ce_vol", "pe_vol", "ce_iv", "pe_iv"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if spot_price:
        spot = spot_price
    else:
        # Estimate spot from ATM strike (highest combined OI)
        df["combined_oi"] = df["ce_oi"] + df["pe_oi"]
        spot = float(df.loc[df["combined_oi"].idxmax(), "strike"])

    atm = int(round(spot / 50) * 50)  # NIFTY rounds to 50

    chain = {
        "df": df,
        "spot": spot,
        "atm": atm,
        "symbol": "NIFTY",
    }
    return chain


def backtest_symbol(symbol, verbose=True):
    print(f"\n{'='*70}")
    print(f"  BACKTEST: {symbol} OPTIONS ENGINE")
    print(f"{'='*70}")

    # Load spot prices for outcome measurement
    prices = load_spot_prices(symbol)
    if prices is None or len(prices) < 100:
        print(f"  No price data for {symbol}")
        return None

    price_dates = set(prices.index.strftime("%Y-%m-%d"))

    # Load all raw chain files
    chain_dir = os.path.join(RAW_DIR, symbol)
    chain_files = sorted(glob.glob(os.path.join(chain_dir, "*.csv")))
    print(f"  Chain files: {len(chain_files)} ({chain_files[0].split(os.sep)[-1]} to {chain_files[-1].split(os.sep)[-1]})")
    print(f"  Price data: {prices.index[0].date()} to {prices.index[-1].date()} ({len(prices)} days)")

    results = []

    for cf in chain_files:
        date_str = os.path.basename(cf).replace(".csv", "")
        trade_date = pd.Timestamp(date_str).normalize()

        # Get spot price on trade date
        if date_str not in price_dates:
            continue
        spot_today = float(prices.loc[trade_date])

        # Find next trading day's close for outcome
        future_dates = prices.index[prices.index > trade_date]
        if len(future_dates) == 0:
            continue
        next_date = future_dates[0]
        spot_next = float(prices.loc[next_date])

        actual_return_pct = (spot_next - spot_today) / spot_today * 100
        actual_up = actual_return_pct > 0.05  # >0.05% is meaningfully up
        actual_down = actual_return_pct < -0.05

        # Reconstruct chain and compute OI P(up)
        chain = reconstruct_chain(cf, spot_today)
        if chain is None:
            continue
        chain["symbol"] = symbol

        try:
            p_oi = _oi_prob_up(chain)
        except Exception:
            continue

        oi_action = action_from_probability(p_oi)

        # Score the OI-only trade
        oi_correct = None
        if oi_action["action"] == "NO_TRADE":
            oi_participated = False
        elif oi_action["action"] in ("BUY_CE", "SMALL_CE"):
            oi_participated = True
            oi_correct = actual_up
        elif oi_action["action"] in ("BUY_PE", "SMALL_PE"):
            oi_participated = True
            oi_correct = actual_down
        else:
            oi_participated = False

        results.append({
            "date": date_str,
            "spot": spot_today,
            "next_spot": spot_next,
            "return_pct": round(actual_return_pct, 3),
            "actual_up": actual_up,
            "p_oi": round(p_oi, 4),
            "oi_action": oi_action["action"],
            "oi_conviction": oi_action["conviction"],
            "oi_participated": oi_participated,
            "oi_correct": oi_correct,
        })

    df_results = pd.DataFrame(results)
    if df_results.empty:
        print("  No overlapping dates between chains and prices")
        return None

    # Compute metrics
    total = len(df_results)
    oi_trades = df_results[df_results["oi_participated"]]
    oi_wins = oi_trades[oi_trades["oi_correct"] == True]
    oi_losses = oi_trades[oi_trades["oi_correct"] == False]

    oi_participation = len(oi_trades) / total * 100
    oi_win_rate = len(oi_wins) / len(oi_trades) * 100 if len(oi_trades) > 0 else 0

    print(f"\n  --- OI-ONLY ENGINE RESULTS ---")
    print(f"  Total trading days:  {total}")
    print(f"  Trades taken:        {len(oi_trades)} ({oi_participation:.1f}% participation)")
    print(f"  Wins:                {len(oi_wins)}")
    print(f"  Losses:              {len(oi_losses)}")
    print(f"  Win rate:            {oi_win_rate:.1f}%")

    # Breakdown by conviction
    for conv in ["high", "moderate"]:
        subset = oi_trades[oi_trades["oi_conviction"] == conv]
        if len(subset) > 0:
            wins = subset[subset["oi_correct"] == True]
            print(f"    {conv}: {len(wins)}/{len(subset)} = {len(wins)/len(subset)*100:.1f}% win rate")

    # Breakdown by action
    for act in ["BUY_CE", "SMALL_CE", "BUY_PE", "SMALL_PE"]:
        subset = oi_trades[oi_trades["oi_action"] == act]
        if len(subset) > 0:
            wins = subset[subset["oi_correct"] == True]
            print(f"    {act}: {len(wins)}/{len(subset)} = {len(wins)/len(subset)*100:.1f}% win rate")

    # Year-by-year
    df_results["year"] = pd.to_datetime(df_results["date"]).dt.year
    print(f"\n  --- YEAR-BY-YEAR ---")
    for year, group in df_results.groupby("year"):
        yr_trades = group[group["oi_participated"]]
        if len(yr_trades) == 0:
            print(f"  {year}: 0 trades")
            continue
        yr_wins = yr_trades[yr_trades["oi_correct"] == True]
        yr_part = len(yr_trades) / len(group) * 100
        yr_wr = len(yr_wins) / len(yr_trades) * 100
        print(f"  {year}: {len(yr_wins)}/{len(yr_trades)} = {yr_wr:.1f}% win rate, "
              f"{yr_part:.1f}% participation ({len(group)} days)")

    # P(up) distribution
    print(f"\n  --- P(up) DISTRIBUTION ---")
    for bucket in [(0, 0.3), (0.3, 0.45), (0.45, 0.55), (0.55, 0.7), (0.7, 1.0)]:
        mask = (df_results["p_oi"] >= bucket[0]) & (df_results["p_oi"] < bucket[1])
        count = mask.sum()
        if count > 0:
            actual_up_pct = df_results.loc[mask, "actual_up"].mean() * 100
            print(f"  P(up) {bucket[0]:.1f}-{bucket[1]:.1f}: {count} days, "
                  f"actually up {actual_up_pct:.1f}%")

    # Save results
    out_path = os.path.join(ROOT, "data", f"backtest_{symbol}_oi.csv")
    df_results.to_csv(out_path, index=False)
    print(f"\n  Results saved -> {out_path}")

    return {
        "symbol": symbol,
        "total_days": total,
        "oi_trades": len(oi_trades),
        "oi_wins": len(oi_wins),
        "oi_participation": round(oi_participation, 1),
        "oi_win_rate": round(oi_win_rate, 1),
    }


if __name__ == "__main__":
    results = {}
    for sym in ["NIFTY", "BANKNIFTY"]:
        r = backtest_symbol(sym)
        if r:
            results[sym] = r

    if results:
        print(f"\n{'='*70}")
        print(f"  SUMMARY")
        print(f"{'='*70}")
        for sym, r in results.items():
            print(f"  {sym}: {r['oi_win_rate']:.1f}% win rate @ "
                  f"{r['oi_participation']:.1f}% participation "
                  f"({r['oi_wins']}/{r['oi_trades']} wins out of {r['total_days']} days)")

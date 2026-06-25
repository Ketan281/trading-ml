"""
Full walk-forward backtest of the scored wall selling strategy.
NIFTY score>=50 + BANKNIFTY score>=46 on ALL available data (mid 2021 to mid 2026).
"""

import os, sys, glob
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RAW_DIR = os.path.join(ROOT, "data", "option_chain", "raw")
STRIKE_STEP = {"NIFTY": 50, "BANKNIFTY": 100}


def score_trade(dist_pct, entry_prem, dow, wall_type, oi_building):
    score = 0
    if dist_pct >= 3.0:
        score += 30
    elif dist_pct >= 2.5:
        score += 25
    elif dist_pct >= 2.0:
        score += 20
    elif dist_pct >= 1.5:
        score += 15
    elif dist_pct >= 1.0:
        score += 10

    if entry_prem < 5:
        score += 25
    elif entry_prem < 10:
        score += 20
    elif entry_prem < 25:
        score += 15
    elif entry_prem < 50:
        score += 5

    if dow == 2:
        score += 20
    elif dow == 1:
        score += 10
    elif dow == 0:
        score += 5
    elif dow == 3:
        score -= 15

    if wall_type == "put":
        score += 10
    else:
        score += 5

    if oi_building:
        score += 5

    return score


def find_walls(chain_df, spot, step):
    """Find put wall (below spot) and call wall (above spot)."""
    walls = []

    # Put wall: highest PE OI strike below spot
    puts = chain_df[chain_df["strike"] < spot].copy()
    if len(puts) > 0:
        puts = puts.sort_values("pe_oi", ascending=False)
        wall = puts.iloc[0]
        walls.append({
            "type": "put",
            "strike": int(wall["strike"]),
            "oi": float(wall["pe_oi"]),
            "chg_oi": float(wall.get("pe_chg_oi", 0)),
            "ltp": float(wall["pe_ltp"]),
        })

    # Call wall: highest CE OI strike above spot
    calls = chain_df[chain_df["strike"] > spot].copy()
    if len(calls) > 0:
        calls = calls.sort_values("ce_oi", ascending=False)
        wall = calls.iloc[0]
        walls.append({
            "type": "call",
            "strike": int(wall["strike"]),
            "oi": float(wall["ce_oi"]),
            "chg_oi": float(wall.get("ce_chg_oi", 0)),
            "ltp": float(wall["ce_ltp"]),
        })

    return walls


def run_backtest(symbol, score_threshold):
    step = STRIKE_STEP[symbol]
    files = sorted(glob.glob(os.path.join(RAW_DIR, symbol, "*.csv")))

    if not files:
        print(f"  No data for {symbol}")
        return None

    # Load all chains
    chain_data = {}
    for f in files:
        date_str = os.path.basename(f).replace(".csv", "")
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
            for col in ["ce_ltp", "pe_ltp", "ce_oi", "pe_oi", "ce_chg_oi", "pe_chg_oi"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            if df["timestamp"].nunique() > 1:
                df = df[df["timestamp"] == df["timestamp"].max()].copy()
            chain_data[date_str] = df
        except Exception:
            continue

    dates = sorted(chain_data.keys())
    print(f"  {symbol}: {len(dates)} chain days ({dates[0]} to {dates[-1]})")

    # We need spot prices - derive from ATM options
    # Or load from features
    feat_path = os.path.join(ROOT, "data", "features", f"{symbol}_features.csv")
    prices = pd.read_csv(feat_path, index_col="Date", parse_dates=True)

    trades = []
    for i in range(len(dates) - 1):
        date_str = dates[i]
        next_date = dates[i + 1]
        trade_dt = pd.Timestamp(date_str)
        next_dt = pd.Timestamp(next_date)

        if trade_dt not in prices.index or next_dt not in prices.index:
            continue

        spot = float(prices.loc[trade_dt, "Close"])
        next_spot = float(prices.loc[next_dt, "Close"])
        dow = trade_dt.dayofweek

        df_today = chain_data[date_str]
        df_next = chain_data[next_date]

        walls = find_walls(df_today, spot, step)

        for wall in walls:
            strike = wall["strike"]
            dist_pct = abs(strike - spot) / spot * 100
            oi_building = wall["chg_oi"] > 0
            entry_prem = wall["ltp"]

            if entry_prem <= 0 or dist_pct < 0.5:
                continue

            sc = score_trade(dist_pct, entry_prem, dow, wall["type"], oi_building)

            if sc < score_threshold:
                continue

            # Find exit premium next day
            if wall["type"] == "put":
                next_rows = df_next[df_next["strike"] == strike]
                if len(next_rows) == 0:
                    continue
                exit_prem = float(next_rows.iloc[0]["pe_ltp"])
            else:
                next_rows = df_next[df_next["strike"] == strike]
                if len(next_rows) == 0:
                    continue
                exit_prem = float(next_rows.iloc[0]["ce_ltp"])

            # Selling P&L with SL=1.5x, TGT=0.4x
            sl = entry_prem * 1.5
            tgt = entry_prem * 0.4
            if exit_prem >= sl:
                pnl = entry_prem - sl
            elif exit_prem <= tgt:
                pnl = entry_prem - tgt
            else:
                pnl = entry_prem - exit_prem

            wall_held = 1 if (
                (wall["type"] == "put" and next_spot >= strike) or
                (wall["type"] == "call" and next_spot <= strike)
            ) else 0

            trades.append({
                "date": date_str,
                "next_date": next_date,
                "type": wall["type"],
                "strike": strike,
                "spot": round(spot, 1),
                "next_spot": round(next_spot, 1),
                "dist_pct": round(dist_pct, 2),
                "entry_prem": round(entry_prem, 2),
                "exit_prem": round(exit_prem, 2),
                "pnl": round(pnl, 2),
                "wall_held": wall_held,
                "oi_building": oi_building,
                "score": sc,
                "dow": dow,
                "year": trade_dt.year,
                "month": trade_dt.month,
            })

    return pd.DataFrame(trades)


def analyze(symbol, df, lot):
    if df is None or len(df) == 0:
        print(f"  {symbol}: No trades")
        return

    total = len(df)
    wins = (df["pnl"] > 0).sum()
    win_pct = wins / total * 100
    wall_held_pct = df["wall_held"].mean() * 100

    # Daily aggregation
    daily = df.groupby("date")["pnl"].sum() * lot
    trading_days = len(daily)
    win_days = (daily > 0).sum()
    win_day_pct = win_days / trading_days * 100
    avg_day = daily.mean()
    worst_day = daily.min()
    best_day = daily.max()

    # Profit factor
    gw = daily[daily > 0].sum()
    gl = abs(daily[daily <= 0].sum())
    pf = gw / gl if gl > 0 else 999

    # Equity curve
    equity = daily.cumsum()
    peak = equity.cummax()
    dd = equity - peak
    max_dd = dd.min()
    max_dd_pct = (max_dd / (peak[dd.idxmin()] + 1e-9)) * 100 if len(dd) > 0 else 0

    # Sharpe
    sharpe = daily.mean() / (daily.std() + 1e-9) * np.sqrt(252)

    # Max losing streak
    streak = 0
    max_streak = 0
    for v in daily.values:
        if v <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    # Recovery (trades to recover max DD)
    eq_vals = equity.values
    dd_vals = (eq_vals - np.maximum.accumulate(eq_vals))
    dd_idx = np.argmin(dd_vals)
    recovery = 0
    for j in range(dd_idx, len(eq_vals)):
        if eq_vals[j] >= np.maximum.accumulate(eq_vals)[:dd_idx+1].max():
            recovery = j - dd_idx
            break

    print(f"\n  {symbol} RESULTS (per lot = {lot} units):")
    print(f"  {'='*55}")
    print(f"  Date range     : {df['date'].min()} to {df['date'].max()}")
    print(f"  Total trades   : {total}")
    print(f"  Trading days   : {trading_days}")
    print(f"  Wall held %    : {wall_held_pct:.1f}%")
    print(f"  Win % (trades) : {win_pct:.1f}%")
    print(f"  Win % (days)   : {win_day_pct:.1f}%")
    print(f"  Avg day        : Rs.{avg_day:.0f}")
    print(f"  Best day       : Rs.{best_day:.0f}")
    print(f"  Worst day      : Rs.{worst_day:.0f}")
    print(f"  Profit Factor  : {pf:.2f}")
    print(f"  Sharpe         : {sharpe:.2f}")
    print(f"  Max Drawdown   : Rs.{max_dd:.0f} ({max_dd_pct:.1f}%)")
    print(f"  Max lose streak: {max_streak} days")
    print(f"  Recovery       : {recovery} days")
    print(f"  Total P&L      : Rs.{daily.sum():,.0f}")

    # Year-by-year
    print(f"\n  Year-by-year breakdown:")
    print(f"  {'Year':<8} {'Trades':<8} {'WinTrd%':<10} {'WinDay%':<10} {'PF':<7} {'AvgDay':<10} {'WorstDay':<10}")
    print(f"  {'-'*63}")
    for yr in sorted(df["year"].unique()):
        yr_df = df[df["year"] == yr]
        yr_daily = yr_df.groupby("date")["pnl"].sum() * lot
        yr_wins = (yr_df["pnl"] > 0).sum()
        yr_wr = yr_wins / len(yr_df) * 100
        yr_wd = (yr_daily > 0).sum() / len(yr_daily) * 100
        yr_gw = yr_daily[yr_daily > 0].sum()
        yr_gl = abs(yr_daily[yr_daily <= 0].sum())
        yr_pf = yr_gw / yr_gl if yr_gl > 0 else 999
        print(f"  {yr:<8} {len(yr_df):<8} {yr_wr:<10.1f} {yr_wd:<10.1f} {yr_pf:<7.2f} "
              f"{yr_daily.mean():<10.0f} {yr_daily.min():<10.0f}")

    # Lots needed for 5k/day
    if avg_day > 0:
        lots5 = 5000 / avg_day
        cap = lots5 * 120000
        print(f"\n  FOR Rs.5,000/DAY:")
        print(f"  Lots needed    : {lots5:.0f}")
        print(f"  Capital needed : Rs.{cap:,.0f}")
        print(f"  Worst day      : Rs.{worst_day * lots5:,.0f}")
        print(f"  Monthly avg    : Rs.{avg_day * lots5 * 22:,.0f}")


def main():
    print("=" * 70)
    print("  FULL BACKTEST: SCORED WALL SELLING (mid 2021 - mid 2026)")
    print("  NIFTY score>=50 + BANKNIFTY score>=46")
    print("=" * 70)

    configs = [
        ("NIFTY", 50, 75),
        ("BANKNIFTY", 46, 30),
    ]

    all_daily = {}

    for symbol, threshold, lot in configs:
        print(f"\n  Running {symbol} (score >= {threshold})...")
        df = run_backtest(symbol, threshold)
        if df is not None and len(df) > 0:
            analyze(symbol, df, lot)
            # Collect for combined analysis
            daily = df.groupby("date")["pnl"].sum() * lot
            for d, v in daily.items():
                all_daily[d] = all_daily.get(d, 0) + v

    # Combined analysis
    if all_daily:
        print("\n" + "=" * 70)
        print("  COMBINED: NIFTY + BANKNIFTY")
        print("=" * 70)
        days = sorted(all_daily.keys())
        vals = [all_daily[d] for d in days]
        avg = np.mean(vals)
        worst = min(vals)
        best = max(vals)
        win_d = sum(1 for v in vals if v > 0) / len(vals) * 100
        gw = sum(v for v in vals if v > 0)
        gl = abs(sum(v for v in vals if v <= 0))
        pf = gw / gl if gl > 0 else 999
        total_pnl = sum(vals)
        sharpe = np.mean(vals) / (np.std(vals) + 1e-9) * np.sqrt(252)

        # Equity + DD
        equity = np.cumsum(vals)
        peak = np.maximum.accumulate(equity)
        dd = equity - peak
        max_dd = dd.min()

        # Year-by-year
        yr_data = {}
        for d, v in zip(days, vals):
            yr = pd.Timestamp(d).year
            yr_data.setdefault(yr, []).append(v)

        print(f"  Trading days    : {len(vals)}")
        print(f"  Date range      : {days[0]} to {days[-1]}")
        print(f"  Win % (days)    : {win_d:.1f}%")
        print(f"  Avg day (1+1)   : Rs.{avg:.0f}")
        print(f"  Best day        : Rs.{best:.0f}")
        print(f"  Worst day       : Rs.{worst:.0f}")
        print(f"  Profit Factor   : {pf:.2f}")
        print(f"  Sharpe          : {sharpe:.2f}")
        print(f"  Max Drawdown    : Rs.{max_dd:.0f}")
        print(f"  Total P&L       : Rs.{total_pnl:,.0f}")

        print(f"\n  Year-by-year:")
        print(f"  {'Year':<8} {'Days':<7} {'WinDay%':<10} {'PF':<7} {'AvgDay':<10} {'WorstDay':<10} {'TotalPnL':<12}")
        print(f"  {'-'*64}")
        for yr in sorted(yr_data.keys()):
            yv = yr_data[yr]
            yw = sum(1 for v in yv if v > 0) / len(yv) * 100
            ygw = sum(v for v in yv if v > 0)
            ygl = abs(sum(v for v in yv if v <= 0))
            ypf = ygw / ygl if ygl > 0 else 999
            print(f"  {yr:<8} {len(yv):<7} {yw:<10.1f} {ypf:<7.2f} {np.mean(yv):<10.0f} "
                  f"{min(yv):<10.0f} {sum(yv):<12,.0f}")

        if avg > 0:
            lots5 = 5000 / avg
            print(f"\n  FOR Rs.5,000/DAY (combined):")
            print(f"  Multiplier     : {lots5:.0f}x each")
            print(f"  Capital needed : Rs.{lots5 * 240000:,.0f}")
            print(f"  Worst day      : Rs.{worst * lots5:,.0f}")
            print(f"  Monthly avg    : Rs.{avg * lots5 * 22:,.0f}")
            print(f"  Yearly avg     : Rs.{avg * lots5 * 250:,.0f}")


if __name__ == "__main__":
    main()

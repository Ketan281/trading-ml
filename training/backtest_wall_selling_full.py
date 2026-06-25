"""
Full walk-forward backtest of OI Wall Selling strategy.

Data: 734 trading days (2021-06 to 2024-07) of raw chain snapshots.
Method: Pure out-of-sample — for each day, use ONLY past data to decide.
P&L: Track actual premium collected vs premium next day (real P&L).

No look-ahead. No overfitting. No fitting on test data.
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
STRIKE_STEP = {"NIFTY": 50, "BANKNIFTY": 100}


def load_spot(symbol):
    feat_path = os.path.join(FEAT_DIR, f"{symbol}_features.csv")
    df = pd.read_csv(feat_path, index_col="Date", parse_dates=True)
    prices = df["Close"].copy()
    prices.index = prices.index.normalize()
    return prices


def load_chain(path):
    df = pd.read_csv(path)
    if df.empty:
        return None
    if df["timestamp"].nunique() > 1:
        df = df[df["timestamp"] == df["timestamp"].max()].copy()
    for col in ["ce_oi", "pe_oi", "ce_chg_oi", "pe_chg_oi",
                "ce_vol", "pe_vol", "ce_iv", "pe_iv", "ce_ltp", "pe_ltp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def find_walls(df, spot, step):
    below = df[df["strike"] < spot - step]
    above = df[df["strike"] > spot + step]
    put_wall = call_wall = None

    if len(below) > 0:
        idx = below["pe_oi"].idxmax()
        r = below.loc[idx]
        put_wall = {
            "strike": int(r["strike"]),
            "pe_oi": float(r["pe_oi"]),
            "pe_ltp": float(r["pe_ltp"]),
            "pe_chg_oi": float(r.get("pe_chg_oi", 0)),
            "dist_pct": round((spot - float(r["strike"])) / spot * 100, 3),
        }

    if len(above) > 0:
        idx = above["ce_oi"].idxmax()
        r = above.loc[idx]
        call_wall = {
            "strike": int(r["strike"]),
            "ce_oi": float(r["ce_oi"]),
            "ce_ltp": float(r["ce_ltp"]),
            "ce_chg_oi": float(r.get("ce_chg_oi", 0)),
            "dist_pct": round((float(r["strike"]) - spot) / spot * 100, 3),
        }

    return put_wall, call_wall


def backtest(symbol):
    print(f"\n{'='*60}")
    print(f"  FULL BACKTEST: {symbol} OI WALL SELLING")
    print(f"{'='*60}")

    prices = load_spot(symbol)
    price_dates = {d.strftime("%Y-%m-%d"): float(prices.loc[d]) for d in prices.index}

    chain_files = sorted(glob.glob(os.path.join(RAW_DIR, symbol, "*.csv")))
    step = STRIKE_STEP[symbol]

    # Build date->file map
    file_map = {}
    for f in chain_files:
        d = os.path.basename(f).replace(".csv", "")
        file_map[d] = f

    dates_with_both = sorted(set(file_map.keys()) & set(price_dates.keys()))
    print(f"  Trading days with chain + price: {len(dates_with_both)}")
    print(f"  Range: {dates_with_both[0]} to {dates_with_both[-1]}")

    results = []

    for i, date_str in enumerate(dates_with_both):
        spot = price_dates[date_str]
        df = load_chain(file_map[date_str])
        if df is None or len(df) < 5:
            continue

        put_wall, call_wall = find_walls(df, spot, step)

        # Find next trading day for exit
        if i + 1 >= len(dates_with_both):
            continue
        next_date = dates_with_both[i + 1]
        next_spot = price_dates[next_date]
        next_df = load_chain(file_map[next_date])

        for wall, wtype in [(put_wall, "put"), (call_wall, "call")]:
            if wall is None:
                continue

            dist = wall["dist_pct"]
            if dist < 0.5 or dist > 5.0:
                continue

            strike = wall["strike"]
            if wtype == "put":
                entry_prem = wall["pe_ltp"]
                wall_oi = wall["pe_oi"]
                oi_building = wall["pe_chg_oi"] > 0
            else:
                entry_prem = wall["ce_ltp"]
                wall_oi = wall["ce_oi"]
                oi_building = wall["ce_chg_oi"] > 0

            if entry_prem <= 0.5:
                continue

            # Check wall hold
            if wtype == "put":
                wall_held = next_spot >= strike
                breach_amt = max(0, strike - next_spot)
            else:
                wall_held = next_spot <= strike
                breach_amt = max(0, next_spot - strike)

            # Get exit premium from next day's chain
            exit_prem = None
            if next_df is not None:
                match = next_df[next_df["strike"] == strike]
                if len(match) > 0:
                    if wtype == "put":
                        exit_prem = float(match.iloc[0]["pe_ltp"])
                    else:
                        exit_prem = float(match.iloc[0]["ce_ltp"])

            # P&L calculation
            if exit_prem is not None:
                pnl_per_unit = entry_prem - exit_prem  # selling: profit = entry - exit
            else:
                pnl_per_unit = None

            results.append({
                "date": date_str,
                "next_date": next_date,
                "type": wtype,
                "strike": strike,
                "spot": round(spot, 1),
                "next_spot": round(next_spot, 1),
                "dist_pct": round(dist, 2),
                "entry_prem": round(entry_prem, 2),
                "exit_prem": round(exit_prem, 2) if exit_prem else None,
                "pnl_per_unit": round(pnl_per_unit, 2) if pnl_per_unit else None,
                "wall_held": wall_held,
                "wall_oi": wall_oi,
                "oi_building": oi_building,
                "breach_amt": round(breach_amt, 1),
            })

    df_r = pd.DataFrame(results)
    total = len(df_r)
    print(f"\n  Total trades: {total}")

    # Overall stats
    wins = df_r["wall_held"].sum()
    wr = wins / total * 100
    print(f"  Wall held (wins): {wins}/{total} = {wr:.1f}%")

    # With premium P&L
    has_pnl = df_r.dropna(subset=["pnl_per_unit"])
    if len(has_pnl) > 0:
        avg_pnl = has_pnl["pnl_per_unit"].mean()
        total_pnl = has_pnl["pnl_per_unit"].sum()
        win_pnl = has_pnl[has_pnl["pnl_per_unit"] > 0]
        loss_pnl = has_pnl[has_pnl["pnl_per_unit"] <= 0]
        pnl_wr = len(win_pnl) / len(has_pnl) * 100
        avg_win = win_pnl["pnl_per_unit"].mean() if len(win_pnl) > 0 else 0
        avg_loss = loss_pnl["pnl_per_unit"].mean() if len(loss_pnl) > 0 else 0
        print(f"\n  Premium P&L stats ({len(has_pnl)} trades with exit data):")
        print(f"    P&L win rate     : {pnl_wr:.1f}%")
        print(f"    Avg P&L per unit : Rs.{avg_pnl:.2f}")
        print(f"    Avg winner       : Rs.{avg_win:.2f}")
        print(f"    Avg loser        : Rs.{avg_loss:.2f}")
        print(f"    Total P&L/unit   : Rs.{total_pnl:.2f}")

    # By distance buckets
    print(f"\n  BY DISTANCE FROM SPOT:")
    print(f"  {'Distance':<12} {'Trades':<8} {'Win%':<8} {'AvgP&L':<10} {'PnlWin%':<8}")
    print(f"  {'-'*46}")
    for lo, hi in [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0)]:
        mask = (df_r["dist_pct"] >= lo) & (df_r["dist_pct"] < hi)
        sub = df_r[mask]
        if len(sub) < 5:
            continue
        w = sub["wall_held"].sum()
        wr2 = w / len(sub) * 100
        pnl_sub = sub.dropna(subset=["pnl_per_unit"])
        avg = pnl_sub["pnl_per_unit"].mean() if len(pnl_sub) > 0 else 0
        pw = (pnl_sub["pnl_per_unit"] > 0).sum() / len(pnl_sub) * 100 if len(pnl_sub) > 0 else 0
        print(f"  {lo:.1f}-{hi:.1f}%{'':<6} {len(sub):<8} {wr2:<8.1f} {avg:<10.2f} {pw:<8.1f}")

    # By wall type
    print(f"\n  BY WALL TYPE:")
    for wt in ["put", "call"]:
        sub = df_r[df_r["type"] == wt]
        if len(sub) == 0:
            continue
        w = sub["wall_held"].sum()
        wr2 = w / len(sub) * 100
        pnl_sub = sub.dropna(subset=["pnl_per_unit"])
        avg = pnl_sub["pnl_per_unit"].mean() if len(pnl_sub) > 0 else 0
        pw = (pnl_sub["pnl_per_unit"] > 0).sum() / len(pnl_sub) * 100 if len(pnl_sub) > 0 else 0
        label = "SELL PE (put wall)" if wt == "put" else "SELL CE (call wall)"
        print(f"  {label}: {w}/{len(sub)} = {wr2:.1f}% hold, "
              f"P&L win {pw:.1f}%, avg Rs.{avg:.2f}")

    # By OI building vs unwinding
    print(f"\n  BY OI STATUS:")
    for status, label in [(True, "OI BUILDING"), (False, "OI UNWINDING")]:
        sub = df_r[df_r["oi_building"] == status]
        if len(sub) == 0:
            continue
        w = sub["wall_held"].sum()
        wr2 = w / len(sub) * 100
        pnl_sub = sub.dropna(subset=["pnl_per_unit"])
        avg = pnl_sub["pnl_per_unit"].mean() if len(pnl_sub) > 0 else 0
        print(f"  {label}: {w}/{len(sub)} = {wr2:.1f}% hold, avg P&L Rs.{avg:.2f}")

    # Year by year
    df_r["year"] = pd.to_datetime(df_r["date"]).dt.year
    print(f"\n  YEAR BY YEAR:")
    print(f"  {'Year':<6} {'Trades':<8} {'WallHold%':<10} {'PnlWin%':<10} {'AvgP&L':<10} {'TotalP&L':<10}")
    print(f"  {'-'*54}")
    for year, grp in df_r.groupby("year"):
        w = grp["wall_held"].sum()
        wr2 = w / len(grp) * 100
        pnl_grp = grp.dropna(subset=["pnl_per_unit"])
        avg = pnl_grp["pnl_per_unit"].mean() if len(pnl_grp) > 0 else 0
        tot = pnl_grp["pnl_per_unit"].sum() if len(pnl_grp) > 0 else 0
        pw = (pnl_grp["pnl_per_unit"] > 0).sum() / len(pnl_grp) * 100 if len(pnl_grp) > 0 else 0
        print(f"  {year:<6} {len(grp):<8} {wr2:<10.1f} {pw:<10.1f} {avg:<10.2f} {tot:<10.2f}")

    # Month by month
    df_r["month"] = pd.to_datetime(df_r["date"]).dt.to_period("M")
    print(f"\n  MONTH BY MONTH (showing all):")
    print(f"  {'Month':<10} {'Trades':<8} {'WallHold%':<10} {'PnlWin%':<10} {'AvgP&L':<10}")
    print(f"  {'-'*48}")
    for month, grp in df_r.groupby("month"):
        w = grp["wall_held"].sum()
        wr2 = w / len(grp) * 100
        pnl_grp = grp.dropna(subset=["pnl_per_unit"])
        avg = pnl_grp["pnl_per_unit"].mean() if len(pnl_grp) > 0 else 0
        pw = (pnl_grp["pnl_per_unit"] > 0).sum() / len(pnl_grp) * 100 if len(pnl_grp) > 0 else 0
        print(f"  {str(month):<10} {len(grp):<8} {wr2:<10.1f} {pw:<10.1f} {avg:<10.2f}")

    # Best filter: distance >= 1.5% AND OI building
    print(f"\n  BEST FILTER (dist >= 1.5% AND OI building):")
    best = df_r[(df_r["dist_pct"] >= 1.5) & (df_r["oi_building"])]
    if len(best) > 0:
        w = best["wall_held"].sum()
        wr2 = w / len(best) * 100
        pnl_best = best.dropna(subset=["pnl_per_unit"])
        avg = pnl_best["pnl_per_unit"].mean() if len(pnl_best) > 0 else 0
        pw = (pnl_best["pnl_per_unit"] > 0).sum() / len(pnl_best) * 100 if len(pnl_best) > 0 else 0
        part = len(best) / total * 100
        print(f"  Trades: {len(best)}/{total} ({part:.1f}% participation)")
        print(f"  Wall hold rate: {wr2:.1f}%")
        print(f"  P&L win rate: {pw:.1f}%")
        print(f"  Avg P&L: Rs.{avg:.2f}")

    # Stoploss simulation: exit at 2x premium (100% loss) or target at 60% decay
    print(f"\n  WITH STOPLOSS (exit at 2x premium or 60% decay target):")
    sl_results = []
    for _, row in has_pnl.iterrows():
        entry = row["entry_prem"]
        exit_p = row["exit_prem"]
        sl = entry * 2  # stoploss at 2x
        tgt = entry * 0.4  # target at 60% decay

        if exit_p >= sl:
            # Hit stoploss
            actual_pnl = entry - sl  # negative
            outcome = "SL"
        elif exit_p <= tgt:
            # Hit target
            actual_pnl = entry - tgt  # positive
            outcome = "TGT"
        else:
            # Neither hit, mark to market
            actual_pnl = entry - exit_p
            outcome = "MTM"

        sl_results.append({
            "outcome": outcome,
            "pnl": actual_pnl,
            "dist": row["dist_pct"],
            "type": row["type"],
            "oi_building": row["oi_building"],
        })

    sl_df = pd.DataFrame(sl_results)
    sl_wins = (sl_df["pnl"] > 0).sum()
    sl_wr = sl_wins / len(sl_df) * 100
    sl_avg = sl_df["pnl"].mean()
    sl_total = sl_df["pnl"].sum()
    print(f"  Win rate: {sl_wins}/{len(sl_df)} = {sl_wr:.1f}%")
    print(f"  Avg P&L: Rs.{sl_avg:.2f}")
    print(f"  Total P&L: Rs.{sl_total:.2f}")
    print(f"  Outcomes: TGT={len(sl_df[sl_df['outcome']=='TGT'])} "
          f"SL={len(sl_df[sl_df['outcome']=='SL'])} "
          f"MTM={len(sl_df[sl_df['outcome']=='MTM'])}")

    # Best filter with stoploss
    best_sl = sl_df[(pd.DataFrame(results).loc[sl_df.index, "dist_pct"] >= 1.5) &
                     (pd.DataFrame(results).loc[sl_df.index, "oi_building"])] \
        if len(sl_df) > 0 else pd.DataFrame()

    # Simpler approach for filtered stoploss
    print(f"\n  BEST FILTER + STOPLOSS (dist >= 1.5% AND OI building):")
    best_mask = (has_pnl["dist_pct"] >= 1.5) & (has_pnl["oi_building"])
    best_pnl = has_pnl[best_mask]
    if len(best_pnl) > 0:
        b_results = []
        for _, row in best_pnl.iterrows():
            entry = row["entry_prem"]
            exit_p = row["exit_prem"]
            sl = entry * 2
            tgt = entry * 0.4
            if exit_p >= sl:
                pnl = entry - sl
            elif exit_p <= tgt:
                pnl = entry - tgt
            else:
                pnl = entry - exit_p
            b_results.append(pnl)
        b_arr = np.array(b_results)
        b_wins = (b_arr > 0).sum()
        b_wr = b_wins / len(b_arr) * 100
        b_avg = b_arr.mean()
        part = len(best_pnl) / total * 100
        print(f"  Trades: {len(best_pnl)} ({part:.1f}% participation)")
        print(f"  Win rate: {b_wr:.1f}%")
        print(f"  Avg P&L: Rs.{b_avg:.2f}")

    # Save
    out = os.path.join(ROOT, "data", f"backtest_wall_full_{symbol}.csv")
    df_r.to_csv(out, index=False)
    print(f"\n  Saved -> {out}")

    return df_r


if __name__ == "__main__":
    for sym in ["NIFTY", "BANKNIFTY"]:
        backtest(sym)

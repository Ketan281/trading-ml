"""
Find the optimal approach for Rs.5000/day with minimum off-day losses.
Tests: different SL levels, diversification, daily loss caps, filtered trades.
"""

import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_backtest(sym):
    df = pd.read_csv(os.path.join(ROOT, f"data/backtest_wall_full_{sym}.csv"))
    return df.dropna(subset=["pnl_per_unit"]).copy()


def compute_managed_pnl(row, lot, sl_mult=2.0, tgt_mult=0.4):
    entry = row["entry_prem"]
    exit_p = row["exit_prem"]
    sl = entry * sl_mult
    tgt = entry * tgt_mult
    if exit_p >= sl:
        return (entry - sl) * lot
    elif exit_p <= tgt:
        return (entry - tgt) * lot
    else:
        return (entry - exit_p) * lot


def daily_stats(daily_pnls):
    days = list(daily_pnls.values())
    if not days:
        return None
    avg = np.mean(days)
    worst = min(days)
    best = max(days)
    wd = sum(1 for d in days if d > 0) / len(days) * 100
    gw = sum(d for d in days if d > 0)
    gl = abs(sum(d for d in days if d <= 0))
    pf = gw / gl if gl > 0 else 999
    return {
        "trading_days": len(days),
        "avg_day": avg,
        "worst_day": worst,
        "best_day": best,
        "win_day_pct": wd,
        "pf": pf,
    }


def main():
    data = {}
    for sym in ["NIFTY", "BANKNIFTY"]:
        data[sym] = {
            "df": load_backtest(sym),
            "lot": 75 if sym == "NIFTY" else 30
        }

    # ===== TEST 1: Different SL levels =====
    print("=" * 70)
    print("  TEST 1: NIFTY WALL SELLING - DIFFERENT STOPLOSS LEVELS (per lot)")
    print("=" * 70)
    print(f"  {'SL':<10} {'WinDays%':<10} {'AvgDay':<10} {'WorstDay':<12} {'PF':<7} {'Lots5k':<8} {'Capital':<12}")
    print(f"  {'-' * 69}")

    for sl in [1.5, 2.0, 2.5, 3.0]:
        daily = {}
        for _, row in data["NIFTY"]["df"].iterrows():
            pnl = compute_managed_pnl(row, 75, sl_mult=sl)
            d = row["date"]
            daily[d] = daily.get(d, 0) + pnl
        s = daily_stats(daily)
        lots = 5000 / s["avg_day"] if s["avg_day"] > 0 else 999
        cap = lots * 120000
        print(f"  {sl}x{'':<6} {s['win_day_pct']:<10.0f} {s['avg_day']:<10.0f} "
              f"{s['worst_day']:<12.0f} {s['pf']:<7.2f} {lots:<8.0f} Rs.{cap:,.0f}")

    # ===== TEST 2: Combined NIFTY + BANKNIFTY =====
    print()
    print("=" * 70)
    print("  TEST 2: NIFTY + BANKNIFTY COMBINED (1 lot each, SL=2x)")
    print("=" * 70)

    daily_comb = {}
    for sym in ["NIFTY", "BANKNIFTY"]:
        for _, row in data[sym]["df"].iterrows():
            pnl = compute_managed_pnl(row, data[sym]["lot"])
            d = row["date"]
            daily_comb[d] = daily_comb.get(d, 0) + pnl

    s = daily_stats(daily_comb)
    lots = 5000 / s["avg_day"] if s["avg_day"] > 0 else 999
    print(f"  Avg daily P&L (1+1 lot) : Rs.{s['avg_day']:.0f}")
    print(f"  Winning days            : {s['win_day_pct']:.0f}%")
    print(f"  Best day                : Rs.{s['best_day']:.0f}")
    print(f"  Worst day               : Rs.{s['worst_day']:.0f}")
    print(f"  PF                      : {s['pf']:.2f}")
    print(f"  Multiplier for 5k/day   : {lots:.0f}x")
    print(f"  Capital needed          : Rs.{lots * 240000:,.0f}")

    # ===== TEST 3: Filtered trades (dist >= 1% AND OI building) =====
    print()
    print("=" * 70)
    print("  TEST 3: FILTERED TRADES ONLY (dist>=1% + OI building)")
    print("=" * 70)

    for sym in ["NIFTY", "BANKNIFTY"]:
        df = data[sym]["df"]
        lot = data[sym]["lot"]
        filtered = df[(df["dist_pct"] >= 1.0) & (df["oi_building"])]

        daily = {}
        for _, row in filtered.iterrows():
            pnl = compute_managed_pnl(row, lot)
            d = row["date"]
            daily[d] = daily.get(d, 0) + pnl

        s = daily_stats(daily)
        if not s:
            continue
        hold_rate = filtered["wall_held"].mean() * 100
        lots = 5000 / s["avg_day"] if s["avg_day"] > 0 else 999
        print(f"\n  {sym}: {len(filtered)} trades across {s['trading_days']} days")
        print(f"    Wall hold rate  : {hold_rate:.1f}%")
        print(f"    Winning days    : {s['win_day_pct']:.0f}%")
        print(f"    Avg day (1 lot) : Rs.{s['avg_day']:.0f}")
        print(f"    Worst day       : Rs.{s['worst_day']:.0f}")
        print(f"    PF              : {s['pf']:.2f}")
        print(f"    Lots for 5k/day : {lots:.0f}")
        print(f"    Capital         : Rs.{lots * 120000:,.0f}")

    # ===== TEST 4: Combined filtered + daily loss cap =====
    print()
    print("=" * 70)
    print("  TEST 4: DIFFERENT DAILY LOSS CAPS (NIFTY filtered, per lot)")
    print("=" * 70)
    print(f"  {'Cap':<14} {'WinDays%':<10} {'AvgDay':<10} {'WorstDay':<12} {'PF':<7}")
    print(f"  {'-' * 53}")

    df_n = data["NIFTY"]["df"]
    df_n_filt = df_n[(df_n["dist_pct"] >= 1.0) & (df_n["oi_building"])]

    for cap in [500, 1000, 2000, 5000, 99999]:
        daily = {}
        for _, row in df_n_filt.iterrows():
            pnl = compute_managed_pnl(row, 75)
            d = row["date"]
            current = daily.get(d, 0)
            if current <= -cap:
                continue
            new = current + pnl
            daily[d] = max(new, -cap)

        s = daily_stats(daily)
        cap_label = f"Rs.{cap}" if cap < 99999 else "No cap"
        print(f"  {cap_label:<14} {s['win_day_pct']:<10.0f} {s['avg_day']:<10.0f} "
              f"{s['worst_day']:<12.0f} {s['pf']:<7.2f}")

    # ===== TEST 5: THE OPTIMAL COMBO =====
    print()
    print("=" * 70)
    print("  TEST 5: OPTIMAL COMBO (filtered + NIFTY+BN + Rs.3000 daily cap)")
    print("=" * 70)

    daily_opt = {}
    for sym in ["NIFTY", "BANKNIFTY"]:
        df = data[sym]["df"]
        lot = data[sym]["lot"]
        filtered = df[(df["dist_pct"] >= 1.0) & (df["oi_building"])]
        for _, row in filtered.iterrows():
            pnl = compute_managed_pnl(row, lot)
            d = row["date"]
            current = daily_opt.get(d, 0)
            if current <= -3000:
                continue
            new = current + pnl
            daily_opt[d] = max(new, -3000)

    s = daily_stats(daily_opt)
    lots = 5000 / s["avg_day"] if s["avg_day"] > 0 else 999
    worst_scaled = s["worst_day"] * lots
    monthly = s["avg_day"] * lots * 22

    print(f"  Trading days       : {s['trading_days']}")
    print(f"  Winning days       : {s['win_day_pct']:.0f}%")
    print(f"  Avg day (1+1 lot)  : Rs.{s['avg_day']:.0f}")
    print(f"  Best day           : Rs.{s['best_day']:.0f}")
    print(f"  Worst day          : Rs.{s['worst_day']:.0f}  (CAPPED)")
    print(f"  Profit factor      : {s['pf']:.2f}")
    print(f"  Lots for 5k/day    : {lots:.0f}x each")
    print(f"  Capital needed     : Rs.{lots * 240000:,.0f}")
    print(f"  Worst day at scale : Rs.{worst_scaled:.0f}")
    print(f"  Monthly avg        : Rs.{monthly:,.0f}")

    # ===== FINAL COMPARISON =====
    print()
    print("=" * 70)
    print("  FINAL COMPARISON: ALL APPROACHES FOR Rs.5000/DAY")
    print("=" * 70)
    print(f"  {'Approach':<40} {'WinDay%':<9} {'WorstDay/lot':<14} {'PF':<7} {'Capital':<12}")
    print(f"  {'-' * 82}")

    approaches = []

    # A: NIFTY only, unfiltered
    daily_a = {}
    for _, row in data["NIFTY"]["df"].iterrows():
        pnl = compute_managed_pnl(row, 75)
        d = row["date"]
        daily_a[d] = daily_a.get(d, 0) + pnl
    sa = daily_stats(daily_a)
    la = 5000 / sa["avg_day"]
    approaches.append(("A: NIFTY unfiltered", sa["win_day_pct"], sa["worst_day"], sa["pf"], la * 120000, sa["worst_day"] * la))

    # B: NIFTY filtered
    daily_b = {}
    for _, row in df_n_filt.iterrows():
        pnl = compute_managed_pnl(row, 75)
        d = row["date"]
        daily_b[d] = daily_b.get(d, 0) + pnl
    sb = daily_stats(daily_b)
    lb = 5000 / sb["avg_day"]
    approaches.append(("B: NIFTY filtered (dist>=1%+OIbuild)", sb["win_day_pct"], sb["worst_day"], sb["pf"], lb * 120000, sb["worst_day"] * lb))

    # C: Combined unfiltered
    sc = daily_stats(daily_comb)
    lc = 5000 / sc["avg_day"]
    approaches.append(("C: NIFTY+BN combined", sc["win_day_pct"], sc["worst_day"], sc["pf"], lc * 240000, sc["worst_day"] * lc))

    # D: Optimal combo
    sd = daily_stats(daily_opt)
    ld = 5000 / sd["avg_day"]
    approaches.append(("D: Filtered+Combined+Cap3k (OPTIMAL)", sd["win_day_pct"], sd["worst_day"], sd["pf"], ld * 240000, sd["worst_day"] * ld))

    for name, wd, worst, pf, cap, worst_scaled in approaches:
        print(f"  {name:<40} {wd:<9.0f} Rs.{worst:<11.0f} {pf:<7.2f} Rs.{cap:,.0f}")

    print()
    print(f"  WORST DAY AT Rs.5k/day SCALE:")
    for name, wd, worst, pf, cap, worst_scaled in approaches:
        print(f"    {name:<40} Rs.{worst_scaled:,.0f}")


if __name__ == "__main__":
    main()

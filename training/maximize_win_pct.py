"""
Maximize win % for wall selling by testing every filter combination.
Filters: SL level, distance, OI building, wall type, day of week, IV level.
"""

import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load(sym):
    df = pd.read_csv(os.path.join(ROOT, f"data/backtest_wall_full_{sym}.csv"))
    df = df.dropna(subset=["pnl_per_unit"]).copy()
    df["date_dt"] = pd.to_datetime(df["date"])
    df["dow"] = df["date_dt"].dt.dayofweek  # 0=Mon, 4=Fri
    return df


def evaluate(df, lot, sl_mult=1.5, label=""):
    daily = {}
    wins = 0
    total = 0
    for _, row in df.iterrows():
        entry = row["entry_prem"]
        exit_p = row["exit_prem"]
        sl = entry * sl_mult
        tgt = entry * 0.4
        if exit_p >= sl:
            pnl = (entry - sl) * lot
        elif exit_p <= tgt:
            pnl = (entry - tgt) * lot
        else:
            pnl = (entry - exit_p) * lot

        total += 1
        if pnl > 0:
            wins += 1
        d = row["date"]
        daily[d] = daily.get(d, 0) + pnl

    if total == 0:
        return None

    days = list(daily.values())
    avg = np.mean(days)
    worst = min(days)
    win_trades = wins / total * 100
    win_days = sum(1 for d in days if d > 0) / len(days) * 100
    gw = sum(d for d in days if d > 0)
    gl = abs(sum(d for d in days if d <= 0))
    pf = gw / gl if gl > 0 else 999

    return {
        "label": label,
        "trades": total,
        "days": len(days),
        "win_trade_pct": round(win_trades, 1),
        "win_day_pct": round(win_days, 1),
        "avg_day": round(avg, 0),
        "worst_day": round(worst, 0),
        "pf": round(pf, 2),
    }


def main():
    for sym in ["NIFTY"]:
        df = load(sym)
        lot = 75 if sym == "NIFTY" else 30

        print("=" * 80)
        print(f"  MAXIMIZING WIN % — {sym} WALL SELLING")
        print("=" * 80)

        # ── FILTER 1: Distance from spot ──
        print("\n  FILTER 1: MINIMUM DISTANCE FROM SPOT")
        print(f"  {'MinDist':<10} {'Trades':<8} {'WinTrd%':<10} {'WinDay%':<10} "
              f"{'AvgDay':<10} {'WorstDay':<12} {'PF':<7}")
        print(f"  {'-' * 67}")
        for d in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            filt = df[df["dist_pct"] >= d]
            r = evaluate(filt, lot, 1.5, f">={d}%")
            if r:
                print(f"  {r['label']:<10} {r['trades']:<8} {r['win_trade_pct']:<10} "
                      f"{r['win_day_pct']:<10} {r['avg_day']:<10} {r['worst_day']:<12} {r['pf']:<7}")

        # ── FILTER 2: OI building ──
        print("\n  FILTER 2: OI BUILDING (fresh money at wall)")
        for oi_flag, label in [(True, "OI building"), (False, "OI unwinding")]:
            filt = df[df["oi_building"] == oi_flag]
            r = evaluate(filt, lot, 1.5, label)
            if r:
                print(f"  {r['label']:<16} {r['trades']:<8} WinTrd={r['win_trade_pct']}% "
                      f"WinDay={r['win_day_pct']}% PF={r['pf']}")

        # ── FILTER 3: Wall type (put vs call) ──
        print("\n  FILTER 3: WALL TYPE")
        type_col = "wall_type" if "wall_type" in df.columns else "type"
        for wt in df[type_col].dropna().unique():
            filt = df[df[type_col] == wt]
            r = evaluate(filt, lot, 1.5, str(wt))
            if r:
                print(f"  {r['label']:<16} {r['trades']:<8} WinTrd={r['win_trade_pct']}% "
                      f"WinDay={r['win_day_pct']}% PF={r['pf']}")

        # ── FILTER 4: Day of week ──
        print("\n  FILTER 4: DAY OF WEEK")
        for dow, name in [(0, "Monday"), (1, "Tuesday"), (2, "Wednesday"),
                          (3, "Thursday"), (4, "Friday")]:
            filt = df[df["dow"] == dow]
            r = evaluate(filt, lot, 1.5, name)
            if r:
                print(f"  {r['label']:<16} {r['trades']:<8} WinTrd={r['win_trade_pct']}% "
                      f"WinDay={r['win_day_pct']}% PF={r['pf']}")

        # ── FILTER 5: Premium level (cheap vs expensive) ──
        print("\n  FILTER 5: PREMIUM LEVEL")
        for lo, hi, label in [(0, 10, "<Rs.10"), (10, 25, "Rs.10-25"),
                              (25, 50, "Rs.25-50"), (50, 100, "Rs.50-100"),
                              (100, 500, ">Rs.100")]:
            filt = df[(df["entry_prem"] >= lo) & (df["entry_prem"] < hi)]
            r = evaluate(filt, lot, 1.5, label)
            if r:
                print(f"  {r['label']:<16} {r['trades']:<8} WinTrd={r['win_trade_pct']}% "
                      f"WinDay={r['win_day_pct']}% PF={r['pf']}")

        # ── COMBINATIONS: Stack winning filters ──
        print("\n" + "=" * 80)
        print("  STACKING FILTERS (best combos)")
        print("=" * 80)
        print(f"  {'Combo':<45} {'Trades':<8} {'WinTrd%':<10} {'WinDay%':<10} "
              f"{'AvgDay':<10} {'WorstDay':<12} {'PF':<7}")
        print(f"  {'-' * 95}")

        combos = [
            ("No filter", df),
            ("dist>=1.5%", df[df["dist_pct"] >= 1.5]),
            ("dist>=2.0%", df[df["dist_pct"] >= 2.0]),
            ("dist>=1.5% + OI building", df[(df["dist_pct"] >= 1.5) & (df["oi_building"])]),
            ("dist>=2.0% + OI building", df[(df["dist_pct"] >= 2.0) & (df["oi_building"])]),
            ("dist>=1.5% + prem<25", df[(df["dist_pct"] >= 1.5) & (df["entry_prem"] < 25)]),
            ("dist>=2.0% + prem<25", df[(df["dist_pct"] >= 2.0) & (df["entry_prem"] < 25)]),
            ("dist>=1.5% + OI build + prem<25",
             df[(df["dist_pct"] >= 1.5) & (df["oi_building"]) & (df["entry_prem"] < 25)]),
            ("dist>=2.0% + OI build + prem<25",
             df[(df["dist_pct"] >= 2.0) & (df["oi_building"]) & (df["entry_prem"] < 25)]),
            ("dist>=2.0% + OI build + no Thu/Fri",
             df[(df["dist_pct"] >= 2.0) & (df["oi_building"]) & (~df["dow"].isin([3, 4]))]),
            ("dist>=1.5% + OI build + put_wall only",
             df[(df["dist_pct"] >= 1.5) & (df["oi_building"]) & (df[type_col] == df[type_col].unique()[0])]),
            ("dist>=2.0% + OI build + SL=1.3x",
             df[(df["dist_pct"] >= 2.0) & (df["oi_building"])]),
        ]

        for label, filt in combos:
            sl = 1.3 if "SL=1.3x" in label else 1.5
            r = evaluate(filt, lot, sl, label)
            if r:
                print(f"  {r['label']:<45} {r['trades']:<8} {r['win_trade_pct']:<10} "
                      f"{r['win_day_pct']:<10} {r['avg_day']:<10} {r['worst_day']:<12} {r['pf']:<7}")

        # ── SWEET SPOT FINDER ──
        print("\n" + "=" * 80)
        print("  SWEET SPOT: WIN% vs TRADES (how many trades do you sacrifice)")
        print("=" * 80)
        print(f"  {'Config':<40} {'WinTrd%':<10} {'Trades':<8} {'Days':<7} "
              f"{'5k/day lots':<12} {'Capital':<12}")
        print(f"  {'-' * 89}")

        sweet = [
            ("Baseline (all trades, SL=1.5x)", df, 1.5),
            ("dist>=1.5% + OI build, SL=1.5x",
             df[(df["dist_pct"] >= 1.5) & (df["oi_building"])], 1.5),
            ("dist>=2.0% + OI build, SL=1.5x",
             df[(df["dist_pct"] >= 2.0) & (df["oi_building"])], 1.5),
            ("dist>=2.0% + OI build, SL=1.3x",
             df[(df["dist_pct"] >= 2.0) & (df["oi_building"])], 1.3),
            ("dist>=2.5% + OI build, SL=1.3x",
             df[(df["dist_pct"] >= 2.5) & (df["oi_building"])], 1.3),
            ("dist>=3.0% + OI build, SL=1.3x",
             df[(df["dist_pct"] >= 3.0) & (df["oi_building"])], 1.3),
        ]

        for label, filt, sl in sweet:
            r = evaluate(filt, lot, sl, label)
            if r and r["avg_day"] > 0:
                lots5 = 5000 / r["avg_day"]
                cap = lots5 * 120000
                print(f"  {r['label']:<40} {r['win_trade_pct']:<10} {r['trades']:<8} "
                      f"{r['days']:<7} {lots5:<12.0f} Rs.{cap:,.0f}")


if __name__ == "__main__":
    main()

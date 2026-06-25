"""
Smart filter scoring: combine all winning filters with OR/scoring logic
to maximize participation while keeping win % at 85%+.

Each trade gets a score based on how many winning conditions it meets.
Trade if score >= threshold.
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
    df["dow"] = df["date_dt"].dt.dayofweek
    return df


def score_trade(row):
    """Score each trade based on proven winning filters.
    Each filter contributes points proportional to its edge."""
    score = 0

    # Filter 1: Distance from spot (biggest edge)
    dist = row["dist_pct"]
    if dist >= 3.0:
        score += 30
    elif dist >= 2.5:
        score += 25  # 83.3% win at >=2.5%
    elif dist >= 2.0:
        score += 20  # 79.4% win
    elif dist >= 1.5:
        score += 15  # 78.4% win
    elif dist >= 1.0:
        score += 10  # 77.2% win
    else:
        score += 0   # close to spot = risky

    # Filter 2: Premium level (cheap = safe)
    prem = row["entry_prem"]
    if prem < 5:
        score += 25  # nearly worthless, will expire OTM
    elif prem < 10:
        score += 20  # 84.7% win
    elif prem < 25:
        score += 15  # 77% win
    elif prem < 50:
        score += 5
    else:
        score += 0   # expensive = close to ATM = risky

    # Filter 3: Day of week
    dow = row["dow"]
    if dow == 2:    # Wednesday
        score += 20  # 90.5% win
    elif dow == 1:  # Tuesday
        score += 10  # 77% win
    elif dow == 0:  # Monday
        score += 5   # 73.6% win
    elif dow == 4:  # Friday
        score += 0   # 66.4% win
    elif dow == 3:  # Thursday (expiry)
        score -= 15  # 0% win (only 9 trades but terrible)

    # Filter 4: Wall type
    wtype = row.get("type", "")
    if wtype == "put":
        score += 10  # 77.6% vs 74.3% for call
    else:
        score += 5

    # Filter 5: OI building
    if row.get("oi_building", False):
        score += 5

    return score


def evaluate(df, lot, sl_mult=1.5):
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
    win_t = wins / total * 100
    win_d = sum(1 for d in days if d > 0) / len(days) * 100
    gw = sum(d for d in days if d > 0)
    gl = abs(sum(d for d in days if d <= 0))
    pf = gw / gl if gl > 0 else 999
    return {
        "trades": total,
        "days": len(days),
        "win_trade": round(win_t, 1),
        "win_day": round(win_d, 1),
        "avg_day": round(avg, 0),
        "worst_day": round(worst, 0),
        "pf": round(pf, 2),
    }


def main():
    for sym in ["NIFTY", "BANKNIFTY"]:
        df = load(sym)
        lot = 75 if sym == "NIFTY" else 30
        total_days = df["date"].nunique()

        # Score every trade
        df["score"] = df.apply(score_trade, axis=1)

        print("=" * 85)
        print(f"  SMART FILTER SCORING: {sym} (total {len(df)} trades, {total_days} days)")
        print("=" * 85)

        # Show score distribution
        print("\n  Score distribution:")
        for lo, hi in [(0, 20), (20, 35), (35, 50), (50, 65), (65, 100)]:
            subset = df[(df["score"] >= lo) & (df["score"] < hi)]
            if len(subset) > 0:
                wr = subset["wall_held"].mean() * 100
                print(f"    Score {lo}-{hi}: {len(subset)} trades, wall hold={wr:.1f}%")

        # Test different thresholds
        print(f"\n  {'Threshold':<12} {'Trades':<8} {'Days':<7} {'Part%':<8} "
              f"{'WinTrd%':<10} {'WinDay%':<10} {'AvgDay':<10} {'WorstDay':<12} "
              f"{'PF':<7} {'5k lots':<9} {'Capital':<12}")
        print(f"  {'-' * 105}")

        for thresh in [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]:
            filt = df[df["score"] >= thresh]
            r = evaluate(filt, lot, 1.5)
            if r and r["trades"] > 10:
                part = r["days"] / total_days * 100
                lots5 = 5000 / r["avg_day"] if r["avg_day"] > 0 else 999
                cap = lots5 * 120000
                marker = ""
                if 84 <= r["win_trade"] <= 86 and part >= 50:
                    marker = " <-- SWEET SPOT"
                elif r["win_trade"] >= 85 and part >= 40:
                    marker = " <-- HIGH WIN"
                print(f"  {'>=' + str(thresh):<12} {r['trades']:<8} {r['days']:<7} "
                      f"{part:<8.0f} {r['win_trade']:<10} {r['win_day']:<10} "
                      f"{r['avg_day']:<10} {r['worst_day']:<12} {r['pf']:<7}"
                      f"  {lots5:<9.0f} Rs.{cap:,.0f}{marker}")

        # Find the exact sweet spot: highest participation with win >= 85%
        print(f"\n  --- FINDING SWEET SPOT: win% >= 85% with max participation ---")
        best = None
        for thresh in range(10, 80):
            filt = df[df["score"] >= thresh]
            r = evaluate(filt, lot, 1.5)
            if r and r["win_trade"] >= 85 and r["trades"] >= 20:
                part = r["days"] / total_days * 100
                if best is None or part > best["part"]:
                    best = {"thresh": thresh, "part": part, **r}

        if best:
            lots5 = 5000 / best["avg_day"] if best["avg_day"] > 0 else 999
            cap = lots5 * 120000
            print(f"  Score >= {best['thresh']}")
            print(f"  Trades          : {best['trades']}")
            print(f"  Trading days    : {best['days']}")
            print(f"  Participation   : {best['part']:.0f}%")
            print(f"  Win % (trades)  : {best['win_trade']}%")
            print(f"  Win % (days)    : {best['win_day']}%")
            print(f"  Avg day (1 lot) : Rs.{best['avg_day']}")
            print(f"  Worst day       : Rs.{best['worst_day']}")
            print(f"  PF              : {best['pf']}")
            print(f"  Lots for 5k/day : {lots5:.0f}")
            print(f"  Capital needed  : Rs.{cap:,.0f}")
            print(f"  Worst at scale  : Rs.{best['worst_day'] * lots5:,.0f}")
            print(f"  Monthly avg     : Rs.{best['avg_day'] * lots5 * 22:,.0f}")

        # Also find sweet spot for win >= 80%
        print(f"\n  --- SWEET SPOT: win% >= 80% with max participation ---")
        best80 = None
        for thresh in range(10, 80):
            filt = df[df["score"] >= thresh]
            r = evaluate(filt, lot, 1.5)
            if r and r["win_trade"] >= 80 and r["trades"] >= 20:
                part = r["days"] / total_days * 100
                if best80 is None or part > best80["part"]:
                    best80 = {"thresh": thresh, "part": part, **r}

        if best80:
            lots5 = 5000 / best80["avg_day"] if best80["avg_day"] > 0 else 999
            cap = lots5 * 120000
            print(f"  Score >= {best80['thresh']}")
            print(f"  Trades          : {best80['trades']}")
            print(f"  Trading days    : {best80['days']}")
            print(f"  Participation   : {best80['part']:.0f}%")
            print(f"  Win % (trades)  : {best80['win_trade']}%")
            print(f"  Win % (days)    : {best80['win_day']}%")
            print(f"  Avg day (1 lot) : Rs.{best80['avg_day']}")
            print(f"  Worst day       : Rs.{best80['worst_day']}")
            print(f"  PF              : {best80['pf']}")
            print(f"  Lots for 5k/day : {lots5:.0f}")
            print(f"  Capital needed  : Rs.{cap:,.0f}")

        print()


if __name__ == "__main__":
    main()

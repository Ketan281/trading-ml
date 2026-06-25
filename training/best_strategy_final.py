"""
Find the absolute best strategy: maximize WIN% x PARTICIPATION% product.
Tests: score tuning, combined symbols, tiered sizing, re-weighted scoring,
strangle (sell both walls), and different SL per tier.
"""

import os, sys, glob
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RAW_DIR = os.path.join(ROOT, "data", "option_chain", "raw")
STRIKE_STEP = {"NIFTY": 50, "BANKNIFTY": 100}


def find_walls(chain_df, spot):
    walls = []
    puts = chain_df[chain_df["strike"] < spot].copy()
    if len(puts) > 0:
        w = puts.sort_values("pe_oi", ascending=False).iloc[0]
        walls.append({"type": "put", "strike": int(w["strike"]),
                       "oi": float(w["pe_oi"]), "chg_oi": float(w.get("pe_chg_oi", 0)),
                       "ltp": float(w["pe_ltp"])})
    calls = chain_df[chain_df["strike"] > spot].copy()
    if len(calls) > 0:
        w = calls.sort_values("ce_oi", ascending=False).iloc[0]
        walls.append({"type": "call", "strike": int(w["strike"]),
                       "oi": float(w["ce_oi"]), "chg_oi": float(w.get("ce_chg_oi", 0)),
                       "ltp": float(w["ce_ltp"])})
    return walls


def load_trades(symbol):
    """Load all wall trades with features."""
    files = sorted(glob.glob(os.path.join(RAW_DIR, symbol, "*.csv")))
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

    prices = pd.read_csv(os.path.join(ROOT, "data", "features", f"{symbol}_features.csv"),
                         index_col="Date", parse_dates=True)
    dates = sorted(chain_data.keys())

    trades = []
    for i in range(len(dates) - 1):
        d, nd = dates[i], dates[i + 1]
        td, ntd = pd.Timestamp(d), pd.Timestamp(nd)
        if td not in prices.index or ntd not in prices.index:
            continue
        spot = float(prices.loc[td, "Close"])
        next_spot = float(prices.loc[ntd, "Close"])
        dow = td.dayofweek

        for wall in find_walls(chain_data[d], spot):
            strike = wall["strike"]
            dist_pct = abs(strike - spot) / spot * 100
            entry_prem = wall["ltp"]
            if entry_prem <= 0 or dist_pct < 0.3:
                continue

            next_rows = chain_data[nd][chain_data[nd]["strike"] == strike]
            if len(next_rows) == 0:
                continue
            col = "pe_ltp" if wall["type"] == "put" else "ce_ltp"
            exit_prem = float(next_rows.iloc[0][col])

            wall_held = 1 if (
                (wall["type"] == "put" and next_spot >= strike) or
                (wall["type"] == "call" and next_spot <= strike)
            ) else 0

            trades.append({
                "date": d, "symbol": symbol, "type": wall["type"],
                "strike": strike, "spot": spot, "next_spot": next_spot,
                "dist_pct": round(dist_pct, 3),
                "entry_prem": round(entry_prem, 2),
                "exit_prem": round(exit_prem, 2),
                "wall_held": wall_held,
                "oi_building": wall["chg_oi"] > 0,
                "dow": dow, "year": td.year,
            })

    return pd.DataFrame(trades)


# ── SCORING FUNCTIONS ──

def score_v1(r):
    """Original scoring."""
    s = 0
    if r["dist_pct"] >= 3.0: s += 30
    elif r["dist_pct"] >= 2.5: s += 25
    elif r["dist_pct"] >= 2.0: s += 20
    elif r["dist_pct"] >= 1.5: s += 15
    elif r["dist_pct"] >= 1.0: s += 10
    if r["entry_prem"] < 5: s += 25
    elif r["entry_prem"] < 10: s += 20
    elif r["entry_prem"] < 25: s += 15
    elif r["entry_prem"] < 50: s += 5
    if r["dow"] == 2: s += 20
    elif r["dow"] == 1: s += 10
    elif r["dow"] == 0: s += 5
    elif r["dow"] == 3: s -= 15
    if r["type"] == "put": s += 10
    else: s += 5
    if r["oi_building"]: s += 5
    return s


def score_v2(r):
    """V2: heavier weight on distance + premium (the two strongest signals)."""
    s = 0
    # Distance (40 points max — biggest predictor)
    if r["dist_pct"] >= 3.0: s += 40
    elif r["dist_pct"] >= 2.5: s += 35
    elif r["dist_pct"] >= 2.0: s += 28
    elif r["dist_pct"] >= 1.5: s += 20
    elif r["dist_pct"] >= 1.0: s += 12
    elif r["dist_pct"] >= 0.7: s += 5

    # Premium (30 points max — second strongest)
    if r["entry_prem"] < 3: s += 30
    elif r["entry_prem"] < 5: s += 25
    elif r["entry_prem"] < 10: s += 20
    elif r["entry_prem"] < 15: s += 15
    elif r["entry_prem"] < 25: s += 10
    elif r["entry_prem"] < 50: s += 5

    # Day of week (15 points max)
    if r["dow"] == 2: s += 15
    elif r["dow"] == 1: s += 10
    elif r["dow"] == 0: s += 5
    elif r["dow"] == 4: s -= 5
    elif r["dow"] == 3: s -= 10

    # Wall type (10 points max)
    if r["type"] == "put": s += 10
    else: s += 5

    # OI building (5 points)
    if r["oi_building"]: s += 5
    return s


def score_v3(r):
    """V3: distance-premium combined score (multiplicative)."""
    # Core = distance * premium decay
    dist_score = min(r["dist_pct"] / 3.0, 1.0) * 50  # 0-50
    prem_score = max(0, 1 - r["entry_prem"] / 50) * 30  # 0-30

    # Day bonus
    day_bonus = {0: 3, 1: 8, 2: 15, 3: -10, 4: 0}.get(r["dow"], 0)

    # Type bonus
    type_bonus = 7 if r["type"] == "put" else 3

    return dist_score + prem_score + day_bonus + type_bonus


def evaluate(df_trades, lot, sl_mult=1.5):
    """Evaluate trades and return daily stats."""
    if len(df_trades) == 0:
        return None

    daily = {}
    wins = 0
    for _, r in df_trades.iterrows():
        entry, exit_p = r["entry_prem"], r["exit_prem"]
        sl, tgt = entry * sl_mult, entry * 0.4
        if exit_p >= sl: pnl = (entry - sl) * lot
        elif exit_p <= tgt: pnl = (entry - tgt) * lot
        else: pnl = (entry - exit_p) * lot
        if pnl > 0: wins += 1
        d = r["date"]
        daily[d] = daily.get(d, 0) + pnl

    days = list(daily.values())
    return {
        "trades": len(df_trades),
        "days": len(days),
        "win_trade": round(wins / len(df_trades) * 100, 1),
        "win_day": round(sum(1 for v in days if v > 0) / len(days) * 100, 1),
        "avg_day": round(np.mean(days), 0),
        "worst_day": round(min(days), 0),
        "pf": round(sum(v for v in days if v > 0) / (abs(sum(v for v in days if v <= 0)) + 1e-9), 2),
    }


def main():
    # Load all trades
    print("Loading trades...")
    all_trades = {}
    total_chain_days = {}
    for sym in ["NIFTY", "BANKNIFTY"]:
        df = load_trades(sym)
        all_trades[sym] = df
        total_chain_days[sym] = df["date"].nunique()
        print(f"  {sym}: {len(df)} trades, {total_chain_days[sym]} unique days")

    # ══════════════════════════════════════════════════════════════════
    # STRATEGY 1: Single symbol, different score versions
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  STRATEGY 1: SCORE VERSION COMPARISON (NIFTY)")
    print(f"{'='*90}")

    df_n = all_trades["NIFTY"]
    lot_n = 75
    total_n = total_chain_days["NIFTY"]

    for name, score_fn in [("V1 (original)", score_v1), ("V2 (dist+prem heavy)", score_v2),
                           ("V3 (multiplicative)", score_v3)]:
        df_n[f"sc_{name[:2]}"] = df_n.apply(score_fn, axis=1)

        print(f"\n  {name}:")
        print(f"  {'Thresh':<10} {'Trades':<8} {'Days':<7} {'Part%':<8} {'WinTrd%':<10} "
              f"{'WinDay%':<10} {'PF':<7} {'WxP':<8}")
        print(f"  {'-'*68}")

        col = f"sc_{name[:2]}"
        for t in range(20, 75, 5):
            filt = df_n[df_n[col] >= t]
            r = evaluate(filt, lot_n)
            if r and r["trades"] >= 15:
                part = r["days"] / total_n * 100
                wxp = r["win_trade"] * part / 100  # win% x participation% product
                print(f"  >={t:<8} {r['trades']:<8} {r['days']:<7} {part:<8.0f} "
                      f"{r['win_trade']:<10} {r['win_day']:<10} {r['pf']:<7} {wxp:<8.1f}")

    # ══════════════════════════════════════════════════════════════════
    # STRATEGY 2: Combined NIFTY + BANKNIFTY
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  STRATEGY 2: COMBINED NIFTY + BANKNIFTY (more participation)")
    print(f"{'='*90}")

    # Use best scoring version
    for sym in ["NIFTY", "BANKNIFTY"]:
        all_trades[sym]["score"] = all_trades[sym].apply(score_v2, axis=1)

    combined = pd.concat([all_trades["NIFTY"], all_trades["BANKNIFTY"]], ignore_index=True)
    total_combined_days = combined["date"].nunique()

    print(f"\n  {'Thresh':<10} {'Trades':<8} {'Days':<7} {'Part%':<8} {'WinTrd%':<10} "
          f"{'WinDay%':<10} {'WorstDay':<12} {'PF':<7} {'WxP':<8}")
    print(f"  {'-'*80}")

    for t in range(20, 75, 5):
        filt_n = all_trades["NIFTY"][all_trades["NIFTY"]["score"] >= t]
        filt_b = all_trades["BANKNIFTY"][all_trades["BANKNIFTY"]["score"] >= t]

        # Compute combined daily P&L
        daily = {}
        wins = 0
        total_trades = 0
        for sym_df, lot in [(filt_n, 75), (filt_b, 30)]:
            for _, r in sym_df.iterrows():
                entry, exit_p = r["entry_prem"], r["exit_prem"]
                sl, tgt = entry * 1.5, entry * 0.4
                if exit_p >= sl: pnl = (entry - sl) * lot
                elif exit_p <= tgt: pnl = (entry - tgt) * lot
                else: pnl = (entry - exit_p) * lot
                if pnl > 0: wins += 1
                total_trades += 1
                daily[r["date"]] = daily.get(r["date"], 0) + pnl

        if total_trades < 15:
            continue
        days = list(daily.values())
        part = len(days) / total_combined_days * 100
        wt = wins / total_trades * 100
        wd = sum(1 for v in days if v > 0) / len(days) * 100
        worst = min(days)
        gw = sum(v for v in days if v > 0)
        gl = abs(sum(v for v in days if v <= 0))
        pf = gw / gl if gl > 0 else 999
        wxp = wt * part / 100

        marker = ""
        if wxp > 50:
            marker = " <-- BEST"

        print(f"  >={t:<8} {total_trades:<8} {len(days):<7} {part:<8.0f} {wt:<10.1f} "
              f"{wd:<10.1f} {worst:<12.0f} {pf:<7.2f} {wxp:<8.1f}{marker}")

    # ══════════════════════════════════════════════════════════════════
    # STRATEGY 3: Tiered sizing (more participation, managed risk)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  STRATEGY 3: TIERED SIZING (trade everything, size by score)")
    print(f"{'='*90}")
    print(f"  Tier 1 (score>=55): Full size  |  Tier 2 (35-54): Half  |  Tier 3 (20-34): Quarter")

    for sym in ["NIFTY", "BANKNIFTY"]:
        df = all_trades[sym]
        lot = 75 if sym == "NIFTY" else 30
        total_d = total_chain_days[sym]

        # Tiered
        daily_tiered = {}
        wins_tiered = 0
        total_tiered = 0
        tier_stats = {1: {"w": 0, "t": 0}, 2: {"w": 0, "t": 0}, 3: {"w": 0, "t": 0}}

        for _, r in df.iterrows():
            sc = r["score"]
            if sc < 20:
                continue

            if sc >= 55:
                size_mult = 1.0
                tier = 1
            elif sc >= 35:
                size_mult = 0.5
                tier = 2
            else:
                size_mult = 0.25
                tier = 3

            entry, exit_p = r["entry_prem"], r["exit_prem"]
            sl, tgt = entry * 1.5, entry * 0.4
            if exit_p >= sl: pnl = (entry - sl) * lot * size_mult
            elif exit_p <= tgt: pnl = (entry - tgt) * lot * size_mult
            else: pnl = (entry - exit_p) * lot * size_mult

            total_tiered += 1
            tier_stats[tier]["t"] += 1
            if pnl > 0:
                wins_tiered += 1
                tier_stats[tier]["w"] += 1
            daily_tiered[r["date"]] = daily_tiered.get(r["date"], 0) + pnl

        days = list(daily_tiered.values())
        part = len(days) / total_d * 100
        wt = wins_tiered / total_tiered * 100
        wd = sum(1 for v in days if v > 0) / len(days) * 100
        avg = np.mean(days)
        worst = min(days)
        gw = sum(v for v in days if v > 0)
        gl = abs(sum(v for v in days if v <= 0))
        pf = gw / gl if gl > 0 else 999

        print(f"\n  {sym}:")
        print(f"    Trades        : {total_tiered}")
        print(f"    Trading days  : {len(days)} ({part:.0f}% participation)")
        print(f"    Win % trades  : {wt:.1f}%")
        print(f"    Win % days    : {wd:.1f}%")
        print(f"    Avg day       : Rs.{avg:.0f}")
        print(f"    Worst day     : Rs.{worst:.0f}")
        print(f"    PF            : {pf:.2f}")
        for tier in [1, 2, 3]:
            ts = tier_stats[tier]
            tw = ts["w"] / ts["t"] * 100 if ts["t"] > 0 else 0
            print(f"    Tier {tier}: {ts['t']} trades, {tw:.1f}% win")

    # ══════════════════════════════════════════════════════════════════
    # STRATEGY 4: Combined + Tiered (THE ULTIMATE)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  STRATEGY 4: COMBINED NIFTY+BN + TIERED SIZING (THE ULTIMATE)")
    print(f"{'='*90}")

    daily_ult = {}
    wins_ult = 0
    total_ult = 0
    tier_stats_ult = {1: {"w": 0, "t": 0}, 2: {"w": 0, "t": 0}, 3: {"w": 0, "t": 0}}

    for sym in ["NIFTY", "BANKNIFTY"]:
        df = all_trades[sym]
        lot = 75 if sym == "NIFTY" else 30

        for _, r in df.iterrows():
            sc = r["score"]
            if sc < 20:
                continue
            if sc >= 55: size_mult, tier = 1.0, 1
            elif sc >= 35: size_mult, tier = 0.5, 2
            else: size_mult, tier = 0.25, 3

            entry, exit_p = r["entry_prem"], r["exit_prem"]
            sl, tgt = entry * 1.5, entry * 0.4
            if exit_p >= sl: pnl = (entry - sl) * lot * size_mult
            elif exit_p <= tgt: pnl = (entry - tgt) * lot * size_mult
            else: pnl = (entry - exit_p) * lot * size_mult

            total_ult += 1
            tier_stats_ult[tier]["t"] += 1
            if pnl > 0:
                wins_ult += 1
                tier_stats_ult[tier]["w"] += 1
            daily_ult[r["date"]] = daily_ult.get(r["date"], 0) + pnl

    days = sorted(daily_ult.keys())
    vals = [daily_ult[d] for d in days]
    part = len(days) / total_combined_days * 100
    wt = wins_ult / total_ult * 100
    wd = sum(1 for v in vals if v > 0) / len(vals) * 100
    avg = np.mean(vals)
    worst = min(vals)
    best = max(vals)
    gw = sum(v for v in vals if v > 0)
    gl = abs(sum(v for v in vals if v <= 0))
    pf = gw / gl if gl > 0 else 999
    sharpe = np.mean(vals) / (np.std(vals) + 1e-9) * np.sqrt(252)

    equity = np.cumsum(vals)
    peak = np.maximum.accumulate(equity)
    max_dd = (equity - peak).min()

    print(f"\n  Date range      : {days[0]} to {days[-1]}")
    print(f"  Total trades    : {total_ult}")
    print(f"  Trading days    : {len(days)} ({part:.0f}% participation)")
    print(f"  Win % (trades)  : {wt:.1f}%")
    print(f"  Win % (days)    : {wd:.1f}%")
    print(f"  Avg day         : Rs.{avg:.0f}")
    print(f"  Best day        : Rs.{best:.0f}")
    print(f"  Worst day       : Rs.{worst:.0f}")
    print(f"  PF              : {pf:.2f}")
    print(f"  Sharpe          : {sharpe:.2f}")
    print(f"  Max DD          : Rs.{max_dd:.0f}")
    print(f"  Total P&L       : Rs.{sum(vals):,.0f}")

    for tier in [1, 2, 3]:
        ts = tier_stats_ult[tier]
        tw = ts["w"] / ts["t"] * 100 if ts["t"] > 0 else 0
        label = ["Full size", "Half size", "Quarter"][tier - 1]
        print(f"  Tier {tier} ({label}): {ts['t']} trades, {tw:.1f}% win")

    # Year-by-year
    print(f"\n  Year-by-year:")
    print(f"  {'Year':<8} {'Days':<7} {'WinDay%':<10} {'PF':<7} {'AvgDay':<10} {'WorstDay':<10}")
    print(f"  {'-'*52}")
    yr_data = {}
    for d, v in zip(days, vals):
        yr = pd.Timestamp(d).year
        yr_data.setdefault(yr, []).append(v)
    for yr in sorted(yr_data.keys()):
        yv = yr_data[yr]
        yw = sum(1 for v in yv if v > 0) / len(yv) * 100
        ygw = sum(v for v in yv if v > 0)
        ygl = abs(sum(v for v in yv if v <= 0))
        ypf = ygw / ygl if ygl > 0 else 999
        print(f"  {yr:<8} {len(yv):<7} {yw:<10.1f} {ypf:<7.2f} {np.mean(yv):<10.0f} {min(yv):<10.0f}")

    # For Rs.1 lakh
    if avg > 0:
        print(f"\n  WITH Rs.1 LAKH (1 lot each, tiered sizing):")
        print(f"    Monthly income : Rs.{avg * 22:,.0f}")
        print(f"    Yearly income  : Rs.{avg * 250:,.0f}")
        print(f"    Monthly ROI    : {avg * 22 / 100000 * 100:.1f}%")
        print(f"    Worst day      : Rs.{worst:,.0f}")

    # ══════════════════════════════════════════════════════════════════
    # FINAL SUMMARY: All strategies compared
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  FINAL RANKING: ALL STRATEGIES")
    print(f"{'='*90}")
    print(f"  {'Strategy':<45} {'WinTrd%':<10} {'Part%':<8} {'WxP':<8} "
          f"{'PF':<7} {'WorstDay':<10}")
    print(f"  {'-'*88}")

    strategies = []

    # A: NIFTY only, score>=50
    filt = all_trades["NIFTY"][all_trades["NIFTY"]["score"] >= 50]
    r = evaluate(filt, 75)
    if r:
        p = r["days"] / total_chain_days["NIFTY"] * 100
        strategies.append(("A: NIFTY score>=50", r["win_trade"], p, r["pf"], r["worst_day"]))

    # B: NIFTY+BN combined, score>=45
    daily_b = {}
    wins_b, tot_b = 0, 0
    for sym in ["NIFTY", "BANKNIFTY"]:
        lot = 75 if sym == "NIFTY" else 30
        filt = all_trades[sym][all_trades[sym]["score"] >= 45]
        for _, row in filt.iterrows():
            entry, exit_p = row["entry_prem"], row["exit_prem"]
            sl, tgt = entry * 1.5, entry * 0.4
            if exit_p >= sl: pnl = (entry - sl) * lot
            elif exit_p <= tgt: pnl = (entry - tgt) * lot
            else: pnl = (entry - exit_p) * lot
            if pnl > 0: wins_b += 1
            tot_b += 1
            daily_b[row["date"]] = daily_b.get(row["date"], 0) + pnl
    if tot_b > 0:
        db = list(daily_b.values())
        p_b = len(db) / total_combined_days * 100
        w_b = wins_b / tot_b * 100
        gw_b = sum(v for v in db if v > 0)
        gl_b = abs(sum(v for v in db if v <= 0))
        strategies.append(("B: NIFTY+BN score>=45", w_b, p_b,
                          gw_b / gl_b if gl_b > 0 else 999, min(db)))

    # C: Combined + Tiered (ultimate)
    strategies.append(("C: NIFTY+BN tiered (ULTIMATE)", wt, part, pf, worst))

    # D: NIFTY+BN score>=35 (high participation)
    daily_d = {}
    wins_d, tot_d = 0, 0
    for sym in ["NIFTY", "BANKNIFTY"]:
        lot = 75 if sym == "NIFTY" else 30
        filt = all_trades[sym][all_trades[sym]["score"] >= 35]
        for _, row in filt.iterrows():
            entry, exit_p = row["entry_prem"], row["exit_prem"]
            sl, tgt = entry * 1.5, entry * 0.4
            if exit_p >= sl: pnl = (entry - sl) * lot
            elif exit_p <= tgt: pnl = (entry - tgt) * lot
            else: pnl = (entry - exit_p) * lot
            if pnl > 0: wins_d += 1
            tot_d += 1
            daily_d[row["date"]] = daily_d.get(row["date"], 0) + pnl
    if tot_d > 0:
        dd_vals = list(daily_d.values())
        p_d = len(dd_vals) / total_combined_days * 100
        w_d = wins_d / tot_d * 100
        gw_d = sum(v for v in dd_vals if v > 0)
        gl_d = abs(sum(v for v in dd_vals if v <= 0))
        strategies.append(("D: NIFTY+BN score>=35", w_d, p_d,
                          gw_d / gl_d if gl_d > 0 else 999, min(dd_vals)))

    for name, wtr, part_pct, pf_val, wd_val in strategies:
        wxp_val = wtr * part_pct / 100
        print(f"  {name:<45} {wtr:<10.1f} {part_pct:<8.0f} {wxp_val:<8.1f} "
              f"{pf_val:<7.2f} Rs.{wd_val:,.0f}")


if __name__ == "__main__":
    main()

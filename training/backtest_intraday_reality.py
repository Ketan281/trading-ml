"""
Reality check: test scored wall selling with INTRADAY data.
Uses the 14 days of multi-snapshot chain data (June 2026).
Checks if SL would have been hit mid-day even if EOD looks like a win.
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
    if dist_pct >= 3.0: score += 30
    elif dist_pct >= 2.5: score += 25
    elif dist_pct >= 2.0: score += 20
    elif dist_pct >= 1.5: score += 15
    elif dist_pct >= 1.0: score += 10
    if entry_prem < 5: score += 25
    elif entry_prem < 10: score += 20
    elif entry_prem < 25: score += 15
    elif entry_prem < 50: score += 5
    if dow == 2: score += 20
    elif dow == 1: score += 10
    elif dow == 0: score += 5
    elif dow == 3: score -= 15
    if wall_type == "put": score += 10
    else: score += 5
    if oi_building: score += 5
    return score


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


def main():
    for sym in ["NIFTY", "BANKNIFTY"]:
        step = STRIKE_STEP[sym]
        lot = 75 if sym == "NIFTY" else 30
        thresh = 50 if sym == "NIFTY" else 46

        files = sorted(glob.glob(os.path.join(RAW_DIR, sym, "*.csv")))

        # Find files with intraday data
        intraday_files = []
        for f in files:
            df = pd.read_csv(f)
            if df["timestamp"].nunique() > 1:
                intraday_files.append(f)

        print(f"\n{'='*80}")
        print(f"  INTRADAY REALITY CHECK: {sym}")
        print(f"  {len(intraday_files)} days with intraday snapshots")
        print(f"{'='*80}")

        if len(intraday_files) < 2:
            print("  Not enough intraday data")
            continue

        # For each pair of consecutive intraday days
        trades = []
        for i in range(len(intraday_files) - 1):
            today_file = intraday_files[i]
            next_file = intraday_files[i + 1]
            date_str = os.path.basename(today_file).replace(".csv", "")
            next_date = os.path.basename(next_file).replace(".csv", "")

            df_today_full = pd.read_csv(today_file)
            df_next_full = pd.read_csv(next_file)

            for col in ["ce_ltp", "pe_ltp", "ce_oi", "pe_oi", "ce_chg_oi", "pe_chg_oi"]:
                if col in df_today_full.columns:
                    df_today_full[col] = pd.to_numeric(df_today_full[col], errors="coerce").fillna(0)
                if col in df_next_full.columns:
                    df_next_full[col] = pd.to_numeric(df_next_full[col], errors="coerce").fillna(0)

            # Use last snapshot of today for entry signal
            timestamps_today = sorted(df_today_full["timestamp"].unique())
            last_ts = timestamps_today[-1]
            df_entry = df_today_full[df_today_full["timestamp"] == last_ts]

            # Get spot from ATM (mid of CE+PE ATM LTP)
            # Use the strike with highest combined OI as proxy for ATM
            df_entry_copy = df_entry.copy()
            df_entry_copy["total_oi"] = df_entry_copy["ce_oi"] + df_entry_copy["pe_oi"]
            if len(df_entry_copy) == 0:
                continue
            atm_row = df_entry_copy.sort_values("total_oi", ascending=False).iloc[0]
            spot = float(atm_row["strike"])

            dow = pd.Timestamp(date_str).dayofweek
            walls = find_walls(df_entry, spot)

            # All intraday snapshots of NEXT day (for SL check)
            timestamps_next = sorted(df_next_full["timestamp"].unique())

            for wall in walls:
                strike = wall["strike"]
                dist_pct = abs(strike - spot) / spot * 100
                entry_prem = wall["ltp"]
                if entry_prem <= 0 or dist_pct < 0.5:
                    continue

                sc = score_trade(dist_pct, entry_prem, dow, wall["type"], wall["chg_oi"] > 0)
                if sc < thresh:
                    continue

                # SL and TGT levels
                sl_level = entry_prem * 1.5
                tgt_level = entry_prem * 0.4

                # Track premium through ALL intraday snapshots of next day
                intraday_premiums = []
                sl_hit_intraday = False
                tgt_hit_intraday = False
                sl_hit_time = None
                tgt_hit_time = None

                for ts in timestamps_next:
                    snap = df_next_full[df_next_full["timestamp"] == ts]
                    strike_row = snap[snap["strike"] == strike]
                    if len(strike_row) == 0:
                        continue

                    if wall["type"] == "put":
                        prem = float(strike_row.iloc[0]["pe_ltp"])
                    else:
                        prem = float(strike_row.iloc[0]["ce_ltp"])

                    intraday_premiums.append({"time": ts, "prem": prem})

                    if prem >= sl_level and not sl_hit_intraday:
                        sl_hit_intraday = True
                        sl_hit_time = ts
                    if prem <= tgt_level and not tgt_hit_intraday:
                        tgt_hit_intraday = True
                        tgt_hit_time = ts

                if not intraday_premiums:
                    continue

                eod_prem = intraday_premiums[-1]["prem"]
                max_prem = max(p["prem"] for p in intraday_premiums)
                min_prem = min(p["prem"] for p in intraday_premiums)

                # EOD P&L (what our backtest measured)
                if eod_prem >= sl_level:
                    eod_pnl = entry_prem - sl_level
                elif eod_prem <= tgt_level:
                    eod_pnl = entry_prem - tgt_level
                else:
                    eod_pnl = entry_prem - eod_prem

                # Intraday P&L (what would really happen)
                if sl_hit_intraday:
                    intraday_pnl = entry_prem - sl_level  # stopped out
                elif tgt_hit_intraday:
                    intraday_pnl = entry_prem - tgt_level  # target hit
                else:
                    intraday_pnl = entry_prem - eod_prem  # held to close

                eod_win = eod_pnl > 0
                intraday_win = intraday_pnl > 0

                trades.append({
                    "date": date_str,
                    "type": wall["type"],
                    "strike": strike,
                    "dist_pct": round(dist_pct, 2),
                    "entry_prem": round(entry_prem, 2),
                    "eod_exit": round(eod_prem, 2),
                    "max_prem_intraday": round(max_prem, 2),
                    "min_prem_intraday": round(min_prem, 2),
                    "sl_level": round(sl_level, 2),
                    "sl_hit_intraday": sl_hit_intraday,
                    "sl_hit_time": sl_hit_time,
                    "tgt_hit_intraday": tgt_hit_intraday,
                    "eod_pnl": round(eod_pnl, 2),
                    "intraday_pnl": round(intraday_pnl, 2),
                    "eod_win": eod_win,
                    "intraday_win": intraday_win,
                    "score": sc,
                    "ghost_sl": sl_hit_intraday and eod_win,  # would have been stopped out but EOD shows win
                })

        if not trades:
            print("  No scored trades found in intraday period")
            continue

        df_t = pd.DataFrame(trades)
        total = len(df_t)
        eod_wins = df_t["eod_win"].sum()
        intraday_wins = df_t["intraday_win"].sum()
        ghost_sls = df_t["ghost_sl"].sum()

        print(f"\n  Total scored trades    : {total}")
        print(f"  EOD win %              : {eod_wins/total*100:.1f}% ({eod_wins}/{total})")
        print(f"  INTRADAY win %         : {intraday_wins/total*100:.1f}% ({intraday_wins}/{total})")
        print(f"  Ghost SL hits          : {ghost_sls} (EOD shows win but SL hit mid-day)")
        print(f"  Win % drop             : {(eod_wins-intraday_wins)/total*100:.1f}%")

        # EOD vs Intraday P&L
        eod_total = df_t["eod_pnl"].sum() * lot
        intra_total = df_t["intraday_pnl"].sum() * lot
        print(f"\n  EOD total P&L (1 lot)  : Rs.{eod_total:,.0f}")
        print(f"  Intraday P&L (1 lot)   : Rs.{intra_total:,.0f}")
        print(f"  Difference             : Rs.{intra_total - eod_total:,.0f}")

        # Detail each trade
        print(f"\n  Trade-by-trade detail:")
        print(f"  {'Date':<12} {'Type':<6} {'Strike':<8} {'Dist%':<7} {'Entry':<8} "
              f"{'MaxIntra':<10} {'EOD':<8} {'SL':<8} {'SLhit?':<7} {'EODwin':<7} {'RealWin':<7} {'Ghost':<6}")
        print(f"  {'-'*97}")
        for _, r in df_t.iterrows():
            print(f"  {r['date']:<12} {r['type']:<6} {r['strike']:<8} {r['dist_pct']:<7} "
                  f"{r['entry_prem']:<8} {r['max_prem_intraday']:<10} {r['eod_exit']:<8} "
                  f"{r['sl_level']:<8} {'YES' if r['sl_hit_intraday'] else 'no':<7} "
                  f"{'WIN' if r['eod_win'] else 'LOSS':<7} "
                  f"{'WIN' if r['intraday_win'] else 'LOSS':<7} "
                  f"{'GHOST' if r['ghost_sl'] else '':<6}")


if __name__ == "__main__":
    main()

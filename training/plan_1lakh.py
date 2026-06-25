"""
Realistic plan for Rs.1 lakh capital using scored wall selling.
Tests: naked selling (1 lot), credit spreads (lower margin), scaling plan.
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
        puts = puts.sort_values("pe_oi", ascending=False)
        w = puts.iloc[0]
        walls.append({"type": "put", "strike": int(w["strike"]),
                       "oi": float(w["pe_oi"]), "chg_oi": float(w.get("pe_chg_oi", 0)),
                       "ltp": float(w["pe_ltp"])})
    calls = chain_df[chain_df["strike"] > spot].copy()
    if len(calls) > 0:
        calls = calls.sort_values("ce_oi", ascending=False)
        w = calls.iloc[0]
        walls.append({"type": "call", "strike": int(w["strike"]),
                       "oi": float(w["ce_oi"]), "chg_oi": float(w.get("ce_chg_oi", 0)),
                       "ltp": float(w["ce_ltp"])})
    return walls


def load_all(symbol, score_thresh):
    files = sorted(glob.glob(os.path.join(RAW_DIR, symbol, "*.csv")))
    chain_data = {}
    for f in files:
        date_str = os.path.basename(f).replace(".csv", "")
        try:
            df = pd.read_csv(f)
            if df.empty: continue
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
    step = STRIKE_STEP[symbol]

    trades = []
    for i in range(len(dates) - 1):
        d, nd = dates[i], dates[i+1]
        td, ntd = pd.Timestamp(d), pd.Timestamp(nd)
        if td not in prices.index or ntd not in prices.index: continue
        spot = float(prices.loc[td, "Close"])
        next_spot = float(prices.loc[ntd, "Close"])
        dow = td.dayofweek

        for wall in find_walls(chain_data[d], spot):
            strike = wall["strike"]
            dist_pct = abs(strike - spot) / spot * 100
            entry_prem = wall["ltp"]
            if entry_prem <= 0 or dist_pct < 0.5: continue

            sc = score_trade(dist_pct, entry_prem, dow, wall["type"], wall["chg_oi"] > 0)
            if sc < score_thresh: continue

            # Find hedge strike (for spread) - next OTM strike
            if wall["type"] == "put":
                hedge_strike = strike - step
                today_rows = chain_data[d][chain_data[d]["strike"] == hedge_strike]
                next_wall = chain_data[nd][chain_data[nd]["strike"] == strike]
                next_hedge = chain_data[nd][chain_data[nd]["strike"] == hedge_strike]
                hedge_prem = float(today_rows.iloc[0]["pe_ltp"]) if len(today_rows) > 0 else 0
                exit_prem = float(next_wall.iloc[0]["pe_ltp"]) if len(next_wall) > 0 else 0
                exit_hedge = float(next_hedge.iloc[0]["pe_ltp"]) if len(next_hedge) > 0 else 0
            else:
                hedge_strike = strike + step
                today_rows = chain_data[d][chain_data[d]["strike"] == hedge_strike]
                next_wall = chain_data[nd][chain_data[nd]["strike"] == strike]
                next_hedge = chain_data[nd][chain_data[nd]["strike"] == hedge_strike]
                hedge_prem = float(today_rows.iloc[0]["ce_ltp"]) if len(today_rows) > 0 else 0
                exit_prem = float(next_wall.iloc[0]["ce_ltp"]) if len(next_wall) > 0 else 0
                exit_hedge = float(next_hedge.iloc[0]["ce_ltp"]) if len(next_hedge) > 0 else 0

            if exit_prem == 0: continue

            wall_held = 1 if (
                (wall["type"] == "put" and next_spot >= strike) or
                (wall["type"] == "call" and next_spot <= strike)
            ) else 0

            trades.append({
                "date": d, "type": wall["type"], "strike": strike,
                "spot": spot, "next_spot": next_spot, "dist_pct": dist_pct,
                "entry_prem": entry_prem, "exit_prem": exit_prem,
                "hedge_prem": hedge_prem, "exit_hedge": exit_hedge,
                "hedge_strike": hedge_strike,
                "wall_held": wall_held, "score": sc,
                "year": td.year, "month": td.month,
            })

    return pd.DataFrame(trades)


def main():
    print("=" * 70)
    print("  Rs.1 LAKH CAPITAL PLAN")
    print("  What can you realistically earn?")
    print("=" * 70)

    for sym in ["NIFTY", "BANKNIFTY"]:
        thresh = 50 if sym == "NIFTY" else 46
        lot = 75 if sym == "NIFTY" else 30
        step = STRIKE_STEP[sym]

        df = load_all(sym, thresh)
        if df is None or len(df) == 0:
            continue

        print(f"\n{'='*70}")
        print(f"  {sym} — {len(df)} scored trades")
        print(f"{'='*70}")

        # ── APPROACH 1: Naked selling (1 lot) ──
        print(f"\n  APPROACH 1: NAKED SELLING (1 lot)")
        print(f"  Margin needed: ~Rs.{100000 if sym == 'NIFTY' else 100000:,}")
        daily_naked = {}
        for _, r in df.iterrows():
            entry, exit_p = r["entry_prem"], r["exit_prem"]
            sl, tgt = entry * 1.5, entry * 0.4
            if exit_p >= sl: pnl = (entry - sl) * lot
            elif exit_p <= tgt: pnl = (entry - tgt) * lot
            else: pnl = (entry - exit_p) * lot
            daily_naked[r["date"]] = daily_naked.get(r["date"], 0) + pnl

        vals = list(daily_naked.values())
        avg = np.mean(vals)
        worst = min(vals)
        wd = sum(1 for v in vals if v > 0) / len(vals) * 100
        gw = sum(v for v in vals if v > 0)
        gl = abs(sum(v for v in vals if v <= 0))
        pf = gw / gl if gl > 0 else 999
        monthly = avg * 22
        yearly = avg * 250

        print(f"    Trading days  : {len(vals)}")
        print(f"    Win days      : {wd:.0f}%")
        print(f"    Avg day       : Rs.{avg:.0f}")
        print(f"    Worst day     : Rs.{worst:.0f}")
        print(f"    PF            : {pf:.2f}")
        print(f"    Monthly       : Rs.{monthly:,.0f}")
        print(f"    Yearly        : Rs.{yearly:,.0f}")
        print(f"    Monthly ROI   : {monthly/100000*100:.1f}%")
        print(f"    Yearly ROI    : {yearly/100000*100:.1f}%")

        # ── APPROACH 2: Credit spread (sell wall + buy next OTM) ──
        print(f"\n  APPROACH 2: CREDIT SPREAD (sell wall + buy {step} OTM)")
        spread_margin = step * lot * 0.3  # approximate spread margin
        max_spreads = int(100000 / spread_margin) if spread_margin > 0 else 1
        max_spreads = max(1, min(max_spreads, 5))

        daily_spread = {}
        spread_wins = 0
        spread_total = 0
        for _, r in df.iterrows():
            net_credit = r["entry_prem"] - r["hedge_prem"]
            if net_credit <= 0: continue
            net_exit = r["exit_prem"] - r["exit_hedge"]

            # Spread P&L = credit - debit to close
            pnl_per_unit = net_credit - net_exit
            # Cap loss at spread width minus credit
            max_loss = step - net_credit
            if pnl_per_unit < -max_loss:
                pnl_per_unit = -max_loss

            pnl = pnl_per_unit * lot * max_spreads
            spread_total += 1
            if pnl > 0: spread_wins += 1
            daily_spread[r["date"]] = daily_spread.get(r["date"], 0) + pnl

        if daily_spread:
            vals_s = list(daily_spread.values())
            avg_s = np.mean(vals_s)
            worst_s = min(vals_s)
            wd_s = sum(1 for v in vals_s if v > 0) / len(vals_s) * 100
            gw_s = sum(v for v in vals_s if v > 0)
            gl_s = abs(sum(v for v in vals_s if v <= 0))
            pf_s = gw_s / gl_s if gl_s > 0 else 999
            monthly_s = avg_s * 22
            yearly_s = avg_s * 250
            wr_s = spread_wins / spread_total * 100 if spread_total > 0 else 0

            print(f"    Margin/spread : Rs.{spread_margin:,.0f}")
            print(f"    Spreads       : {max_spreads} (with Rs.1L)")
            print(f"    Trades        : {spread_total}")
            print(f"    Win % trades  : {wr_s:.1f}%")
            print(f"    Win days      : {wd_s:.0f}%")
            print(f"    Avg day       : Rs.{avg_s:.0f}")
            print(f"    Worst day     : Rs.{worst_s:.0f}")
            print(f"    PF            : {pf_s:.2f}")
            print(f"    Monthly       : Rs.{monthly_s:,.0f}")
            print(f"    Yearly        : Rs.{yearly_s:,.0f}")
            print(f"    Monthly ROI   : {monthly_s/100000*100:.1f}%")
            print(f"    Yearly ROI    : {yearly_s/100000*100:.1f}%")

    # ── COMPOUNDING PLAN ──
    print(f"\n{'='*70}")
    print(f"  COMPOUNDING PLAN: Rs.1 LAKH to Rs.X in 12 MONTHS")
    print(f"  (NIFTY naked selling, add 1 lot per Rs.1.2L capital)")
    print(f"{'='*70}")

    # Use NIFTY avg daily per lot from backtest
    avg_per_lot = 314  # from backtest
    capital = 100000
    lots = 1
    margin_per_lot = 120000

    print(f"\n  {'Month':<8} {'Capital':<14} {'Lots':<7} {'Monthly':<12} {'Cumulative':<14}")
    print(f"  {'-'*55}")

    cumulative = 0
    for m in range(1, 13):
        # Trading days per month ~22
        monthly_pnl = avg_per_lot * lots * 22 * 0.7  # 70% participation
        capital += monthly_pnl
        cumulative += monthly_pnl

        # Can we add lots?
        new_lots = int(capital / margin_per_lot)
        new_lots = max(lots, new_lots)  # never reduce
        lots = new_lots

        print(f"  {m:<8} Rs.{capital:>10,.0f} {lots:<7} Rs.{monthly_pnl:>9,.0f} Rs.{cumulative:>11,.0f}")

    print(f"\n  Started with    : Rs.1,00,000")
    print(f"  After 12 months : Rs.{capital:,.0f}")
    print(f"  Total earned    : Rs.{cumulative:,.0f}")
    print(f"  Return          : {cumulative/100000*100:.0f}%")
    print(f"  Final lots      : {lots}")
    print(f"  Final monthly   : Rs.{avg_per_lot * lots * 22 * 0.7:,.0f}")


if __name__ == "__main__":
    main()

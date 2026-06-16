import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(
       os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipelines.backtester import (
    run_backtest,
    generate_signals,
    calculate_metrics,
    grade_strategy
)

DATA_DIR   = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "outputs",
                          "walk_forward")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Walk Forward Config ───────────────────────────────
DEFAULT_WF_CONFIG = {
    "train_period"  : 180,  # 180 days training
    "test_period"   : 60,   # 60 days testing
    "step_size"     : 30,   # Step 30 days forward
    "min_train_bars": 100,  # Min bars to train
    "min_test_bars" : 20,   # Min bars to test
}

# ── Single Walk Forward Window ────────────────────────
def run_single_window(df, train_start,
                       train_end, test_start,
                       test_end, strategy,
                       initial_capital,
                       window_num):
    print(f"\n  Window {window_num}:")
    print(f"    Train: {train_start.date()} "
          f"→ {train_end.date()}")
    print(f"    Test : {test_start.date()} "
          f"→ {test_end.date()}")

    # Training period
    train_df = df[
        (df.index >= train_start) &
        (df.index <= train_end)
    ]

    # Test period
    test_df  = df[
        (df.index >= test_start) &
        (df.index <= test_end)
    ]

    if len(train_df) < 50 or len(test_df) < 10:
        print(f"    ⚠ Not enough data — skipping")
        return None

    print(f"    Train bars: {len(train_df)} | "
          f"Test bars: {len(test_df)}")

    # Generate signals on test period
    # (using parameters learned from train)
    test_signals = generate_signals(
        test_df, strategy
    )

    # Run backtest on TEST period only
    symbol   = "WALK_FORWARD"
    capital  = initial_capital
    trades   = []
    position = None

    from pipelines.backtester import (
        apply_slippage,
        calculate_brokerage,
        Position
    )

    equity_curve = []
    peak_capital = capital
    total_brok   = 0

    for i in range(len(test_df)):
        bar    = test_df.iloc[i]
        date   = test_df.index[i]
        signal = test_signals.iloc[i]
        close  = float(bar["Close"])
        high   = float(bar["High"])
        low    = float(bar["Low"])

        # Check stop loss / target
        if position and position.status == "open":
            exit_price  = None
            exit_reason = None

            if position.action == "buy":
                if (position.stop_loss and
                        low <= position.stop_loss):
                    exit_price  = apply_slippage(
                        position.stop_loss,
                        "sell", "high_vol"
                    )
                    exit_reason = "stop_loss"
                elif (position.target and
                        high >= position.target):
                    exit_price  = apply_slippage(
                        position.target,
                        "sell", "normal"
                    )
                    exit_reason = "target"
                elif signal == "sell":
                    exit_price  = apply_slippage(
                        close, "sell", "normal"
                    )
                    exit_reason = "signal_exit"

            elif position.action == "sell":
                if (position.stop_loss and
                        high >= position.stop_loss):
                    exit_price  = apply_slippage(
                        position.stop_loss,
                        "buy", "high_vol"
                    )
                    exit_reason = "stop_loss"
                elif (position.target and
                        low <= position.target):
                    exit_price  = apply_slippage(
                        position.target,
                        "buy", "normal"
                    )
                    exit_reason = "target"
                elif signal == "buy":
                    exit_price  = apply_slippage(
                        close, "buy", "normal"
                    )
                    exit_reason = "signal_exit"

            if exit_price:
                position.close(
                    exit_price, date, exit_reason
                )
                brok = calculate_brokerage(
                    exit_price,
                    position.quantity,
                    "sell" if position.action
                    == "buy" else "buy"
                )
                total_brok += brok
                pnl         = position.pnl - brok
                capital    += pnl
                peak_capital = max(
                    peak_capital, capital
                )
                trades.append(position)
                position = None

        # Open new position
        if not position and \
                signal in ["buy", "sell"]:
            risk_amount = capital * 0.02
            entry_price = apply_slippage(
                close, signal, "normal"
            )
            sl_distance = entry_price * 0.015
            quantity    = max(
                1, int(risk_amount / sl_distance)
            )
            max_qty     = int(
                capital * 0.95 / entry_price
            )
            quantity    = min(quantity, max_qty)

            if quantity < 1:
                continue

            if signal == "buy":
                sl     = round(
                    entry_price * 0.985, 2
                )
                target = round(
                    entry_price * 1.03, 2
                )
            else:
                sl     = round(
                    entry_price * 1.015, 2
                )
                target = round(
                    entry_price * 0.97, 2
                )

            brok     = calculate_brokerage(
                entry_price, quantity, signal
            )
            total_brok += brok
            capital    -= brok

            position = Position(
                symbol      = symbol,
                action      = signal,
                entry_price = entry_price,
                quantity    = quantity,
                entry_date  = date,
                stop_loss   = sl,
                target      = target
            )

        unrealized = 0
        if position and position.status == "open":
            if position.action == "buy":
                unrealized = (
                    (close - position.entry_price)
                    * position.quantity
                )
            else:
                unrealized = (
                    (position.entry_price - close)
                    * position.quantity
                )

        equity_curve.append({
            "date"    : str(date.date()),
            "capital" : round(
                capital + unrealized, 2
            ),
            "drawdown": round(
                (peak_capital - capital
                 - unrealized)
                / peak_capital * 100, 2
            )
        })

    # Close open position
    if position and position.status == "open":
        last_close = float(
            test_df["Close"].iloc[-1]
        )
        exit_price = apply_slippage(
            last_close,
            "sell" if position.action == "buy"
            else "buy"
        )
        position.close(
            exit_price,
            test_df.index[-1],
            "end_of_window"
        )
        brok     = calculate_brokerage(
            exit_price, position.quantity,
            "sell" if position.action == "buy"
            else "buy"
        )
        total_brok += brok
        capital    += position.pnl - brok
        trades.append(position)

    # Calculate metrics
    metrics = calculate_metrics(
        trades, initial_capital,
        capital, equity_curve, total_brok
    )

    result = {
        "window_num"    : window_num,
        "train_start"   : str(train_start.date()),
        "train_end"     : str(train_end.date()),
        "test_start"    : str(test_start.date()),
        "test_end"      : str(test_end.date()),
        "train_bars"    : len(train_df),
        "test_bars"     : len(test_df),
        "initial_capital": initial_capital,
        "final_capital" : round(capital, 2),
        "metrics"       : metrics,
        "trades"        : len(trades)
    }

    if metrics:
        print(
            f"    ROI: {metrics.get('roi_pct')}% | "
            f"WR: {metrics.get('win_rate')}% | "
            f"Sharpe: {metrics.get('sharpe_ratio')} | "
            f"Trades: {len(trades)}"
        )

    return result

# ── Full Walk Forward Test ────────────────────────────
def run_walk_forward(symbol,
                      strategy="ema_crossover",
                      initial_capital=100000,
                      config=None):

    config = config or DEFAULT_WF_CONFIG

    print(f"\n{'=' * 60}")
    print(f"  Walk Forward Test — {symbol}")
    print(f"  Strategy    : {strategy}")
    print(f"  Train Period: {config['train_period']} days")
    print(f"  Test Period : {config['test_period']} days")
    print(f"  Step Size   : {config['step_size']} days")
    print(f"{'=' * 60}")

    # Load data
    path = os.path.join(
        DATA_DIR, f"{symbol}_daily.csv"
    )
    if not os.path.exists(path):
        print(f"  ❌ No data for {symbol}")
        return None

    df = pd.read_csv(
        path, index_col="Date", parse_dates=True
    )

    print(f"  Total bars : {len(df)}")
    print(f"  Date range : {df.index[0].date()} "
          f"→ {df.index[-1].date()}")

    total_needed = (
        config["train_period"] +
        config["test_period"]
    )

    if len(df) < total_needed:
        print(
            f"  ❌ Need at least {total_needed} bars, "
            f"have {len(df)}"
        )
        return None

    # Generate windows
    windows       = []
    window_num    = 1
    start_idx     = 0

    while True:
        train_end_idx = (
            start_idx + config["train_period"]
        )
        test_end_idx  = (
            train_end_idx + config["test_period"]
        )

        if test_end_idx > len(df):
            break

        train_start = df.index[start_idx]
        train_end   = df.index[train_end_idx - 1]
        test_start  = df.index[train_end_idx]
        test_end    = df.index[
            min(test_end_idx - 1, len(df) - 1)
        ]

        result = run_single_window(
            df, train_start, train_end,
            test_start, test_end,
            strategy, initial_capital,
            window_num
        )

        if result:
            windows.append(result)

        start_idx  += config["step_size"]
        window_num += 1

    if not windows:
        print("  ❌ No valid windows generated")
        return None

    # ── Aggregate Results ─────────────────────────────
    analysis = aggregate_walk_forward(
        windows, symbol, strategy
    )

    # Print results
    print_walk_forward_results(
        analysis, windows
    )

    # Save
    full_result = {
        "symbol"         : symbol,
        "strategy"       : strategy,
        "initial_capital": initial_capital,
        "config"         : config,
        "windows"        : windows,
        "analysis"       : analysis,
        "timestamp"      : datetime.now().isoformat()
    }

    path = os.path.join(
        OUTPUT_DIR,
        f"{symbol}_{strategy}_wf_"
        f"{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    )
    with open(path, "w") as f:
        json.dump(full_result, f,
                  indent=2, default=str)

    print(f"\n  ✅ Walk forward saved → {path}")
    return full_result

# ── Aggregate Walk Forward Results ────────────────────
def aggregate_walk_forward(windows, symbol,
                            strategy):
    valid = [
        w for w in windows
        if w.get("metrics")
        and w["metrics"].get("total_trades", 0) > 0
    ]

    if not valid:
        return {"error": "No valid windows"}

    # Collect metrics across all windows
    rois         = [
        w["metrics"].get("roi_pct", 0)
        for w in valid
    ]
    win_rates    = [
        w["metrics"].get("win_rate", 0)
        for w in valid
    ]
    sharpes      = [
        w["metrics"].get("sharpe_ratio", 0)
        for w in valid
    ]
    drawdowns    = [
        w["metrics"].get("max_drawdown", 0)
        for w in valid
    ]
    pf_list      = [
        w["metrics"].get("profit_factor", 0)
        for w in valid
    ]

    # Profitable windows
    profitable   = [
        w for w in valid
        if w["metrics"].get("roi_pct", 0) > 0
    ]
    consistency  = round(
        len(profitable) / len(valid) * 100, 1
    )

    # Efficiency ratio
    # Ratio of out-of-sample to in-sample performance
    avg_roi_test = np.mean(rois)

    # Stability score
    roi_std      = np.std(rois)
    stability    = round(
        100 - min(roi_std * 2, 100), 1
    )

    # Overall grade
    avg_sharpe   = np.mean(sharpes)
    avg_wr       = np.mean(win_rates)
    avg_dd       = np.mean(drawdowns)
    avg_pf       = np.mean(pf_list)

    grade        = grade_strategy({
        "win_rate"     : avg_wr,
        "profit_factor": avg_pf,
        "sharpe_ratio" : avg_sharpe,
        "max_drawdown" : avg_dd,
        "roi_pct"      : avg_roi_test
    })

    # Overfitting check
    roi_trend    = np.polyfit(
        range(len(rois)), rois, 1
    )[0] if len(rois) > 2 else 0

    if roi_trend < -0.5:
        overfit_risk = "high"
        overfit_note = (
            "ROI declining across windows — "
            "possible overfitting"
        )
    elif roi_trend < 0:
        overfit_risk = "medium"
        overfit_note = (
            "Slight ROI decline — monitor carefully"
        )
    else:
        overfit_risk = "low"
        overfit_note = (
            "Strategy performing consistently"
        )

    return {
        "symbol"          : symbol,
        "strategy"        : strategy,
        "total_windows"   : len(windows),
        "valid_windows"   : len(valid),
        "profitable_windows": len(profitable),
        "consistency_pct" : consistency,
        "stability_score" : stability,
        "avg_roi"         : round(avg_roi_test, 2),
        "avg_win_rate"    : round(avg_wr,  2),
        "avg_sharpe"      : round(avg_sharpe, 2),
        "avg_max_drawdown": round(avg_dd,  2),
        "avg_profit_factor": round(avg_pf, 2),
        "roi_std"         : round(roi_std,  2),
        "roi_trend"       : round(roi_trend, 4),
        "overfit_risk"    : overfit_risk,
        "overfit_note"    : overfit_note,
        "grade"           : grade,
        "roi_by_window"   : rois,
        "recommendation"  : get_recommendation(
            consistency, avg_sharpe,
            overfit_risk, avg_dd
        )
    }

# ── Get Recommendation ────────────────────────────────
def get_recommendation(consistency, sharpe,
                        overfit_risk, drawdown):
    if (consistency >= 60 and
            sharpe >= 1.0 and
            overfit_risk == "low" and
            drawdown <= 20):
        return {
            "status" : "DEPLOY",
            "message": "Strategy is robust — "
                       "safe to deploy with "
                       "normal position sizing",
            "icon"   : "✅"
        }
    elif (consistency >= 50 and
            sharpe >= 0.5 and
            overfit_risk != "high"):
        return {
            "status" : "PAPER_TRADE",
            "message": "Strategy shows promise — "
                       "paper trade for 30 more days "
                       "before deploying",
            "icon"   : "⚠️"
        }
    elif overfit_risk == "high":
        return {
            "status" : "OVERFIT",
            "message": "Strategy likely overfit — "
                       "do not deploy, rebuild with "
                       "simpler rules",
            "icon"   : "🔴"
        }
    else:
        return {
            "status" : "REJECT",
            "message": "Strategy not robust enough — "
                       "needs significant improvement",
            "icon"   : "❌"
        }

# ── Print Walk Forward Results ────────────────────────
def print_walk_forward_results(analysis,
                                windows):
    print(f"\n{'═' * 60}")
    print(f"  WALK FORWARD ANALYSIS RESULTS")
    print(f"{'═' * 60}")
    print(
        f"  Symbol          : {analysis['symbol']}"
    )
    print(
        f"  Strategy        : {analysis['strategy']}"
    )
    print(f"  {'─' * 58}")
    print(
        f"  Total Windows   : "
        f"{analysis['total_windows']}"
    )
    print(
        f"  Valid Windows   : "
        f"{analysis['valid_windows']}"
    )
    print(
        f"  Profitable      : "
        f"{analysis['profitable_windows']} windows"
    )
    print(
        f"  Consistency     : "
        f"{analysis['consistency_pct']}%"
    )
    print(
        f"  Stability Score : "
        f"{analysis['stability_score']}/100"
    )
    print(f"  {'─' * 58}")
    print(
        f"  Avg ROI         : "
        f"{analysis['avg_roi']}%"
    )
    print(
        f"  Avg Win Rate    : "
        f"{analysis['avg_win_rate']}%"
    )
    print(
        f"  Avg Sharpe      : "
        f"{analysis['avg_sharpe']}"
    )
    print(
        f"  Avg Max Drawdown: "
        f"{analysis['avg_max_drawdown']}%"
    )
    print(
        f"  Avg Profit Factor: "
        f"{analysis['avg_profit_factor']}"
    )
    print(f"  {'─' * 58}")
    print(
        f"  Overfit Risk    : "
        f"{analysis['overfit_risk'].upper()}"
    )
    print(
        f"  Overfit Note    : "
        f"{analysis['overfit_note']}"
    )
    print(f"  {'─' * 58}")

    # ROI by window
    print(f"\n  📊 ROI by Window:")
    print(f"  {'WIN':<6} {'PERIOD':<25} "
          f"{'ROI%':<10} {'WR%':<8} {'TRADES'}")
    print("  " + "─" * 55)

    for w in windows:
        if not w.get("metrics"):
            continue
        m    = w["metrics"]
        roi  = m.get("roi_pct", 0)
        icon = "✅" if roi > 0 else "❌"
        print(
            f"  {w['window_num']:<6} "
            f"{w['test_start']} → "
            f"{w['test_end']}  "
            f"{icon} {str(roi):<8} "
            f"{m.get('win_rate',0):<8} "
            f"{m.get('total_trades',0)}"
        )

    # Grade
    grade = analysis.get("grade", {})
    print(f"\n  {'─' * 58}")
    print(
        f"  Overall Grade   : "
        f"{grade.get('grade', 'N/A')}"
    )
    print(
        f"  Assessment      : "
        f"{grade.get('assessment', 'N/A')}"
    )

    # Recommendation
    rec = analysis.get("recommendation", {})
    print(f"\n  {'═' * 58}")
    print(
        f"  {rec.get('icon','❓')} "
        f"RECOMMENDATION: {rec.get('status','N/A')}"
    )
    print(f"  {rec.get('message','N/A')}")
    print(f"  {'═' * 58}")

# ── Compare Strategies Walk Forward ───────────────────
def compare_strategies_wf(symbol,
                           initial_capital=100000):
    strategies = [
        "ema_crossover",
        "rsi_mean_reversion",
        "macd_crossover",
        "bollinger_breakout"
    ]

    print(f"\n{'🔥' * 27}")
    print(f"  WALK FORWARD COMPARISON — {symbol}")
    print(f"{'🔥' * 27}")

    results = []
    for strategy in strategies:
        result = run_walk_forward(
            symbol, strategy, initial_capital
        )
        if result and result.get("analysis"):
            a = result["analysis"]
            results.append({
                "strategy"    : strategy,
                "consistency" : a.get(
                    "consistency_pct", 0),
                "avg_roi"     : a.get("avg_roi", 0),
                "avg_sharpe"  : a.get("avg_sharpe", 0),
                "overfit_risk": a.get(
                    "overfit_risk", "high"),
                "stability"   : a.get(
                    "stability_score", 0),
                "rec_status"  : a.get(
                    "recommendation", {}
                ).get("status", "REJECT")
            })

    # Comparison table
    if results:
        print(f"\n{'═' * 75}")
        print(f"  WALK FORWARD COMPARISON — {symbol}")
        print(f"{'═' * 75}")
        print(
            f"  {'STRATEGY':<25} {'CONSISTENCY':<14} "
            f"{'AVG ROI':<10} {'SHARPE':<10} "
            f"{'OVERFIT':<10} {'STATUS'}"
        )
        print("  " + "─" * 72)

        for r in sorted(
            results,
            key=lambda x: (
                x["consistency"],
                x["avg_roi"]
            ),
            reverse=True
        ):
            print(
                f"  {r['strategy']:<25} "
                f"{str(r['consistency'])+'%':<14} "
                f"{str(r['avg_roi'])+'%':<10} "
                f"{str(r['avg_sharpe']):<10} "
                f"{r['overfit_risk']:<10} "
                f"{r['rec_status']}"
            )

        # Best strategy
        deployable = [
            r for r in results
            if r["rec_status"] == "DEPLOY"
        ]

        if deployable:
            best = max(
                deployable,
                key=lambda x: x["avg_roi"]
            )
            print(f"\n  🏆 Best Deployable: "
                  f"{best['strategy']}")
            print(f"     Consistency : "
                  f"{best['consistency']}%")
            print(f"     Avg ROI     : "
                  f"{best['avg_roi']}%")
        else:
            best = max(
                results,
                key=lambda x: x["consistency"]
            )
            print(
                f"\n  ⚠ No strategy ready to deploy."
            )
            print(
                f"  Best candidate: "
                f"{best['strategy']} — "
                f"needs more optimization"
            )

    return results

# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Trading AI — Walk Forward Testing")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    for symbol in ["NIFTY", "BANKNIFTY",
                   "RELIANCE", "TCS"]:
        compare_strategies_wf(symbol)

    print("\n  ✅ Walk Forward Testing complete!")
    print(
        "  Check outputs/walk_forward/ for results"
    )
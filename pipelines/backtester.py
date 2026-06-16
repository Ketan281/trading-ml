import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(
       os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA_DIR   = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "outputs",
                          "backtests")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Brokerage Config ──────────────────────────────────
BROKERAGE = {
    "zerodha": {
        "equity_intraday" : 0.0003,  # 0.03%
        "equity_delivery" : 0.0,     # Free
        "futures"         : 0.0003,
        "options_buy"     : 20.0,    # Flat ₹20
        "options_sell"    : 20.0,
        "stt_equity"      : 0.001,
        "stt_futures"     : 0.0001,
        "stt_options"     : 0.0005,
        "exchange_charge" : 0.0000325,
        "gst"             : 0.18,
        "sebi_charge"     : 0.000001,
        "stamp_duty"      : 0.00003
    }
}

# ── Slippage Config ───────────────────────────────────
SLIPPAGE = {
    "low_vol"   : 0.0005,  # 0.05%
    "normal"    : 0.001,   # 0.10%
    "high_vol"  : 0.002,   # 0.20%
    "extreme"   : 0.004    # 0.40%
}

# ── Position Class ────────────────────────────────────
class Position:
    def __init__(self, symbol, action, entry_price,
                 quantity, entry_date, stop_loss=None,
                 target=None, trade_type="intraday"):
        self.symbol      = symbol
        self.action      = action
        self.entry_price = entry_price
        self.quantity    = quantity
        self.entry_date  = entry_date
        self.stop_loss   = stop_loss
        self.target      = target
        self.trade_type  = trade_type
        self.exit_price  = None
        self.exit_date   = None
        self.pnl         = None
        self.exit_reason = None
        self.status      = "open"

    def close(self, exit_price, exit_date,
              exit_reason):
        self.exit_price  = exit_price
        self.exit_date   = exit_date
        self.exit_reason = exit_reason
        self.status      = "closed"

        if self.action == "buy":
            self.pnl = (
                (exit_price - self.entry_price)
                * self.quantity
            )
        else:
            self.pnl = (
                (self.entry_price - exit_price)
                * self.quantity
            )

    def to_dict(self):
        return {
            "symbol"      : self.symbol,
            "action"      : self.action,
            "entry_price" : self.entry_price,
            "exit_price"  : self.exit_price,
            "quantity"    : self.quantity,
            "entry_date"  : str(self.entry_date),
            "exit_date"   : str(self.exit_date),
            "stop_loss"   : self.stop_loss,
            "target"      : self.target,
            "pnl"         : round(self.pnl, 2)
                            if self.pnl else 0,
            "exit_reason" : self.exit_reason,
            "trade_type"  : self.trade_type,
            "status"      : self.status
        }

# ── Brokerage Calculator ──────────────────────────────
def calculate_brokerage(price, quantity,
                         action, instrument="equity"):
    broker   = BROKERAGE["zerodha"]
    value    = price * quantity

    if instrument == "options":
        brokerage = broker["options_buy"] \
                    if action == "buy" \
                    else broker["options_sell"]
        stt       = value * broker["stt_options"] \
                    if action == "sell" else 0
    elif instrument == "futures":
        brokerage = value * broker["futures"]
        stt       = value * broker["stt_futures"]
    else:
        brokerage = value * broker["equity_intraday"]
        stt       = value * broker["stt_equity"]

    exchange = value * broker["exchange_charge"]
    sebi     = value * broker["sebi_charge"]
    stamp    = value * broker["stamp_duty"]
    gst      = (brokerage + exchange) * broker["gst"]

    total    = brokerage + stt + exchange + \
               sebi + stamp + gst

    return round(total, 2)

# ── Slippage Calculator ───────────────────────────────
def apply_slippage(price, action, vol_regime="normal"):
    slip_pct = SLIPPAGE.get(vol_regime,
                             SLIPPAGE["normal"])
    if action == "buy":
        return round(price * (1 + slip_pct), 2)
    else:
        return round(price * (1 - slip_pct), 2)

# ── Signal Generator ──────────────────────────────────
def generate_signals(df, strategy="ema_crossover"):
    signals = pd.Series("hold", index=df.index)
    close   = df["Close"]

    if strategy == "ema_crossover":
        ema9  = close.ewm(span=9).mean()
        ema21 = close.ewm(span=21).mean()

        for i in range(1, len(df)):
            if (ema9.iloc[i]  > ema21.iloc[i] and
                    ema9.iloc[i-1] <= ema21.iloc[i-1]):
                signals.iloc[i] = "buy"
            elif (ema9.iloc[i]  < ema21.iloc[i] and
                    ema9.iloc[i-1] >= ema21.iloc[i-1]):
                signals.iloc[i] = "sell"

    elif strategy == "rsi_mean_reversion":
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = 100 - (100 / (1 + gain / loss))

        for i in range(1, len(df)):
            if rsi.iloc[i] < 30:
                signals.iloc[i] = "buy"
            elif rsi.iloc[i] > 70:
                signals.iloc[i] = "sell"

    elif strategy == "macd_crossover":
        ema12  = close.ewm(span=12).mean()
        ema26  = close.ewm(span=26).mean()
        macd   = ema12 - ema26
        signal = macd.ewm(span=9).mean()

        for i in range(1, len(df)):
            if (macd.iloc[i]   > signal.iloc[i] and
                    macd.iloc[i-1] <= signal.iloc[i-1]):
                signals.iloc[i] = "buy"
            elif (macd.iloc[i]   < signal.iloc[i] and
                    macd.iloc[i-1] >= signal.iloc[i-1]):
                signals.iloc[i] = "sell"

    elif strategy == "bollinger_breakout":
        ma20  = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20

        for i in range(1, len(df)):
            if close.iloc[i] > upper.iloc[i]:
                signals.iloc[i] = "buy"
            elif close.iloc[i] < lower.iloc[i]:
                signals.iloc[i] = "sell"

    return signals

# ── Backtest Engine ───────────────────────────────────
def run_backtest(symbol, strategy="ema_crossover",
                 initial_capital=100000,
                 risk_per_trade=0.02,
                 stop_loss_pct=0.015,
                 target_pct=0.03,
                 start_date=None, end_date=None):

    print(f"\n{'=' * 55}")
    print(f"  Backtester — {symbol} | {strategy}")
    print(f"  Capital: ₹{initial_capital:,} | "
          f"Risk/Trade: {risk_per_trade*100}%")
    print(f"{'=' * 55}")

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

    # Filter date range
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]

    if len(df) < 50:
        print(f"  ❌ Not enough data ({len(df)} rows)")
        return None

    print(f"  Data range : {df.index[0].date()} "
          f"→ {df.index[-1].date()}")
    print(f"  Total bars : {len(df)}")

    # Generate signals
    signals  = generate_signals(df, strategy)

    # Portfolio tracking
    capital      = initial_capital
    peak_capital = initial_capital
    trades       = []
    equity_curve = []
    position     = None
    total_brok   = 0

    # Candle by candle replay
    for i in range(len(df)):
        bar        = df.iloc[i]
        date       = df.index[i]
        signal     = signals.iloc[i]
        close      = float(bar["Close"])
        high       = float(bar["High"])
        low        = float(bar["Low"])

        # Check stop loss and target on open position
        if position and position.status == "open":
            exit_price  = None
            exit_reason = None

            if position.action == "buy":
                # Stop loss hit
                if (position.stop_loss and
                        low <= position.stop_loss):
                    exit_price  = apply_slippage(
                        position.stop_loss,
                        "sell", "high_vol"
                    )
                    exit_reason = "stop_loss"
                # Target hit
                elif (position.target and
                        high >= position.target):
                    exit_price  = apply_slippage(
                        position.target,
                        "sell", "normal"
                    )
                    exit_reason = "target"
                # New sell signal
                elif signal == "sell":
                    exit_price  = apply_slippage(
                        close, "sell", "normal"
                    )
                    exit_reason = "signal_exit"

            elif position.action == "sell":
                # Stop loss hit
                if (position.stop_loss and
                        high >= position.stop_loss):
                    exit_price  = apply_slippage(
                        position.stop_loss,
                        "buy", "high_vol"
                    )
                    exit_reason = "stop_loss"
                # Target hit
                elif (position.target and
                        low <= position.target):
                    exit_price  = apply_slippage(
                        position.target,
                        "buy", "normal"
                    )
                    exit_reason = "target"
                # New buy signal
                elif signal == "buy":
                    exit_price  = apply_slippage(
                        close, "buy", "normal"
                    )
                    exit_reason = "signal_exit"

            # Close position
            if exit_price:
                position.close(
                    exit_price, date, exit_reason
                )
                brok = calculate_brokerage(
                    exit_price,
                    position.quantity,
                    "sell" if position.action == "buy"
                    else "buy"
                )
                total_brok += brok
                pnl         = position.pnl - brok
                capital    += pnl
                peak_capital = max(
                    peak_capital, capital
                )
                trades.append(position)
                position = None

        # Open new position on signal
        if not position and signal in ["buy", "sell"]:
            # Position sizing
            risk_amount = capital * risk_per_trade
            entry_price = apply_slippage(
                close, signal, "normal"
            )
            sl_distance = entry_price * stop_loss_pct
            quantity    = max(
                1, int(risk_amount / sl_distance)
            )

            # Cap quantity to affordable
            max_qty = int(
                capital * 0.95 / entry_price
            )
            quantity = min(quantity, max_qty)

            if quantity < 1:
                continue

            # Set SL and target
            if signal == "buy":
                sl     = round(
                    entry_price * (1 - stop_loss_pct),
                    2
                )
                target = round(
                    entry_price * (1 + target_pct), 2
                )
            else:
                sl     = round(
                    entry_price * (1 + stop_loss_pct),
                    2
                )
                target = round(
                    entry_price * (1 - target_pct), 2
                )

            # Entry brokerage
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

        # Track equity
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
            "date"      : str(date.date()),
            "capital"   : round(capital + unrealized, 2),
            "drawdown"  : round(
                (peak_capital - capital - unrealized)
                / peak_capital * 100, 2
            )
        })

    # Close any open position at end
    if position and position.status == "open":
        last_close = float(df["Close"].iloc[-1])
        exit_price = apply_slippage(
            last_close, "sell"
            if position.action == "buy" else "buy"
        )
        position.close(
            exit_price,
            df.index[-1],
            "end_of_backtest"
        )
        brok     = calculate_brokerage(
            exit_price, position.quantity,
            "sell" if position.action == "buy"
            else "buy"
        )
        total_brok += brok
        capital    += position.pnl - brok
        trades.append(position)

    # ── Performance Metrics ───────────────────────────
    metrics = calculate_metrics(
        trades, initial_capital,
        capital, equity_curve, total_brok
    )

    # Print results
    print_backtest_results(
        symbol, strategy, metrics, trades
    )

    # Save results
    result = {
        "symbol"        : symbol,
        "strategy"      : strategy,
        "initial_capital": initial_capital,
        "final_capital" : round(capital, 2),
        "total_brokerage": round(total_brok, 2),
        "metrics"       : metrics,
        "trades"        : [
            t.to_dict() for t in trades
        ],
        "equity_curve"  : equity_curve[-50:]
    }

    path = os.path.join(
        OUTPUT_DIR,
        f"{symbol}_{strategy}_backtest_"
        f"{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    )
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  ✅ Backtest saved → {path}")
    return result

# ── Performance Metrics ───────────────────────────────
def calculate_metrics(trades, initial_capital,
                       final_capital, equity_curve,
                       total_brokerage):
    if not trades:
        return {}

    closed = [t for t in trades
              if t.status == "closed"
              and t.pnl is not None]

    if not closed:
        return {}

    pnls        = [t.pnl for t in closed]
    wins        = [p for p in pnls if p > 0]
    losses      = [p for p in pnls if p < 0]

    total_pnl   = sum(pnls)
    win_rate    = len(wins) / len(pnls) * 100
    avg_win     = sum(wins)   / len(wins)   \
                  if wins   else 0
    avg_loss    = sum(losses) / len(losses) \
                  if losses else 0

    profit_factor = (
        abs(sum(wins)) / abs(sum(losses))
        if losses else 99.0
    )

    # Expectancy
    expectancy  = (
        win_rate / 100 * avg_win +
        (1 - win_rate / 100) * avg_loss
    )

    # Max drawdown
    drawdowns   = [
        e["drawdown"] for e in equity_curve
    ]
    max_dd      = max(drawdowns) if drawdowns else 0

    # Sharpe ratio
    returns     = []
    caps        = [e["capital"] for e in equity_curve]
    for i in range(1, len(caps)):
        ret = (caps[i] - caps[i-1]) / caps[i-1]
        returns.append(ret)

    if returns:
        avg_ret = np.mean(returns)
        std_ret = np.std(returns)
        sharpe  = round(
            avg_ret / std_ret * np.sqrt(252), 2
        ) if std_ret > 0 else 0
    else:
        sharpe  = 0

    # Average RR
    rr_list = []
    for t in closed:
        if (t.stop_loss and t.entry_price and
                t.target):
            risk   = abs(
                t.entry_price - t.stop_loss
            )
            reward = abs(
                t.target - t.entry_price
            )
            if risk > 0:
                rr_list.append(reward / risk)

    avg_rr = round(
        sum(rr_list) / len(rr_list), 2
    ) if rr_list else 0

    # Return on capital
    roi = round(
        (final_capital - initial_capital)
        / initial_capital * 100, 2
    )

    # Exit reasons
    exit_reasons = {}
    for t in closed:
        r = t.exit_reason or "unknown"
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    return {
        "total_trades"   : len(closed),
        "winning_trades" : len(wins),
        "losing_trades"  : len(losses),
        "win_rate"       : round(win_rate,       2),
        "total_pnl"      : round(total_pnl,      2),
        "avg_win"        : round(avg_win,         2),
        "avg_loss"       : round(avg_loss,        2),
        "profit_factor"  : round(profit_factor,   2),
        "expectancy"     : round(expectancy,      2),
        "sharpe_ratio"   : sharpe,
        "max_drawdown"   : round(max_dd,          2),
        "avg_rr"         : avg_rr,
        "roi_pct"        : roi,
        "total_brokerage": round(total_brokerage, 2),
        "exit_reasons"   : exit_reasons
    }

# ── Print Results ─────────────────────────────────────
def print_backtest_results(symbol, strategy,
                            metrics, trades):
    if not metrics:
        print("  ⚠ No metrics available")
        return

    print(f"\n  {'═' * 50}")
    print(f"  📊 BACKTEST RESULTS")
    print(f"  {'═' * 50}")
    print(f"  Symbol       : {symbol}")
    print(f"  Strategy     : {strategy}")
    print(f"  {'─' * 50}")
    print(f"  Total Trades : {metrics['total_trades']}")
    print(f"  Win Rate     : {metrics['win_rate']}%")
    print(f"  Profit Factor: {metrics['profit_factor']}")
    print(f"  Sharpe Ratio : {metrics['sharpe_ratio']}")
    print(f"  Max Drawdown : {metrics['max_drawdown']}%")
    print(f"  {'─' * 50}")
    print(f"  Total PnL    : ₹{metrics['total_pnl']:,.2f}")
    print(f"  Avg Win      : ₹{metrics['avg_win']:,.2f}")
    print(f"  Avg Loss     : ₹{metrics['avg_loss']:,.2f}")
    print(f"  Expectancy   : ₹{metrics['expectancy']:,.2f}")
    print(f"  Avg RR       : {metrics['avg_rr']}")
    print(f"  ROI          : {metrics['roi_pct']}%")
    print(f"  Brokerage    : ₹{metrics['total_brokerage']:,.2f}")
    print(f"  {'─' * 50}")

    # Exit reason breakdown
    print(f"  Exit Reasons:")
    for reason, count in metrics.get(
        "exit_reasons", {}
    ).items():
        pct = round(count / metrics["total_trades"]
                    * 100, 1)
        print(f"     {reason:<20} : "
              f"{count} ({pct}%)")

    # Grade the strategy
    grade = grade_strategy(metrics)
    print(f"\n  {'─' * 50}")
    print(f"  Strategy Grade : {grade['grade']}")
    print(f"  Assessment     : {grade['assessment']}")
    print(f"  {'═' * 50}")

# ── Strategy Grader ───────────────────────────────────
def grade_strategy(metrics):
    score = 0

    # Win rate scoring
    wr = metrics.get("win_rate", 0)
    if wr > 60:   score += 25
    elif wr > 50: score += 15
    elif wr > 40: score += 5

    # Profit factor
    pf = metrics.get("profit_factor", 0)
    if pf > 2.0:  score += 25
    elif pf > 1.5: score += 15
    elif pf > 1.0: score += 5

    # Sharpe ratio
    sr = metrics.get("sharpe_ratio", 0)
    if sr > 2.0:  score += 20
    elif sr > 1.0: score += 10
    elif sr > 0.5: score += 5

    # Max drawdown
    md = metrics.get("max_drawdown", 100)
    if md < 10:   score += 20
    elif md < 20: score += 10
    elif md < 30: score += 5

    # ROI
    roi = metrics.get("roi_pct", 0)
    if roi > 30:  score += 10
    elif roi > 15: score += 5

    # Grade
    if score >= 80:
        grade      = "A+ 🏆"
        assessment = "Excellent — deploy with confidence"
    elif score >= 65:
        grade      = "A  ✅"
        assessment = "Good — deploy with normal sizing"
    elif score >= 50:
        grade      = "B  📊"
        assessment = "Average — optimize before deploying"
    elif score >= 35:
        grade      = "C  ⚠️"
        assessment = "Weak — needs significant improvement"
    else:
        grade      = "D  ❌"
        assessment = "Poor — do not deploy"

    return {
        "score"     : score,
        "grade"     : grade,
        "assessment": assessment
    }

# ── Run Multiple Strategies ───────────────────────────
def run_all_strategies(symbol, initial_capital=100000):
    strategies = [
        "ema_crossover",
        "rsi_mean_reversion",
        "macd_crossover",
        "bollinger_breakout"
    ]

    print(f"\n{'🔥' * 27}")
    print(f"  STRATEGY COMPARISON — {symbol}")
    print(f"{'🔥' * 27}")

    results = []
    for strategy in strategies:
        result = run_backtest(
            symbol, strategy, initial_capital
        )
        if result and result.get("metrics"):
            results.append({
                "strategy"     : strategy,
                "roi"          : result["metrics"].get(
                                     "roi_pct"),
                "win_rate"     : result["metrics"].get(
                                     "win_rate"),
                "sharpe"       : result["metrics"].get(
                                     "sharpe_ratio"),
                "max_dd"       : result["metrics"].get(
                                     "max_drawdown"),
                "profit_factor": result["metrics"].get(
                                     "profit_factor"),
                "total_trades" : result["metrics"].get(
                                     "total_trades")
            })

    # Comparison table
    if results:
        print(f"\n{'═' * 70}")
        print(f"  STRATEGY COMPARISON TABLE — {symbol}")
        print(f"{'═' * 70}")
        print(
            f"  {'STRATEGY':<25} {'ROI%':<8} "
            f"{'WR%':<8} {'SHARPE':<8} "
            f"{'MAX DD':<8} {'PF'}"
        )
        print("  " + "─" * 65)
        for r in sorted(
            results,
            key=lambda x: x["roi"],
            reverse=True
        ):
            print(
                f"  {r['strategy']:<25} "
                f"{str(r['roi']):<8} "
                f"{str(r['win_rate']):<8} "
                f"{str(r['sharpe']):<8} "
                f"{str(r['max_dd']):<8} "
                f"{r['profit_factor']}"
            )

        # Best strategy
        best = max(results, key=lambda x: (
            x["roi"] or 0
        ))
        print(f"\n  🏆 Best Strategy : "
              f"{best['strategy']}")
        print(f"     ROI          : {best['roi']}%")
        print(f"     Win Rate     : {best['win_rate']}%")
        print(f"     Sharpe       : {best['sharpe']}")

    return results

# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Trading AI — Backtesting Engine")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    # Run for all symbols
    for symbol in ["NIFTY", "BANKNIFTY",
                   "RELIANCE", "TCS"]:
        run_all_strategies(symbol)

    print("\n  ✅ Backtesting complete!")
    print("  Check outputs/backtests/ for results")
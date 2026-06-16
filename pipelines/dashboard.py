import os
import sys
import json
import sqlite3
import time
from datetime import datetime, date, timedelta

ROOT = os.path.dirname(os.path.dirname(
       os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MEMORY_DB     = os.path.join(ROOT, "memory",
                              "trading_memory.db")
OUTPUT_DIR    = os.path.join(ROOT, "outputs")
STRATEGIES_DIR = os.path.join(ROOT, "strategies")

# ── Terminal Colors ───────────────────────────────────
class Colors:
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"

def colored(text, color):
    return f"{color}{text}{Colors.RESET}"

def bold(text):
    return f"{Colors.BOLD}{text}{Colors.RESET}"

# ── Clear Screen ──────────────────────────────────────
def clear():
    os.system(
        "cls" if os.name == "nt" else "clear"
    )

# ── Load Latest Signals ───────────────────────────────
def load_latest_signals():
    signals = []
    symbols = [
        "NIFTY", "BANKNIFTY",
        "RELIANCE", "TCS"
    ]

    for symbol in symbols:
        path = os.path.join(
            STRATEGIES_DIR,
            f"{symbol}_strategy.json"
        )
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                    data["symbol"] = symbol
                    signals.append(data)
            except Exception:
                pass

    return signals

# ── Load Daily Report ─────────────────────────────────
def load_daily_report():
    path = os.path.join(
        STRATEGIES_DIR, "daily_report.json"
    )
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return None

# ── Load Memory Stats ─────────────────────────────────
def load_memory_stats():
    if not os.path.exists(MEMORY_DB):
        return {}

    try:
        conn  = sqlite3.connect(MEMORY_DB)
        c     = conn.cursor()
        today = str(date.today())

        # Total analyses
        c.execute("SELECT COUNT(*) FROM analyses")
        total = c.fetchone()[0]

        # Today's analyses
        c.execute("""
            SELECT COUNT(*) FROM analyses
            WHERE DATE(timestamp) = ?
        """, (today,))
        today_count = c.fetchone()[0]

        # Win/loss stats
        c.execute("""
            SELECT
                result,
                COUNT(*),
                ROUND(AVG(pnl), 2),
                ROUND(SUM(pnl), 2)
            FROM outcomes
            GROUP BY result
        """)
        outcome_stats = {}
        for row in c.fetchall():
            outcome_stats[row[0]] = {
                "count"  : row[1],
                "avg_pnl": row[2],
                "sum_pnl": row[3]
            }

        # Today's PnL
        c.execute("""
            SELECT COALESCE(SUM(o.pnl), 0)
            FROM outcomes o
            JOIN analyses a
                ON a.id = o.analysis_id
            WHERE DATE(a.timestamp) = ?
        """, (today,))
        today_pnl = float(
            c.fetchone()[0] or 0
        )

        # Recent trades
        c.execute("""
            SELECT
                a.symbol,
                a.action,
                a.timestamp,
                o.result,
                o.pnl
            FROM analyses a
            JOIN outcomes o
                ON o.analysis_id = a.id
            ORDER BY a.timestamp DESC
            LIMIT 5
        """)
        recent = c.fetchall()

        conn.close()

        wins   = outcome_stats.get(
            "profit", {}
        )
        losses = outcome_stats.get(
            "loss", {}
        )
        total_trades = (
            wins.get("count", 0) +
            losses.get("count", 0)
        )
        win_rate = round(
            wins.get("count", 0) /
            total_trades * 100, 1
        ) if total_trades > 0 else 0

        return {
            "total_analyses": total,
            "today_analyses": today_count,
            "total_trades"  : total_trades,
            "wins"          : wins.get("count", 0),
            "losses"        : losses.get("count", 0),
            "win_rate"      : win_rate,
            "total_pnl"     : round(
                wins.get("sum_pnl", 0) +
                losses.get("sum_pnl", 0), 2
            ),
            "today_pnl"     : today_pnl,
            "avg_win"       : wins.get("avg_pnl", 0),
            "avg_loss"      : losses.get(
                "avg_pnl", 0
            ),
            "recent_trades" : recent
        }

    except Exception as e:
        return {"error": str(e)}

# ── Load Latest Evaluation ────────────────────────────
def load_latest_evaluation():
    eval_dir = os.path.join(
        OUTPUT_DIR, "evaluations"
    )
    if not os.path.exists(eval_dir):
        return None

    files = sorted([
        f for f in os.listdir(eval_dir)
        if f.endswith(".json")
    ])

    if not files:
        return None

    try:
        with open(
            os.path.join(eval_dir, files[-1])
        ) as f:
            return json.load(f)
    except Exception:
        return None

# ── Load Event Context ────────────────────────────────
def load_event_context():
    path = os.path.join(
        OUTPUT_DIR,
        f"event_context_"
        f"{datetime.now().strftime('%Y%m%d')}.json"
    )
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return None

# ── Load Latest Regime ────────────────────────────────
def load_latest_regime(symbol):
    files = []
    if os.path.exists(OUTPUT_DIR):
        files = sorted([
            f for f in os.listdir(OUTPUT_DIR)
            if f.startswith(f"{symbol}_regime")
            and f.endswith(".json")
        ])

    if files:
        try:
            with open(
                os.path.join(
                    OUTPUT_DIR, files[-1]
                )
            ) as f:
                return json.load(f)
        except Exception:
            pass
    return None

# ── Draw Box ──────────────────────────────────────────
def box(title, width=60):
    line = "─" * (width - 2)
    print(colored(f"┌{line}┐", Colors.CYAN))
    pad  = width - 4 - len(title)
    print(colored(
        f"│ {bold(title)}"
        f"{' ' * pad} │",
        Colors.CYAN
    ))
    print(colored(f"├{line}┤", Colors.CYAN))

def box_end(width=60):
    line = "─" * (width - 2)
    print(colored(f"└{line}┘", Colors.CYAN))

def box_row(label, value,
            width=60, value_color=None):
    value_str = str(value)
    if value_color:
        value_display = colored(
            value_str, value_color
        )
    else:
        value_display = value_str

    label_w  = 22
    value_w  = width - label_w - 6
    label_pad = label_w - len(label)
    print(
        colored("│ ", Colors.CYAN) +
        colored(label, Colors.WHITE) +
        " " * label_pad +
        ": " +
        value_display +
        colored(" │", Colors.CYAN)
    )

def divider(width=60):
    line = "─" * (width - 2)
    print(colored(f"├{line}┤", Colors.CYAN))

# ── Signal Color ──────────────────────────────────────
def signal_color(action):
    return {
        "buy"           : Colors.GREEN,
        "sell"          : Colors.RED,
        "hold"          : Colors.YELLOW,
        "avoid"         : Colors.RED,
        "reduce_exposure": Colors.YELLOW
    }.get(action, Colors.WHITE)

def status_color(status):
    return {
        "CLEAN"       : Colors.GREEN,
        "MINOR_ISSUES": Colors.YELLOW,
        "DEGRADED"    : Colors.YELLOW,
        "HIGH_RISK"   : Colors.RED,
        "BLOCKED"     : Colors.RED,
        "APPROVED"    : Colors.GREEN,
        "REDUCED"     : Colors.YELLOW
    }.get(status, Colors.WHITE)

# ── Header ────────────────────────────────────────────
def print_header():
    now = datetime.now()
    print()
    print(colored(
        "╔══════════════════════════════════════"
        "════════════════════╗",
        Colors.CYAN
    ))
    print(colored(
        "║          🤖 TRADING AI — INTELLIGENCE"
        " DASHBOARD           ║",
        Colors.CYAN
    ))
    print(colored(
        "╚══════════════════════════════════════"
        "════════════════════╝",
        Colors.CYAN
    ))
    print(colored(
        f"  📅 {now.strftime('%A, %d %B %Y')}  "
        f"⏰ {now.strftime('%H:%M:%S')}  "
        f"v10.0",
        Colors.DIM
    ))
    print()

# ── Market Status Panel ───────────────────────────────
def print_market_status(event_ctx):
    box("📅 MARKET STATUS & EVENTS", 60)

    now       = datetime.now()
    hour      = now.hour
    minute    = now.minute
    t         = hour * 100 + minute

    if 915 <= t <= 1530:
        market_status = colored(
            "🟢 OPEN", Colors.GREEN
        )
    elif t < 915:
        market_status = colored(
            "🔴 PRE-MARKET", Colors.YELLOW
        )
    else:
        market_status = colored(
            "🔴 CLOSED", Colors.RED
        )

    box_row("Market Status", market_status)
    box_row("Time", now.strftime("%H:%M:%S"))

    if event_ctx:
        risk     = event_ctx.get(
            "event_risk", "normal"
        )
        risk_col = (
            Colors.RED    if risk == "extreme"
            else Colors.YELLOW if risk == "elevated"
            else Colors.GREEN
        )
        box_row(
            "Event Risk",
            colored(risk.upper(), risk_col)
        )

        is_event = event_ctx.get(
            "is_event_day", False
        )
        box_row(
            "Event Day",
            colored(
                "🔴 YES", Colors.RED
            ) if is_event else colored(
                "✅ NO", Colors.GREEN
            )
        )

        results = event_ctx.get(
            "results_season", {}
        )
        if results.get("in_results_season"):
            box_row(
                "Results Season",
                colored(
                    f"📊 YES — "
                    f"{results['quarter']}",
                    Colors.YELLOW
                )
            )

        alerts = event_ctx.get("alerts", [])
        if alerts:
            divider()
            for alert in alerts[:2]:
                print(colored(
                    f"│ ⚠ {alert[:54]:<54} │",
                    Colors.YELLOW
                ))
    else:
        box_row(
            "Events",
            colored("Loading...", Colors.DIM)
        )

    box_end()
    print()

# ── Signals Panel ─────────────────────────────────────
def print_signals_panel(signals):
    box("🎯 LIVE TRADING SIGNALS", 60)

    if not signals:
        print(colored(
            "│  No signals loaded yet.              "
            "                       │",
            Colors.CYAN
        ))
        print(colored(
            "│  Run: python pipelines/intelligence.py"
            "                     │",
            Colors.DIM
        ))
        box_end()
        print()
        return

    for signal in signals:
        symbol    = signal.get("symbol", "?")
        action    = signal.get("action",  "hold")
        conf      = signal.get("confidence", 0)
        vstatus   = signal.get(
            "validation_status", "UNKNOWN"
        )
        pstatus   = signal.get(
            "portfolio_status", ""
        )
        size      = signal.get(
            "recommended_size", 0
        )
        strategy  = signal.get("strategy", "N/A")
        entry     = signal.get("entry_zone", "N/A")
        sl        = signal.get("stop_loss",  "N/A")
        target    = signal.get("target",     "N/A")

        a_col     = signal_color(action)
        v_col     = status_color(vstatus)
        p_col     = status_color(pstatus)

        # Confidence bar
        conf_pct  = int(float(conf) * 20)
        conf_bar  = (
            "█" * conf_pct +
            "░" * (20 - conf_pct)
        )

        divider()
        print(colored(
            f"│  {bold(symbol):<12} "
            f"{colored(action.upper(), a_col):<20}"
            f"Conf: {float(conf):.0%}             │",
            Colors.CYAN
        ))
        print(colored(
            f"│  [{conf_bar}]"
            f"                           │",
            Colors.DIM
        ))
        print(colored(
            f"│  Strategy   : "
            f"{str(strategy)[:40]:<40}│",
            Colors.CYAN
        ))
        print(colored(
            f"│  Entry      : {str(entry):<20}"
            f"SL: {str(sl):<15}        │",
            Colors.CYAN
        ))
        print(colored(
            f"│  Target     : {str(target):<20}"
            f"Size: ₹{float(size):>10,.0f}      │",
            Colors.CYAN
        ))

        # Status row
        v_str = colored(vstatus, v_col)
        p_str = colored(pstatus, p_col) \
                if pstatus else ""
        print(colored(
            f"│  Valid: {vstatus:<14}"
            f"Port: {pstatus:<14}"
            f"               │",
            Colors.CYAN
        ))

        # Warnings
        warnings = [
            w for w in signal.get(
                "warnings", []
            ) if w
        ][:1]
        for w in warnings:
            print(colored(
                f"│  ⚠ {str(w)[:54]:<54} │",
                Colors.YELLOW
            ))

    box_end()
    print()

# ── Regime Panel ──────────────────────────────────────
def print_regime_panel():
    box("📊 MARKET REGIME", 60)

    for symbol in ["NIFTY", "BANKNIFTY"]:
        regime = load_latest_regime(symbol)
        if regime:
            fusion = regime.get("fusion",     {})
            trend  = regime.get("trend",      {})
            vol    = regime.get("volatility", {})
            expiry = regime.get("expiry",     {})

            bias      = fusion.get(
                "primary_bias", "neutral"
            )
            bias_col  = (
                Colors.GREEN  if bias == "bullish"
                else Colors.RED if bias == "bearish"
                else Colors.YELLOW
            )

            vol_reg   = vol.get(
                "vol_regime", "normal"
            )
            vol_col   = (
                Colors.RED    if vol_reg == "extreme"
                else Colors.YELLOW if vol_reg == "high"
                else Colors.GREEN
            )

            divider()
            print(colored(
                f"│  {bold(symbol):<52}│",
                Colors.CYAN
            ))
            print(colored(
                f"│  Bias    : "
                f"{colored(bias.upper(), bias_col):<30}"
                f"Score: "
                f"{fusion.get('regime_score',0)}/100    │",
                Colors.CYAN
            ))
            print(colored(
                f"│  Trend   : "
                f"{trend.get('trend_regime','N/A'):<20}"
                f"ADX: "
                f"{trend.get('adx', 0):<18}    │",
                Colors.CYAN
            ))
            print(colored(
                f"│  Vol     : "
                f"{colored(vol_reg.upper(), vol_col):<30}"
                f"HV20: "
                f"{vol.get('hv20',0)}%               │",
                Colors.CYAN
            ))
            print(colored(
                f"│  Expiry  : "
                f"{expiry.get('expiry_type','N/A'):<20}"
                f"DTE: "
                f"{expiry.get('days_to_expiry',0)}d"
                f"               │",
                Colors.CYAN
            ))
        else:
            divider()
            print(colored(
                f"│  {symbol}: No regime data yet"
                f"                              │",
                Colors.DIM
            ))

    box_end()
    print()

# ── Memory Stats Panel ────────────────────────────────
def print_memory_panel(stats):
    box("🧠 MEMORY & PERFORMANCE", 60)

    if not stats or "error" in stats:
        print(colored(
            "│  No memory data yet.                 "
            "                       │",
            Colors.DIM
        ))
        box_end()
        print()
        return

    total_pnl = stats.get("total_pnl", 0)
    today_pnl = stats.get("today_pnl", 0)
    win_rate  = stats.get("win_rate",  0)

    pnl_col   = (
        Colors.GREEN if total_pnl >= 0
        else Colors.RED
    )
    today_col = (
        Colors.GREEN if today_pnl >= 0
        else Colors.RED
    )
    wr_col    = (
        Colors.GREEN if win_rate >= 55
        else Colors.YELLOW if win_rate >= 45
        else Colors.RED
    )

    box_row(
        "Total Analyses",
        stats.get("total_analyses", 0)
    )
    box_row(
        "Today's Analyses",
        stats.get("today_analyses", 0)
    )
    divider()
    box_row(
        "Total Trades",
        stats.get("total_trades", 0)
    )
    box_row(
        "Win Rate",
        colored(f"{win_rate}%", wr_col)
    )
    box_row(
        "Total PnL",
        colored(f"₹{total_pnl:,.2f}", pnl_col)
    )
    box_row(
        "Today PnL",
        colored(f"₹{today_pnl:,.2f}", today_col)
    )
    box_row(
        "Avg Win",
        colored(
            f"₹{stats.get('avg_win',0):,.2f}",
            Colors.GREEN
        )
    )
    box_row(
        "Avg Loss",
        colored(
            f"₹{stats.get('avg_loss',0):,.2f}",
            Colors.RED
        )
    )

    # Recent trades
    recent = stats.get("recent_trades", [])
    if recent:
        divider()
        print(colored(
            "│  Recent Trades:                      "
            "                       │",
            Colors.CYAN
        ))
        for trade in recent[:4]:
            sym    = trade[0]
            action = trade[1]
            result = trade[3]
            pnl    = float(trade[4] or 0)

            r_col  = (
                Colors.GREEN if result == "profit"
                else Colors.RED
            )
            a_col  = signal_color(action)
            icon   = "✅" if result == "profit" \
                     else "❌"

            print(colored(
                f"│  {icon} "
                f"{colored(sym, Colors.WHITE):<10}"
                f"{colored(action.upper(), a_col):<8}"
                f"{colored(f'₹{pnl:,.0f}', r_col):<15}"
                f"              │",
                Colors.CYAN
            ))

    box_end()
    print()

# ── Model Health Panel ────────────────────────────────
def print_model_health(evaluation):
    box("🤖 MODEL HEALTH", 60)

    if not evaluation:
        print(colored(
            "│  No evaluation data yet.             "
            "                       │",
            Colors.DIM
        ))
        print(colored(
            "│  Run: python pipelines/"
            "model_evaluator.py              │",
            Colors.DIM
        ))
        box_end()
        print()
        return

    quality  = evaluation.get("quality") or {}
    drift    = evaluation.get("drift",   {})
    retrain  = evaluation.get(
        "retraining", {}
    )

    score    = quality.get("score", 0)
    grade    = quality.get("grade", "N/A")
    status   = quality.get("status", "N/A")

    score_col = (
        Colors.GREEN  if score >= 65
        else Colors.YELLOW if score >= 50
        else Colors.RED
    )

    drift_level = drift.get(
        "drift_level", "none"
    )
    drift_col   = (
        Colors.GREEN  if drift_level == "none"
        else Colors.YELLOW
        if drift_level in ["minor", "moderate"]
        else Colors.RED
    )

    urgency    = retrain.get("urgency", "none")
    urgency_col = (
        Colors.GREEN  if urgency == "none"
        else Colors.YELLOW if urgency == "planned"
        else Colors.RED
    )

    box_row(
        "Signal Quality",
        colored(
            f"{score}/100 ({grade})", score_col
        )
    )
    box_row(
        "Status",
        colored(status.upper(), score_col)
    )
    divider()
    box_row(
        "Drift Level",
        colored(drift_level.upper(), drift_col)
    )
    box_row(
        "Drift Score",
        f"{drift.get('drift_score',0):.1%}"
    )
    box_row(
        "Win Rate Trend",
        drift.get("trend_direction", "N/A")
    )
    divider()
    box_row(
        "Retrain Urgency",
        colored(urgency.upper(), urgency_col)
    )

    reasons = retrain.get("reasons", [])
    if reasons:
        for reason in reasons[:2]:
            print(colored(
                f"│  ⚠ {str(reason)[:54]:<54} │",
                Colors.YELLOW
            ))

    box_end()
    print()

# ── Portfolio Panel ───────────────────────────────────
def print_portfolio_panel(signals):
    box("💼 PORTFOLIO OVERVIEW", 60)

    actionable = [
        s for s in signals
        if s.get("action") in ["buy", "sell"]
        and s.get("validation_status") in [
            "CLEAN", "MINOR_ISSUES"
        ]
        and float(s.get("confidence", 0)) >= 0.6
    ]

    total_allocated = sum(
        float(s.get("recommended_size", 0))
        for s in actionable
    )

    box_row(
        "Active Signals",
        f"{len(actionable)}"
    )
    box_row(
        "Total Allocated",
        colored(
            f"₹{total_allocated:,.0f}",
            Colors.CYAN
        )
    )

    if actionable:
        divider()
        print(colored(
            "│  ACTIONABLE SIGNALS:                 "
            "                       │",
            Colors.CYAN
        ))
        for s in actionable:
            sym   = s.get("symbol",  "?")
            act   = s.get("action",  "?")
            conf  = float(
                s.get("confidence", 0)
            )
            size  = float(
                s.get("recommended_size", 0)
            )
            entry = s.get("entry_zone", "N/A")
            sl    = s.get("stop_loss",  "N/A")

            a_col = signal_color(act)
            print(colored(
                f"│  "
                f"{colored(act.upper(), a_col):<10}"
                f"{sym:<12}"
                f"₹{size:>8,.0f}  "
                f"Conf:{conf:.0%}          │",
                Colors.CYAN
            ))
            print(colored(
                f"│  Entry: {str(entry):<20}"
                f"SL: {str(sl):<20}"
                f"        │",
                Colors.DIM
            ))
    else:
        divider()
        print(colored(
            "│  ⏳ No actionable signals right now   "
            "                       │",
            Colors.YELLOW
        ))
        print(colored(
            "│  Stay patient — wait for clean setups"
            "                       │",
            Colors.DIM
        ))

    box_end()
    print()

# ── Quick Stats Bar ───────────────────────────────────
def print_stats_bar(stats, signals):
    total_pnl = stats.get("total_pnl", 0)
    win_rate  = stats.get("win_rate",  0)
    total_sig = len(signals)
    active    = len([
        s for s in signals
        if s.get("action") in ["buy", "sell"]
    ])
    blocked   = len([
        s for s in signals
        if s.get("action") == "avoid"
    ])

    pnl_col = (
        Colors.GREEN if total_pnl >= 0
        else Colors.RED
    )

    print(colored(
        "─" * 62, Colors.DIM
    ))
    print(
        f"  📊 Signals: "
        f"{colored(str(total_sig), Colors.WHITE)} | "
        f"Active: "
        f"{colored(str(active), Colors.GREEN)} | "
        f"Blocked: "
        f"{colored(str(blocked), Colors.RED)} | "
        f"WR: "
        f"{colored(f'{win_rate}%', Colors.CYAN)} | "
        f"PnL: "
        f"{colored(f'₹{total_pnl:,.0f}', pnl_col)}"
    )
    print(colored(
        "─" * 62, Colors.DIM
    ))
    print()

# ── Footer ────────────────────────────────────────────
def print_footer(refresh_secs=60):
    next_refresh = datetime.now().replace(
        microsecond=0
    ) + timedelta(seconds=refresh_secs)

    print(colored(
        f"  🔄 Next refresh: "
        f"{next_refresh.strftime('%H:%M:%S')} | "
        f"Press Ctrl+C to exit",
        Colors.DIM
    ))
    print()

# ── Run Full Dashboard ────────────────────────────────
def run_dashboard(refresh_secs=60,
                   single_run=False):
    while True:
        try:
            clear()

            # Load all data
            signals    = load_latest_signals()
            report     = load_daily_report()
            mem_stats  = load_memory_stats()
            event_ctx  = load_event_context()
            evaluation = load_latest_evaluation()

            # Print dashboard
            print_header()
            print_stats_bar(mem_stats, signals)
            print_market_status(event_ctx)
            print_signals_panel(signals)
            print_regime_panel()
            print_portfolio_panel(signals)
            print_memory_panel(mem_stats)
            print_model_health(evaluation)
            print_footer(refresh_secs)

            if single_run:
                break

            time.sleep(refresh_secs)

        except KeyboardInterrupt:
            print(colored(
                "\n  Dashboard stopped. Goodbye! 👋",
                Colors.CYAN
            ))
            break
        except Exception as e:
            print(colored(
                f"\n  Dashboard error: {e}",
                Colors.RED
            ))
            if single_run:
                break
            time.sleep(30)

# ── Snapshot Mode ─────────────────────────────────────
def run_snapshot():
    """Single run — no refresh loop"""
    run_dashboard(single_run=True)

# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Trading AI Dashboard"
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Single snapshot, no refresh"
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=60,
        help="Refresh interval in seconds"
    )

    args = parser.parse_args()

    if args.snapshot:
        run_snapshot()
    else:
        print(colored(
            "\n  🚀 Starting Trading AI Dashboard...",
            Colors.CYAN
        ))
        print(colored(
            f"  Refreshing every "
            f"{args.refresh} seconds.",
            Colors.DIM
        ))
        print(colored(
            "  Press Ctrl+C to exit.\n",
            Colors.DIM
        ))
        time.sleep(1)
        run_dashboard(args.refresh)
import os
import sys
import json
import sqlite3
from datetime import datetime, date

ROOT = os.path.dirname(os.path.dirname(
       os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MEMORY_DB  = os.path.join(ROOT, "memory",
                           "trading_memory.db")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Portfolio Config ──────────────────────────────────
DEFAULT_CONFIG = {
    "total_capital"        : 500000,   # ₹5 Lakhs
    "max_risk_per_trade"   : 0.02,     # 2% per trade
    "max_daily_loss"       : 0.03,     # 3% daily loss
    "max_weekly_loss"      : 0.06,     # 6% weekly loss
    "max_open_positions"   : 5,        # Max 5 trades
    "max_sector_exposure"  : 0.40,     # 40% one sector
    "max_correlated_trades": 2,        # Max 2 correlated
    "min_confidence"       : 0.60,     # Min 60% conf
    "max_position_size"    : 0.20,     # Max 20% trade
}

# ── Sector Map ────────────────────────────────────────
SECTOR_MAP = {
    "NIFTY"     : "index",
    "BANKNIFTY" : "banking",
    "HDFCBANK"  : "banking",
    "ICICIBANK" : "banking",
    "AXISBANK"  : "banking",
    "KOTAKBANK" : "banking",
    "SBIN"      : "banking",
    "RELIANCE"  : "energy",
    "ONGC"      : "energy",
    "BPCL"      : "energy",
    "TCS"       : "technology",
    "INFY"      : "technology",
    "WIPRO"     : "technology",
    "HCLTECH"   : "technology",
    "TECHM"     : "technology",
    "TATAMOTORS": "auto",
    "MARUTI"    : "auto",
    "BAJAJ-AUTO": "auto",
    "SUNPHARMA" : "pharma",
    "DRREDDY"   : "pharma",
    "CIPLA"     : "pharma",
}

# ── Correlation Groups ────────────────────────────────
CORRELATION_GROUPS = [
    ["NIFTY", "BANKNIFTY"],
    ["HDFCBANK", "ICICIBANK",
     "AXISBANK", "KOTAKBANK"],
    ["TCS", "INFY", "WIPRO",
     "HCLTECH", "TECHM"],
    ["RELIANCE", "ONGC", "BPCL"],
]

# ── Portfolio State ───────────────────────────────────
class PortfolioState:
    def __init__(self, config=None):
        self.config            = config or DEFAULT_CONFIG
        self.open_positions    = []
        self.daily_pnl         = 0.0
        self.weekly_pnl        = 0.0
        self.total_capital     = self.config[
            "total_capital"
        ]
        self.available_capital = self.total_capital
        self.daily_loss_hit    = False
        self.weekly_loss_hit   = False

    def add_position(self, position):
        self.open_positions.append(position)

    def get_sector_exposure(self):
        sectors = {}
        for pos in self.open_positions:
            sector = SECTOR_MAP.get(
                pos.get("symbol", ""), "other"
            )
            val    = pos.get("value", 0)
            sectors[sector] = \
                sectors.get(sector, 0) + val
        return sectors

    def get_total_exposure(self):
        return sum(
            p.get("value", 0)
            for p in self.open_positions
        )

    def to_dict(self):
        return {
            "total_capital"    : self.total_capital,
            "available_capital": self.available_capital,
            "open_positions"   : len(
                self.open_positions
            ),
            "daily_pnl"        : self.daily_pnl,
            "weekly_pnl"       : self.weekly_pnl,
            "total_exposure"   : self.get_total_exposure(),
            "sector_exposure"  : self.get_sector_exposure(),
            "daily_loss_hit"   : self.daily_loss_hit,
            "weekly_loss_hit"  : self.weekly_loss_hit
        }

# ── Load Open Positions ───────────────────────────────
def load_open_positions():
    if not os.path.exists(MEMORY_DB):
        return []

    try:
        conn  = sqlite3.connect(MEMORY_DB)
        c     = conn.cursor()
        today = str(date.today())

        c.execute("""
            SELECT
                a.symbol,
                a.action,
                a.timestamp,
                o.entry_price,
                o.result
            FROM analyses a
            LEFT JOIN outcomes o
                ON o.analysis_id = a.id
            WHERE DATE(a.timestamp) = ?
            AND a.action IN ('buy', 'sell')
        """, (today,))

        rows      = c.fetchall()
        positions = []

        for row in rows:
            if row[4] is None:
                positions.append({
                    "symbol"     : row[0],
                    "action"     : row[1],
                    "timestamp"  : row[2],
                    "entry_price": row[3] or 0,
                    "value"      : (row[3] or 0) * 50
                })

        conn.close()
        return positions

    except Exception as e:
        print(f"  ⚠ Position load failed: {e}")
        return []

# ── Load Daily PnL ────────────────────────────────────
def load_daily_pnl():
    if not os.path.exists(MEMORY_DB):
        return 0.0

    try:
        conn  = sqlite3.connect(MEMORY_DB)
        c     = conn.cursor()
        today = str(date.today())

        c.execute("""
            SELECT COALESCE(SUM(o.pnl), 0)
            FROM outcomes o
            JOIN analyses a
                ON a.id = o.analysis_id
            WHERE DATE(a.timestamp) = ?
        """, (today,))

        result = c.fetchone()
        conn.close()
        return float(result[0]) if result else 0.0

    except Exception as e:
        print(f"  ⚠ PnL load failed: {e}")
        return 0.0

# ── Position Sizing Engine ────────────────────────────
def calculate_position_size(signal, portfolio,
                              config=None):
    config     = config or DEFAULT_CONFIG
    capital    = portfolio.total_capital
    confidence = float(
        signal.get("confidence", 0.5)
    )
    risk_level = signal.get("risk_level", "medium")
    action     = signal.get("action",     "hold")

    if action not in ["buy", "sell"]:
        return {
            "recommended_size": 0,
            "size_pct"        : 0,
            "reasoning"       : "No actionable signal"
        }

    # Base risk
    base_risk = config["max_risk_per_trade"]

    # Confidence multiplier
    conf_mult = confidence / 0.7
    conf_mult = max(0.5, min(1.5, conf_mult))

    # Risk level multiplier
    risk_mult = {
        "low"    : 1.2,
        "medium" : 1.0,
        "high"   : 0.7,
        "extreme": 0.3
    }.get(risk_level, 1.0)

    # Position size hint multiplier
    size_hint = signal.get("position_size", "full")
    hint_mult = {
        "full"   : 1.0,
        "half"   : 0.5,
        "quarter": 0.25,
        "avoid"  : 0.0
    }.get(size_hint, 1.0)

    # Final risk
    final_risk = (
        base_risk * conf_mult *
        risk_mult * hint_mult
    )
    final_risk = min(
        final_risk,
        config["max_position_size"]
    )

    # Position value
    position_value = capital * final_risk
    available      = portfolio.available_capital
    position_value = min(
        position_value, available * 0.95
    )

    return {
        "recommended_size": round(position_value, 2),
        "size_pct"        : round(
            final_risk * 100, 2
        ),
        "base_risk_pct"   : round(
            base_risk * 100, 2
        ),
        "conf_multiplier" : round(conf_mult, 2),
        "risk_multiplier" : round(risk_mult, 2),
        "hint_multiplier" : round(hint_mult, 2),
        "reasoning"       : (
            f"₹{position_value:,.0f} = "
            f"{final_risk*100:.1f}% of capital"
        )
    }

# ── Correlation Check ─────────────────────────────────
def check_correlation(symbol, open_positions,
                       direction):
    correlated = []

    for group in CORRELATION_GROUPS:
        if symbol in group:
            for pos in open_positions:
                pos_symbol = pos.get("symbol", "")
                if (pos_symbol in group and
                        pos_symbol != symbol):
                    correlated.append({
                        "symbol"   : pos_symbol,
                        "direction": pos.get("action"),
                        "group"    : group
                    })

    same_dir = [
        c for c in correlated
        if c["direction"] == direction
    ]

    return {
        "correlated_symbols": [
            c["symbol"] for c in correlated
        ],
        "same_direction"    : [
            c["symbol"] for c in same_dir
        ],
        "correlation_risk"  : (
            "high"   if len(same_dir) >= 2
            else "medium" if len(same_dir) == 1
            else "low"
        )
    }

# ── Sector Concentration Check ────────────────────────
def check_sector_concentration(symbol,
                                 portfolio,
                                 new_value,
                                 config=None):
    config       = config or DEFAULT_CONFIG
    max_exposure = config["max_sector_exposure"]
    capital      = portfolio.total_capital
    sector       = SECTOR_MAP.get(symbol, "other")
    current_exp  = portfolio.get_sector_exposure()
    current_val  = current_exp.get(sector, 0)
    new_total    = current_val + new_value
    new_pct      = new_total / capital

    return {
        "sector"          : sector,
        "current_exposure": round(
            current_val / capital * 100, 1
        ),
        "new_exposure"    : round(new_pct * 100, 1),
        "max_allowed"     : round(
            max_exposure * 100, 1
        ),
        "breaches_limit"  : new_pct > max_exposure,
        "headroom_pct"    : round(
            (max_exposure - new_pct) * 100, 1
        )
    }

# ── Daily Loss Check ──────────────────────────────────
def check_daily_loss_limit(portfolio, config=None):
    config    = config or DEFAULT_CONFIG
    max_loss  = (
        portfolio.total_capital *
        config["max_daily_loss"]
    )
    daily_pnl = portfolio.daily_pnl
    loss_pct  = (
        abs(daily_pnl) /
        portfolio.total_capital * 100
        if daily_pnl < 0 else 0
    )

    return {
        "daily_pnl"       : daily_pnl,
        "max_allowed_loss": round(-max_loss, 2),
        "loss_pct"        : round(loss_pct,  2),
        "limit_breached"  : daily_pnl < -max_loss,
        "remaining_risk"  : round(
            max_loss + daily_pnl, 2
        ) if daily_pnl < 0 else round(max_loss, 2)
    }

# ── Full Portfolio Risk Check ─────────────────────────
def check_portfolio_risk(signal, config=None):
    config  = config or DEFAULT_CONFIG
    symbol  = signal.get("symbol", "")
    action  = signal.get("action",  "hold")

    print(f"\n  {'=' * 50}")
    print(f"  Portfolio Risk — {symbol}")
    print(f"  {'=' * 50}")

    # Load state
    open_positions         = load_open_positions()
    daily_pnl              = load_daily_pnl()
    portfolio              = PortfolioState(config)
    portfolio.open_positions = open_positions
    portfolio.daily_pnl      = daily_pnl
    portfolio.available_capital = (
        config["total_capital"] + daily_pnl
    )

    risks    = []
    blocks   = []
    warnings = []

    # ── Check 1 — Daily Loss ──────────────────────────
    loss_check = check_daily_loss_limit(
        portfolio, config
    )
    if loss_check["limit_breached"]:
        blocks.append(
            f"Daily loss limit breached — "
            f"PnL: ₹{daily_pnl:,.0f}"
        )
        portfolio.daily_loss_hit = True

    print(f"\n  📊 Daily PnL    : ₹{daily_pnl:,.2f} | "
          f"{'🔴 BREACHED' if loss_check['limit_breached'] else '✅ OK'}")

    # ── Check 2 — Max Positions ───────────────────────
    pos_count = len(open_positions)
    max_pos   = config["max_open_positions"]

    if pos_count >= max_pos:
        blocks.append(
            f"Max positions reached "
            f"({pos_count}/{max_pos})"
        )

    print(f"  📋 Positions    : "
          f"{pos_count}/{max_pos} | "
          f"{'🔴 FULL' if pos_count >= max_pos else '✅ OK'}")

    # ── Check 3 — Position Sizing ─────────────────────
    sizing = calculate_position_size(
        signal, portfolio, config
    )

    print(f"  💰 Size         : "
          f"₹{sizing['recommended_size']:,.2f} "
          f"({sizing['size_pct']}%)")

    # ── Check 4 — Correlation ─────────────────────────
    corr = check_correlation(
        symbol, open_positions, action
    )

    if corr["correlation_risk"] == "high":
        blocks.append(
            f"Too many correlated trades: "
            f"{corr['same_direction']}"
        )
    elif corr["correlation_risk"] == "medium":
        warnings.append(
            f"Correlated position exists: "
            f"{corr['same_direction']}"
        )

    print(f"  🔗 Correlation  : "
          f"{corr['correlation_risk'].upper()} | "
          f"Same dir: {corr['same_direction']}")

    # ── Check 5 — Sector ──────────────────────────────
    sector_check = check_sector_concentration(
        symbol, portfolio,
        sizing["recommended_size"], config
    )

    if sector_check["breaches_limit"]:
        warnings.append(
            f"Sector {sector_check['sector']} "
            f"at {sector_check['new_exposure']}% "
            f"(max {sector_check['max_allowed']}%)"
        )

    print(f"  🏭 Sector       : "
          f"{sector_check['sector']} | "
          f"{sector_check['new_exposure']}% | "
          f"{'⚠ HIGH' if sector_check['breaches_limit'] else '✅ OK'}")

    # ── Check 6 — Confidence ──────────────────────────
    confidence = float(signal.get("confidence", 0))
    min_conf   = config["min_confidence"]

    if confidence < min_conf:
        blocks.append(
            f"Confidence {confidence} below "
            f"minimum {min_conf}"
        )

    print(f"  🎯 Confidence   : "
          f"{confidence} | "
          f"{'🔴 LOW' if confidence < min_conf else '✅ OK'}")

    # ── Final Decision ────────────────────────────────
    is_blocked = len(blocks) > 0

    if is_blocked:
        final_action     = "avoid"
        final_size       = 0
        portfolio_status = "BLOCKED"
    elif warnings:
        final_action     = action
        final_size       = (
            sizing["recommended_size"] * 0.5
        )
        portfolio_status = "REDUCED"
    else:
        final_action     = action
        final_size       = sizing["recommended_size"]
        portfolio_status = "APPROVED"

    status_icon = {
        "APPROVED": "✅",
        "REDUCED" : "⚠️",
        "BLOCKED" : "❌"
    }.get(portfolio_status, "❓")

    print(f"\n  {status_icon} Portfolio : "
          f"{portfolio_status} | "
          f"Final Size: ₹{final_size:,.2f}")

    if blocks:
        print(f"  🔴 Blocked by:")
        for b in blocks:
            print(f"     → {b}")

    if warnings:
        print(f"  ⚠ Warnings:")
        for w in warnings:
            print(f"     → {w}")

    # Build result
    result = {
        "symbol"           : symbol,
        "timestamp"        : datetime.now().isoformat(),
        "portfolio_status" : portfolio_status,
        "final_action"     : final_action,
        "recommended_size" : round(final_size, 2),
        "size_pct"         : sizing["size_pct"],
        "blocks"           : blocks,
        "warnings"         : warnings,
        "checks"           : {
            "daily_loss"  : loss_check,
            "positions"   : {
                "current": pos_count,
                "max"    : max_pos,
                "ok"     : pos_count < max_pos
            },
            "correlation" : corr,
            "sector"      : sector_check,
            "confidence"  : {
                "value"  : confidence,
                "minimum": min_conf,
                "ok"     : confidence >= min_conf
            }
        },
        "portfolio_state"  : portfolio.to_dict()
    }

    # Save
    path = os.path.join(
        OUTPUT_DIR,
        f"portfolio_risk_"
        f"{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    )
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result

# ── Run Portfolio Check on All Signals ────────────────
def run_portfolio_risk_all(signals, config=None):
    config   = config or DEFAULT_CONFIG

    print("\n" + "🔥" * 27)
    print("  PORTFOLIO RISK ENGINE")
    print("🔥" * 27)

    approved = []
    blocked  = []
    reduced  = []

    for signal in signals:
        if signal.get("action") not in [
            "buy", "sell"
        ]:
            continue

        result = check_portfolio_risk(
            signal, config
        )

        status = result["portfolio_status"]
        if status == "APPROVED":
            approved.append(result)
            signal["portfolio_approved"] = True
            signal["recommended_size"]   = \
                result["recommended_size"]
        elif status == "REDUCED":
            reduced.append(result)
            signal["portfolio_approved"] = True
            signal["recommended_size"]   = \
                result["recommended_size"]
            signal["portfolio_reduced"]  = True
        else:
            blocked.append(result)
            signal["portfolio_approved"] = False
            signal["action"]             = "avoid"

    # Summary
    print(f"\n{'=' * 55}")
    print(f"  PORTFOLIO SUMMARY")
    print(f"{'=' * 55}")
    print(f"  ✅ Approved : {len(approved)}")
    print(f"  ⚠️  Reduced  : {len(reduced)}")
    print(f"  ❌ Blocked  : {len(blocked)}")

    if approved or reduced:
        print(f"\n  Tradeable Signals:")
        for r in approved + reduced:
            icon = "✅" if r[
                "portfolio_status"
            ] == "APPROVED" else "⚠️"
            print(
                f"     {icon} {r['symbol']:<12} → "
                f"₹{r['recommended_size']:,.0f} | "
                f"{r['portfolio_status']}"
            )

    if blocked:
        print(f"\n  Blocked Signals:")
        for r in blocked:
            print(
                f"     ❌ {r['symbol']:<12} → "
                f"{r['blocks'][0] if r['blocks'] else 'blocked'}"
            )

    return {
        "approved": approved,
        "reduced" : reduced,
        "blocked" : blocked,
        "signals" : signals
    }

# ── Main Test ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Trading AI — Portfolio Risk Engine")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    # Test signals
    test_signals = [
        {
            "symbol"       : "NIFTY",
            "action"       : "buy",
            "confidence"   : 0.72,
            "risk_level"   : "medium",
            "position_size": "half",
            "entry_zone"   : "23700-23750",
            "stop_loss"    : "23500"
        },
        {
            "symbol"       : "BANKNIFTY",
            "action"       : "buy",
            "confidence"   : 0.65,
            "risk_level"   : "medium",
            "position_size": "full",
            "entry_zone"   : "51500-51600",
            "stop_loss"    : "51000"
        },
        {
            "symbol"       : "RELIANCE",
            "action"       : "sell",
            "confidence"   : 0.45,
            "risk_level"   : "high",
            "position_size": "quarter",
            "entry_zone"   : "1380-1400",
            "stop_loss"    : "1420"
        }
    ]

    results = run_portfolio_risk_all(test_signals)
    print(f"\n  ✅ Portfolio Risk Engine complete!")
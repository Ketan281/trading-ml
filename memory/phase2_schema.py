"""
Phase 2 database schema — adds 8 tables to trading_memory.db.

Safe to call multiple times (CREATE TABLE IF NOT EXISTS).
Call migrate() at server startup alongside memory_store.init_db().
"""

import sqlite3
import os

MEMORY_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(MEMORY_DIR, "trading_memory.db")


def migrate():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    c = conn.cursor()

    # ── Module 1: Psychology Engine ────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS psychology_state (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            date                TEXT NOT NULL,
            daily_loss          REAL DEFAULT 0,
            daily_trades        INTEGER DEFAULT 0,
            weekly_loss         REAL DEFAULT 0,
            monthly_loss        REAL DEFAULT 0,
            consecutive_losses  INTEGER DEFAULT 0,
            last_trade_ts       TEXT,
            cooldown_until      TEXT,
            risk_state          TEXT DEFAULT 'normal',
            psychology_score    REAL DEFAULT 100,
            discipline_score    REAL DEFAULT 100,
            notes               TEXT,
            updated_at          TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS psychology_events (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp           TEXT NOT NULL,
            event_type          TEXT NOT NULL,
            details             TEXT,
            risk_state_before   TEXT,
            risk_state_after    TEXT
        )
    """)

    # ── Module 2: Reflection V2 / Trade Journal ────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS trade_journal (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id            TEXT UNIQUE,
            timestamp           TEXT,
            symbol              TEXT,
            segment             TEXT,
            direction           TEXT,
            confidence          REAL,
            conviction          REAL,
            grade               TEXT,
            trade_quality       REAL,
            breadth_score       REAL,
            rs_percentile       REAL,
            sector_phase        TEXT,
            options_flow        REAL,
            regime              TEXT,
            regime_day_type     TEXT,
            mtf_alignment       REAL,
            psychology_score    REAL,
            entry_price         REAL,
            exit_price          REAL,
            stop_loss           REAL,
            target              REAL,
            position_size       REAL,
            capital_pct         REAL,
            pnl                 REAL,
            pnl_pct             REAL,
            r_multiple          REAL,
            max_favorable       REAL,
            max_adverse         REAL,
            hold_duration       TEXT,
            exit_reason         TEXT,
            enrichment_json     TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS reflection_reports (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            report_type         TEXT NOT NULL,
            period_start        TEXT,
            period_end          TEXT,
            generated_at        TEXT,
            report_json         TEXT,
            reflection_score    REAL,
            edge_score          REAL,
            recommendations     TEXT
        )
    """)

    # ── Module 8: Paper Trading V2 ─────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades_v2 (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id            TEXT UNIQUE,
            symbol              TEXT,
            segment             TEXT,
            direction           TEXT,
            entry_price         REAL,
            entry_time          TEXT,
            entry_signal        TEXT,
            shares              INTEGER,
            lots                INTEGER,
            capital_allocated   REAL,
            initial_stop        REAL,
            current_stop        REAL,
            target_1            REAL,
            target_2            REAL,
            target_3            REAL,
            grade               TEXT,
            conviction          REAL,
            trade_quality       REAL,
            regime_at_entry     TEXT,
            psychology_at_entry REAL,
            status              TEXT DEFAULT 'open',
            partial_exits       TEXT,
            exit_price          REAL,
            exit_time           TEXT,
            exit_reason         TEXT,
            slippage_entry      REAL DEFAULT 0,
            slippage_exit       REAL DEFAULT 0,
            brokerage           REAL DEFAULT 0,
            total_costs         REAL DEFAULT 0,
            gross_pnl           REAL,
            net_pnl             REAL,
            pnl_pct             REAL,
            r_multiple          REAL,
            max_favorable_excursion REAL,
            max_adverse_excursion   REAL,
            last_price          REAL,
            last_update         TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS paper_equity_curve (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            date                TEXT NOT NULL,
            cash                REAL,
            positions_value     REAL,
            total_equity        REAL,
            daily_return        REAL,
            drawdown            REAL,
            max_drawdown        REAL,
            open_positions      INTEGER
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS paper_metrics (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            computed_at         TEXT,
            total_trades        INTEGER,
            win_rate            REAL,
            profit_factor       REAL,
            sharpe              REAL,
            sortino             REAL,
            max_drawdown        REAL,
            cagr                REAL,
            expectancy          REAL,
            avg_r_multiple      REAL,
            avg_win_r           REAL,
            avg_loss_r          REAL,
            best_trade_r        REAL,
            worst_trade_r       REAL,
            avg_hold_days       REAL,
            metrics_json        TEXT
        )
    """)

    # ── Module 10: Quant Lab ───────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS quant_experiments (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id       TEXT UNIQUE,
            experiment_type     TEXT,
            name                TEXT,
            config_json         TEXT,
            started_at          TEXT,
            completed_at        TEXT,
            status              TEXT DEFAULT 'pending',
            results_json        TEXT,
            summary             TEXT
        )
    """)

    conn.commit()
    conn.close()
    print("  Phase 2 schema migrated")


if __name__ == "__main__":
    migrate()

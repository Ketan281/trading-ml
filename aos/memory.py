"""
Trade Memory — the system's long-term memory and meta-learning substrate.

A single SQLite database that records EVERYTHING the agentic OS does, so the
system can learn from its own history:

  • regime_snapshots — the market state at each decision (regime, breadth, vol)
  • signals          — every signal produced (source, score, confidence, regime,
                       sentiment, a JSON snapshot) and, later, its OUTCOME
  • decisions        — each orchestrated decision (final action, veto, conviction)
  • agent_reports    — every agent's vote / confidence / evidence per decision
  • trades           — opened/closed trades with net P&L, fees, exit reason
  • lessons          — post-market lessons learned, categorised

`meta_dataset()` joins signals to their realised outcomes — the table the
Meta-Learning Layer trains on (which signals work in which regimes, when
confidence is unreliable, recurring causes of loss).

Pure stdlib sqlite3 (no new dependency). Safe for 24/7 use: WAL mode, one file.
"""

import os
import json
import sqlite3
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(ROOT, "data", "aos")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "memory.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS regime_snapshots(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT, regime TEXT, breadth_score REAL, vol_pctile REAL,
  sector_top TEXT, extra TEXT);

CREATE TABLE IF NOT EXISTS signals(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT, symbol TEXT, asset TEXT, source TEXT,
  score REAL, confidence REAL, regime TEXT, sentiment REAL,
  snapshot TEXT, outcome_ret REAL, outcome_label INTEGER, outcome_ts TEXT);

CREATE TABLE IF NOT EXISTS decisions(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT, symbol TEXT, asset TEXT, proposed_action TEXT, final_action TEXT,
  conviction REAL, regime TEXT, vetoed INTEGER, veto_reason TEXT, evidence TEXT);

CREATE TABLE IF NOT EXISTS agent_reports(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  decision_id INTEGER, agent TEXT, role TEXT, vote TEXT, confidence REAL,
  evidence TEXT, flags TEXT, rationale TEXT,
  FOREIGN KEY(decision_id) REFERENCES decisions(id));

CREATE TABLE IF NOT EXISTS trades(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  decision_id INTEGER, symbol TEXT, segment TEXT, side TEXT,
  entry REAL, qty INTEGER, stop REAL, targets TEXT,
  status TEXT, net_pnl REAL, fees REAL, exit_reason TEXT,
  opened_at TEXT, closed_at TEXT);

CREATE TABLE IF NOT EXISTS lessons(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT, category TEXT, text TEXT, evidence TEXT);

CREATE INDEX IF NOT EXISTS ix_signals_regime ON signals(regime);
CREATE INDEX IF NOT EXISTS ix_signals_source ON signals(source);
CREATE INDEX IF NOT EXISTS ix_trades_status ON trades(status);
"""


def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    return con


def init_db():
    with connect() as con:
        con.executescript(SCHEMA)
    return DB_PATH


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _j(x):
    return json.dumps(x, default=str)


# ── writes ────────────────────────────────────────────
def record_regime(regime, breadth_score=None, vol_pctile=None, sector_top=None, extra=None):
    with connect() as con:
        cur = con.execute(
            "INSERT INTO regime_snapshots(ts,regime,breadth_score,vol_pctile,sector_top,extra)"
            " VALUES(?,?,?,?,?,?)",
            (_now(), regime, breadth_score, vol_pctile, _j(sector_top), _j(extra)))
        return cur.lastrowid


def record_signal(symbol, asset, source, score=None, confidence=None,
                  regime=None, sentiment=None, snapshot=None):
    with connect() as con:
        cur = con.execute(
            "INSERT INTO signals(ts,symbol,asset,source,score,confidence,regime,"
            "sentiment,snapshot) VALUES(?,?,?,?,?,?,?,?,?)",
            (_now(), symbol, asset, source, score, confidence, regime,
             sentiment, _j(snapshot)))
        return cur.lastrowid


def set_signal_outcome(signal_id, outcome_ret, outcome_label=None):
    with connect() as con:
        con.execute("UPDATE signals SET outcome_ret=?,outcome_label=?,outcome_ts=?"
                    " WHERE id=?", (outcome_ret, outcome_label, _now(), signal_id))


def record_decision(symbol, asset, proposed_action, final_action, conviction,
                    regime, vetoed=False, veto_reason=None, evidence=None):
    with connect() as con:
        cur = con.execute(
            "INSERT INTO decisions(ts,symbol,asset,proposed_action,final_action,"
            "conviction,regime,vetoed,veto_reason,evidence) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (_now(), symbol, asset, proposed_action, final_action, conviction,
             regime, int(vetoed), veto_reason, _j(evidence)))
        return cur.lastrowid


def record_reports(decision_id, reports):
    with connect() as con:
        for r in reports:
            con.execute(
                "INSERT INTO agent_reports(decision_id,agent,role,vote,confidence,"
                "evidence,flags,rationale) VALUES(?,?,?,?,?,?,?,?)",
                (decision_id, r.get("agent"), r.get("role"), r.get("vote"),
                 r.get("confidence"), _j(r.get("evidence")), _j(r.get("flags")),
                 r.get("rationale")))


def record_trade(decision_id, symbol, segment, side, entry, qty, stop, targets,
                 status="open", opened_at=None):
    with connect() as con:
        cur = con.execute(
            "INSERT INTO trades(decision_id,symbol,segment,side,entry,qty,stop,"
            "targets,status,opened_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (decision_id, symbol, segment, side, entry, qty, stop, _j(targets),
             status, opened_at or _now()))
        return cur.lastrowid


def close_trade(trade_id, net_pnl, fees, exit_reason):
    with connect() as con:
        con.execute("UPDATE trades SET status='closed',net_pnl=?,fees=?,"
                    "exit_reason=?,closed_at=? WHERE id=?",
                    (net_pnl, fees, exit_reason, _now(), trade_id))


def record_lesson(category, text, evidence=None):
    with connect() as con:
        cur = con.execute("INSERT INTO lessons(ts,category,text,evidence)"
                          " VALUES(?,?,?,?)", (_now(), category, text, _j(evidence)))
        return cur.lastrowid


# ── reads / meta-learning dataset ─────────────────────
def query(sql, params=()):
    with connect() as con:
        return [dict(r) for r in con.execute(sql, params).fetchall()]


def meta_dataset():
    """Signals joined with realised outcomes — the meta-learning training set."""
    return query("SELECT source,regime,score,confidence,sentiment,outcome_ret,"
                 "outcome_label FROM signals WHERE outcome_ret IS NOT NULL")


def recent_lessons(n=10):
    return query("SELECT ts,category,text FROM lessons ORDER BY id DESC LIMIT ?", (n,))


def stats():
    with connect() as con:
        c = con.cursor()
        return {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("regime_snapshots", "signals", "decisions",
                          "agent_reports", "trades", "lessons")}


if __name__ == "__main__":
    init_db()
    print("=" * 56)
    print(f"  TRADE MEMORY  →  {DB_PATH}")
    print("=" * 56)
    print("  Tables & row counts:")
    for t, n in stats().items():
        print(f"     {t:<20} {n}")

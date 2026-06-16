# Agentic Trading OS — Architecture & Deployment

A hedge-fund-style **autonomous intelligence system** built on top of the
existing trading platform. It deliberates as a committee of agents, controls
risk strictly, learns from its own mistakes, and runs continuously in
**paper-trading mode**. The cardinal rule is enforced structurally: **LLMs only
reason / summarise / explain — they never produce a price, probability, signal,
or trade decision.** Those flow only from quantitative engines.

> Status: paper trading only. No broker integration. No live capital. The
> `BrokerExecutor` interface is the single seam where real execution would
> later attach (out of scope here).

---

## 1. System layers

```
┌──────────────────────────────────────────────────────────────────────┐
│  CLIENTS         frontend / CLI / cron                                 │
├──────────────────────────────────────────────────────────────────────┤
│  API             api/server.py (FastAPI)  +  intent router            │
│                  /query  /options/{sym}  /book  /screen  /health       │
├──────────────────────────────────────────────────────────────────────┤
│  AGENTIC OS (aos/)                                                     │
│   orchestrator ── debate + Risk-Officer VETO ── final decision        │
│        │                                                               │
│   9 agents (intel · sentiment · quant · options · pm · risk ·         │
│             execution · review · model-improvement)                    │
│        │                                                               │
│   allocator (wallet-aware sizing)   meta-learning (size policy)        │
│   pre-market · post-market · monitoring · 24/7 scheduler              │
├──────────────────────────────────────────────────────────────────────┤
│  EXECUTION       agents/ (TradeManager · Wallet · Brokerage · Position)│
├──────────────────────────────────────────────────────────────────────┤
│  ENGINES         models/ · pipelines/ (ranker · ensemble · regime ·   │
│                  breadth · sector · options chain · strategy selector) │
├──────────────────────────────────────────────────────────────────────┤
│  BACKBONE        infra/ (feature-store · model-registry · drift ·     │
│                  data-quality · retrain · backtests)                   │
├──────────────────────────────────────────────────────────────────────┤
│  MEMORY & DATA   Trade Memory (SQLite) · feature store · model        │
│                  registry · paper ledger · collectors (bars, chain)    │
├──────────────────────────────────────────────────────────────────────┤
│  LLM             Ollama (qwen2.5:3b) — reasoning / prose ONLY          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. The nine agents (`aos/agents.py`)

| Agent | Wraps (quant source) | Output |
|---|---|---|
| Market Intelligence | regime classifier + breadth + sector RS | regime/breadth/leaders, vote |
| News & Sentiment | *(no feed — honest stub)* | neutral, flagged `no_news_feed` |
| Quant Decision | screener / ensemble ranker | equity candidate + conviction |
| Options Strategy | options dashboard + strategy selector | structure + prob_up |
| Portfolio Manager | risk policy + wallet | sized proposal |
| Risk Officer | risk policy + heat + earnings | **VETO power** |
| Trade Execution | paper TradeManager | fill event |
| Trade Review | Trade Memory (closed trades) | error stats |
| Model Improvement | drift monitor | retrain recommendation |

Each returns an `AgentReport{vote, confidence, evidence, flags, rationale}`.
`analyze()` produces vote/confidence/evidence from engines; `narrate()` adds
LLM prose to `rationale` only (failure-safe, cannot alter numbers).

### Agent workflow (one deliberation)
```
1 INTELLIGENCE  market_intel, news_sentiment, quant, options   → shared ctx
2 PROPOSAL      portfolio_mgr sizes the candidate               → ctx.proposal
3 RISK REVIEW   risk_officer enforces limits                    → may VETO
4 EXECUTION     execution opens the trade  (only if not vetoed)
   ↳ debate: orchestrator detects conflicts (e.g. quant BUY vs risk-off regime)
   ↳ conviction adjusted by conflicts/regime (quantitatively, not by LLM)
   ↳ everything logged to Trade Memory
```

---

## 3. Database schema (Trade Memory — `data/aos/memory.db`, SQLite/WAL)

```sql
regime_snapshots(id, ts, regime, breadth_score, vol_pctile, sector_top, extra)
signals(id, ts, symbol, asset, source, score, confidence, regime, sentiment,
        snapshot, outcome_ret, outcome_label, outcome_ts)
decisions(id, ts, symbol, asset, proposed_action, final_action, conviction,
          regime, vetoed, veto_reason, evidence)
agent_reports(id, decision_id→decisions, agent, role, vote, confidence,
              evidence, flags, rationale)
trades(id, decision_id→decisions, symbol, segment, side, entry, qty, stop,
       targets, status, net_pnl, fees, exit_reason, opened_at, closed_at)
lessons(id, ts, category, text, evidence)
```
`meta_dataset()` = signals ⋈ outcomes — the meta-learning training table.
Other persistent stores: feature store (`data/feature_store/`), model registry
(`models/registry/registry.json`), paper ledger (`data/agents/state.json`),
API cache (`data/api_cache/`), collectors (`data/intraday/`, `data/option_chain/`).

---

## 4. API surface (`api/server.py`)

| Method | Path | Purpose |
|---|---|---|
| POST | `/query` | NL command → `{answer, answer_raw, data, intent}` |
| GET | `/options/{symbol}` | options dashboard |
| GET | `/book` | constructed portfolio book |
| GET | `/screen` | swing shortlist |
| GET | `/health` | liveness |

Planned AOS endpoints (thin wrappers over `aos/`):
`POST /aos/deliberate`, `GET /aos/decisions`, `GET /aos/lessons`,
`GET /aos/health` (monitoring), `GET /aos/premarket`, `GET /agents/status`.

---

## 5. Background jobs (`aos/scheduler.py`)

| Job | Schedule | Action |
|---|---|---|
| premarket | daily 08:30 (wkday) | global+regime+breadth briefing |
| deliberate | every 15m, market hrs | committee considers a new trade |
| manage | every 2m, market hrs | tick open positions (stops/targets) |
| monitor | hourly | health dashboard, log WARN/ALERT |
| postmarket | daily 16:00 (wkday) | self-review → lessons |
| metalearn | daily 18:00 | re-learn size policy |
| retrain | weekly Sat 07:00 | walk-forward retrain (promote-if-better) |
| collectors | 5m, market hrs | intraday bars + option chain |
| precompute | 5m, market hrs | warm API cache |

Recovery: `last_run.json` + catch-up on missed daily jobs; per-job try/except;
idempotent jobs; persistent state ⇒ restart resumes mid-session.

---

## 6. Memory & learning systems

- **Trade Memory** (§3) — every signal/decision/vote/trade/lesson.
- **Meta-Learning** (`aos/meta_learning.py`) — learns signal×regime edge,
  confidence reliability per regime, loss causes → `meta_policy.json` →
  `sizing_multiplier()` consumed by the allocator.
- **Feature store / model registry** — reusable features + versioned models
  with promote/rollback.
- **Monitoring** (`aos/monitoring.py`) — drift + performance + data-quality +
  **agent performance scoring** (down-weight chronically-wrong agents).

---

## 7. Deployment structure

```
trading-ai/
  api/         FastAPI server + router + narrate + precompute
  aos/         agents, orchestrator, memory, premarket, postmarket,
               meta_learning, monitoring, allocator, scheduler
  agents/      TradeManager, Wallet, Brokerage, Position, auto_trader
  models/ pipelines/ infra/   engines + backbone
  data/        memory.db, feature_store, option_chain, intraday, agents, aos
  models/registry/             versioned models
  logs/aos/                    structured logs
  scripts/                     *.bat wrappers (Windows Task Scheduler)
```

Processes (3 long-running):
1. **API** — `uvicorn api.server:app` (stateless, horizontally scalable).
2. **Scheduler** — `python aos/scheduler.py run-forever` (single instance).
3. **Ollama** — local LLM for prose (optional; degrades gracefully).

---

## 8. Docker services (`docker-compose.yml`)

- `ollama` — LLM model server (qwen2.5:3b), volume-cached.
- `api` — FastAPI; depends on shared `./data` + `./models` volumes.
- `scheduler` — the 24/7 runner (premarket → trade loop → postmarket → learn).

All three share the project volume so Trade Memory, model registry and the
paper ledger are consistent. SQLite-on-a-volume is fine for a single node; the
scalability roadmap moves it to Postgres for multi-node.

---

## 9. Source-code implementation plan / status

| # | Component | Module | Status |
|---|---|---|---|
| 1 | Agent base | `aos/base.py` | ✅ |
| 2 | Trade Memory | `aos/memory.py` | ✅ |
| 3 | 9 agents | `aos/agents.py` | ✅ |
| 4 | Orchestration (debate+veto) | `aos/orchestrator.py` | ✅ |
| 5 | Pre-market briefing | `aos/premarket.py` | ✅ |
| 6 | Post-market self-review | `aos/postmarket.py` | ✅ |
| 7 | Meta-learning | `aos/meta_learning.py` | ✅ |
| 8 | Monitoring + agent scoring | `aos/monitoring.py` | ✅ |
| 9 | Capital allocator | `aos/allocator.py` | ✅ |
| 10 | 24/7 scheduler + recovery | `aos/scheduler.py` | ✅ |
| 11 | Architecture + Docker | `docs/`, `Dockerfile`, compose | ✅ |
| — | News/sentiment feed | (needs data source) | ⏳ honest stub |
| — | AOS REST endpoints | `api/server.py` | ⏳ next |
| — | Broker executor (live) | `agents/` interface | 🚫 out of scope (paper only) |

---

## 10. Scalability roadmap

**Now (single node, paper):** SQLite + file stores, 3 processes, local LLM.

1. **Data layer** → Postgres (decisions/trades/signals) + TimescaleDB for
   bars/chain; keep feature store in Parquet on object storage (S3/MinIO).
2. **Queue / event-driven** → Redis or NATS: collectors publish ticks, the
   `manage`/`deliberate` jobs become consumers (true event-driven vs polling).
3. **API** → horizontal scale behind nginx; move the precompute cache to Redis.
4. **Scheduler** → a proper orchestrator (Celery beat / Temporal / Airflow) for
   distributed, observable jobs with retries.
5. **LLM** → a GPU node (or hosted free-tier) so prose is instant; batch
   narration.
6. **Universe / multi-strategy** → shard the ranker by sector; run multiple
   strategy "books" each with its own allocator slice and risk budget.
7. **Observability** → Prometheus + Grafana on the monitoring metrics; alert on
   drift/perf/agent-score degradation.
8. **Backtest/meta-learning at scale** → nightly Spark/Dask job over the full
   Trade Memory to refresh the size policy and agent weights.

**Goal preserved at every step:** statistical rigor (quant produces numbers),
strict risk control (Risk Officer veto + allocator caps + circuit-breaker),
learning from experience (Trade Memory → meta-learning), paper-only safety.
```

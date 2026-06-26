# Trading-AI — Project Overview

## What It Does

Automated Indian stock market trading system that combines ML models with institutional-grade options strategies. Runs on paper capital of ₹10,00,000 with two independent income streams:

### Stream 1: OI Wall Selling (Primary — 71-89% Win Rate)
Sells options at strikes where heavy Open Interest creates "walls" — levels institutional sellers defend. An XGBoost model predicts the probability each wall holds. Each trade is scored 0-100 and sized by tier:

| Score | Tier | Sizing | Action |
|-------|------|--------|--------|
| ≥ 55 | 1 | Full lot | Trade with full conviction |
| 35-54 | 2 | Half lot | Trade with reduced size |
| 20-34 | 3 | Quarter | Small exploratory trade |
| < 20 | — | Skip | No trade |

**Backtested Results (2021-2026)**:
- NIFTY: 83.5% win, Profit Factor 6.05, Sharpe 9.51
- BANKNIFTY: 87.5% win, Profit Factor 16.58, Sharpe 14.95

### Stream 2: Intraday Direction ML (86.9% Win Rate)
XGBoost model trained on 500+ NSE stocks predicts intraday direction. Top 5 highest-confidence picks bought at open, exited at close.

## How It Works

```
Market Opens (09:15 IST)
    ↓
market_scheduler runs at 09:20
    ↓
auto_trader.place_best_trades()
    ├── fetch_chain(NIFTY) → live NSE option chain
    ├── fetch_chain(BANKNIFTY) → live NSE option chain
    ├── simple_signal() → wall selling signals with ML P(hold)
    ├── _score_wall_trade() → smart score 0-100
    ├── Select best 2 wall trades (1 per index) + 1 stock trade
    └── Record in DB with entry, SL, target, score, tier
    ↓
tick_user() runs on /me/wallet requests
    ├── _option_ltp() → live premium from NSE chain
    ├── _stock_price() → live price from yfinance
    ├── Update pnl_series with [time, ltp, gross_pnl]
    └── Auto-exit on SL/target hit
    ↓
market_scheduler runs at 15:20
    ↓
auto_trader.close_trades() → square off all, record P&L
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, FastAPI, uvicorn |
| ML | XGBoost, scikit-learn, 1300+ trained models |
| Data | jugaad_data (NSE live), yfinance (stocks), pandas |
| Database | SQLite (trading_memory.db) |
| Frontend | React 18, Vite, lightweight-charts |
| Hosting | Oracle Cloud 1GB (backend), Vercel (frontend) |
| Proxy | nginx with Let's Encrypt SSL |
| Auth | JWT tokens |

## API Endpoints (Key)

| Endpoint | Purpose |
|----------|---------|
| `GET /me/wallet` | Wallet status + open positions with live P&L |
| `GET /phase2/auto/dashboard` | Auto-trader account, trades, daily P&L |
| `GET /phase2/auto/wall-signals` | Live scored wall selling signals |
| `POST /phase2/auto/trade` | Place highest-probability trades |
| `POST /phase2/auto/close` | Close all open positions |
| `GET /options/{symbol}` | Full options chain analysis |
| `GET /candles/{symbol}` | OHLCV candlestick data |
| `GET /phase2/regime` | Market regime classification |
| `GET /phase2/psychology` | Psychology gate status |

## ML Models

| Model | Location | Purpose | Win Rate |
|-------|----------|---------|----------|
| OI Wall 1d | `models/oi_wall_selling/` | Will wall hold for 1 day? | NIFTY 92.7%, BN 88.8% |
| OI Wall 2d/3d/5d | `models/oi_wall_selling/` | Multi-day wall holding | 75-89% |
| Intraday Direction | `models/intraday/latest_direction.pkl` | Stock up or down today? | 86.9% |
| Intraday Range | `models/intraday/` | Premium range prediction | — |
| Strike Ranker | `models/intraday/latest_strike_ranker.pkl` | Best strike selection | — |
| Index Direction | `models/index_direction/` | NIFTY/BN direction | — |
| Per-Stock | `models/{SYMBOL}_rf.pkl` | 436 stocks × 3 models | Varies |

## Phase 2 Intelligence Layer

Built on top of Phase 1 pipelines (which remain untouched):

| Engine | Role |
|--------|------|
| `psychology_engine.py` | THE GATE — blocks trading after losses/revenge/overtrade |
| `regime_v2.py` | Market day-type detection (trend/range/panic/etc) |
| `conviction_v2.py` | Grade signals A+/A/B/C/NO_TRADE |
| `trade_quality.py` | Score trade setups 0-100 |
| `capital_allocation.py` | Quarter-Kelly position sizing |
| `explainability.py` | "Why This Trade?" in plain English |
| `recommendation_engine.py` | Final orchestrator combining all engines |
| `paper_trading_v2.py` | Full trade simulation with slippage/brokerage |
| `reflection_v2.py` | Weekly/monthly self-learning reports |
| `quant_lab.py` | A/B testing, stress testing, calibration |

# CLAUDE.md — Trading-AI

## Critical Rules

- **DO NOT redesign or rebuild existing modules** — the backend pipeline layer (65+ files in `pipelines/`) is mature and working. Only extend or fix.
- **DO NOT git add/commit/push** — the user handles all commits, pushes, and deployment manually.
- **DO NOT deploy on Oracle server** — the user deploys manually via SSH.
- **DO NOT create documentation files** unless explicitly asked.

## Project Overview

Indian stock market trading intelligence platform combining 1300+ ML models with rule-based options strategies. Two independent income streams:

1. **OI Wall Selling** on NIFTY + BANKNIFTY — sell options at heavy OI walls (71-89% win rate, PF 5.8-16.5)
2. **Intraday Direction ML** — buy top stocks at open, exit at close (86.9% win rate)

The system auto-trades with ₹10,00,000 paper capital, managed by the backend's market scheduler.

## Architecture

### Repos
- **Backend**: `C:\Users\KetanMohite\OneDrive - INTELLINUM\Desktop\git\allGit\trading-ai` (this repo)
- **Frontend**: `C:\Users\KetanMohite\OneDrive - INTELLINUM\Desktop\git\allGit\trading-ai-frontend` (React, deployed to Vercel)

### Server (Oracle Cloud — 1GB RAM)
- Host: `140.245.5.148` / `ketan-trading.duckdns.org`
- SSH key: `C:\Users\KetanMohite\Downloads\ssh-key-2026-06-23.key`
- Server directory: `~/trading-ml` (NOT `~/trading-ai`)
- User: `ubuntu`
- Environment: `AOS_PROFILE=micro` (skips heavy jobs on 1GB server)
- API: uvicorn on port 8000, 1 worker, behind nginx
- Frontend: https://trading-ai-frontend-iota.vercel.app/

### Backend Structure
```
trading-ai/
├── api/                    # FastAPI server
│   ├── server.py           # Main app, mounts routers
│   ├── cache.py            # TTL cache (300s default)
│   ├── auth.py             # JWT auth
│   ├── market.py           # Market data helpers
│   └── routers/
│       ├── public_routes.py    # /health, /options, /candles, /recommendation
│       ├── user_routes.py      # /me/wallet, /me/trade, /me/mode
│       ├── portfolio_routes.py # /portfolio/brief, /portfolio/risk
│       ├── phase2_routes.py    # /phase2/* (regime, psychology, auto-trader)
│       ├── auth_routes.py      # /auth/login, /auth/register
│       ├── live_routes.py      # WebSocket live stream
│       └── admin_routes.py     # Admin endpoints
├── engines/                # Phase 2 — higher-order intelligence (consumes pipelines)
│   ├── auto_trader.py          # Auto-trading engine (wall selling + ML stock picks)
│   ├── market_scheduler.py     # Background loop: auto-trade at 09:20, close at 15:20
│   ├── psychology_engine.py    # Trading discipline gate
│   ├── regime_v2.py            # Market regime / day-type detection
│   ├── conviction_v2.py        # A+/A/B/C/NO_TRADE grading
│   ├── trade_quality.py        # Setup scoring (0-100)
│   ├── capital_allocation.py   # Quarter-Kelly sizing
│   ├── explainability.py       # "Why This Trade?" explanations
│   ├── recommendation_engine.py # Final orchestrator
│   ├── paper_trading_v2.py     # Simulation with full lifecycle
│   ├── reflection_v2.py        # Self-learning journal
│   ├── quant_lab.py            # Research / A-B testing
│   ├── strike_ranker.py        # ML strike ranking model
│   ├── intraday_trainer.py     # Train intraday direction/range models
│   └── intraday_inference.py   # Run inference on trained models
├── pipelines/              # Phase 1 — 44 signal generation modules (DO NOT MODIFY unless fixing bugs)
│   ├── options_action_engine.py  # Core: simple_signal(), wall_selling_plan()
│   ├── options/
│   │   ├── chain_live_intel.py   # Live NSE chain fetch (fetch_chain with 3min cache)
│   │   ├── options_dashboard.py  # Options analysis dashboard
│   │   └── ...13 files
│   ├── combined_intelligence.py
│   ├── institutional_engine.py
│   ├── screener.py
│   └── ...
├── aos/                    # Autonomous Operating System
│   ├── user_wallet.py          # Multi-user paper wallet (tick_user for live P&L)
│   ├── sim_wallet.py           # Single autonomous wallet
│   └── scheduler.py            # Legacy scheduler
├── models/                 # Trained ML models (1300+ .pkl files)
│   ├── oi_wall_selling/        # NIFTY/BANKNIFTY wall holding prediction (XGBoost)
│   ├── intraday/               # Direction + range + strike ranker models
│   ├── index_direction/        # Index direction models
│   └── {SYMBOL}_rf.pkl etc     # Per-stock RF/XGBoost/LabelEncoder
├── agents/                 # Trading agents
│   ├── auto_trader.py          # Agent-level auto trader (_stock_price with cache)
│   └── brokerage.py            # Brokerage fee calculator
├── broker/                 # Broker integration (AngelOne)
│   └── executor.py
├── memory/                 # SQLite databases
│   └── trading_memory.db       # All state: wallets, trades, psychology, journal
├── deploy/                 # Deployment configs
│   ├── nginx.conf              # Reverse proxy (timeout 120s)
│   └── trading-ai-api.service  # systemd unit
└── training/               # Training scripts and data
```

### Frontend Structure
```
trading-ai-frontend/src/
├── App.jsx                 # Main app, nav, hooks (useWallet, useCandles)
├── CandleChart.jsx         # Lightweight candlestick chart
├── api.js                  # API client (apiGet, apiPost, JWT auth)
└── components/
    ├── DailyBrief.jsx          # Home page — wall signals, auto-trader summary, risk
    ├── AutoTrader.jsx          # Auto Trader tab — full ML mode dashboard
    ├── Portfolio.jsx           # Positions, wallet, deposit, trade modes
    ├── OptionsHub.jsx          # Options analysis (NIFTY/BANKNIFTY chain)
    ├── EquityHub.jsx           # Equity screener + analysis
    ├── SwingHub.jsx            # Swing trading ideas
    ├── Performance.jsx         # Performance analytics
    ├── RiskDashboard.jsx       # Risk metrics
    ├── Alerts.jsx              # Alert management
    ├── RegimeStrip.jsx         # Market regime status bar
    └── TradeExplainer.jsx      # Trade explanation modal
```

## Key Technical Details

### OI Wall Selling Strategy
- Sells options at strikes with heavy Open Interest (institutional "walls")
- ML model (`models/oi_wall_selling/`) predicts P(wall holds) using XGBoost
- **Smart Score (0-100)**: distance (0-30), premium (0-25), day-of-week (-15 to +20), wall type (5-10), OI building (0-5)
- **Tiered Sizing**: Score≥55 = full lot, 35-54 = half, 20-34 = quarter
- Backtested 2021-2026: NIFTY 83.5% win PF 6.05, BANKNIFTY 87.5% win PF 16.58

### Core Functions
- `pipelines/options_action_engine.py:simple_signal(symbol, capital)` — returns scored wall selling signals
- `pipelines/options/chain_live_intel.py:fetch_chain(symbol)` — live NSE option chain (3min cache)
- `engines/auto_trader.py:place_best_trades()` — places highest-probability trades
- `engines/auto_trader.py:close_trades()` — EOD position close
- `engines/market_scheduler.py:background_loop()` — auto-trade at 09:20, close at 15:20
- `aos/user_wallet.py:tick_user(uid)` — updates open positions with live LTP + P&L
- `aos/user_wallet.py:status(uid)` — full wallet status for frontend

### Auto-Trader DB Tables (in trading_memory.db)
- `auto_trader_account` — capital, deployed, P&L, win/loss counts
- `auto_trader_trades` — individual trade records with score, tier, entry/exit
- `auto_trader_daily` — daily P&L aggregates

### API Performance
- All cache TTLs set to 300s (5min) to prevent 504s on 1GB server
- `fetch_chain()` has 3-min in-memory cache — multiple endpoints share one NSE call
- nginx proxy_read_timeout = 120s
- Frontend polls every 60-120s (not 15-30s)
- Single uvicorn worker — NSE calls block, so caching is critical

### Live Price Flow
- `/me/wallet` calls `status(uid, do_tick=True)` → `tick_user()` → `_live_price()`
- Options LTP: `_option_ltp()` → `fetch_chain()` → NSE live chain
- Equity LTP: `_stock_price()` → yfinance (cached)
- P&L tracked in `pnl_series` array: `[[time, price, gross_pnl], ...]`
- Frontend reads `pnl_series[-1]` for current LTP and P&L

## Common Tasks

### Deploy to Oracle Server
```bash
ssh -i C:\Users\KetanMohite\Downloads\ssh-key-2026-06-23.key ubuntu@140.245.5.148
cd ~/trading-ml
git pull
sudo systemctl restart trading-ai-api
# If nginx.conf changed:
sudo cp deploy/nginx.conf /etc/nginx/sites-available/trading-ai
sudo nginx -t && sudo systemctl reload nginx
```

### Run Locally
```bash
# Backend
python -m uvicorn api.server:app --reload --port 8000

# Frontend (separate repo)
cd trading-ai-frontend
npm run dev
```

### Test Wall Selling Signals
```bash
python engines/auto_trader.py signals   # Generate today's signals
python engines/auto_trader.py trade     # Place best trades
python engines/auto_trader.py close     # Close all open
python engines/auto_trader.py dashboard # Full account stats
```

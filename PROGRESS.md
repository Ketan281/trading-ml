# PROGRESS.md — Trading-AI

## Completed

### Phase 1 — Signal Generation (pre-existing)
- 44 pipeline modules for technical analysis, options flow, breadth, screener
- 1300+ per-stock ML models (RF + XGBoost + LabelEncoder)
- NSE live option chain integration (jugaad_data)
- yfinance intraday data
- FastAPI server with auth, WebSocket, 15+ endpoints
- React frontend with 11 views

### Phase 2 — Trading Intelligence (all complete)
- **2A Foundation**: DB schema migration, regime_v2, trade_quality, psychology_engine
- **2B Grading**: conviction_v2, capital_allocation, explainability
- **2C Output**: recommendation_engine, paper_trading_v2
- **2D Learning**: reflection_v2, quant_lab
- **2E Wiring**: phase2_routes.py (all /phase2/* endpoints), frontend components

### ML Model Training (complete)
- **OI Wall Selling**: XGBoost models for NIFTY + BANKNIFTY, 1d/2d/3d/5d horizons
  - NIFTY 1d: 92.7% win, 80.2% participation
  - BANKNIFTY 1d: 88.8% win, 87.4% participation
- **Intraday Direction**: 86.9% win rate on stock direction prediction
- **Intraday Range**: Options premium range prediction
- **Strike Ranker**: ML-based strike selection

### Auto-Trader System (complete)
- `engines/auto_trader.py` — full auto-trading engine with smart scoring
- `engines/market_scheduler.py` — auto-trade at 09:20, auto-close at 15:20
- DB tables: auto_trader_account, auto_trader_trades, auto_trader_daily
- Paper capital: ₹10,00,000
- API routes: /phase2/auto/dashboard, /phase2/auto/wall-signals, /phase2/auto/trade, /phase2/auto/close
- Frontend: AutoTrader.jsx (full dashboard), DailyBrief.jsx (home page signals)

### Performance Fixes (complete)
- API cache TTL: 60s → 300s
- `fetch_chain()`: added 3-min in-memory cache (prevents duplicate NSE calls)
- nginx proxy_read_timeout: 60s → 120s
- Frontend polling: 15-30s → 60-120s
- Wall-signals endpoint cached 300s

### Live Price Fix (complete)
- `/me/wallet` changed from `do_tick=False` to `do_tick=True`
- Open positions now update with live LTP from NSE (options) and yfinance (equity)
- P&L updates in real-time via `pnl_series`

### Wallet Fixes (complete)
- Default balance: ₹10,000 → ₹10,00,000
- Deposit cap: ₹1,00,000 → ₹1,00,00,000
- Reset endpoint: POST /me/wallet/reset
- ML mode persistence via localStorage

---

## Uncommitted Changes (needs push)

### Backend (`trading-ai`)
- `api/routers/user_routes.py` — `do_tick=False` → `do_tick=True` (live P&L for open positions)

### Frontend (`trading-ai-frontend`)
- All changes committed and pushed

---

## Pending / Known Issues

### Must Do (before next trading day)
1. **Push backend** `user_routes.py` change to Oracle — without this, positions show ₹0 P&L
2. **Update nginx** on server — copy `deploy/nginx.conf`, reload nginx
3. **Restart API** on server after git pull

### Known Bugs
1. **3 open BUY positions** (NIFTY 24000 CE, BANKNIFTY 58200 CE, IOC) — these are from the OLD buy-side system, not the trained wall selling model. They should be squared off manually since they were placed by the old logic.
2. **Chart not loading after market hours** — `useCandles` fetches 5m/1d from yfinance which returns empty when market is closed. This is expected behavior, not a bug.
3. **Mega backtest unicode error** — `UnicodeEncodeError` on `★` character in final ranking print. Data is fine, just the console output fails. Non-critical.

### Future Improvements
1. **Live chart during market hours** — verify candlestick chart renders when market opens
2. **WebSocket for live prices** — instead of polling /me/wallet every 60s, push price updates via WebSocket
3. **Broker integration** — AngelOne API for real live trading (broker/executor.py scaffold exists)
4. **Multi-worker uvicorn** — currently 1 worker on 1GB server; upgrade to 2GB + 2 workers would eliminate most 504s
5. **Background price ticker** — run tick_user in scheduler instead of on API request to avoid blocking
6. **Precompute wall signals** — scheduler could cache wall signals every 3min instead of computing on API hit

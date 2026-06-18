# Graph Report - .  (2026-06-18)

## Corpus Check
- Large corpus: 192 files · ~18,794,533 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 1180 nodes · 2461 edges · 75 communities (67 shown, 8 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 65 edges (avg confidence: 0.67)
- Token cost: 41,248 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Capital Allocation & Trade Memory|Capital Allocation & Trade Memory]]
- [[_COMMUNITY_Memory Store & ML Models|Memory Store & ML Models]]
- [[_COMMUNITY_Decision Engine & Calibration|Decision Engine & Calibration]]
- [[_COMMUNITY_API Precompute & Router|API Precompute & Router]]
- [[_COMMUNITY_Autonomous Paper Wallet|Autonomous Paper Wallet]]
- [[_COMMUNITY_Nine-Agent Committee|Nine-Agent Committee]]
- [[_COMMUNITY_Futures & Greeks Analysis|Futures & Greeks Analysis]]
- [[_COMMUNITY_Brokerage & Trade Manager|Brokerage & Trade Manager]]
- [[_COMMUNITY_Agentic OS Architecture (docs)|Agentic OS Architecture (docs)]]
- [[_COMMUNITY_Market Data & Intraday Signals|Market Data & Intraday Signals]]
- [[_COMMUNITY_Frontend Web App|Frontend Web App]]
- [[_COMMUNITY_Terminal Dashboard|Terminal Dashboard]]
- [[_COMMUNITY_Index Options Model|Index Options Model]]
- [[_COMMUNITY_Options Strategy Selector|Options Strategy Selector]]
- [[_COMMUNITY_Point-in-Time Fundamentals|Point-in-Time Fundamentals]]
- [[_COMMUNITY_Memory Retrieval & Similarity|Memory Retrieval & Similarity]]
- [[_COMMUNITY_Execution Simulator|Execution Simulator]]
- [[_COMMUNITY_Backtester & Walk-Forward|Backtester & Walk-Forward]]
- [[_COMMUNITY_Options Action Engine|Options Action Engine]]
- [[_COMMUNITY_Auth & User Store|Auth & User Store]]
- [[_COMMUNITY_Event Awareness & Narration|Event Awareness & Narration]]
- [[_COMMUNITY_Label Building (training)|Label Building (training)]]
- [[_COMMUNITY_Dealer Exposure & Order Flow|Dealer Exposure & Order Flow]]
- [[_COMMUNITY_Portfolio Risk Limits|Portfolio Risk Limits]]
- [[_COMMUNITY_Explainability & Portfolio Book|Explainability & Portfolio Book]]
- [[_COMMUNITY_Sector Strength & Optimizer|Sector Strength & Optimizer]]
- [[_COMMUNITY_Hallucination Control|Hallucination Control]]
- [[_COMMUNITY_Model Evaluator|Model Evaluator]]
- [[_COMMUNITY_Feature Building (training)|Feature Building (training)]]
- [[_COMMUNITY_Data Quality & Retrain|Data Quality & Retrain]]
- [[_COMMUNITY_Model Registry|Model Registry]]
- [[_COMMUNITY_Backtest & Market Regime|Backtest & Market Regime]]
- [[_COMMUNITY_Advanced Options Reads|Advanced Options Reads]]
- [[_COMMUNITY_Live Chain Intelligence|Live Chain Intelligence]]
- [[_COMMUNITY_Auto-Trader Loop|Auto-Trader Loop]]
- [[_COMMUNITY_Feature Store|Feature Store]]
- [[_COMMUNITY_Ensemble Meta-Model|Ensemble Meta-Model]]
- [[_COMMUNITY_Regime Classifier & Breadth|Regime Classifier & Breadth]]
- [[_COMMUNITY_PIT Fundamentals Fetch|PIT Fundamentals Fetch]]
- [[_COMMUNITY_AngelOne Broker Connect|AngelOne Broker Connect]]
- [[_COMMUNITY_Learning-to-Rank Ranker|Learning-to-Rank Ranker]]
- [[_COMMUNITY_Fundamental Quality Scoring|Fundamental Quality Scoring]]
- [[_COMMUNITY_Paper-Trading Environment|Paper-Trading Environment]]
- [[_COMMUNITY_Walk-Forward Dataset Builder|Walk-Forward Dataset Builder]]
- [[_COMMUNITY_Dataset Validation|Dataset Validation]]
- [[_COMMUNITY_Pre-Market Briefing|Pre-Market Briefing]]
- [[_COMMUNITY_Probability Calibration|Probability Calibration]]
- [[_COMMUNITY_Earnings-Event Risk|Earnings-Event Risk]]
- [[_COMMUNITY_Regime Detector|Regime Detector]]
- [[_COMMUNITY_Risk Policy & Drawdown Guard|Risk Policy & Drawdown Guard]]
- [[_COMMUNITY_Option-Chain Collector|Option-Chain Collector]]
- [[_COMMUNITY_Deep Ranker (MLP)|Deep Ranker (MLP)]]
- [[_COMMUNITY_Slippage Estimator|Slippage Estimator]]
- [[_COMMUNITY_Fundamentals Fetch|Fundamentals Fetch]]
- [[_COMMUNITY_IV Surface|IV Surface]]
- [[_COMMUNITY_Dynamic Stops|Dynamic Stops]]
- [[_COMMUNITY_Colab Prep|Colab Prep]]
- [[_COMMUNITY_Drift Monitor|Drift Monitor]]
- [[_COMMUNITY_Performance Attribution|Performance Attribution]]
- [[_COMMUNITY_Monte Carlo Risk|Monte Carlo Risk]]
- [[_COMMUNITY_Universe Fetch|Universe Fetch]]
- [[_COMMUNITY_Historical Data Fetch|Historical Data Fetch]]
- [[_COMMUNITY_Small cluster 65|Small cluster 65]]
- [[_COMMUNITY_Small cluster 66|Small cluster 66]]
- [[_COMMUNITY_Small cluster 67|Small cluster 67]]
- [[_COMMUNITY_Small cluster 68|Small cluster 68]]
- [[_COMMUNITY_Small cluster 69|Small cluster 69]]
- [[_COMMUNITY_Small cluster 72|Small cluster 72]]
- [[_COMMUNITY_Small cluster 73|Small cluster 73]]

## God Nodes (most connected - your core abstractions)
1. `load_prices()` - 44 edges
2. `$()` - 35 edges
3. `AgentReport` - 24 edges
4. `fetch_chain()` - 23 edges
5. `dashboard()` - 20 edges
6. `TradeManager` - 19 edges
7. `Orchestrator` - 18 edges
8. `_silent()` - 18 edges
9. `run_dashboard()` - 18 edges
10. `live_trade_plan()` - 17 edges

## Surprising Connections (you probably didn't know these)
- `Orchestrator` --uses--> `TradeManager`  [INFERRED]
  aos/orchestrator.py → agents/manager.py
- `job_retrain()` --calls--> `run()`  [EXTRACTED]
  aos/scheduler.py → infra/retrain_pipeline.py
- `auto_open_swing()` --calls--> `cached_screen()`  [EXTRACTED]
  agents/auto_trader.py → api/precompute.py
- `auto_open_option()` --calls--> `fetch_chain()`  [EXTRACTED]
  agents/auto_trader.py → pipelines/options/chain_live_intel.py
- `auto_open_option()` --calls--> `chain_prob_up()`  [EXTRACTED]
  agents/auto_trader.py → pipelines/options_action_engine.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **The Nine Deliberation Agents** — docs_agentic_os_market_intelligence_agent, docs_agentic_os_news_sentiment_agent, docs_agentic_os_quant_decision_agent, docs_agentic_os_options_strategy_agent, docs_agentic_os_portfolio_manager_agent, docs_agentic_os_risk_officer_agent, docs_agentic_os_trade_execution_agent, docs_agentic_os_trade_review_agent, docs_agentic_os_model_improvement_agent [EXTRACTED 1.00]
- **One Deliberation Flow** — docs_agentic_os_quant_decision_agent, docs_agentic_os_portfolio_manager_agent, docs_agentic_os_risk_officer_agent, docs_agentic_os_trade_execution_agent, docs_agentic_os_trade_memory [EXTRACTED 1.00]
- **Deployment Stack** — docs_deploy_oracle_always_free, docs_deploy_nginx_reverse_proxy, docs_deploy_letsencrypt_ssl, docs_deploy_cors_lockdown [EXTRACTED 1.00]

## Communities (75 total, 8 thin omitted)

### Community 0 - "Capital Allocation & Trade Memory"
Cohesion: 0.06
Nodes (61): account_tier(), allocate(), Wallet-aware capital allocator — how many rupees actually go on each trade.  Com, candidate needs entry + stop (equity) or entry premium (options)., close_trade(), connect(), init_db(), _j() (+53 more)

### Community 1 - "Memory Store & ML Models"
Cohesion: 0.07
Nodes (51): init_db(), print_summary(), save_analysis(), save_outcome(), compute_features(), create_labels(), _make_rf(), _make_xgb() (+43 more)

### Community 2 - "Decision Engine & Calibration"
Cohesion: 0.06
Nodes (50): calibrate_scores(), Map raw ranker probabilities → calibrated probabilities. Falls back to     the r, _build_reasoning(), decide(), _decided_by(), _market_condition(), _position_size(), Deterministic trading decision engine.  This module makes the actual trade decis (+42 more)

### Community 3 - "API Precompute & Router"
Cohesion: 0.07
Nodes (51): cached(), cached_book(), cached_dashboard(), cached_screen(), load_fresh(), _path(), Pre-compute cache — make API responses instant.  The engines hit live NSE + mode, run_precompute() (+43 more)

### Community 4 - "Autonomous Paper Wallet"
Cohesion: 0.11
Nodes (46): Robust last price: fast_info, else latest 1-day close (works even when     the m, _stock_price(), _analysis(), _close(), deposit(), _fresh(), get_wallet(), _live_price() (+38 more)

### Community 5 - "Nine-Agent Committee"
Cohesion: 0.11
Nodes (22): MarketIntelligenceAgent, ModelImprovementAgent, NewsSentimentAgent, OptionsStrategyAgent, PortfolioManagerAgent, QuantDecisionAgent, The nine agents — each wraps a REAL quantitative engine and returns an AgentRepo, RiskOfficerAgent (+14 more)

### Community 6 - "Futures & Greeks Analysis"
Cohesion: 0.09
Nodes (39): analyze_rollover(), calculate_basis(), full_futures_analysis(), futures_oi_analysis(), get_futures_data(), get_futures_live_fno(), get_spot(), atm_greeks_summary() (+31 more)

### Community 7 - "Brokerage & Trade Manager"
Cohesion: 0.08
Nodes (17): charges(), Brokerage / cost agent — accurate Indian (NSE) trading charges.  Every entry and, side: 'buy' or 'sell'. price = per-share / per-premium-unit. qty = shares     (o, Total cost of a full in-and-out trade (buy then sell)., round_trip(), Trade manager — the orchestrator that coordinates the agents.  Wires together th, targets: list of (price, fraction). Charges entry fees, reserves capital., TradeManager (+9 more)

### Community 8 - "Agentic OS Architecture (docs)"
Cohesion: 0.07
Nodes (35): AgentReport, Agentic Trading OS, Capital Allocator, FastAPI Server (api/server.py), BrokerExecutor interface, docker-compose services, LLMs Reason Only Rule, Market Intelligence Agent (+27 more)

### Community 9 - "Market Data & Intraday Signals"
Cohesion: 0.12
Nodes (30): candles(), Market data helpers for the frontend: candlesticks + today's best recommendation, Return OHLCV candles for lightweight-charts: a list of     {time(epoch s, UTC),, Today's single best disciplined trade, as text + a submit-ready spec., recommendation(), _atr(), _detect(), _ema() (+22 more)

### Community 10 - "Frontend Web App"
Cohesion: 0.16
Nodes (33): $(), api(), bindCloseButtons(), boot(), cardEl(), doDeposit(), drawLevels(), fmt() (+25 more)

### Community 11 - "Terminal Dashboard"
Cohesion: 0.22
Nodes (28): bold(), box(), box_end(), box_row(), clear(), colored(), Colors, divider() (+20 more)

### Community 12 - "Index Options Model"
Cohesion: 0.14
Nodes (25): build_features(), build_labels(), evaluate(), load_index(), _mlp_predict(), Index options strategy model — NIFTY / BANKNIFTY (daily → options structure).  W, Fit on all history, save models, and emit the current weekly view +     options, Merge the two daily files we keep per index into the longest clean     daily ser (+17 more)

### Community 13 - "Options Strategy Selector"
Cohesion: 0.13
Nodes (23): build_structure(), expiry_profile(), _leg(), _net_greeks(), _net_premium(), _payoff(), print_recommendation(), Options strategy auto-selector — the "what do I actually put on" brain.  Combine (+15 more)

### Community 14 - "Point-in-Time Fundamentals"
Cohesion: 0.14
Nodes (20): _rank_ic(), Spearman (rank) correlation of prediction vs actual relative return,     compute, asof_join(), _available_date(), _load_pit(), pit_feature_panel(), Point-in-time fundamental FEATURES — roadmap #3 (feature layer).  Takes the raw, t vs ~4 quarters earlier, only if that earlier point is 270-460 days     back (i (+12 more)

### Community 15 - "Memory Retrieval & Similarity"
Cohesion: 0.16
Nodes (21): build_feature_vector(), build_memory_context(), combined_similarity(), cosine_similarity(), euclidean_similarity(), find_best_setups(), find_similar_conditions(), find_similar_failures() (+13 more)

### Community 16 - "Execution Simulator"
Cohesion: 0.15
Nodes (10): calculate_total_charges(), ExecutionEngine, Order, PartialFillModel, Simulates partial order fills based on     liquidity and market conditions., Models bid-ask spread based on     volatility and liquidity., Realistic slippage based on:     - Market conditions     - Order size     - T, simulate_day_execution() (+2 more)

### Community 17 - "Backtester & Walk-Forward"
Cohesion: 0.25
Nodes (15): apply_slippage(), calculate_brokerage(), calculate_metrics(), generate_signals(), grade_strategy(), Position, print_backtest_results(), run_all_strategies() (+7 more)

### Community 18 - "Options Action Engine"
Cohesion: 0.15
Nodes (17): greeks_for(), Compact, serialisable summary for embedding in a trade plan., structure_summary(), _atm_premiums(), build_trade(), chain_prob_up(), demo(), interim_chain_bias() (+9 more)

### Community 19 - "Auth & User Store"
Cohesion: 0.25
Nodes (15): _conn(), current_user(), decode_token(), get_user(), _hash_pw(), init_db(), list_users(), login() (+7 more)

### Community 20 - "Event Awareness & Narration"
Cohesion: 0.23
Nodes (13): apply_event_adjustments(), build_event_context(), check_pre_event_window(), check_results_season(), check_today_events(), check_upcoming_events(), apply_memory_adjustment(), build_narration_prompt() (+5 more)

### Community 21 - "Label Building (training)"
Cohesion: 0.18
Nodes (14): build_action_label(), build_all_labels(), build_confidence_score(), build_direction_label(), build_entry_label(), build_labels(), build_risk_label(), build_strategy_label() (+6 more)

### Community 22 - "Dealer Exposure & Order Flow"
Cohesion: 0.19
Nodes (13): pick_trade(), fetch_chain(), dealer_exposure(), _greeks(), Dealer gamma / vanna / charm exposure — the second-order positioning map.  Exten, Return gamma, vanna, charm (per-share, leg-agnostic where symmetric)., report(), chain_flow() (+5 more)

### Community 23 - "Portfolio Risk Limits"
Cohesion: 0.25
Nodes (9): calculate_position_size(), check_correlation(), check_daily_loss_limit(), check_portfolio_risk(), check_sector_concentration(), load_daily_pnl(), load_open_positions(), PortfolioState (+1 more)

### Community 24 - "Explainability & Portfolio Book"
Cohesion: 0.23
Nodes (12): _symbol_factors(), explain(), narrative(), _ranker_shap(), Explainable-AI layer — WHY was each name picked?  Every trade should come with a, One-paragraph plain-English reason from the structured explanation., Per-feature SHAP contributions from the saved ranker. SHAP is computed     over, Return a per-symbol explanation dict combining ranker SHAP + ensemble. (+4 more)

### Community 25 - "Sector Strength & Optimizer"
Cohesion: 0.21
Nodes (11): attach_sector_rs(), _blended_momentum(), load_industries(), Sector-relative strength — which SECTORS lead, and each stock's RS vs its peers., Return (sector_table, stock_table). sector_table ranks sectors by RS;     stock_, Merge sector RS onto a ranked dataframe (expects a 'symbol' column).     Adds: s, sector_strength(), optimize() (+3 more)

### Community 26 - "Hallucination Control"
Cohesion: 0.26
Nodes (12): batch_validate(), rule_bollinger_action(), rule_confidence_risk(), rule_macd_action(), rule_numeric_consistency(), rule_oi_contradiction(), rule_regime_contradiction(), rule_rsi_action() (+4 more)

### Community 27 - "Model Evaluator"
Cohesion: 0.27
Nodes (12): analyze_by_condition(), calculate_rolling_performance(), calculate_signal_quality(), check_confidence_calibration(), check_retraining_needed(), detect_performance_drift(), get_retrain_recommendation(), load_trade_history() (+4 more)

### Community 28 - "Feature Building (training)"
Cohesion: 0.29
Nodes (12): add_adx(), add_calendar_features(), add_iv_proxy_features(), add_momentum_features(), add_price_action_features(), add_regime_features(), add_trend_features(), add_volatility_features() (+4 more)

### Community 29 - "Data Quality & Retrain"
Cohesion: 0.30
Nodes (10): Data-quality validation + anomaly detection.  Garbage in, garbage out — a single, validate_prices(), validate_universe(), Hash a {symbol: df} price dict by symbol set + last dates + row counts —     cha, universe_hash(), Automated walk-forward retrain pipeline — the self-maintaining loop.  Chains the, run(), load_prices() (+2 more)

### Community 30 - "Model Registry"
Cohesion: 0.39
Nodes (11): best(), compare(), _find(), get(), list_versions(), _load_index(), promote(), Model registry + experiment tracking (lightweight — no MLflow needed).  Every ti (+3 more)

### Community 31 - "Backtest & Market Regime"
Cohesion: 0.30
Nodes (10): build_panel(), _metrics(), Full-system walk-forward backtest — the evidence layer (roadmap #1).  The ranker, run_backtest(), _turnover(), current_regime(), _load_index(), Market-regime overlay for the cross-sectional book — roadmap #5 (part A).  model (+2 more)

### Community 32 - "Advanced Options Reads"
Cohesion: 0.36
Nodes (10): advanced_read(), gamma_exposure(), iv_skew(), oi_iv_velocity(), pin_risk(), Advanced options reads — gamma regime, IV skew, OI/IV velocity, pin risk.  These, bs_greeks(), Compute delta/theta/gamma/vega from spot, strike, days-to-expiry and     IV%, be (+2 more)

### Community 33 - "Live Chain Intelligence"
Cohesion: 0.26
Nodes (11): analyze(), buildup(), expected_range(), liquidity(), oi_walls(), _prev_close(), _quad(), Options live-intelligence — NIFTY / BANKNIFTY (works on TODAY's chain).  Everyth (+3 more)

### Community 34 - "Auto-Trader Loop"
Cohesion: 0.35
Nodes (10): auto_open_option(), auto_open_swing(), _held(), live_prices(), market_open(), monitor_once(), _num(), Auto-trader — wire the signals to the agents and tick them on live prices.  Clos (+2 more)

### Community 35 - "Feature Store"
Cohesion: 0.33
Nodes (10): cached(), hash_inputs(), is_fresh(), list_features(), load(), _paths(), Feature store — compute features ONCE, reuse everywhere, consistently.  Models,, Stable content hash from arbitrary inputs (paths, mtimes, params). (+2 more)

### Community 36 - "Ensemble Meta-Model"
Cohesion: 0.25
Nodes (10): rank_today(), Return the model's current top-ranked stocks (best relative-strength     candida, contribution_breakdown(), ensemble_score(), Ensemble meta-model — blend every signal into ONE selection score.  Combines the, Which signals drove this stock's ensemble score (for explainability)., Blend signals into one score. If `regime` is given (or auto-detected),     facto, top_picks() (+2 more)

### Community 37 - "Regime Classifier & Breadth"
Cohesion: 0.24
Nodes (9): _components(), _load_index(), 4-class market-regime classifier — bull / bear / sideways / volatile.  The earli, Merge data/historical/<SYM>.csv and data/<SYM>_daily.csv into the     longest up, breadth_read(), breadth_series(), _close_matrix(), Market-breadth model — is the WHOLE market participating, or just a few names? (+1 more)

### Community 38 - "PIT Fundamentals Fetch"
Cohesion: 0.27
Nodes (10): fetch_all(), _fetch_symbol(), _harvest(), _load_existing(), _pick(), Point-in-time (PIT) fundamentals fetch — roadmap #3.  WHY this exists (and how i, All three quarterly statements, merged by period-end date., Return the row Series for the first candidate label present in df. (+2 more)

### Community 39 - "AngelOne Broker Connect"
Cohesion: 0.24
Nodes (3): Exception, AngelOneConnection, get_connection()

### Community 40 - "Learning-to-Rank Ranker"
Cohesion: 0.29
Nodes (9): _long_short(), Average (top-quintile minus bottom-quintile) actual relative return., _fit_ranker(), _graded_relevance(), _make_ranker(), Learning-to-rank ranker (LambdaMART) — roadmap #4 (part A).  The production rank, Per-date forward-return quintile (0=worst .. 4=best). LambdaMART's     relevance, XGBRanker needs rows grouped by query (date) and a qid array. Sort by     date s (+1 more)

### Community 41 - "Fundamental Quality Scoring"
Cohesion: 0.33
Nodes (9): load_fundamentals(), load_industries(), quality_scores(), Fundamental quality scoring (goal #4).  Turns the raw fundamentals snapshot (tra, Median debt/equity PER SECTOR across the whole cached universe, so the     lever, Cross-sectional z-score, winsorised at ±3., Return a DataFrame indexed by symbol with per-category scores, a     composite z, _sector_de_median() (+1 more)

### Community 42 - "Paper-Trading Environment"
Cohesion: 0.49
Nodes (9): _close(), close_all(), enter_book(), _latest_prices(), _load(), Paper-trading environment — forward-test the system before risking money.  A bac, _save(), status() (+1 more)

### Community 43 - "Walk-Forward Dataset Builder"
Cohesion: 0.33
Nodes (8): balance_classes(), build_all_windows(), build_ideal_response(), build_training_prompt(), build_window_dataset(), build_window_stats(), load_symbol_data(), Ensure no action class dominates dataset

### Community 44 - "Dataset Validation"
Cohesion: 0.36
Nodes (8): check_class_balance(), check_file_integrity(), check_jsonl_validity(), check_output_validity(), check_prompt_quality(), check_year_coverage(), validate_all_windows(), validate_window()

### Community 45 - "Pre-Market Briefing"
Cohesion: 0.50
Nodes (8): global_markets(), _no_feed(), Daily pre-market analysis — the briefing the agents wake up to.  A scheduled job, run_premarket(), _safe(), sector_rotation(), volatility(), classify()

### Community 46 - "Probability Calibration"
Cohesion: 0.33
Nodes (7): _brier(), _ece(), Probability calibration for the ranker's confidence — roadmap #4 (part B).  The, Expected calibration error: bin by predicted prob, average the gap     between m, run(), _make_model(), Cross-sectional stock ranking model.  Instead of predicting a single stock's abs

### Community 47 - "Earnings-Event Risk"
Cohesion: 0.42
Nodes (8): assess(), earnings_risk(), fetch_earnings(), _fetch_one(), _load(), next_earnings(), Earnings-event risk model — don't get gapped by a results print.  Quarterly resu, _save()

### Community 48 - "Regime Detector"
Cohesion: 0.42
Nodes (8): compute_adx(), detect_expiry_day(), detect_full_regime(), detect_intraday_phase(), detect_market_breadth(), detect_trend_regime(), detect_volatility_regime(), fuse_regimes()

### Community 49 - "Risk Policy & Drawdown Guard"
Cohesion: 0.28
Nodes (8): drawdown_scaler(), effective_limits(), portfolio_heat(), Regime-aware risk policy + portfolio drawdown guard.  The portfolio's RISK BUDGE, current_dd ≤ 0 (e.g., -0.08). Returns a gross multiplier in [0,1]., Combine regime policy + drawdown guard (+ optional breadth haircut)     into the, positions: list of dicts with 'risk_rupees' and a shared capital base.     Retur, regime_policy()

### Community 50 - "Option-Chain Collector"
Cohesion: 0.33
Nodes (8): _aggregate(), _append_csv(), collect_once(), _india_vix(), _parse_chain(), Live option-chain collector — NIFTY / BANKNIFTY (builds the ML dataset).  WHY TH, Return (spot, nearest_expiry, list-of-strike-rows) for the nearest     expiry, o, Collapse the strike chain into one ML-ready feature row.

### Community 51 - "Deep Ranker (MLP)"
Cohesion: 0.50
Nodes (7): _build_mlp(), _predict_mlp(), Deep-learning cross-sectional ranker — honest head-to-head vs XGBoost.  The user, Train the MLP with early stopping; return the best-state model., run(), _torch(), _train_mlp()

### Community 52 - "Slippage Estimator"
Cohesion: 0.36
Nodes (7): _adv_and_vol(), estimate(), _half_spread_bps(), Slippage estimator — what the fill REALLY costs, before you trade.  Backtests as, Wider spread for thinner names. ~1bp for very liquid, capped at 60bp., Estimate one-way slippage in bps + ₹ for an order of `order_value` ₹., slippage_for_book()

### Community 53 - "Fundamentals Fetch"
Cohesion: 0.36
Nodes (7): _extract(), fetch_all(), _load_existing(), Fetch fundamental data (~23 params) for the whole universe.  Why: pure price-mom, Every symbol we have price history for (so the score aligns 1:1 with     the ran, Pull just our FIELDS out of a yfinance .info dict, keeping only     numeric valu, _universe_symbols()

### Community 54 - "IV Surface"
Cohesion: 0.43
Nodes (6): atm_iv_history(), iv_percentile(), Historical IV surface database — query the volatility the market is pricing.  Bu, Latest strike-level IV smile (CE & PE IV vs strike)., report(), smile()

### Community 55 - "Dynamic Stops"
Cohesion: 0.43
Nodes (6): _atr(), dynamic_stops(), Dynamic stop-loss selection engine.  A fixed % stop ignores how each stock actua, All four stops + a recommended pick for a LONG. Returns None if data     is too, _trend(), _why()

### Community 56 - "Colab Prep"
Cohesion: 0.52
Nodes (6): generate_colab_notebook(), generate_deploy_script(), generate_instructions(), generate_modelfile(), package_datasets(), prepare_all()

### Community 57 - "Drift Monitor"
Cohesion: 0.47
Nodes (5): _backtest_ic(), _psi(), Model drift & edge-decay monitor — roadmap #5 (part B).  A model with a measured, Population Stability Index between two samples of one feature., run()

### Community 58 - "Performance Attribution"
Cohesion: 0.47
Nodes (5): attribute(), Performance attribution — WHERE did the P&L come from?  Total return tells you n, Extract the dominant ensemble signal from the explanation text., report(), _signal_of()

### Community 59 - "Monte Carlo Risk"
Cohesion: 0.47
Nodes (5): from_book(), Monte Carlo risk engine — what could this book actually do over the next month?, weights: dict {symbol: weight_fraction} (sum ≤ 1; remainder = cash).     Bootstr, _returns(), simulate()

### Community 60 - "Universe Fetch"
Cohesion: 0.47
Nodes (5): fetch_all(), get_constituents(), Fetch the full Nifty 500 universe (~500 stocks) of daily OHLCV history.  Why: th, Return list of (symbol, industry). Tries NSE live, falls back to     a cached co, _save_frame()

## Knowledge Gaps
- **17 isolated node(s):** `Colors`, `NAV`, `LINES`, `News & Sentiment Agent`, `Options Strategy Agent` (+12 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_prices()` connect `Data Quality & Retrain` to `Decision Engine & Calibration`, `Ensemble Meta-Model`, `Regime Classifier & Breadth`, `Learning-to-Rank Ranker`, `Market Data & Intraday Signals`, `Paper-Trading Environment`, `Probability Calibration`, `Point-in-Time Fundamentals`, `Deep Ranker (MLP)`, `Slippage Estimator`, `Dynamic Stops`, `Explainability & Portfolio Book`, `Drift Monitor`, `Monte Carlo Risk`, `Sector Strength & Optimizer`, `Backtest & Market Regime`?**
  _High betweenness centrality (0.173) - this node is a cross-community bridge._
- **Why does `screen()` connect `Decision Engine & Calibration` to `Memory Store & ML Models`, `API Precompute & Router`, `Ensemble Meta-Model`, `Data Quality & Retrain`?**
  _High betweenness centrality (0.120) - this node is a cross-community bridge._
- **Why does `decide()` connect `Decision Engine & Calibration` to `Memory Store & ML Models`, `Event Awareness & Narration`?**
  _High betweenness centrality (0.083) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `AgentReport` (e.g. with `MarketIntelligenceAgent` and `ModelImprovementAgent`) actually correct?**
  _`AgentReport` has 9 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Auto-trader — wire the signals to the agents and tick them on live prices.  Clos`, `Robust last price: fast_info, else latest 1-day close (works even when     the m`, `Brokerage / cost agent — accurate Indian (NSE) trading charges.  Every entry and` to the rest of the system?**
  _241 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Capital Allocation & Trade Memory` be split into smaller, more focused modules?**
  _Cohesion score 0.05608322026232474 - nodes in this community are weakly interconnected._
- **Should `Memory Store & ML Models` be split into smaller, more focused modules?**
  _Cohesion score 0.06597222222222222 - nodes in this community are weakly interconnected._
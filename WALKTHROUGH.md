# Complete Walkthrough: BTC Alpha Engine

## Table of Contents
1. [What This Project Does](#what-this-project-does)
2. [How It Works (Technical Deep Dive)](#how-it-works)
3. [Prerequisites](#prerequisites)
4. [Step-by-Step Setup](#step-by-step-setup)
5. [Running the System](#running-the-system)
6. [Understanding the Output](#understanding-the-output)
7. [Monitoring with Grafana](#monitoring-with-grafana)
8. [File-by-File Map](#file-by-file-map)
9. [Troubleshooting](#troubleshooting)

---

## What This Project Does

This is an **autonomous BTC trading research system** that:

1. Pulls derivatives data from CoinGlass (funding rates, open interest, liquidations, leverage ratios, taker flow)
2. Computes ~80 quantitative features from that data
3. Trains multiple ML models across 5 time horizons (1h, 4h, 8h, 12h, 24h)
4. Uses an AI agent (powered by Chutes.ai LLMs) that continuously analyzes its own performance, identifies weaknesses, and proposes improvements
5. Outputs **signal recommendations only** — long/short direction, position size, entry price, take-profit, stop-loss, expected return

**It does NOT execute live trades.** The output is purely advisory.

---

## How It Works

### The Data Flow

```
CoinGlass API → Raw Data (DB) → Feature Pipeline (80+ features) → Models → Ensemble → Signal
```

**Step 1: Data Ingestion** (`src/data/`)
- `coinglass_client.py` calls the CoinGlass v3 API for:
  - Funding rate OHLC (how much longs pay shorts, or vice versa)
  - Open interest OHLC (total outstanding contracts)
  - Liquidation history (forced closures of positions)
  - Long/short account ratios (crowd positioning)
  - Taker buy/sell volumes (aggressive order flow)
- Data is validated (`validators.py`), aligned to hourly UTC timestamps (`resampler.py`), and stored in SQLite/Postgres

**Step 2: Feature Engineering** (`src/features/`)
- `price_features.py`: Returns at 1h/4h/8h/12h/24h, realized volatility, RSI, price vs moving averages
- `funding_features.py`: Funding rate level, change, z-score, persistence (how many consecutive hours same sign), extreme flags
- `oi_features.py`: OI change, acceleration, divergence vs price, correlation with price
- `liquidation_features.py`: Long/short imbalance, shock z-scores, cascade detection (sustained high liquidations)
- `flow_features.py`: Long/short ratio changes, taker buy/sell imbalances, net flow divergence from price
- `regime_features.py`: Classifies the market as trend_up, trend_down, mean_revert, crowded_long, crowded_short, panic_flush, or squeeze
- `feature_pipeline.py`: Orchestrates all of the above + computes interaction features (funding × OI, liquidation × taker flow, etc.)

**Step 3: Labels** (`src/labels/`)
- Forward returns at each horizon (what actually happened 1h/4h/8h/12h/24h later)
- Ternary labels: +1 (profitable long), 0 (no trade), -1 (profitable short) — thresholded to account for trading costs
- MFE/MAE (maximum favorable/adverse excursion) — how much the trade went in your favor vs against you
- Optional triple-barrier labels (take-profit barrier, stop-loss barrier, time barrier)

**Step 4: Model Training** (`src/models/`, `src/research/`)
- Four model types: Logistic Regression, Random Forest, LightGBM, XGBoost
- Each trained per horizon (5 horizons × 4 model types = 20 base models)
- **Walk-forward validation** (`purged_walk_forward.py`):
  - Train on 90 days, test on 14 days
  - 48-hour purge gap between train end and test start (prevents leakage)
  - 24-hour embargo after test
  - Roll forward 14 days, repeat
  - Simulates realistic P&L with 5bps slippage + 2bps commission
  - **Enforces the 5% daily loss and 5% EOD trailing loss rules during simulation**
- **Evolutionary search** (`evolutionary_search.py`): Genetic algorithm that explores feature subsets and hyperparameters, breeding and mutating configurations
- Models that pass thresholds (Sharpe > 0.5, accuracy > 52%, zero breach rate) get promoted to production

**Step 5: Signal Generation** (`src/portfolio/`, `src/risk/`)
- Each promoted model produces a signal (long/short/flat + probability + confidence)
- `ensemble.py`: Aggregates signals using OOS-Sharpe-weighted voting
- `consensus.py`: Checks agreement across time horizons (need ≥2 horizons agreeing)
- `sizing.py`: Volatility-targeted position sizing with drawdown-headroom awareness
- `constraints.py`: Hard caps on position size (25% of equity) and gross exposure
- Risk manager enforces 5% daily loss limit and 5% EOD trailing loss limit

### The Agent Loop (The Brain)

The autonomous agent (`src/agent/alpha_agent.py`) is inspired by [Arbos](https://github.com/unconst/Arbos) — a "ralph-loop" that calls the same prompt repeatedly, forever improving.

Each iteration (default: every hour):

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: DESIGN                                             │
│ LLM analyzes current system performance + known weaknesses  │
│ Proposes concrete changes: new features, hyperparameter     │
│ adjustments, model additions/removals, risk tuning          │
│ → Uses Chutes.ai for inference                              │
└──────────────┬──────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: RUN                                                │
│ Loads latest features from DB                               │
│ Runs all promoted models, collects signals per horizon      │
│ Aggregates via ensemble + cross-horizon consensus           │
└──────────────┬──────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: MEASURE                                            │
│ Pulls walk-forward metrics from DB (Sharpe, drawdown, etc.) │
│ Counts errors, model population, signal accuracy            │
└──────────────┬──────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 4: REFLECT                                            │
│ LLM performs deep analysis:                                 │
│ - What are specific weaknesses right now?                   │
│ - Are previous improvements working?                        │
│ - What regime is the market in?                             │
│ - Where is alpha leaking?                                   │
│ → Uses Chutes.ai for inference                              │
└──────────────┬──────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 5: IMPROVE                                            │
│ Applies top 3 proposed changes from design phase            │
│ Updates weaknesses list from reflection                     │
│ Persists state to artifacts/agent_state.json                │
└──────────────┬──────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────┐
│ OUTPUT: SignalOutput                                         │
│ direction: long / short / flat                              │
│ entry_price, take_profit, stop_loss                         │
│ position_size_pct, position_size_usd                        │
│ expected_return_pct, risk_reward_ratio                      │
│ confidence, regime, reasoning                               │
│ → Printed to console + exposed via Prometheus               │
└─────────────────────────────────────────────────────────────┘
```

### Decentralized Compute

| Provider | What it does | When it's used |
|---|---|---|
| **Chutes.ai** | Serverless LLM inference | Agent design/reflect/improve phases (the thinking) |
| **Targon** | OpenAI-compatible inference + CPU | Fallback for LLM calls + CPU-bound sklearn training |
| **Lium** | GPU pod rental (A100/H100/H200) | Heavy model training (LightGBM GPU, PyTorch sequence models) |

The `ComputeDispatcher` (`src/compute/dispatcher.py`) routes tasks automatically: reasoning → Chutes, CPU → Targon, GPU → Lium. If one provider is unavailable, it falls back to the next.

### Risk Rules

Two hard rules enforced in **both backtests and live operation**:

**Rule 1 — Daily Loss Limit (5%)**
- At 00:00 UTC, "opening equity" is captured
- If equity drops below `opening_equity × 0.95` at any point during the day → breach
- All positions flattened, trading disabled until next day (or operator intervention)

**Rule 2 — EOD Trailing Loss Limit (5%)**
- "EOD high water mark" updates **only** at end-of-day (00:00 UTC), not from intraday peaks
- If equity drops below `eod_hwm × 0.95` → breach
- Same consequences as Rule 1

---

## Prerequisites

You need:
- **Python 3.11+**
- **A CoinGlass API key** — sign up at [coinglass.com](https://www.coinglass.com/) and get a v3 API key
- **A Chutes.ai API key** — sign up at [chutes.ai](https://chutes.ai/) for agent LLM reasoning
- **Docker + Docker Compose** (optional, for Prometheus + Grafana monitoring)
- **Lium CLI** (optional, for GPU training): `pip install lium.io && lium init`
- **Targon API key** (optional, fallback inference): get from [targon.com](https://targon.com/)

**Minimum to get started**: CoinGlass key + Chutes.ai key. Everything else is optional.

---

## Step-by-Step Setup

### Step 1: Clone and install

```bash
# Untar the project
tar xzf btc_alpha_engine.tar.gz
cd btc_alpha_engine

# Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install all dependencies
pip install -e .

# Or install specific extras
pip install -e ".[postgres]"   # if using PostgreSQL
pip install lium.io            # if using Lium GPU rental
pip install prometheus-client  # for proper Prometheus metrics (optional but recommended)
```

### Step 2: Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```bash
# REQUIRED
COINGLASS_API_KEY=your_actual_key_here
CHUTES_API_KEY=your_actual_key_here

# OPTIONAL (but recommended)
TARGON_API_KEY=your_key_here           # fallback inference
CHUTES_BASE_URL=https://llm.chutes.ai/v1
CHUTES_MODEL=deepseek-ai/DeepSeek-V3-0324

# OPTIONAL
DATABASE_URL=sqlite:///./btc_alpha.db  # default is fine for dev
```

### Step 3: Initialize the database

```bash
python scripts/bootstrap_db.py
```

This creates all 19 tables (instruments, price bars, funding, OI, liquidations, long/short, taker flow, feature store, labels, regimes, model registry, walk-forward runs, account snapshots, day state, signals, orders, fills, positions) and seeds the BTC instrument.

### Step 4: Backfill historical data

```bash
# Backfill from Jan 2024 to now (adjust dates as needed)
python scripts/backfill_coinglass.py --symbol BTC --start 2024-01-01

# For a quick test with less data
python scripts/backfill_coinglass.py --symbol BTC --start 2024-10-01
```

This pulls hourly data from CoinGlass and stores it in your database. Depending on the date range and your API plan rate limits, this takes 5-30 minutes.

### Step 5: Train initial models

```bash
# Train models across all horizons with walk-forward validation
python scripts/train_walk_forward.py --horizons 1,4,8,12,24

# Optional: also run evolutionary search (slower, finds better configs)
python scripts/train_walk_forward.py --horizons 4,8,24 --evo-search
```

This runs walk-forward validation for each model type × horizon combination. For 500+ hours of data, expect 5-15 minutes for baseline models.

### Step 6: Promote the best models

```bash
# See what would be promoted (dry run)
python scripts/promote_models.py --dry-run --min-sharpe 0.3

# Actually promote
python scripts/promote_models.py --min-sharpe 0.3
```

Only promoted models are used by the agent for signal generation.

### Step 7: Start monitoring (optional but recommended)

```bash
# Start Prometheus + Grafana
docker-compose -f docker-compose.monitoring.yml up -d

# Grafana is at http://localhost:3000
# Login: admin / alpha_engine
# The BTC Alpha Agent Dashboard is auto-provisioned
```

### Step 8: Run the agent

```bash
# Test with a single iteration first
python scripts/run_agent.py --once

# Start the perpetual loop (hourly iterations)
python scripts/run_agent.py

# Custom: 30-minute intervals, $50k sizing basis
python scripts/run_agent.py --delay 1800 --equity 50000
```

---

## Running the System

### What happens when you run `python scripts/run_agent.py`

1. **Startup**: Loads settings from `.env`, initializes database connection, starts Prometheus metrics server on port 9090, restores agent state from `artifacts/agent_state.json` if it exists

2. **Every iteration** (default: hourly):
   - Fetches current BTC price from Hyperliquid public API
   - **Design phase**: Sends system state + performance metrics to Chutes.ai LLM, receives analysis and proposed improvements
   - **Run phase**: Loads latest features from DB, runs all promoted models, produces per-horizon signals, aggregates via ensemble, checks cross-horizon consensus
   - **Measure phase**: Pulls walk-forward metrics from DB, computes rolling Sharpe, drawdown, accuracy
   - **Reflect phase**: Sends performance data to LLM, receives weakness analysis and regime assessment
   - **Improve phase**: Applies top 3 proposed changes, updates state
   - **Signal output**: Computes entry/TP/SL levels, position size, expected return, prints to console

3. **Between iterations**: Sleeps until next interval (interruptible with Ctrl+C)

4. **Metrics**: Updated after every iteration, scraped by Prometheus every 15s

### What happens without API keys

- **No CoinGlass key**: Backfill fails, but you can still test with synthetic data (the test fixtures create synthetic datasets)
- **No Chutes.ai key**: Agent's design/reflect/improve phases fall back to default behavior (no LLM reasoning, but the run/measure/signal phases still work)
- **No Targon key**: No fallback inference (Chutes is primary)
- **No Lium**: GPU training dispatching is skipped; all training runs locally

---

## Understanding the Output

When the agent runs, you'll see output like:

```
============================================================
[2025-06-15 14:00] LONG BTC
  Entry: $65,000.00 | Size: 12.0% ($12,000)
  TP: $67,600.00 (+4.00%) | SL: $63,700.00 (-2.00%)
  Expected: +4.00% over 8h | R:R 2.0
  Confidence: 0.78 | Regime: trend_up
  Horizons: [4, 8, 12] | Sharpe: 1.23
  Reasoning: strong consensus across medium horizons
============================================================
```

What each field means:
- **Direction**: LONG (buy), SHORT (sell), or FLAT (no trade)
- **Entry**: Current price at time of signal
- **Size**: Recommended position as % of equity and USD amount
- **TP (Take Profit)**: Price level to close for profit (based on 2× ATR estimate)
- **SL (Stop Loss)**: Price level to close for loss protection (based on 1× ATR)
- **Expected Return**: Estimated return % based on signal strength × average move
- **R:R**: Risk/reward ratio (how much you stand to gain vs lose)
- **Confidence**: Model ensemble agreement strength (0 to 1)
- **Regime**: Detected market state (trend_up, crowded_long, panic_flush, etc.)
- **Horizons**: Which time horizons agree on the direction
- **Sharpe**: System's rolling out-of-sample Sharpe ratio
- **Reasoning**: Why the consensus gate passed or failed

The full signal is also available as JSON via `signal.model_dump_json()`.

---

## Monitoring with Grafana

After starting the monitoring stack, open http://localhost:3000 (admin / alpha_engine).

The auto-provisioned dashboard shows:

| Panel | What it shows |
|---|---|
| System Sharpe Ratio (gauge) | Current rolling Sharpe — green >1.0, yellow >0.5, red below |
| Max Drawdown (gauge) | Worst drawdown — green >-3%, yellow >-5%, red below |
| Signal Direction (stat) | LONG / FLAT / SHORT with color coding |
| Signal Confidence (gauge) | How confident the ensemble is (0-1) |
| Position Size (stat) | Recommended position in USD |
| Sharpe Over Time | Time series of Sharpe evolution |
| Drawdown Over Time | Time series of drawdown |
| Entry / TP / SL | Price levels chart |
| Expected Return & R:R | Return and risk/reward time series |
| Iteration Duration | How long each agent iteration takes |
| Errors & Iterations | Counter of iterations and errors |
| Active Models | Number of promoted models |
| Risk Headroom | How much room before a risk breach |
| Equity | Current account equity |

---

## File-by-File Map

### Config (`config/`)
| File | Purpose |
|---|---|
| `assets.yaml` | Defines BTC instrument with tick size, lot size, max position |
| `data_sources.yaml` | CoinGlass and Hyperliquid API endpoints and settings |
| `model_registry.yaml` | Model types, hyperparameter search spaces, promotion criteria |
| `risk_limits.yaml` | 5% daily loss, 5% EOD trailing, kill switch thresholds, vol target |
| `walk_forward.yaml` | 90-day train, 14-day test, 48h purge, cost assumptions |
| `live_trading.yaml` | Feature flags (live trading OFF, paper mode ON) |
| `prometheus.yml` | Prometheus scrape config |
| `grafana/` | Dashboard JSON + Grafana provisioning |

### Data (`src/data/`)
| File | Purpose |
|---|---|
| `coinglass_client.py` | CoinGlass v3 API: funding, OI, liquidations, long/short, taker flow |
| `hyperliquid_client.py` | Hyperliquid info API for price data (public, no auth) |
| `ingest_jobs.py` | Backfill + incremental refresh for all data types |
| `validators.py` | Schema validation, null checks, high/low fixing, gap detection |
| `resampler.py` | Align everything to hourly UTC, fill gaps, merge dataframes |

### Features (`src/features/`)
| File | Features computed |
|---|---|
| `price_features.py` | Returns (1-24h), volatility, RSI, price vs MA, volume ratios |
| `funding_features.py` | Funding level/change/zscore/persistence/extreme flags/cumulative |
| `oi_features.py` | OI level/change/acceleration/zscore/divergence vs price |
| `liquidation_features.py` | Imbalance, shock zscore, cascade flag, momentum |
| `flow_features.py` | Long/short ratio changes, taker buy/sell, net flow |
| `regime_features.py` | Trend up/down, mean revert, crowded long/short, panic, squeeze |
| `feature_pipeline.py` | Orchestrates all above + interaction features |

### Models (`src/models/`)
| File | Purpose |
|---|---|
| `base.py` | Unified BaseAlphaModel interface (fit/predict/save/load/get_signal) |
| `baseline.py` | Logistic Regression + Random Forest (sklearn pipelines with scaling) |
| `gradient_boost.py` | LightGBM + XGBoost wrappers |
| `regime.py` | Regime gate model + signal multipliers per regime |
| `sequence.py` | Optional PyTorch LSTM (feature-flagged, for v2) |
| `registry.py` | Model artifact save/load/promote/retire + DB integration |

### Research (`src/research/`)
| File | Purpose |
|---|---|
| `datasets.py` | Build aligned feature + label datasets from DB |
| `purged_walk_forward.py` | Walk-forward splitter with purge + embargo gaps |
| `scoring.py` | Fold metrics: Sharpe, accuracy, profit factor, drawdown, breach count |
| `selection.py` | Filter candidates by min Sharpe, min accuracy, max breach rate |
| `evolutionary_search.py` | Genetic algorithm over feature subsets + hyperparameters |
| `reports.py` | Save fold reports to DB, generate summary reports |

### Agent (`src/agent/`)
| File | Purpose |
|---|---|
| `alpha_agent.py` | The autonomous agent: design/run/measure/reflect/improve/signal |
| `loop.py` | Perpetual ralph-loop runner with graceful shutdown |
| `signal_output.py` | SignalOutput + AgentState Pydantic models |

### Compute (`src/compute/`)
| File | Purpose |
|---|---|
| `targon_client.py` | Targon OpenAI-compatible inference + CPU compute |
| `lium_client.py` | Lium GPU pod management (create/upload/execute/destroy) |
| `chutes_client.py` | Chutes.ai serverless LLM inference |
| `dispatcher.py` | Routes: reasoning → Chutes, CPU → Targon, GPU → Lium |

### Monitoring (`src/monitoring/`)
| File | Purpose |
|---|---|
| `prometheus_metrics.py` | 15+ Prometheus gauges/counters/histograms + HTTP server |

### Risk (`src/risk/`)
| File | Purpose |
|---|---|
| `account_state.py` | Track equity, day reset at 00:00 UTC, EOD HWM, persist to DB |
| `drawdown_rules.py` | Exact implementations of Rule 1 (daily) and Rule 2 (EOD trailing) |
| `kill_switch.py` | Rate limiting (orders/fills per hour) + consecutive loss detection |
| `exposure.py` | Gross/net exposure tracking and limits |
| `risk_manager.py` | Orchestrates all risk checks |

---

## Troubleshooting

**"lium CLI not installed"**
→ `pip install lium.io && lium init` — Lium is optional; the system works without it

**"No inference provider available"**
→ Set at least `CHUTES_API_KEY` in `.env`. Without it, the agent's LLM reasoning phases fall back to defaults.

**Backfill returns 0 rows**
→ Verify your `COINGLASS_API_KEY` is valid and has the right plan tier. Some endpoints may require a paid plan.

**"Model not fitted" errors during agent run**
→ You need to train and promote models first: `python scripts/train_walk_forward.py` then `python scripts/promote_models.py`

**Prometheus shows no data in Grafana**
→ Check that the agent is running and metrics are exposed: `curl http://localhost:9090/metrics`
→ Check Prometheus can reach the agent: ensure `host.docker.internal:9090` resolves (or use `172.17.0.1:9090` on Linux)

**Agent produces only FLAT signals**
→ This means either: no promoted models exist, or models disagree across horizons (consensus gate blocks). Train more models, lower the consensus threshold, or lower the min-sharpe for promotion.

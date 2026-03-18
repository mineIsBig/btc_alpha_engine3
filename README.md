# BTC Alpha Engine

Autonomous BTC alpha engine with derivatives-based signals, walk-forward research, decentralized compute, and an Arbos-inspired self-improving agent loop.

## Architecture Overview

```
                           ┌─────────────────────────────────────────────────────────┐
                           │  AUTONOMOUS AGENT (Arbos Ralph-Loop)                    │
                           │  design → run → measure → reflect → improve → signal    │
                           │        ↕ Chutes.ai (LLM reasoning)                     │
                           └──────────────┬──────────────────────────────────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────────┐
              ▼                           ▼                               ▼
    ┌─────────────────┐     ┌──────────────────────┐       ┌──────────────────────┐
    │ Data Ingestion   │     │ Model Training        │       │ Signal Generation    │
    │ CoinGlass API    │     │ Lium GPUs (training)  │       │ Ensemble + Consensus │
    │ Hyperliquid      │     │ Targon CPU (sklearn)  │       │ Risk + Sizing        │
    └────────┬────────┘     └──────────┬───────────┘       └──────────┬───────────┘
             │                         │                               │
             ▼                         ▼                               ▼
    ┌─────────────────┐     ┌──────────────────────┐       ┌──────────────────────┐
    │ Feature Pipeline │     │ Walk-Forward + Evo    │       │ OUTPUT: Signal Only  │
    │ 80+ features     │     │ Selection + Registry  │       │ Long/Short + Size    │
    └─────────────────┘     └──────────────────────┘       │ TP/SL + Expected Ret │
                                                            └──────────┬───────────┘
                                                                       │
                                                            ┌──────────▼───────────┐
                                                            │ Prometheus + Grafana  │
                                                            │ Performance Monitor   │
                                                            └──────────────────────┘
```

**Output**: Signal recommendations only (long/short, position size, TP/SL, expected returns). No live trades.

**Compute**: Targon (CPU inference), Lium (GPU training), Chutes.ai (agent reasoning)
**Horizons**: 1h, 4h, 8h, 12h, 24h
**Models**: Logistic Regression, Random Forest, LightGBM, XGBoost (+ optional LSTM)
**Data Sources**: CoinGlass (funding, OI, liquidations, long/short, taker flow), Hyperliquid (price)
**Monitoring**: Prometheus metrics → Grafana dashboards

## Autonomous Agent (Arbos-Inspired)

The system is driven by an autonomous agent that runs a perpetual **ralph-loop** (inspired by [Arbos](https://github.com/unconst/Arbos)):

```
loop t = 1..∞
    S_t = design_or_modify(S_{t-1})   # LLM analyzes system, proposes changes
    O_t = run(S_t)                      # Train models, evaluate, ensemble
    P_t = measure(O_t)                  # Sharpe, PnL, drawdown, regime
    Δ_t = reflect(P_t)                  # Find weaknesses via LLM reflection
    S_{t+1} = improve(S_t, Δ_t)        # Apply improvements
    signal = generate_signal()          # Output: long/short + size + TP/SL
end
```

The agent uses **Chutes.ai** for LLM reasoning (design, reflect, improve phases) and dispatches GPU training to **Lium** when needed.

### Signal Output Format

Each iteration produces a `SignalOutput`:

```
[2025-06-15 14:00] LONG BTC
  Entry: $65,000.00 | Size: 12.0% ($12,000)
  TP: $67,600.00 (+4.00%) | SL: $63,700.00 (-2.00%)
  Expected: +4.00% over 8h | R:R 2.0
  Confidence: 0.78 | Regime: trend_up
  Horizons: [4, 8, 12] | Sharpe: 1.23
  Reasoning: strong consensus across medium horizons
```

## Decentralized Compute Integration

| Provider | Role | What it does |
|---|---|---|
| **Targon** (targon.com) | CPU inference + compute | OpenAI-compatible LLM endpoint, CPU model training |
| **Lium** (lium.io) | GPU training | A100/H100/H200 pod rental for LightGBM/XGBoost/PyTorch training |
| **Chutes.ai** | Agent inference | Serverless LLM for agent reasoning loop (design/reflect/improve) |

All three are Bittensor-based decentralized compute networks.

## Risk Rule Semantics

### Rule 1: Daily Loss Limit (5%)

- **Opening equity** is captured at **00:00 UTC** each trading day
- **Daily loss floor** = `opening_equity × 0.95`
- Checked on **every account update** (every fill and every periodic equity check)
- If `current_equity < daily_loss_floor` → **breach**: flatten positions (if configured), disable trading

### Rule 2: EOD Trailing Loss Limit (5%)

- **EOD high water mark** updates **only from end-of-day equity** at 00:00 UTC, **not from intraday peaks**
- **EOD trailing floor** = `eod_high_water_mark × 0.95`
- Checked on every account update
- If `current_equity < eod_trailing_floor` → **breach**: flatten positions (if configured), disable trading

### On Breach

1. `can_trade` is set to `False`
2. If `flatten_on_breach: true`, all positions are closed immediately
3. Breach reason and timestamp are logged and persisted to the `day_state` table
4. Trading remains disabled until the next 00:00 UTC reset (if `auto_reset_next_day: true`) or until operator intervention

### Trading Day

- Resets at **00:00 UTC**
- New day: opening equity is captured, EOD HWM is updated from previous day's closing equity, daily floor is recalculated, `can_trade` is re-enabled

## Setup

### Prerequisites

- Python 3.11+
- Poetry or pip
- PostgreSQL (recommended) or SQLite (default for dev)

### Installation

```bash
# Clone the repo
cd btc_alpha_engine

# Install dependencies with pip
pip install -e ".[postgres]"  # with PostgreSQL support
# or
pip install -e .              # SQLite only

# Copy environment file
cp .env.example .env
# Edit .env with your API keys
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | No | DB connection string (default: SQLite) |
| `COINGLASS_API_KEY` | Yes (for data) | CoinGlass API v3 key |
| `HYPERLIQUID_API_KEY` | No | Hyperliquid API key (for live trading only) |
| `HYPERLIQUID_API_SECRET` | No | Hyperliquid API secret |
| `HYPERLIQUID_WALLET_ADDRESS` | No | Hyperliquid wallet address |
| `HYPERLIQUID_BASE_URL` | No | Default: testnet URL |
| `LIVE_TRADING_ENABLED` | No | `false` (default) |
| `PAPER_MODE` | No | `true` (default) |
| `FLATTEN_ON_BREACH` | No | `true` (default) |

### Database Bootstrap

```bash
# Create tables and seed instruments
python scripts/bootstrap_db.py

# Or use Alembic for migrations
alembic upgrade head
```

### Historical Backfill

```bash
# Backfill all CoinGlass data from 2023-01-01
python scripts/backfill_coinglass.py --symbol BTC --start 2023-01-01

# Custom date range
python scripts/backfill_coinglass.py --symbol BTC --start 2024-01-01 --end 2024-06-01
```

### Train Models

```bash
# Run walk-forward training with default settings
python scripts/train_walk_forward.py

# Specific horizons
python scripts/train_walk_forward.py --horizons 4,8,24

# With evolutionary search (slower but finds better configs)
python scripts/train_walk_forward.py --evo-search
```

### Promote Models

```bash
# Auto-promote candidates with Sharpe > 0.5
python scripts/promote_models.py --min-sharpe 0.5

# Dry run
python scripts/promote_models.py --dry-run

# Promote specific model
python scripts/promote_models.py --model-id lgbm_h4
```

### Run Paper Trading

```bash
# Start paper trading with $100k
python scripts/paper_trade.py --equity 100000
```

This starts an hourly decision loop that:
1. Refreshes data from CoinGlass
2. Runs all promoted models
3. Aggregates signals with Sharpe-weighted ensemble
4. Checks cross-horizon consensus
5. Sizes position based on volatility targeting and drawdown headroom
6. Executes via paper broker
7. Monitors equity and risk limits every 60 seconds
8. Resets day state at 00:00 UTC

### Enable Shadow/Live Mode

**WARNING**: Live trading submits real orders. Proceed with extreme caution.

1. Set environment variables:
```bash
LIVE_TRADING_ENABLED=true
PAPER_MODE=false
HYPERLIQUID_WALLET_ADDRESS=0x...
HYPERLIQUID_API_SECRET=...
```

2. Run with confirmation:
```bash
python scripts/shadow_live.py --confirm
```

**Note**: Live order submission requires EIP-712 message signing which depends on the Hyperliquid Python SDK. The adapter is structured but signing is marked as TODO. Use paper mode until signing is implemented and verified.

## Project Structure

```
btc_alpha_engine/
├── pyproject.toml              # Dependencies and project config
├── .env.example                # Environment variable template
├── alembic.ini                 # Alembic migration config
├── docker-compose.monitoring.yml  # Prometheus + Grafana stack
├── config/
│   ├── assets.yaml             # Instrument universe
│   ├── data_sources.yaml       # API endpoints and settings
│   ├── model_registry.yaml     # Model types, hyperparams, promotion criteria
│   ├── risk_limits.yaml        # Risk rules, kill switch, exposure limits
│   ├── walk_forward.yaml       # Walk-forward validation settings
│   ├── live_trading.yaml       # Live/paper trading settings
│   ├── prometheus.yml          # Prometheus scrape config
│   └── grafana/                # Grafana dashboard + provisioning
├── migrations/                 # Alembic migrations
├── scripts/
│   ├── bootstrap_db.py         # Create DB and seed instruments
│   ├── backfill_coinglass.py   # Historical data backfill
│   ├── train_walk_forward.py   # Model training pipeline
│   ├── run_agent.py            # ★ Start autonomous agent loop
│   ├── paper_trade.py          # Start paper trading
│   ├── shadow_live.py          # Shadow/live trading
│   └── promote_models.py       # Promote trained models
├── src/
│   ├── common/                 # Config, logging, types, time utilities
│   ├── data/                   # CoinGlass client, Hyperliquid client, ingestion
│   ├── storage/                # SQLAlchemy models, DB session management
│   ├── features/               # Feature computation (price, funding, OI, liquidation, flow, regime)
│   ├── labels/                 # Forward-return labels, triple-barrier
│   ├── models/                 # BaseModel, logistic, RF, LightGBM, XGBoost, regime gate
│   ├── research/               # Walk-forward, scoring, selection, evolutionary search
│   ├── portfolio/              # Ensemble, consensus, sizing, stops, constraints
│   ├── risk/                   # Account state, drawdown rules, kill switch, exposure
│   ├── execution/              # Order router, paper broker, Hyperliquid adapter
│   ├── live/                   # Inference loop, trade loop, health checks
│   ├── orchestrator/           # Scheduler, research/training/live cycles
│   ├── compute/                # ★ Targon, Lium, Chutes.ai clients + dispatcher
│   ├── agent/                  # ★ Autonomous alpha agent (Arbos ralph-loop)
│   └── monitoring/             # ★ Prometheus metrics exposition
└── tests/
    ├── conftest.py             # Fixtures with synthetic data
    ├── unit/                   # Unit tests (including test_agent.py)
    └── integration/            # Integration test stubs
```

## Running the Agent

```bash
# One-shot iteration (test)
python scripts/run_agent.py --once

# Perpetual loop (1-hour intervals)
python scripts/run_agent.py --delay 3600

# Custom interval and equity
python scripts/run_agent.py --delay 1800 --equity 50000

# Prometheus metrics exposed on port 9090
curl http://localhost:9090/metrics
```

## Monitoring (Prometheus + Grafana)

```bash
# Start monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Access Grafana
open http://localhost:3000  # admin / alpha_engine

# Start agent with metrics
python scripts/run_agent.py
```

The Grafana dashboard shows:
- System Sharpe ratio (gauge + time series)
- Maximum drawdown (gauge + time series)
- Signal direction (LONG/SHORT/FLAT)
- Signal confidence, position size, expected return
- Entry price, take profit, stop loss levels
- Risk/reward ratio
- Agent iteration duration and error rate
- Model count and risk headroom
- Equity tracking

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=term-missing

# Specific test file
pytest tests/unit/test_risk.py -v
pytest tests/unit/test_agent.py -v

# Skip slow integration tests
pytest -m "not slow"
```

## Configuration

All configuration is in YAML files under `config/`. Key settings:

- **risk_limits.yaml**: Daily loss limit (5%), EOD trailing limit (5%), max leverage, kill switch thresholds
- **walk_forward.yaml**: Train/test/purge/embargo windows, cost assumptions, simulation settings
- **model_registry.yaml**: Model types, hyperparameter spaces, promotion criteria, evolutionary search settings
- **live_trading.yaml**: Feature flags, order settings, health check intervals

## Feature List

| Category | Features |
|---|---|
| Price | Returns (1h-24h), realized vol, RSI, price vs MA, momentum, range |
| Funding | Level, change, z-score, persistence, extreme flags, cumulative |
| Open Interest | Level, change, acceleration, z-score, divergence vs price, correlation |
| Liquidation | Imbalance, magnitude, rolling sums, shock z-score, cascade flag |
| Flow | Long/short ratio, taker buy/sell ratio, net flow, z-scores, extremes |
| Regime | Trend up/down, mean revert, crowded long/short, panic flush, squeeze |
| Interaction | Funding × OI, liquidation × taker flow, funding × LS ratio |

## Checklist: Manual Steps Required

- [ ] Obtain a CoinGlass API v3 key and set `COINGLASS_API_KEY`
- [ ] Obtain a Chutes.ai API key from chutes.ai and set `CHUTES_API_KEY`
- [ ] Obtain a Targon API key from targon.com and set `TARGON_API_KEY`
- [ ] Install Lium CLI: `pip install lium.io && lium init`
- [ ] Verify CoinGlass endpoint payloads match the v3 docs
- [ ] For PostgreSQL: create the database and set `DATABASE_URL`
- [ ] Review risk limits in `config/risk_limits.yaml`
- [ ] Run backfill and verify data quality before training models
- [ ] Start monitoring: `docker-compose -f docker-compose.monitoring.yml up -d`
- [ ] Run agent: `python scripts/run_agent.py`

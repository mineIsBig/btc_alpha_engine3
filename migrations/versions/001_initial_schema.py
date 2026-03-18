"""initial schema

Revision ID: 001
Revises:
Create Date: 2025-01-01 00:00:00.000000
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Instruments
    op.create_table(
        "instruments",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("exchange_symbol", sa.String(20)),
        sa.Column("coinalyze_symbol", sa.String(30)),
        sa.Column("hyperliquid_symbol", sa.String(20)),
        sa.Column("tick_size", sa.Float(), default=0.1),
        sa.Column("lot_size", sa.Float(), default=0.001),
        sa.Column("min_notional", sa.Float(), default=10.0),
        sa.Column("max_position_usd", sa.Float(), default=100000.0),
        sa.Column("enabled", sa.Boolean(), default=True),
        sa.Column("created_at", sa.DateTime()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol"),
    )
    op.create_index("ix_instruments_symbol", "instruments", ["symbol"])

    # Price bars
    op.create_table(
        "price_bars_1h",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("open", sa.Float(), nullable=False),
        sa.Column("high", sa.Float(), nullable=False),
        sa.Column("low", sa.Float(), nullable=False),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("volume", sa.Float(), default=0.0),
        sa.Column("source", sa.String(20), default="coinalyze"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "timestamp", name="uq_price_1h"),
    )
    op.create_index("ix_price_1h_ts", "price_bars_1h", ["timestamp"])

    # CG Funding
    op.create_table(
        "cg_funding_1h",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("exchange", sa.String(20), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("open", sa.Float()),
        sa.Column("high", sa.Float()),
        sa.Column("low", sa.Float()),
        sa.Column("close", sa.Float()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "exchange", "timestamp", name="uq_funding_1h"),
    )
    op.create_index("ix_funding_1h_ts", "cg_funding_1h", ["timestamp"])

    # CG OI
    op.create_table(
        "cg_oi_1h",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("exchange", sa.String(20), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("open", sa.Float()),
        sa.Column("high", sa.Float()),
        sa.Column("low", sa.Float()),
        sa.Column("close", sa.Float()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "exchange", "timestamp", name="uq_oi_1h"),
    )
    op.create_index("ix_oi_1h_ts", "cg_oi_1h", ["timestamp"])

    # CG Liquidations
    op.create_table(
        "cg_liquidations_1h",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("long_liquidations_usd", sa.Float(), default=0.0),
        sa.Column("short_liquidations_usd", sa.Float(), default=0.0),
        sa.Column("total_liquidations_usd", sa.Float(), default=0.0),
        sa.Column("count", sa.Integer(), default=0),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "timestamp", name="uq_liq_1h"),
    )
    op.create_index("ix_liq_1h_ts", "cg_liquidations_1h", ["timestamp"])

    # CG Long/Short
    op.create_table(
        "cg_long_short_1h",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("exchange", sa.String(20), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("long_ratio", sa.Float()),
        sa.Column("short_ratio", sa.Float()),
        sa.Column("long_short_ratio", sa.Float()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "exchange", "timestamp", name="uq_ls_1h"),
    )
    op.create_index("ix_ls_1h_ts", "cg_long_short_1h", ["timestamp"])

    # CG Taker Flow
    op.create_table(
        "cg_taker_flow_1h",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("buy_volume", sa.Float(), default=0.0),
        sa.Column("sell_volume", sa.Float(), default=0.0),
        sa.Column("buy_sell_ratio", sa.Float()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "timestamp", name="uq_taker_1h"),
    )
    op.create_index("ix_taker_1h_ts", "cg_taker_flow_1h", ["timestamp"])

    # Feature Store
    op.create_table(
        "feature_store_1h",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("features_json", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "timestamp", name="uq_feat_1h"),
    )
    op.create_index("ix_feat_1h_ts", "feature_store_1h", ["timestamp"])

    # Labels
    op.create_table(
        "labels_1h",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("horizon", sa.Integer(), nullable=False),
        sa.Column("fwd_return", sa.Float()),
        sa.Column("side_label", sa.Integer()),
        sa.Column("mfe", sa.Float()),
        sa.Column("mae", sa.Float()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "timestamp", "horizon", name="uq_labels_1h"),
    )
    op.create_index("ix_labels_1h_ts", "labels_1h", ["timestamp"])

    # Regimes
    op.create_table(
        "regimes_1h",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("regime", sa.String(30), nullable=False),
        sa.Column("confidence", sa.Float(), default=0.5),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "timestamp", name="uq_regime_1h"),
    )
    op.create_index("ix_regime_1h_ts", "regimes_1h", ["timestamp"])

    # Model Registry
    op.create_table(
        "model_registry",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_id", sa.String(100), nullable=False),
        sa.Column("model_type", sa.String(50), nullable=False),
        sa.Column("horizon", sa.Integer(), nullable=False),
        sa.Column("features_json", sa.Text()),
        sa.Column("params_json", sa.Text()),
        sa.Column("artifact_path", sa.String(500)),
        sa.Column("train_start", sa.DateTime()),
        sa.Column("train_end", sa.DateTime()),
        sa.Column("oos_sharpe", sa.Float()),
        sa.Column("oos_accuracy", sa.Float()),
        sa.Column("oos_profit_factor", sa.Float()),
        sa.Column("breach_rate", sa.Float(), default=0.0),
        sa.Column("status", sa.String(20), default="candidate"),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("promoted_at", sa.DateTime()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_id"),
    )

    # Walk Forward Runs
    op.create_table(
        "walk_forward_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(100), nullable=False),
        sa.Column("model_id", sa.String(100), nullable=False),
        sa.Column("fold_idx", sa.Integer(), nullable=False),
        sa.Column("train_start", sa.DateTime(), nullable=False),
        sa.Column("train_end", sa.DateTime(), nullable=False),
        sa.Column("test_start", sa.DateTime(), nullable=False),
        sa.Column("test_end", sa.DateTime(), nullable=False),
        sa.Column("n_train_samples", sa.Integer()),
        sa.Column("n_test_samples", sa.Integer()),
        sa.Column("sharpe", sa.Float()),
        sa.Column("accuracy", sa.Float()),
        sa.Column("profit_factor", sa.Float()),
        sa.Column("max_drawdown", sa.Float()),
        sa.Column("n_trades", sa.Integer()),
        sa.Column("breach_count", sa.Integer(), default=0),
        sa.Column("metrics_json", sa.Text()),
        sa.Column("created_at", sa.DateTime()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id"),
    )

    # Model Live Scores
    op.create_table(
        "model_live_scores",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_id", sa.String(100), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("rolling_sharpe", sa.Float()),
        sa.Column("rolling_accuracy", sa.Float()),
        sa.Column("rolling_pnl", sa.Float()),
        sa.Column("n_signals", sa.Integer()),
        sa.Column("window_hours", sa.Integer(), default=168),
        sa.PrimaryKeyConstraint("id"),
    )

    # Account Snapshots
    op.create_table(
        "account_snapshots",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("equity", sa.Float(), nullable=False),
        sa.Column("cash", sa.Float(), nullable=False),
        sa.Column("unrealized_pnl", sa.Float(), default=0.0),
        sa.Column("realized_pnl_today", sa.Float(), default=0.0),
        sa.Column("gross_exposure", sa.Float(), default=0.0),
        sa.Column("net_exposure", sa.Float(), default=0.0),
        sa.Column("source", sa.String(20), default="paper"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_acct_ts", "account_snapshots", ["timestamp"])

    # Day State
    op.create_table(
        "day_state",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("trading_date", sa.String(10), nullable=False),
        sa.Column("opening_equity", sa.Float(), nullable=False),
        sa.Column("eod_high_water_mark", sa.Float(), nullable=False),
        sa.Column("daily_loss_floor", sa.Float(), nullable=False),
        sa.Column("eod_trailing_floor", sa.Float(), nullable=False),
        sa.Column("can_trade", sa.Boolean(), default=True),
        sa.Column("breach_reason", sa.String(200)),
        sa.Column("breach_time", sa.DateTime()),
        sa.Column("last_updated", sa.DateTime()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("trading_date"),
    )

    # Signals
    op.create_table(
        "signals",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("horizon", sa.Integer(), nullable=False),
        sa.Column("model_id", sa.String(100), nullable=False),
        sa.Column("side", sa.Integer(), nullable=False),
        sa.Column("probability", sa.Float()),
        sa.Column("confidence", sa.Float()),
        sa.Column("regime", sa.String(30)),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_sig_ts", "signals", ["timestamp"])

    # Orders
    op.create_table(
        "orders",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("order_id", sa.String(100), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("side", sa.String(10), nullable=False),
        sa.Column("order_type", sa.String(10), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("price", sa.Float()),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("filled_qty", sa.Float(), default=0.0),
        sa.Column("filled_price", sa.Float()),
        sa.Column("exchange_order_id", sa.String(100)),
        sa.Column("broker", sa.String(20), default="paper"),
        sa.Column("reason", sa.String(200)),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("updated_at", sa.DateTime()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("order_id"),
    )
    op.create_index("ix_order_ts", "orders", ["timestamp"])

    # Fills
    op.create_table(
        "fills",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("fill_id", sa.String(100), nullable=False),
        sa.Column("order_id", sa.String(100), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("side", sa.String(10), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("commission", sa.Float(), default=0.0),
        sa.Column("slippage", sa.Float(), default=0.0),
        sa.Column("broker", sa.String(20), default="paper"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("fill_id"),
    )
    op.create_index("ix_fill_ts", "fills", ["timestamp"])

    # Positions
    op.create_table(
        "positions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("side", sa.String(10), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False, default=0.0),
        sa.Column("avg_entry_price", sa.Float()),
        sa.Column("unrealized_pnl", sa.Float(), default=0.0),
        sa.Column("realized_pnl", sa.Float(), default=0.0),
        sa.Column("broker", sa.String(20), default="paper"),
        sa.Column("updated_at", sa.DateTime()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "broker", name="uq_position"),
    )


def downgrade() -> None:
    tables = [
        "positions", "fills", "orders", "signals", "day_state",
        "account_snapshots", "model_live_scores", "walk_forward_runs",
        "model_registry", "regimes_1h", "labels_1h", "feature_store_1h",
        "cg_taker_flow_1h", "cg_long_short_1h", "cg_liquidations_1h",
        "cg_oi_1h", "cg_funding_1h", "price_bars_1h", "instruments",
    ]
    for t in tables:
        op.drop_table(t)

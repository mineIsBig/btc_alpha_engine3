"""SQLAlchemy ORM models for all tables."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# ── Instruments ──────────────────────────────────────────────


class Instrument(Base):
    __tablename__ = "instruments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    exchange_symbol = Column(String(20))
    coinalyze_symbol = Column(String(30))
    hyperliquid_symbol = Column(String(20))
    tick_size = Column(Float, default=0.1)
    lot_size = Column(Float, default=0.001)
    min_notional = Column(Float, default=10.0)
    max_position_usd = Column(Float, default=100000.0)
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ── Price Bars ───────────────────────────────────────────────


class PriceBar1h(Base):
    __tablename__ = "price_bars_1h"
    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", name="uq_price_1h"),
        Index("ix_price_1h_ts", "timestamp"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, default=0.0)
    source = Column(String(20), default="coinalyze")


# ── Derivatives Data Tables (sourced from Coinalyze) ─────────


class CGFunding1h(Base):
    __tablename__ = "cg_funding_1h"
    __table_args__ = (
        UniqueConstraint("symbol", "exchange", "timestamp", name="uq_funding_1h"),
        Index("ix_funding_1h_ts", "timestamp"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    exchange = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)


class CGOI1h(Base):
    __tablename__ = "cg_oi_1h"
    __table_args__ = (
        UniqueConstraint("symbol", "exchange", "timestamp", name="uq_oi_1h"),
        Index("ix_oi_1h_ts", "timestamp"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    exchange = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)


class CGLiquidations1h(Base):
    __tablename__ = "cg_liquidations_1h"
    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", name="uq_liq_1h"),
        Index("ix_liq_1h_ts", "timestamp"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    long_liquidations_usd = Column(Float, default=0.0)
    short_liquidations_usd = Column(Float, default=0.0)
    total_liquidations_usd = Column(Float, default=0.0)
    count = Column(Integer, default=0)


class CGLongShort1h(Base):
    __tablename__ = "cg_long_short_1h"
    __table_args__ = (
        UniqueConstraint("symbol", "exchange", "timestamp", name="uq_ls_1h"),
        Index("ix_ls_1h_ts", "timestamp"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    exchange = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    long_ratio = Column(Float)
    short_ratio = Column(Float)
    long_short_ratio = Column(Float)


class CGTakerFlow1h(Base):
    __tablename__ = "cg_taker_flow_1h"
    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", name="uq_taker_1h"),
        Index("ix_taker_1h_ts", "timestamp"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    buy_volume = Column(Float, default=0.0)
    sell_volume = Column(Float, default=0.0)
    buy_sell_ratio = Column(Float)


# ── Feature Store ────────────────────────────────────────────


class FeatureStore1h(Base):
    __tablename__ = "feature_store_1h"
    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", name="uq_feat_1h"),
        Index("ix_feat_1h_ts", "timestamp"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    features_json = Column(Text, nullable=False)  # JSON blob of feature dict


class Labels1h(Base):
    __tablename__ = "labels_1h"
    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", "horizon", name="uq_labels_1h"),
        Index("ix_labels_1h_ts", "timestamp"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    horizon = Column(Integer, nullable=False)
    fwd_return = Column(Float)
    side_label = Column(Integer)  # -1, 0, 1
    mfe = Column(Float)
    mae = Column(Float)


class Regimes1h(Base):
    __tablename__ = "regimes_1h"
    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", name="uq_regime_1h"),
        Index("ix_regime_1h_ts", "timestamp"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    regime = Column(String(30), nullable=False)
    confidence = Column(Float, default=0.5)


# ── Model Registry & Research ────────────────────────────────


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(100), unique=True, nullable=False, index=True)
    model_type = Column(String(50), nullable=False)
    horizon = Column(Integer, nullable=False)
    features_json = Column(Text)  # JSON list of feature names
    params_json = Column(Text)  # JSON dict of hyperparams
    artifact_path = Column(String(500))
    train_start = Column(DateTime)
    train_end = Column(DateTime)
    oos_sharpe = Column(Float)
    oos_accuracy = Column(Float)
    oos_profit_factor = Column(Float)
    breach_rate = Column(Float, default=0.0)
    status = Column(String(20), default="candidate")  # candidate, promoted, retired
    created_at = Column(DateTime, default=datetime.utcnow)
    promoted_at = Column(DateTime)


class WalkForwardRun(Base):
    __tablename__ = "walk_forward_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(100), unique=True, nullable=False, index=True)
    model_id = Column(String(100), nullable=False, index=True)
    fold_idx = Column(Integer, nullable=False)
    train_start = Column(DateTime, nullable=False)
    train_end = Column(DateTime, nullable=False)
    test_start = Column(DateTime, nullable=False)
    test_end = Column(DateTime, nullable=False)
    n_train_samples = Column(Integer)
    n_test_samples = Column(Integer)
    sharpe = Column(Float)
    accuracy = Column(Float)
    profit_factor = Column(Float)
    max_drawdown = Column(Float)
    n_trades = Column(Integer)
    breach_count = Column(Integer, default=0)
    metrics_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelLiveScore(Base):
    __tablename__ = "model_live_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    rolling_sharpe = Column(Float)
    rolling_accuracy = Column(Float)
    rolling_pnl = Column(Float)
    n_signals = Column(Integer)
    window_hours = Column(Integer, default=168)


# ── Account & Risk State ─────────────────────────────────────


class AccountSnapshot(Base):
    __tablename__ = "account_snapshots"
    __table_args__ = (Index("ix_acct_ts", "timestamp"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    equity = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl_today = Column(Float, default=0.0)
    gross_exposure = Column(Float, default=0.0)
    net_exposure = Column(Float, default=0.0)
    source = Column(String(20), default="paper")


class DayState(Base):
    __tablename__ = "day_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trading_date = Column(
        String(10), unique=True, nullable=False, index=True
    )  # YYYY-MM-DD
    opening_equity = Column(Float, nullable=False)
    eod_high_water_mark = Column(Float, nullable=False)
    daily_loss_floor = Column(Float, nullable=False)
    eod_trailing_floor = Column(Float, nullable=False)
    can_trade = Column(Boolean, default=True)
    breach_reason = Column(String(200))
    breach_time = Column(DateTime)
    last_updated = Column(DateTime, default=datetime.utcnow)


# ── Trading Objects ──────────────────────────────────────────


class SignalRecord(Base):
    __tablename__ = "signals"
    __table_args__ = (Index("ix_sig_ts", "timestamp"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String(20), nullable=False)
    horizon = Column(Integer, nullable=False)
    model_id = Column(String(100), nullable=False)
    side = Column(Integer, nullable=False)
    probability = Column(Float)
    confidence = Column(Float)
    regime = Column(String(30))


class Order(Base):
    __tablename__ = "orders"
    __table_args__ = (Index("ix_order_ts", "timestamp"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(100), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)
    order_type = Column(String(10), nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float)
    status = Column(String(20), nullable=False)
    filled_qty = Column(Float, default=0.0)
    filled_price = Column(Float)
    exchange_order_id = Column(String(100))
    broker = Column(String(20), default="paper")
    reason = Column(String(200))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime)


class Fill(Base):
    __tablename__ = "fills"
    __table_args__ = (Index("ix_fill_ts", "timestamp"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    fill_id = Column(String(100), unique=True, nullable=False)
    order_id = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    commission = Column(Float, default=0.0)
    slippage = Column(Float, default=0.0)
    broker = Column(String(20), default="paper")


class Position(Base):
    __tablename__ = "positions"
    __table_args__ = (UniqueConstraint("symbol", "broker", name="uq_position"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)
    quantity = Column(Float, nullable=False, default=0.0)
    avg_entry_price = Column(Float)
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    broker = Column(String(20), default="paper")
    updated_at = Column(DateTime, default=datetime.utcnow)

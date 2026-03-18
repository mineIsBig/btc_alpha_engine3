"""Shared type definitions and enums."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class BreachType(str, Enum):
    DAILY_LOSS = "daily_loss"
    EOD_TRAILING = "eod_trailing"
    KILL_SWITCH = "kill_switch"
    MANUAL = "manual"


class Horizon(int, Enum):
    H1 = 1
    H4 = 4
    H8 = 8
    H12 = 12
    H24 = 24


class RegimeLabel(str, Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    MEAN_REVERT = "mean_revert"
    CROWDED_LONG = "crowded_long"
    CROWDED_SHORT = "crowded_short"
    PANIC_FLUSH = "panic_flush"
    SQUEEZE = "squeeze"
    NEUTRAL = "neutral"


class Signal(BaseModel):
    """A trading signal from one model for one horizon."""

    timestamp: datetime
    symbol: str = "BTC"
    horizon: int
    model_id: str
    side: int  # -1, 0, 1
    probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    regime: str = "neutral"
    meta: dict = Field(default_factory=dict)


class TradeDecision(BaseModel):
    """Final aggregated trade decision."""

    timestamp: datetime
    symbol: str = "BTC"
    target_side: int  # -1, 0, 1
    target_size_usd: float
    target_size_coin: float
    reason: str
    signals: list[Signal] = Field(default_factory=list)
    risk_approved: bool = True
    meta: dict = Field(default_factory=dict)

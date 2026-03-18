"""Signal schema and helpers."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ModelSignal(BaseModel):
    """Signal from a single model."""

    model_id: str
    horizon: int
    side: int  # -1, 0, 1
    probability: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1, default=0.5)
    regime: str = "neutral"
    oos_sharpe: float = 0.0
    calibrated: bool = False


class AggregatedSignal(BaseModel):
    """Aggregated signal from ensemble."""

    timestamp: datetime
    symbol: str = "BTC"
    target_side: int  # -1, 0, 1
    raw_score: float  # continuous signal strength
    consensus_pct: float  # % of models agreeing
    avg_probability: float
    avg_confidence: float
    regime: str = "neutral"
    component_signals: list[ModelSignal] = Field(default_factory=list)
    reason: str = ""

"""Signal output format: long/short with position size, expected returns, TP/SL.

This is the agent's output. No live trades are executed — only signal recommendations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SignalOutput(BaseModel):
    """Standardized trading signal output from the autonomous agent.

    This is purely advisory — no orders are placed.
    """

    timestamp: datetime
    symbol: str = "BTC"

    # Direction and sizing
    direction: str = Field(..., description="'long', 'short', or 'flat'")
    position_size_pct: float = Field(
        0.0, ge=0.0, le=1.0, description="Position size as fraction of equity"
    )
    position_size_usd: float = Field(0.0, ge=0.0, description="Notional USD size")

    # Price levels
    entry_price: float = Field(0.0, description="Recommended entry price")
    take_profit: float = Field(0.0, description="Take profit level")
    stop_loss: float = Field(0.0, description="Stop loss level")

    # Expected returns
    expected_return_pct: float = Field(0.0, description="Expected return in %")
    expected_holding_hours: int = Field(
        0, description="Expected holding period in hours"
    )
    risk_reward_ratio: float = Field(0.0, description="Risk/reward ratio")

    # Confidence and regime
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    regime: str = "neutral"
    consensus_horizons: list[int] = Field(default_factory=list)

    # Agent reasoning
    reasoning: str = ""
    model_contributions: list[dict[str, Any]] = Field(default_factory=list)
    risk_assessment: str = ""

    # Iteration metadata
    agent_iteration: int = 0
    system_sharpe: float = 0.0
    system_drawdown: float = 0.0

    def to_summary(self) -> str:
        """Human-readable signal summary."""
        if self.direction == "flat":
            return f"[{self.timestamp.strftime('%Y-%m-%d %H:%M')}] FLAT — no trade recommended. Reason: {self.reasoning}"

        tp_dist = (
            abs(self.take_profit - self.entry_price) / self.entry_price * 100
            if self.entry_price > 0
            else 0
        )
        sl_dist = (
            abs(self.stop_loss - self.entry_price) / self.entry_price * 100
            if self.entry_price > 0
            else 0
        )

        return (
            f"[{self.timestamp.strftime('%Y-%m-%d %H:%M')}] {self.direction.upper()} BTC\n"
            f"  Entry: ${self.entry_price:,.2f} | Size: {self.position_size_pct*100:.1f}% (${self.position_size_usd:,.0f})\n"
            f"  TP: ${self.take_profit:,.2f} (+{tp_dist:.2f}%) | SL: ${self.stop_loss:,.2f} (-{sl_dist:.2f}%)\n"
            f"  Expected: {self.expected_return_pct:+.2f}% over {self.expected_holding_hours}h | R:R {self.risk_reward_ratio:.1f}\n"
            f"  Confidence: {self.confidence:.2f} | Regime: {self.regime}\n"
            f"  Horizons: {self.consensus_horizons} | Sharpe: {self.system_sharpe:.2f}\n"
            f"  Reasoning: {self.reasoning}"
        )


class AgentState(BaseModel):
    """Persisted state of the autonomous agent between iterations."""

    iteration: int = 0
    cumulative_pnl: float = 0.0
    rolling_sharpe: float = 0.0
    max_drawdown: float = 0.0
    total_signals: int = 0
    correct_signals: int = 0
    last_signal: SignalOutput | None = None
    weaknesses: list[str] = Field(default_factory=list)
    improvements_applied: list[str] = Field(default_factory=list)
    model_population_size: int = 0
    active_features: list[str] = Field(default_factory=list)
    last_reflection: str = ""
    system_version: str = "v0.1"

    # Evolution tracking
    evolution_version: int = 0
    retrain_pending: bool = False
    last_retrain_iteration: int = 0
    total_changes_executed: int = 0
    total_retrains: int = 0

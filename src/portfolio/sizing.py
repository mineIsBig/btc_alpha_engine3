"""Position sizing: volatility-targeting, drawdown-headroom-aware, with hard caps."""
from __future__ import annotations

import numpy as np

from src.common.config import load_yaml_config
from src.common.logging import get_logger

logger = get_logger(__name__)


class VolatilitySizer:
    """Position sizer using volatility targeting and drawdown headroom."""

    def __init__(
        self,
        vol_target: float | None = None,
        max_position_pct: float | None = None,
        max_leverage: float | None = None,
        min_headroom_pct: float | None = None,
    ):
        cfg = load_yaml_config("risk_limits.yaml")
        self.vol_target = vol_target or cfg.get("vol_target_annualized", 0.30)
        self.max_position_pct = max_position_pct or cfg.get("max_position_pct", 0.25)
        self.max_leverage = max_leverage or cfg.get("max_leverage", 3.0)
        self.min_headroom_pct = min_headroom_pct or cfg.get("min_headroom_pct", 0.02)

    def compute_size(
        self,
        equity: float,
        realized_vol: float,
        signal_strength: float,
        headroom_to_breach: float,
        current_price: float,
    ) -> dict[str, float]:
        """Compute target position size.

        Args:
            equity: current account equity
            realized_vol: annualized realized volatility of the asset
            signal_strength: signal confidence in [0, 1]
            headroom_to_breach: remaining headroom before risk breach (fraction)
            current_price: current asset price

        Returns:
            dict with target_size_usd, target_size_coin, sizing_reason
        """
        if equity <= 0 or current_price <= 0:
            return {"target_size_usd": 0.0, "target_size_coin": 0.0, "reason": "zero_equity_or_price"}

        # Check headroom
        if headroom_to_breach < self.min_headroom_pct:
            return {"target_size_usd": 0.0, "target_size_coin": 0.0, "reason": "insufficient_headroom"}

        # Volatility-based sizing
        if realized_vol > 0:
            vol_size_pct = self.vol_target / realized_vol
        else:
            vol_size_pct = self.max_position_pct

        # Scale by signal strength
        raw_size_pct = vol_size_pct * signal_strength

        # Headroom-aware scaling: reduce size as we approach breach
        headroom_factor = min(1.0, headroom_to_breach / (self.min_headroom_pct * 3))
        raw_size_pct *= headroom_factor

        # Apply hard caps
        capped_size_pct = min(raw_size_pct, self.max_position_pct, self.max_leverage)
        capped_size_pct = max(capped_size_pct, 0.0)

        target_usd = equity * capped_size_pct
        target_coin = target_usd / current_price

        reason = (
            f"vol_size={vol_size_pct:.3f}, signal={signal_strength:.3f}, "
            f"headroom={headroom_to_breach:.4f}, final_pct={capped_size_pct:.3f}"
        )

        return {
            "target_size_usd": round(target_usd, 2),
            "target_size_coin": round(target_coin, 6),
            "reason": reason,
        }

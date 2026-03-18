"""Portfolio constraints: position limits, exposure limits."""

from __future__ import annotations

from src.common.config import load_yaml_config
from src.common.logging import get_logger

logger = get_logger(__name__)


class PortfolioConstraints:
    """Enforce portfolio-level constraints."""

    def __init__(self):
        cfg = load_yaml_config("risk_limits.yaml")
        self.max_position_pct = cfg.get("max_position_pct", 0.25)
        self.max_gross_exposure_pct = cfg.get("max_gross_exposure_pct", 1.0)
        self.max_leverage = cfg.get("max_leverage", 3.0)

    def check_position_limit(
        self, target_size_usd: float, equity: float
    ) -> tuple[float, str]:
        """Ensure position doesn't exceed max percentage of equity.

        Returns (adjusted_size_usd, reason).
        """
        if equity <= 0:
            return 0.0, "zero_equity"

        max_size = equity * self.max_position_pct
        if abs(target_size_usd) > max_size:
            adjusted = max_size if target_size_usd > 0 else -max_size
            return adjusted, f"position_capped_at_{self.max_position_pct*100:.0f}%"
        return target_size_usd, "within_limits"

    def check_gross_exposure(
        self,
        current_gross_exposure: float,
        additional_exposure: float,
        equity: float,
    ) -> tuple[float, str]:
        """Check if adding exposure stays within gross exposure limits.

        Returns (allowed_additional, reason).
        """
        if equity <= 0:
            return 0.0, "zero_equity"

        max_exposure = equity * self.max_gross_exposure_pct
        current_pct = current_gross_exposure / equity
        available = max_exposure - current_gross_exposure

        if additional_exposure > available:
            return max(0, available), f"gross_exposure_capped current={current_pct:.2f}"
        return additional_exposure, "within_limits"

    def validate_order(
        self,
        target_size_usd: float,
        equity: float,
        current_gross_exposure: float,
        can_trade: bool,
    ) -> tuple[float, bool, str]:
        """Full order validation.

        Returns (adjusted_size_usd, approved, reason).
        """
        if not can_trade:
            return 0.0, False, "trading_disabled"

        if abs(target_size_usd) < 10.0:
            return 0.0, False, "below_minimum_size"

        # Position limit
        size, reason = self.check_position_limit(target_size_usd, equity)

        # Gross exposure
        size, exp_reason = self.check_gross_exposure(
            current_gross_exposure,
            abs(size),
            equity,
        )
        if "capped" in exp_reason:
            reason = exp_reason

        if size > 0 or target_size_usd < 0:
            return size if target_size_usd > 0 else -size, True, reason
        return 0.0, False, reason

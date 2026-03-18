"""Stop-loss and take-profit logic."""
from __future__ import annotations

from src.common.logging import get_logger

logger = get_logger(__name__)


class StopManager:
    """Manage stop-loss and take-profit levels."""

    def __init__(
        self,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        trailing_stop_pct: float | None = None,
    ):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct

    def check_stop(
        self,
        entry_price: float,
        current_price: float,
        side: int,
        high_since_entry: float | None = None,
    ) -> tuple[bool, str]:
        """Check if stop conditions are met.

        Returns (should_stop, reason).
        """
        if side == 0 or entry_price <= 0:
            return False, ""

        if side == 1:  # Long
            pnl_pct = (current_price - entry_price) / entry_price
            if pnl_pct <= -self.stop_loss_pct:
                return True, f"stop_loss_long ({pnl_pct:.4f})"
            if pnl_pct >= self.take_profit_pct:
                return True, f"take_profit_long ({pnl_pct:.4f})"
            if self.trailing_stop_pct and high_since_entry:
                trail_pct = (current_price - high_since_entry) / high_since_entry
                if trail_pct <= -self.trailing_stop_pct:
                    return True, f"trailing_stop_long ({trail_pct:.4f})"
        elif side == -1:  # Short
            pnl_pct = (entry_price - current_price) / entry_price
            if pnl_pct <= -self.stop_loss_pct:
                return True, f"stop_loss_short ({pnl_pct:.4f})"
            if pnl_pct >= self.take_profit_pct:
                return True, f"take_profit_short ({pnl_pct:.4f})"

        return False, ""

"""Kill switch: circuit breaker for runaway trading activity."""
from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta

from src.common.config import load_yaml_config
from src.common.logging import get_logger
from src.common.time_utils import utc_now

logger = get_logger(__name__)


class KillSwitch:
    """Circuit breaker that disables trading on anomalous activity."""

    def __init__(
        self,
        max_orders_per_hour: int | None = None,
        max_fills_per_hour: int | None = None,
        max_consecutive_losses: int | None = None,
        cooldown_minutes: int | None = None,
    ):
        cfg = load_yaml_config("risk_limits.yaml").get("kill_switch", {})
        self.max_orders_per_hour = max_orders_per_hour or cfg.get("max_orders_per_hour", 20)
        self.max_fills_per_hour = max_fills_per_hour or cfg.get("max_fills_per_hour", 10)
        self.max_consecutive_losses = max_consecutive_losses or cfg.get("max_consecutive_losses", 5)
        self.cooldown_minutes = cooldown_minutes or cfg.get("cooldown_minutes", 30)

        self._order_times: deque[datetime] = deque()
        self._fill_times: deque[datetime] = deque()
        self._consecutive_losses: int = 0
        self._triggered: bool = False
        self._trigger_time: datetime | None = None
        self._trigger_reason: str = ""

    @property
    def is_triggered(self) -> bool:
        """Check if kill switch is active, considering cooldown."""
        if not self._triggered:
            return False
        if self._trigger_time and self.cooldown_minutes > 0:
            elapsed = (utc_now() - self._trigger_time).total_seconds() / 60
            if elapsed > self.cooldown_minutes:
                self.reset()
                return False
        return True

    @property
    def trigger_reason(self) -> str:
        return self._trigger_reason

    def record_order(self) -> bool:
        """Record an order submission. Returns False if kill switch triggers."""
        now = utc_now()
        self._order_times.append(now)
        self._prune_window(self._order_times)

        if len(self._order_times) > self.max_orders_per_hour:
            self._trigger(f"max_orders_exceeded ({len(self._order_times)}/{self.max_orders_per_hour})")
            return False
        return True

    def record_fill(self, pnl: float) -> bool:
        """Record a fill with PnL. Returns False if kill switch triggers."""
        now = utc_now()
        self._fill_times.append(now)
        self._prune_window(self._fill_times)

        if len(self._fill_times) > self.max_fills_per_hour:
            self._trigger(f"max_fills_exceeded ({len(self._fill_times)}/{self.max_fills_per_hour})")
            return False

        # Track consecutive losses
        if pnl < 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self.max_consecutive_losses:
                self._trigger(f"consecutive_losses ({self._consecutive_losses})")
                return False
        else:
            self._consecutive_losses = 0

        return True

    def _trigger(self, reason: str) -> None:
        self._triggered = True
        self._trigger_time = utc_now()
        self._trigger_reason = reason
        logger.error("KILL_SWITCH_TRIGGERED", reason=reason)

    def reset(self) -> None:
        """Reset the kill switch."""
        self._triggered = False
        self._trigger_time = None
        self._trigger_reason = ""
        self._consecutive_losses = 0
        logger.warning("kill_switch_reset")

    def _prune_window(self, times: deque[datetime]) -> None:
        """Remove entries older than 1 hour."""
        cutoff = utc_now() - timedelta(hours=1)
        while times and times[0] < cutoff:
            times.popleft()

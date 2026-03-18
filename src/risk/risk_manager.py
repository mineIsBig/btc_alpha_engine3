"""Unified risk manager combining all risk components."""
from __future__ import annotations

from typing import Any

from src.common.logging import get_logger
from src.risk.account_state import AccountState
from src.risk.drawdown_rules import DrawdownRuleEngine
from src.risk.kill_switch import KillSwitch
from src.risk.exposure import ExposureTracker

logger = get_logger(__name__)


class RiskManager:
    """Centralized risk manager for pre-trade and intraday checks."""

    def __init__(self, initial_equity: float = 100000.0, source: str = "paper"):
        self.account = AccountState(initial_equity=initial_equity, source=source)
        self.drawdown = DrawdownRuleEngine()
        self.kill_switch = KillSwitch()
        self.exposure = ExposureTracker()

        # Initialize the first day
        self.drawdown.initialize_day(initial_equity)

    @property
    def can_trade(self) -> bool:
        """Check all risk gates."""
        if self.kill_switch.is_triggered:
            return False
        return self.drawdown.can_trade

    def pre_trade_check(self, target_size_usd: float) -> tuple[bool, str]:
        """Full pre-trade risk check.

        Returns (approved, reason).
        """
        if self.kill_switch.is_triggered:
            return False, f"kill_switch: {self.kill_switch.trigger_reason}"

        can, reason = self.drawdown.check_intraday(self.account.equity)
        if not can:
            return False, f"drawdown: {reason}"

        ok, exp_reason = self.exposure.check(
            self.account.gross_exposure + abs(target_size_usd),
            self.account.net_exposure + target_size_usd,
            self.account.equity,
        )
        if not ok:
            return False, f"exposure: {exp_reason}"

        headroom = self.drawdown.get_headroom(self.account.equity)
        if headroom < 0.02:
            return False, f"headroom_too_small: {headroom:.4f}"

        return True, "approved"

    def on_equity_update(self, equity: float) -> tuple[bool, str | None]:
        """Called whenever equity changes (fill, mark-to-market).

        Returns (can_trade, breach_reason).
        """
        self.account.equity = equity
        return self.drawdown.check_intraday(equity)

    def on_fill(self, pnl: float) -> tuple[bool, str | None]:
        """Called on every fill.

        Returns (can_continue, reason).
        """
        self.account.record_fill(pnl)

        # Record in kill switch
        if not self.kill_switch.record_fill(pnl):
            return False, f"kill_switch: {self.kill_switch.trigger_reason}"

        # Check drawdown rules
        return self.drawdown.check_intraday(self.account.equity)

    def on_order_submitted(self) -> bool:
        """Record order submission in kill switch."""
        return self.kill_switch.record_order()

    def on_day_reset(self, equity: float) -> None:
        """Called at 00:00 UTC."""
        self.drawdown.end_of_day_update(equity)
        self.drawdown.initialize_day(equity)
        self.account.reset_daily()

    def get_headroom(self) -> float:
        """Get current headroom to breach."""
        return self.drawdown.get_headroom(self.account.equity)

    def get_state_summary(self) -> dict[str, Any]:
        return {
            "equity": self.account.equity,
            "can_trade": self.can_trade,
            "headroom": self.get_headroom(),
            "daily_floor": self.drawdown.daily_loss_floor,
            "eod_floor": self.drawdown.eod_trailing_floor,
            "eod_hwm": self.drawdown.eod_high_water_mark,
            "kill_switch": self.kill_switch.is_triggered,
            "breach_reason": self.drawdown.breach_reason,
        }

    def load_persisted_state(self) -> None:
        """Load risk state from database on restart."""
        self.drawdown.load_state()
        logger.info("risk_state_loaded", can_trade=self.can_trade)

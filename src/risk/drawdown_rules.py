"""Drawdown rule engine implementing the two hard risk rules.

Rule 1: Daily Loss Limit (5%)
  - Account equity cannot drop more than 5% from the day's opening equity
    at any point during the day.

Rule 2: EOD Trailing Loss Limit (5%)
  - End-of-day account equity cannot drop more than 5% from the
    end-of-day high water mark.

Trading day resets at 00:00 UTC.

EOD high water mark updates ONLY from end-of-day equity, NOT intraday peaks.
Intraday checks happen on every account update.
"""

from __future__ import annotations

from datetime import datetime

from src.common.config import get_settings
from src.common.logging import get_logger
from src.common.time_utils import utc_now, trading_day_start, is_new_trading_day
from src.storage.database import session_scope
from src.storage.models import DayState

logger = get_logger(__name__)


class DrawdownRuleEngine:
    """Enforces daily loss and EOD trailing loss rules."""

    def __init__(
        self,
        daily_loss_limit_pct: float | None = None,
        eod_trailing_loss_limit_pct: float | None = None,
        flatten_on_breach: bool = True,
        auto_reset_next_day: bool = False,
    ):
        settings = get_settings()
        self.daily_loss_limit_pct = (
            daily_loss_limit_pct or settings.daily_loss_limit_pct
        )
        self.eod_trailing_loss_limit_pct = (
            eod_trailing_loss_limit_pct or settings.eod_trailing_loss_limit_pct
        )
        self.flatten_on_breach = flatten_on_breach
        self.auto_reset_next_day = auto_reset_next_day

        # In-memory state
        self.opening_equity: float = 0.0
        self.eod_high_water_mark: float = 0.0
        self.daily_loss_floor: float = 0.0
        self.eod_trailing_floor: float = 0.0
        self.can_trade: bool = True
        self.breach_reason: str | None = None
        self.breach_time: datetime | None = None
        self.current_trading_date: str | None = None
        self._last_reset: datetime | None = None

    def initialize_day(self, equity: float, force: bool = False) -> None:
        """Initialize or reset for a new trading day.

        Called at 00:00 UTC or on first startup.
        """
        today = trading_day_start().strftime("%Y-%m-%d")

        if self.current_trading_date == today and not force:
            return  # Already initialized

        self.opening_equity = equity
        self.daily_loss_floor = equity * (1.0 - self.daily_loss_limit_pct)

        # EOD HWM: on new day, update from previous day's closing equity
        if self.eod_high_water_mark == 0.0:
            self.eod_high_water_mark = equity
        else:
            # Update HWM only from end-of-day equity (the equity we see at day reset)
            self.eod_high_water_mark = max(self.eod_high_water_mark, equity)

        self.eod_trailing_floor = self.eod_high_water_mark * (
            1.0 - self.eod_trailing_loss_limit_pct
        )

        # Reset trading permission if auto-reset enabled
        if self.auto_reset_next_day or self.current_trading_date is None:
            self.can_trade = True
            self.breach_reason = None
            self.breach_time = None

        self.current_trading_date = today
        self._last_reset = utc_now()

        logger.info(
            "day_initialized",
            date=today,
            opening_equity=equity,
            daily_floor=self.daily_loss_floor,
            eod_hwm=self.eod_high_water_mark,
            eod_floor=self.eod_trailing_floor,
            can_trade=self.can_trade,
        )

        self._persist_state()

    def check_intraday(self, current_equity: float) -> tuple[bool, str | None]:
        """Check intraday risk rules. Called on every equity update.

        Returns (can_trade, breach_reason).
        """
        if not self.can_trade:
            return False, self.breach_reason

        # Check for new day
        if is_new_trading_day(self._last_reset):
            self.initialize_day(current_equity)

        # Rule 1: Daily Loss Limit
        if current_equity < self.daily_loss_floor:
            self._trigger_breach(
                f"DAILY_LOSS: equity {current_equity:.2f} < floor {self.daily_loss_floor:.2f} "
                f"(opening {self.opening_equity:.2f}, limit {self.daily_loss_limit_pct*100:.1f}%)"
            )
            return False, self.breach_reason

        # Rule 2: EOD Trailing Loss Limit
        if current_equity < self.eod_trailing_floor:
            self._trigger_breach(
                f"EOD_TRAILING: equity {current_equity:.2f} < floor {self.eod_trailing_floor:.2f} "
                f"(hwm {self.eod_high_water_mark:.2f}, limit {self.eod_trailing_loss_limit_pct*100:.1f}%)"
            )
            return False, self.breach_reason

        return True, None

    def end_of_day_update(self, closing_equity: float) -> None:
        """Called at end of trading day to update EOD HWM.

        EOD HWM updates ONLY here, not from intraday peaks.
        """
        self.eod_high_water_mark = max(self.eod_high_water_mark, closing_equity)
        self.eod_trailing_floor = self.eod_high_water_mark * (
            1.0 - self.eod_trailing_loss_limit_pct
        )
        logger.info(
            "eod_update",
            closing_equity=closing_equity,
            eod_hwm=self.eod_high_water_mark,
            eod_floor=self.eod_trailing_floor,
        )
        self._persist_state()

    def _trigger_breach(self, reason: str) -> None:
        """Handle a rule breach."""
        self.can_trade = False
        self.breach_reason = reason
        self.breach_time = utc_now()
        logger.error("RISK_BREACH", reason=reason, time=self.breach_time.isoformat())
        self._persist_state()

    def force_disable(self, reason: str = "manual_disable") -> None:
        """Manually disable trading."""
        self._trigger_breach(reason)

    def force_enable(self) -> None:
        """Manually re-enable trading (operator intervention)."""
        self.can_trade = True
        self.breach_reason = None
        self.breach_time = None
        logger.warning("trading_force_enabled")
        self._persist_state()

    def get_headroom(self, current_equity: float) -> float:
        """Get minimum headroom to any breach as a fraction of equity.

        Returns the fraction of equity remaining before the tighter rule triggers.
        """
        if current_equity <= 0:
            return 0.0

        daily_headroom = (current_equity - self.daily_loss_floor) / current_equity
        eod_headroom = (current_equity - self.eod_trailing_floor) / current_equity

        return max(0.0, min(daily_headroom, eod_headroom))

    def _persist_state(self) -> None:
        """Persist current day state to database."""
        if not self.current_trading_date:
            return

        try:
            with session_scope() as session:
                existing = (
                    session.query(DayState)
                    .filter_by(trading_date=self.current_trading_date)
                    .first()
                )

                if existing:
                    existing.opening_equity = self.opening_equity
                    existing.eod_high_water_mark = self.eod_high_water_mark
                    existing.daily_loss_floor = self.daily_loss_floor
                    existing.eod_trailing_floor = self.eod_trailing_floor
                    existing.can_trade = self.can_trade
                    existing.breach_reason = self.breach_reason
                    existing.breach_time = self.breach_time
                    existing.last_updated = utc_now()
                else:
                    session.add(
                        DayState(
                            trading_date=self.current_trading_date,
                            opening_equity=self.opening_equity,
                            eod_high_water_mark=self.eod_high_water_mark,
                            daily_loss_floor=self.daily_loss_floor,
                            eod_trailing_floor=self.eod_trailing_floor,
                            can_trade=self.can_trade,
                            breach_reason=self.breach_reason,
                            breach_time=self.breach_time,
                            last_updated=utc_now(),
                        )
                    )
        except Exception as e:
            logger.error("persist_day_state_failed", error=str(e))

    def load_state(self) -> None:
        """Load persisted day state from database."""
        today = trading_day_start().strftime("%Y-%m-%d")
        try:
            with session_scope() as session:
                state = session.query(DayState).filter_by(trading_date=today).first()
                if state:
                    self.opening_equity = state.opening_equity
                    self.eod_high_water_mark = state.eod_high_water_mark
                    self.daily_loss_floor = state.daily_loss_floor
                    self.eod_trailing_floor = state.eod_trailing_floor
                    self.can_trade = state.can_trade
                    self.breach_reason = state.breach_reason
                    self.breach_time = state.breach_time
                    self.current_trading_date = today
                    logger.info(
                        "day_state_loaded", date=today, can_trade=self.can_trade
                    )
        except Exception as e:
            logger.error("load_day_state_failed", error=str(e))

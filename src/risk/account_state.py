"""Account state tracking for risk management."""

from __future__ import annotations

from typing import Any

from src.common.logging import get_logger
from src.common.time_utils import utc_now
from src.storage.database import session_scope
from src.storage.models import AccountSnapshot

logger = get_logger(__name__)


class AccountState:
    """Track and persist account equity state."""

    def __init__(self, initial_equity: float = 100000.0, source: str = "paper"):
        self.equity = initial_equity
        self.cash = initial_equity
        self.unrealized_pnl = 0.0
        self.realized_pnl_today = 0.0
        self.gross_exposure = 0.0
        self.net_exposure = 0.0
        self.source = source

    def update_from_positions(
        self, positions: list[dict[str, Any]], current_prices: dict[str, float]
    ) -> None:
        """Update account state from current positions and prices."""
        total_unrealized = 0.0
        gross = 0.0
        net = 0.0

        for pos in positions:
            symbol = pos["symbol"]
            qty = pos["quantity"]
            side_mult = 1 if pos["side"] == "long" else -1
            entry_price = pos.get("avg_entry_price", 0)
            current_price = current_prices.get(symbol, entry_price)

            if entry_price > 0 and current_price > 0:
                pnl = side_mult * qty * (current_price - entry_price)
                total_unrealized += pnl

            notional = abs(qty * current_price)
            gross += notional
            net += side_mult * notional

        self.unrealized_pnl = total_unrealized
        self.equity = self.cash + total_unrealized
        self.gross_exposure = gross
        self.net_exposure = net

    def record_fill(self, pnl: float) -> None:
        """Record realized PnL from a fill."""
        self.cash += pnl
        self.realized_pnl_today += pnl
        self.equity = self.cash + self.unrealized_pnl

    def snapshot(self) -> dict[str, Any]:
        return {
            "timestamp": utc_now(),
            "equity": self.equity,
            "cash": self.cash,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl_today": self.realized_pnl_today,
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "source": self.source,
        }

    def persist_snapshot(self) -> None:
        snap = self.snapshot()
        with session_scope() as session:
            session.add(AccountSnapshot(**snap))

    def reset_daily(self) -> None:
        self.realized_pnl_today = 0.0
        logger.info("daily_reset", equity=self.equity)

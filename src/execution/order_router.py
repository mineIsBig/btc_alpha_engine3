"""Order router: dispatches orders to paper broker or live exchange."""

from __future__ import annotations

from typing import Any

from src.common.config import get_settings
from src.common.logging import get_logger
from src.execution.paper_broker import PaperBroker
from src.execution.hyperliquid_adapter import HyperliquidAdapter

logger = get_logger(__name__)


class OrderRouter:
    """Routes orders to the appropriate broker."""

    def __init__(self):
        self.settings = get_settings()
        self.paper_broker = PaperBroker()
        self._hl_adapter: HyperliquidAdapter | None = None

    @property
    def _live_adapter(self) -> HyperliquidAdapter:
        if self._hl_adapter is None:
            self._hl_adapter = HyperliquidAdapter()
        return self._hl_adapter

    @property
    def is_paper(self) -> bool:
        return self.settings.paper_mode or not self.settings.live_trading_enabled

    def set_price(self, symbol: str, price: float) -> None:
        self.paper_broker.set_price(symbol, price)

    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        price: float | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        if self.is_paper:
            return self.paper_broker.submit_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price,
                reason=reason,
            )
        else:
            logger.info("live_order_attempt", symbol=symbol, side=side, qty=quantity)
            return self._live_adapter.submit_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price,
            )

    def flatten_all(self, reason: str = "risk_breach") -> list[dict[str, Any]]:
        if self.is_paper:
            return self.paper_broker.flatten_all(reason)
        else:
            try:
                self._live_adapter.cancel_all()
            except Exception as e:
                logger.error("live_cancel_all_failed", error=str(e))
            return []

    def get_positions(self) -> list[dict[str, Any]]:
        if self.is_paper:
            return self.paper_broker.get_all_positions()
        else:
            state = self._live_adapter.get_account_state()
            return state.get("positions", [])

    def get_mid_price(self, symbol: str) -> float:
        if self.is_paper:
            return self.paper_broker.get_mid_price(symbol)
        else:
            return self._live_adapter.get_mid_price(symbol) or 0.0

    def close(self) -> None:
        if self._hl_adapter:
            self._hl_adapter.close()

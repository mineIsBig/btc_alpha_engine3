"""Hyperliquid exchange adapter - safely wrapped behind feature flag.

WARNING: Live order submission requires EIP-712 signing which is not
fully implemented. This adapter is for shadow/monitoring mode only.
Set LIVE_TRADING_ENABLED=true only after verifying signing logic.
"""
from __future__ import annotations

from typing import Any

from src.common.config import get_settings
from src.common.logging import get_logger
from src.data.hyperliquid_client import HyperliquidClient, HyperliquidError

logger = get_logger(__name__)


class HyperliquidAdapter:
    """Adapter for Hyperliquid exchange operations.

    All mutating operations are gated behind live_trading_enabled flag.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = HyperliquidClient()
        self.live_enabled = self.settings.live_trading_enabled and not self.settings.paper_mode

    @property
    def is_live(self) -> bool:
        return self.live_enabled and self.client.is_authenticated

    def get_mid_price(self, symbol: str = "BTC") -> float | None:
        return self.client.get_mid_price(symbol)

    def get_account_state(self) -> dict[str, Any]:
        if not self.client.wallet_address:
            return {"equity": 0, "positions": []}
        state = self.client.get_user_state()
        margin = state.get("marginSummary", {})
        positions = []
        for ap in state.get("assetPositions", []):
            p = ap.get("position", {})
            if float(p.get("szi", 0)) != 0:
                positions.append({
                    "symbol": p.get("coin", ""),
                    "side": "long" if float(p.get("szi", 0)) > 0 else "short",
                    "quantity": abs(float(p.get("szi", 0))),
                    "avg_entry_price": float(p.get("entryPx", 0)),
                    "unrealized_pnl": float(p.get("unrealizedPnl", 0)),
                })
        return {
            "equity": float(margin.get("accountValue", 0)),
            "margin_used": float(margin.get("totalMarginUsed", 0)),
            "positions": positions,
        }

    def submit_order(self, symbol: str, side: str, quantity: float,
                     order_type: str = "market", price: float | None = None) -> dict[str, Any]:
        """Submit order to Hyperliquid.

        Only works if live_trading_enabled=True and wallet is configured.
        Otherwise raises HyperliquidError.
        """
        if not self.is_live:
            raise HyperliquidError(
                "Live trading not enabled. Use paper_broker instead. "
                "Set LIVE_TRADING_ENABLED=true and configure wallet to enable."
            )

        is_buy = side == "buy"
        # TODO: Implement when EIP-712 signing is verified
        return self.client.place_order(
            symbol=symbol, is_buy=is_buy, size=quantity,
            price=price, order_type=order_type,
        )

    def cancel_all(self, symbol: str | None = None) -> dict[str, Any]:
        if not self.is_live:
            raise HyperliquidError("Live trading not enabled")
        return self.client.cancel_all_orders(symbol)

    def close(self) -> None:
        self.client.close()

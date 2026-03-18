"""Hyperliquid exchange client.

Implements info and exchange endpoints per Hyperliquid API docs.
Live trading is OFF by default behind config flag.
Falls back to paper broker if auth not configured.

Hyperliquid API:
- Info endpoint (POST /info): public market data, no auth required
- Exchange endpoint (POST /exchange): order management, requires auth
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.common.config import get_settings, load_yaml_config
from src.common.logging import get_logger

logger = get_logger(__name__)


class HyperliquidError(Exception):
    pass


class HyperliquidClient:
    """Client for Hyperliquid DEX API."""

    def __init__(self) -> None:
        settings = get_settings()
        ds_cfg = load_yaml_config("data_sources.yaml")["hyperliquid"]

        self.base_url = settings.hyperliquid_base_url or ds_cfg.get("testnet_url", "")
        self.info_url = f"{self.base_url}{ds_cfg['info_endpoint']}"
        self.exchange_url = f"{self.base_url}{ds_cfg['exchange_endpoint']}"
        self.wallet_address = settings.hyperliquid_wallet_address
        self.api_key = settings.hyperliquid_api_key
        self.api_secret = settings.hyperliquid_api_secret
        self.live_enabled = settings.live_trading_enabled and not settings.paper_mode

        self._client = httpx.Client(timeout=30.0)

        if self.live_enabled and not self.wallet_address:
            logger.warning("live_enabled_but_no_wallet", msg="Disabling live trading")
            self.live_enabled = False

    def close(self) -> None:
        self._client.close()

    @property
    def is_authenticated(self) -> bool:
        return bool(self.wallet_address and self.api_secret)

    # ── Info Endpoints (Public) ──────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def _post_info(self, payload: dict[str, Any]) -> Any:
        """POST to /info endpoint."""
        resp = self._client.post(self.info_url, json=payload)
        resp.raise_for_status()
        return resp.json()

    def get_meta(self) -> dict[str, Any]:
        """Get exchange metadata including asset info."""
        return self._post_info({"type": "meta"})

    def get_all_mids(self) -> dict[str, str]:
        """Get all mid prices. Returns {symbol: mid_price}."""
        return self._post_info({"type": "allMids"})

    def get_candles(
        self,
        symbol: str = "BTC",
        interval: str = "1h",
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get candle data.

        Interval options: 1m, 5m, 15m, 1h, 4h, 1d
        start_time/end_time in milliseconds.
        """
        payload: dict[str, Any] = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "interval": interval,
                "startTime": start_time or 0,
            },
        }
        if end_time:
            payload["req"]["endTime"] = end_time
        return self._post_info(payload)

    def get_user_state(self, address: str | None = None) -> dict[str, Any]:
        """Get user's account state including positions."""
        addr = address or self.wallet_address
        if not addr:
            return {"marginSummary": {}, "assetPositions": []}
        return self._post_info({"type": "clearinghouseState", "user": addr})

    def get_user_fills(self, address: str | None = None) -> list[dict[str, Any]]:
        """Get user's recent fills."""
        addr = address or self.wallet_address
        if not addr:
            return []
        return self._post_info({"type": "userFills", "user": addr})

    def get_open_orders(self, address: str | None = None) -> list[dict[str, Any]]:
        """Get user's open orders."""
        addr = address or self.wallet_address
        if not addr:
            return []
        return self._post_info({"type": "openOrders", "user": addr})

    def get_l2_book(self, symbol: str = "BTC") -> dict[str, Any]:
        """Get L2 order book snapshot."""
        return self._post_info({"type": "l2Book", "coin": symbol})

    def get_mid_price(self, symbol: str = "BTC") -> float | None:
        """Get mid price for a symbol."""
        mids = self.get_all_mids()
        mid = mids.get(symbol)
        return float(mid) if mid else None

    # ── Exchange Endpoints (Authenticated) ───────────────────
    # NOTE: Live order submission requires proper EIP-712 signing.
    # This is feature-flagged OFF by default.
    # The adapter pattern in execution/ wraps this safely.

    def place_order(
        self,
        symbol: str,
        is_buy: bool,
        size: float,
        price: float | None = None,
        order_type: str = "limit",
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        """Place an order on Hyperliquid.

        WARNING: This requires live_trading_enabled=True and valid auth.
        Returns order response or raises if not enabled.
        """
        if not self.live_enabled:
            raise HyperliquidError(
                "Live trading is disabled. Set LIVE_TRADING_ENABLED=true and configure wallet."
            )
        if not self.is_authenticated:
            raise HyperliquidError("Not authenticated. Configure wallet address and API secret.")

        # TODO: Implement EIP-712 message signing for production.
        # The exact signing flow depends on the Hyperliquid SDK version.
        # For now, this is a structured placeholder that documents the payload format.
        order_payload = {
            "type": "order",
            "orders": [{
                "a": self._coin_to_asset_id(symbol),  # asset index
                "b": is_buy,
                "p": str(price) if price else "0",
                "s": str(size),
                "r": reduce_only,
                "t": {"limit": {"tif": "Gtc"}} if order_type == "limit" else {"market": {}},
            }],
            "grouping": "na",
        }
        logger.info("hyperliquid_order_submit", payload=order_payload)

        # In production, this would be signed and POSTed to exchange_url
        raise NotImplementedError(
            "EIP-712 signing not implemented. Use paper_broker for now. "
            "See Hyperliquid Python SDK for signing reference."
        )

    def cancel_order(self, symbol: str, order_id: int) -> dict[str, Any]:
        """Cancel an order."""
        if not self.live_enabled:
            raise HyperliquidError("Live trading is disabled.")

        raise NotImplementedError("EIP-712 signing required. Use paper_broker.")

    def cancel_all_orders(self, symbol: str | None = None) -> dict[str, Any]:
        """Cancel all open orders."""
        if not self.live_enabled:
            raise HyperliquidError("Live trading is disabled.")

        raise NotImplementedError("EIP-712 signing required. Use paper_broker.")

    def _coin_to_asset_id(self, symbol: str) -> int:
        """Map coin symbol to Hyperliquid asset index."""
        meta = self.get_meta()
        universe = meta.get("universe", [])
        for i, asset in enumerate(universe):
            if asset.get("name", "").upper() == symbol.upper():
                return i
        raise HyperliquidError(f"Unknown symbol: {symbol}")

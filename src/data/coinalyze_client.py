"""Coinalyze API v1 client for derivatives data.

Endpoints implemented per Coinalyze API docs (https://api.coinalyze.net/v1/doc/):
- Funding rate history
- Open interest history
- Liquidation history
- Long/short ratio history
- OHLCV history (used for taker buy/sell volume)

Rate limiting: 40 requests/minute with retry + exponential backoff.

NOTE: Coinalyze uses unix timestamps in seconds (not milliseconds).
      Symbols use exchange-specific format, e.g. "BTCUSDT_PERP.A" for Binance.
      Intraday data retention is limited to 1500-2000 datapoints.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import httpx
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.common.config import get_settings, load_yaml_config
from src.common.logging import get_logger

logger = get_logger(__name__)

# Mapping from generic symbol to Coinalyze exchange-specific symbol.
# Coinalyze requires exchange-suffixed symbols like "BTCUSDT_PERP.A" (Binance).
SYMBOL_MAP = {
    "BTC": "BTCUSDT_PERP.A",
    "ETH": "ETHUSDT_PERP.A",
    "SOL": "SOLUSDT_PERP.A",
}

# Mapping from our interval codes to Coinalyze interval strings.
INTERVAL_MAP = {
    "h1": "1hour",
    "1h": "1hour",
    "h4": "4hour",
    "4h": "4hour",
    "d1": "daily",
    "1d": "daily",
    "1hour": "1hour",
    "4hour": "4hour",
    "daily": "daily",
}


class CoinalyzeAPIError(Exception):
    """Raised on non-success Coinalyze API responses."""

    pass


class CoinalyzeRateLimitError(CoinalyzeAPIError):
    """Raised when rate limited (HTTP 429)."""

    pass


class CoinalyzeClient:
    """Client for Coinalyze REST API v1."""

    def __init__(self) -> None:
        settings = get_settings()
        ds_cfg = load_yaml_config("data_sources.yaml")["coinalyze"]

        self.base_url = ds_cfg["base_url"]
        self.api_key = settings.coinalyze_api_key
        self.rate_limit_per_min = ds_cfg.get("rate_limit_per_minute", 40)
        self.endpoints = ds_cfg["endpoints"]

        self._request_times: list[float] = []
        self._client = httpx.Client(
            timeout=30.0,
            headers={"accept": "application/json"},
        )

    def close(self) -> None:
        self._client.close()

    def _resolve_symbol(self, symbol: str) -> str:
        """Convert generic symbol (e.g. 'BTC') to Coinalyze format."""
        return SYMBOL_MAP.get(symbol, symbol)

    def _resolve_interval(self, interval: str) -> str:
        """Convert interval code to Coinalyze format."""
        return INTERVAL_MAP.get(interval, interval)

    def _throttle(self) -> None:
        """Simple sliding-window rate limiter."""
        now = time.monotonic()
        window = 60.0
        self._request_times = [t for t in self._request_times if now - t < window]
        if len(self._request_times) >= self.rate_limit_per_min:
            sleep_time = window - (now - self._request_times[0]) + 0.5
            logger.info("coinalyze_rate_limit_sleep", seconds=sleep_time)
            time.sleep(sleep_time)
        self._request_times.append(time.monotonic())

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=30),
        retry=retry_if_exception_type((httpx.HTTPStatusError, CoinalyzeRateLimitError)),
    )
    def _get(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Make a GET request with retry and rate limiting.

        Coinalyze returns a JSON array directly (not wrapped in a 'data' field).
        Each element has a 'symbol' and 'history' array.
        """
        self._throttle()
        url = f"{self.base_url}{endpoint}"
        # Add API key as query parameter
        all_params = {"api_key": self.api_key}
        if params:
            all_params.update(params)

        logger.debug(
            "coinalyze_request",
            url=url,
            params={k: v for k, v in all_params.items() if k != "api_key"},
        )

        resp = self._client.get(url, params=all_params)

        if resp.status_code == 429:
            raise CoinalyzeRateLimitError("Rate limited by Coinalyze")
        resp.raise_for_status()

        return resp.json()

    # ── Funding Rate History ────────────────────────────────────

    def fetch_funding_ohlc(
        self,
        symbol: str = "BTC",
        exchange: str | None = None,
        interval: str = "h1",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch historical funding rate OHLC data.

        Returns DataFrame with columns: timestamp, open, high, low, close
        """
        ca_symbol = self._resolve_symbol(symbol)
        ca_interval = self._resolve_interval(interval)
        params: dict[str, Any] = {
            "symbols": ca_symbol,
            "interval": ca_interval,
        }
        if start_time:
            params["from"] = int(start_time.timestamp())
        if end_time:
            params["to"] = int(end_time.timestamp())

        data = self._get(self.endpoints["funding_rate_history"], params)
        return self._parse_ohlc(data)

    # ── Open Interest History ───────────────────────────────────

    def fetch_oi_ohlc(
        self,
        symbol: str = "BTC",
        exchange: str | None = None,
        interval: str = "h1",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch historical OI OHLC data."""
        ca_symbol = self._resolve_symbol(symbol)
        ca_interval = self._resolve_interval(interval)
        params: dict[str, Any] = {
            "symbols": ca_symbol,
            "interval": ca_interval,
            "convert_to_usd": "true",
        }
        if start_time:
            params["from"] = int(start_time.timestamp())
        if end_time:
            params["to"] = int(end_time.timestamp())

        data = self._get(self.endpoints["oi_history"], params)
        return self._parse_ohlc(data)

    # ── Liquidation History ──────────────────────────────────────

    def fetch_liquidation_history(
        self,
        symbol: str = "BTC",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch liquidation history.

        Returns DataFrame with: timestamp, long_liquidations_usd, short_liquidations_usd,
        total_liquidations_usd, count
        """
        ca_symbol = self._resolve_symbol(symbol)
        params: dict[str, Any] = {
            "symbols": ca_symbol,
            "interval": "1hour",
        }
        if start_time:
            params["from"] = int(start_time.timestamp())
        if end_time:
            params["to"] = int(end_time.timestamp())

        data = self._get(self.endpoints["liquidation_history"], params)
        return self._parse_liquidations(data)

    # ── Long/Short Ratio History ─────────────────────────────────

    def fetch_long_short_ratio(
        self,
        symbol: str = "BTC",
        exchange: str | None = None,
        interval: str = "h1",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch long/short ratio history."""
        ca_symbol = self._resolve_symbol(symbol)
        ca_interval = self._resolve_interval(interval)
        params: dict[str, Any] = {
            "symbols": ca_symbol,
            "interval": ca_interval,
        }
        if start_time:
            params["from"] = int(start_time.timestamp())
        if end_time:
            params["to"] = int(end_time.timestamp())

        data = self._get(self.endpoints["long_short_ratio_history"], params)
        return self._parse_long_short(data)

    # ── Taker Buy/Sell (via OHLCV) ──────────────────────────────

    def fetch_taker_buy_sell(
        self,
        symbol: str = "BTC",
        interval: str = "h1",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch taker buy/sell volume via OHLCV history.

        Coinalyze OHLCV response includes 'v' (total volume) and 'bv' (buy volume).
        Sell volume is derived as v - bv.
        """
        ca_symbol = self._resolve_symbol(symbol)
        ca_interval = self._resolve_interval(interval)
        params: dict[str, Any] = {
            "symbols": ca_symbol,
            "interval": ca_interval,
        }
        if start_time:
            params["from"] = int(start_time.timestamp())
        if end_time:
            params["to"] = int(end_time.timestamp())

        data = self._get(self.endpoints["ohlcv_history"], params)
        return self._parse_taker_flow(data)

    # ── Parsers ──────────────────────────────────────────────────

    def _parse_ohlc(self, data: list[dict[str, Any]]) -> pd.DataFrame:
        """Parse OHLC response from Coinalyze API.

        Response format: [{"symbol": "...", "history": [{"t": ts, "o": val, "h": val, "l": val, "c": val}, ...]}]
        Timestamps are unix seconds.
        """
        if not data:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])

        rows = []
        for symbol_block in data:
            history = symbol_block.get("history", [])
            for item in history:
                rows.append(
                    {
                        "timestamp": datetime.fromtimestamp(item["t"], tz=timezone.utc),
                        "open": float(item.get("o", 0)),
                        "high": float(item.get("h", 0)),
                        "low": float(item.get("l", 0)),
                        "close": float(item.get("c", 0)),
                    }
                )

        df = pd.DataFrame(rows)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _parse_liquidations(self, data: list[dict[str, Any]]) -> pd.DataFrame:
        """Parse liquidation response.

        Response format: [{"symbol": "...", "history": [{"t": ts, "l": long_liq, "s": short_liq}, ...]}]
        """
        if not data:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "long_liquidations_usd",
                    "short_liquidations_usd",
                    "total_liquidations_usd",
                    "count",
                ]
            )

        rows = []
        for symbol_block in data:
            history = symbol_block.get("history", [])
            for item in history:
                long_liq = float(item.get("l", 0))
                short_liq = float(item.get("s", 0))
                rows.append(
                    {
                        "timestamp": datetime.fromtimestamp(item["t"], tz=timezone.utc),
                        "long_liquidations_usd": long_liq,
                        "short_liquidations_usd": short_liq,
                        "total_liquidations_usd": long_liq + short_liq,
                        "count": 0,  # Coinalyze doesn't provide count; default to 0
                    }
                )

        df = pd.DataFrame(rows)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _parse_long_short(self, data: list[dict[str, Any]]) -> pd.DataFrame:
        """Parse long/short ratio response.

        Response format: [{"symbol": "...", "history": [{"t": ts, "r": ratio, "l": long%, "s": short%}, ...]}]
        """
        if not data:
            return pd.DataFrame(
                columns=["timestamp", "long_ratio", "short_ratio", "long_short_ratio"]
            )

        rows = []
        for symbol_block in data:
            history = symbol_block.get("history", [])
            for item in history:
                long_pct = float(item.get("l", 0.5))
                short_pct = float(item.get("s", 0.5))
                ratio = float(
                    item.get("r", long_pct / short_pct if short_pct > 0 else 1.0)
                )
                rows.append(
                    {
                        "timestamp": datetime.fromtimestamp(item["t"], tz=timezone.utc),
                        "long_ratio": long_pct,
                        "short_ratio": short_pct,
                        "long_short_ratio": ratio,
                    }
                )

        df = pd.DataFrame(rows)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _parse_taker_flow(self, data: list[dict[str, Any]]) -> pd.DataFrame:
        """Parse OHLCV response to extract taker buy/sell volumes.

        Response format: [{"symbol": "...", "history": [{"t": ts, "o": .., "h": .., "l": .., "c": .., "v": total_vol, "bv": buy_vol, ...}, ...]}]
        Sell volume = v - bv.
        """
        if not data:
            return pd.DataFrame(
                columns=["timestamp", "buy_volume", "sell_volume", "buy_sell_ratio"]
            )

        rows = []
        for symbol_block in data:
            history = symbol_block.get("history", [])
            for item in history:
                total_vol = float(item.get("v", 0))
                buy_vol = float(item.get("bv", 0))
                sell_vol = total_vol - buy_vol
                ratio = buy_vol / sell_vol if sell_vol > 0 else 1.0
                rows.append(
                    {
                        "timestamp": datetime.fromtimestamp(item["t"], tz=timezone.utc),
                        "buy_volume": buy_vol,
                        "sell_volume": sell_vol,
                        "buy_sell_ratio": ratio,
                    }
                )

        df = pd.DataFrame(rows)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

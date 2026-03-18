"""CoinGlass API v3 client for derivatives data.

Endpoints implemented per CoinGlass Open API v3 docs:
- Funding rate OHLC history
- Open interest OHLC history
- Liquidation history
- Global long/short account ratio history
- Taker buy/sell history

Rate limiting: respects per-minute limits with retry + exponential backoff.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.common.config import get_settings, load_yaml_config
from src.common.logging import get_logger
from src.common.time_utils import ts_to_ms, ms_to_dt

logger = get_logger(__name__)


class CoinGlassAPIError(Exception):
    """Raised on non-success CoinGlass API responses."""
    pass


class CoinGlassRateLimitError(CoinGlassAPIError):
    """Raised when rate limited."""
    pass


class CoinGlassClient:
    """Client for CoinGlass Open API v3."""

    def __init__(self) -> None:
        settings = get_settings()
        ds_cfg = load_yaml_config("data_sources.yaml")["coinglass"]

        self.base_url = ds_cfg["base_url"]
        self.api_key = settings.coinglass_api_key
        self.rate_limit_per_min = ds_cfg.get("rate_limit_per_minute", 30)
        self.default_exchange = ds_cfg.get("default_exchange", "Binance")
        self.endpoints = ds_cfg["endpoints"]

        self._request_times: list[float] = []
        self._client = httpx.Client(
            timeout=30.0,
            headers={
                "accept": "application/json",
                "CG-API-KEY": self.api_key,
            },
        )

    def close(self) -> None:
        self._client.close()

    def _throttle(self) -> None:
        """Simple sliding-window rate limiter."""
        now = time.monotonic()
        window = 60.0
        self._request_times = [t for t in self._request_times if now - t < window]
        if len(self._request_times) >= self.rate_limit_per_min:
            sleep_time = window - (now - self._request_times[0]) + 0.5
            logger.info("coinglass_rate_limit_sleep", seconds=sleep_time)
            time.sleep(sleep_time)
        self._request_times.append(time.monotonic())

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=30),
        retry=retry_if_exception_type((httpx.HTTPStatusError, CoinGlassRateLimitError)),
    )
    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make a GET request with retry and rate limiting."""
        self._throttle()
        url = f"{self.base_url}{endpoint}"
        logger.debug("coinglass_request", url=url, params=params)

        resp = self._client.get(url, params=params or {})

        if resp.status_code == 429:
            raise CoinGlassRateLimitError("Rate limited by CoinGlass")
        resp.raise_for_status()

        data = resp.json()
        if data.get("code") not in (None, "0", 0, "20000"):
            # CoinGlass returns code "0" or "20000" on success depending on endpoint
            if str(data.get("code")) not in ("0", "20000"):
                raise CoinGlassAPIError(f"API error: {data.get('msg', data)}")
        return data

    # ── Funding Rate OHLC ────────────────────────────────────

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
        exchange = exchange or self.default_exchange
        params: dict[str, Any] = {
            "symbol": symbol,
            "exchange": exchange,
            "interval": interval,
            "limit": limit,
        }
        if start_time:
            params["startTime"] = ts_to_ms(start_time)
        if end_time:
            params["endTime"] = ts_to_ms(end_time)

        data = self._get(self.endpoints["funding_rate_ohlc"], params)
        return self._parse_ohlc(data, exchange)

    # ── Open Interest OHLC ───────────────────────────────────

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
        exchange = exchange or self.default_exchange
        params: dict[str, Any] = {
            "symbol": symbol,
            "exchange": exchange,
            "interval": interval,
            "limit": limit,
        }
        if start_time:
            params["startTime"] = ts_to_ms(start_time)
        if end_time:
            params["endTime"] = ts_to_ms(end_time)

        data = self._get(self.endpoints["oi_ohlc"], params)
        return self._parse_ohlc(data, exchange)

    # ── Liquidation History ──────────────────────────────────

    def fetch_liquidation_history(
        self,
        symbol: str = "BTC",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch liquidation history.

        Returns DataFrame with: timestamp, long_liquidations_usd, short_liquidations_usd, count
        """
        params: dict[str, Any] = {
            "symbol": symbol,
            "limit": limit,
        }
        if start_time:
            params["startTime"] = ts_to_ms(start_time)
        if end_time:
            params["endTime"] = ts_to_ms(end_time)

        data = self._get(self.endpoints["liquidation_history"], params)
        return self._parse_liquidations(data)

    # ── Long/Short Ratio ─────────────────────────────────────

    def fetch_long_short_ratio(
        self,
        symbol: str = "BTC",
        exchange: str | None = None,
        interval: str = "h1",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch global long/short account ratio history."""
        exchange = exchange or self.default_exchange
        params: dict[str, Any] = {
            "symbol": symbol,
            "exchange": exchange,
            "interval": interval,
            "limit": limit,
        }
        if start_time:
            params["startTime"] = ts_to_ms(start_time)
        if end_time:
            params["endTime"] = ts_to_ms(end_time)

        data = self._get(self.endpoints["long_short_ratio"], params)
        return self._parse_long_short(data)

    # ── Taker Buy/Sell ───────────────────────────────────────

    def fetch_taker_buy_sell(
        self,
        symbol: str = "BTC",
        interval: str = "h1",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch taker buy/sell volume data."""
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_time:
            params["startTime"] = ts_to_ms(start_time)
        if end_time:
            params["endTime"] = ts_to_ms(end_time)

        data = self._get(self.endpoints["taker_buy_sell"], params)
        return self._parse_taker_flow(data)

    # ── Parsers ──────────────────────────────────────────────

    def _parse_ohlc(self, data: dict[str, Any], exchange: str) -> pd.DataFrame:
        """Parse OHLC response from CoinGlass v3 API.

        The v3 API returns data in 'data' field, typically as a list of lists
        or list of dicts. We handle both formats.
        """
        raw = data.get("data", [])
        if not raw:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])

        rows = []
        # Handle list-of-lists format: [[timestamp, o, h, l, c], ...]
        if isinstance(raw, list) and len(raw) > 0:
            if isinstance(raw[0], list):
                for row in raw:
                    if len(row) >= 5:
                        rows.append({
                            "timestamp": ms_to_dt(int(row[0])),
                            "open": float(row[1]),
                            "high": float(row[2]),
                            "low": float(row[3]),
                            "close": float(row[4]),
                        })
            elif isinstance(raw[0], dict):
                for row in raw:
                    ts_key = next((k for k in ("t", "time", "timestamp", "createTime") if k in row), None)
                    if ts_key is None:
                        continue
                    ts_val = row[ts_key]
                    if isinstance(ts_val, (int, float)):
                        ts = ms_to_dt(int(ts_val)) if ts_val > 1e12 else ms_to_dt(int(ts_val * 1000))
                    else:
                        ts = pd.to_datetime(ts_val, utc=True)
                    rows.append({
                        "timestamp": ts,
                        "open": float(row.get("o", row.get("open", 0))),
                        "high": float(row.get("h", row.get("high", 0))),
                        "low": float(row.get("l", row.get("low", 0))),
                        "close": float(row.get("c", row.get("close", 0))),
                    })

        df = pd.DataFrame(rows)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _parse_liquidations(self, data: dict[str, Any]) -> pd.DataFrame:
        raw = data.get("data", [])
        if not raw:
            return pd.DataFrame(columns=[
                "timestamp", "long_liquidations_usd", "short_liquidations_usd",
                "total_liquidations_usd", "count"
            ])

        rows = []
        items = raw if isinstance(raw, list) else [raw]
        for item in items:
            if isinstance(item, dict):
                ts_key = next((k for k in ("t", "time", "timestamp", "createTime") if k in item), None)
                if ts_key is None:
                    continue
                ts_val = item[ts_key]
                if isinstance(ts_val, (int, float)):
                    ts = ms_to_dt(int(ts_val)) if ts_val > 1e12 else ms_to_dt(int(ts_val * 1000))
                else:
                    ts = pd.to_datetime(ts_val, utc=True)

                long_liq = float(item.get("longLiquidationUsd", item.get("buyVolUsd", 0)) or 0)
                short_liq = float(item.get("shortLiquidationUsd", item.get("sellVolUsd", 0)) or 0)
                rows.append({
                    "timestamp": ts,
                    "long_liquidations_usd": long_liq,
                    "short_liquidations_usd": short_liq,
                    "total_liquidations_usd": long_liq + short_liq,
                    "count": int(item.get("count", item.get("liquidationOrders", 0)) or 0),
                })

        df = pd.DataFrame(rows)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _parse_long_short(self, data: dict[str, Any]) -> pd.DataFrame:
        raw = data.get("data", [])
        if not raw:
            return pd.DataFrame(columns=["timestamp", "long_ratio", "short_ratio", "long_short_ratio"])

        rows = []
        items = raw if isinstance(raw, list) else [raw]
        for item in items:
            if isinstance(item, dict):
                ts_key = next((k for k in ("t", "time", "timestamp", "createTime") if k in item), None)
                if ts_key is None:
                    continue
                ts_val = item[ts_key]
                if isinstance(ts_val, (int, float)):
                    ts = ms_to_dt(int(ts_val)) if ts_val > 1e12 else ms_to_dt(int(ts_val * 1000))
                else:
                    ts = pd.to_datetime(ts_val, utc=True)

                long_r = float(item.get("longRate", item.get("longAccount", 0.5)) or 0.5)
                short_r = float(item.get("shortRate", item.get("shortAccount", 0.5)) or 0.5)
                ratio = long_r / short_r if short_r > 0 else 1.0
                rows.append({
                    "timestamp": ts,
                    "long_ratio": long_r,
                    "short_ratio": short_r,
                    "long_short_ratio": ratio,
                })

        df = pd.DataFrame(rows)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _parse_taker_flow(self, data: dict[str, Any]) -> pd.DataFrame:
        raw = data.get("data", [])
        if not raw:
            return pd.DataFrame(columns=["timestamp", "buy_volume", "sell_volume", "buy_sell_ratio"])

        rows = []
        items = raw if isinstance(raw, list) else [raw]
        for item in items:
            if isinstance(item, dict):
                ts_key = next((k for k in ("t", "time", "timestamp", "createTime") if k in item), None)
                if ts_key is None:
                    continue
                ts_val = item[ts_key]
                if isinstance(ts_val, (int, float)):
                    ts = ms_to_dt(int(ts_val)) if ts_val > 1e12 else ms_to_dt(int(ts_val * 1000))
                else:
                    ts = pd.to_datetime(ts_val, utc=True)

                buy_vol = float(item.get("buyVol", item.get("buyVolUsd", 0)) or 0)
                sell_vol = float(item.get("sellVol", item.get("sellVolUsd", 0)) or 0)
                ratio = buy_vol / sell_vol if sell_vol > 0 else 1.0
                rows.append({
                    "timestamp": ts,
                    "buy_volume": buy_vol,
                    "sell_volume": sell_vol,
                    "buy_sell_ratio": ratio,
                })

        df = pd.DataFrame(rows)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

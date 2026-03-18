"""Data ingestion jobs: backfill and incremental refresh."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import select, func

from src.common.logging import get_logger
from src.common.time_utils import utc_now
from src.data.coinglass_client import CoinGlassClient
from src.data.validators import validate_ohlc, validate_ratio_data, validate_liquidation_data
from src.data.resampler import align_to_hourly
from src.storage.database import session_scope
from src.storage.models import (
    PriceBar1h, CGFunding1h, CGOI1h, CGLiquidations1h,
    CGLongShort1h, CGTakerFlow1h,
)

logger = get_logger(__name__)

DEFAULT_EXCHANGE = "Binance"


def _upsert_rows(session, model_class, rows: list[dict], unique_keys: list[str]) -> int:
    """Insert rows, skipping duplicates based on unique keys."""
    inserted = 0
    for row in rows:
        # Check if exists
        filters = [getattr(model_class, k) == row[k] for k in unique_keys if k in row]
        existing = session.query(model_class).filter(*filters).first()
        if existing is None:
            session.add(model_class(**row))
            inserted += 1
    return inserted


def backfill_price_data(
    symbol: str = "BTC",
    start: datetime | None = None,
    end: datetime | None = None,
    exchange: str = DEFAULT_EXCHANGE,
) -> int:
    """Backfill hourly price bars from CoinGlass OHLC."""
    client = CoinGlassClient()
    try:
        if start is None:
            start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        if end is None:
            end = utc_now()

        total_inserted = 0
        chunk_days = 30
        current = start

        while current < end:
            chunk_end = min(current + timedelta(days=chunk_days), end)
            logger.info("backfill_price", start=current.isoformat(), end=chunk_end.isoformat())

            # Use OI OHLC as a proxy for price if price endpoint unavailable,
            # or funding OHLC for the funding-specific data.
            # CoinGlass v3 doesn't have a dedicated price OHLC, so we use their
            # candle data or fall back to Hyperliquid candles.
            df = client.fetch_oi_ohlc(
                symbol=symbol, exchange=exchange,
                start_time=current, end_time=chunk_end, limit=500,
            )

            if df.empty:
                logger.warning("no_price_data", start=current.isoformat())
                current = chunk_end
                continue

            df = validate_ohlc(df, source="price_backfill")
            df = align_to_hourly(df)

            rows = []
            for _, r in df.iterrows():
                rows.append({
                    "symbol": symbol,
                    "timestamp": r["timestamp"].to_pydatetime(),
                    "open": r["open"],
                    "high": r["high"],
                    "low": r["low"],
                    "close": r["close"],
                    "volume": r.get("volume", 0),
                    "source": "coinglass",
                })

            with session_scope() as session:
                n = _upsert_rows(session, PriceBar1h, rows, ["symbol", "timestamp"])
                total_inserted += n
                logger.info("price_inserted", count=n)

            current = chunk_end

        return total_inserted
    finally:
        client.close()


def backfill_funding(
    symbol: str = "BTC",
    start: datetime | None = None,
    end: datetime | None = None,
    exchange: str = DEFAULT_EXCHANGE,
) -> int:
    """Backfill funding rate OHLC data."""
    client = CoinGlassClient()
    try:
        if start is None:
            start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        if end is None:
            end = utc_now()

        total = 0
        current = start
        while current < end:
            chunk_end = min(current + timedelta(days=30), end)
            df = client.fetch_funding_ohlc(symbol=symbol, exchange=exchange,
                                           start_time=current, end_time=chunk_end)
            df = validate_ohlc(df, source="funding_backfill")
            df = align_to_hourly(df)

            rows = [{
                "symbol": symbol, "exchange": exchange,
                "timestamp": r["timestamp"].to_pydatetime(),
                "open": r["open"], "high": r["high"],
                "low": r["low"], "close": r["close"],
            } for _, r in df.iterrows()]

            with session_scope() as session:
                n = _upsert_rows(session, CGFunding1h, rows, ["symbol", "exchange", "timestamp"])
                total += n
            current = chunk_end
        return total
    finally:
        client.close()


def backfill_oi(
    symbol: str = "BTC",
    start: datetime | None = None,
    end: datetime | None = None,
    exchange: str = DEFAULT_EXCHANGE,
) -> int:
    """Backfill OI OHLC data."""
    client = CoinGlassClient()
    try:
        if start is None:
            start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        if end is None:
            end = utc_now()

        total = 0
        current = start
        while current < end:
            chunk_end = min(current + timedelta(days=30), end)
            df = client.fetch_oi_ohlc(symbol=symbol, exchange=exchange,
                                      start_time=current, end_time=chunk_end)
            df = validate_ohlc(df, source="oi_backfill")
            df = align_to_hourly(df)

            rows = [{
                "symbol": symbol, "exchange": exchange,
                "timestamp": r["timestamp"].to_pydatetime(),
                "open": r["open"], "high": r["high"],
                "low": r["low"], "close": r["close"],
            } for _, r in df.iterrows()]

            with session_scope() as session:
                n = _upsert_rows(session, CGOI1h, rows, ["symbol", "exchange", "timestamp"])
                total += n
            current = chunk_end
        return total
    finally:
        client.close()


def backfill_liquidations(
    symbol: str = "BTC",
    start: datetime | None = None,
    end: datetime | None = None,
) -> int:
    """Backfill liquidation data."""
    client = CoinGlassClient()
    try:
        if start is None:
            start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        if end is None:
            end = utc_now()

        total = 0
        current = start
        while current < end:
            chunk_end = min(current + timedelta(days=30), end)
            df = client.fetch_liquidation_history(symbol=symbol,
                                                  start_time=current, end_time=chunk_end)
            df = validate_liquidation_data(df, source="liq_backfill")
            df = align_to_hourly(df)

            rows = [{
                "symbol": symbol,
                "timestamp": r["timestamp"].to_pydatetime(),
                "long_liquidations_usd": r.get("long_liquidations_usd", 0),
                "short_liquidations_usd": r.get("short_liquidations_usd", 0),
                "total_liquidations_usd": r.get("total_liquidations_usd", 0),
                "count": r.get("count", 0),
            } for _, r in df.iterrows()]

            with session_scope() as session:
                n = _upsert_rows(session, CGLiquidations1h, rows, ["symbol", "timestamp"])
                total += n
            current = chunk_end
        return total
    finally:
        client.close()


def backfill_long_short(
    symbol: str = "BTC",
    start: datetime | None = None,
    end: datetime | None = None,
    exchange: str = DEFAULT_EXCHANGE,
) -> int:
    """Backfill long/short ratio data."""
    client = CoinGlassClient()
    try:
        if start is None:
            start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        if end is None:
            end = utc_now()

        total = 0
        current = start
        while current < end:
            chunk_end = min(current + timedelta(days=30), end)
            df = client.fetch_long_short_ratio(symbol=symbol, exchange=exchange,
                                               start_time=current, end_time=chunk_end)
            df = validate_ratio_data(df, source="ls_backfill")
            df = align_to_hourly(df)

            rows = [{
                "symbol": symbol, "exchange": exchange,
                "timestamp": r["timestamp"].to_pydatetime(),
                "long_ratio": r.get("long_ratio"),
                "short_ratio": r.get("short_ratio"),
                "long_short_ratio": r.get("long_short_ratio"),
            } for _, r in df.iterrows()]

            with session_scope() as session:
                n = _upsert_rows(session, CGLongShort1h, rows, ["symbol", "exchange", "timestamp"])
                total += n
            current = chunk_end
        return total
    finally:
        client.close()


def backfill_taker_flow(
    symbol: str = "BTC",
    start: datetime | None = None,
    end: datetime | None = None,
) -> int:
    """Backfill taker buy/sell data."""
    client = CoinGlassClient()
    try:
        if start is None:
            start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        if end is None:
            end = utc_now()

        total = 0
        current = start
        while current < end:
            chunk_end = min(current + timedelta(days=30), end)
            df = client.fetch_taker_buy_sell(symbol=symbol,
                                             start_time=current, end_time=chunk_end)
            df = validate_ratio_data(df, source="taker_backfill")
            df = align_to_hourly(df)

            rows = [{
                "symbol": symbol,
                "timestamp": r["timestamp"].to_pydatetime(),
                "buy_volume": r.get("buy_volume", 0),
                "sell_volume": r.get("sell_volume", 0),
                "buy_sell_ratio": r.get("buy_sell_ratio"),
            } for _, r in df.iterrows()]

            with session_scope() as session:
                n = _upsert_rows(session, CGTakerFlow1h, rows, ["symbol", "timestamp"])
                total += n
            current = chunk_end
        return total
    finally:
        client.close()


def backfill_all(
    symbol: str = "BTC",
    start: datetime | None = None,
    end: datetime | None = None,
) -> dict[str, int]:
    """Run all backfill jobs."""
    results = {}
    for name, fn in [
        ("price", backfill_price_data),
        ("funding", backfill_funding),
        ("oi", backfill_oi),
        ("liquidations", backfill_liquidations),
        ("long_short", backfill_long_short),
        ("taker_flow", backfill_taker_flow),
    ]:
        try:
            count = fn(symbol=symbol, start=start, end=end)
            results[name] = count
            logger.info("backfill_complete", dataset=name, rows=count)
        except Exception as e:
            logger.error("backfill_failed", dataset=name, error=str(e))
            results[name] = -1
    return results


def incremental_refresh(symbol: str = "BTC") -> dict[str, int]:
    """Refresh data from last known timestamp to now."""
    # Find latest timestamps per table and backfill from there
    from src.storage.database import get_session

    session = get_session()
    results = {}

    table_map = {
        "price": (PriceBar1h, backfill_price_data),
        "funding": (CGFunding1h, backfill_funding),
        "oi": (CGOI1h, backfill_oi),
        "liquidations": (CGLiquidations1h, backfill_liquidations),
        "long_short": (CGLongShort1h, backfill_long_short),
        "taker_flow": (CGTakerFlow1h, backfill_taker_flow),
    }

    for name, (model, fn) in table_map.items():
        try:
            latest = session.query(func.max(model.timestamp)).scalar()
            start = latest - timedelta(hours=2) if latest else None
            count = fn(symbol=symbol, start=start)
            results[name] = count
        except Exception as e:
            logger.error("refresh_failed", dataset=name, error=str(e))
            results[name] = -1

    session.close()
    return results

"""Time utility functions for UTC-based trading day logic."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta


def utc_now() -> datetime:
    """Current UTC datetime."""
    return datetime.now(timezone.utc)


def trading_day_start(dt: datetime | None = None) -> datetime:
    """Return 00:00 UTC of the trading day for given datetime."""
    if dt is None:
        dt = utc_now()
    return dt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)


def is_new_trading_day(last_reset: datetime | None) -> bool:
    """Check if we've crossed 00:00 UTC since last_reset."""
    now = utc_now()
    if last_reset is None:
        return True
    return trading_day_start(now) > trading_day_start(last_reset)


def hours_to_td(hours: int) -> timedelta:
    return timedelta(hours=hours)


def floor_to_hour(dt: datetime) -> datetime:
    """Floor datetime to the start of its hour."""
    return dt.replace(minute=0, second=0, microsecond=0)


def ts_to_ms(dt: datetime) -> int:
    """Convert datetime to millisecond timestamp."""
    return int(dt.timestamp() * 1000)


def ms_to_dt(ms: int) -> datetime:
    """Convert millisecond timestamp to UTC datetime."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)

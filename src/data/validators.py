"""Data validation utilities for ingested data."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.common.logging import get_logger

logger = get_logger(__name__)


class DataValidationError(Exception):
    pass


def validate_ohlc(df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """Validate OHLC DataFrame: check schema, nulls, price sanity."""
    required = ["timestamp", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns {missing} in {source}")

    if df.empty:
        logger.warning("empty_ohlc", source=source)
        return df

    # Drop rows with null timestamps
    null_ts = df["timestamp"].isna().sum()
    if null_ts > 0:
        logger.warning("null_timestamps", count=null_ts, source=source)
        df = df.dropna(subset=["timestamp"])

    # Check for negative prices
    for col in ["open", "high", "low", "close"]:
        neg = (df[col] < 0).sum()
        if neg > 0:
            logger.warning("negative_prices", column=col, count=neg, source=source)
            df = df[df[col] >= 0]

    # Check high >= low
    bad_hl = (df["high"] < df["low"]).sum()
    if bad_hl > 0:
        logger.warning("high_lt_low", count=bad_hl, source=source)
        # Swap them
        mask = df["high"] < df["low"]
        df.loc[mask, ["high", "low"]] = df.loc[mask, ["low", "high"]].values

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    if len(df) < before:
        logger.info("dropped_duplicates", count=before - len(df), source=source)

    return df.sort_values("timestamp").reset_index(drop=True)


def validate_ratio_data(df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """Validate ratio data (long/short, taker flow)."""
    if df.empty:
        return df

    if "timestamp" not in df.columns:
        raise DataValidationError(f"Missing timestamp in {source}")

    df = df.dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    return df.sort_values("timestamp").reset_index(drop=True)


def validate_liquidation_data(df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """Validate liquidation data."""
    if df.empty:
        return df

    if "timestamp" not in df.columns:
        raise DataValidationError(f"Missing timestamp in {source}")

    df = df.dropna(subset=["timestamp"])

    # Liquidation values should be non-negative
    for col in [
        "long_liquidations_usd",
        "short_liquidations_usd",
        "total_liquidations_usd",
    ]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    return df.sort_values("timestamp").reset_index(drop=True)


def check_data_freshness(
    df: pd.DataFrame,
    max_gap_hours: int = 4,
    source: str = "",
) -> bool:
    """Check if the most recent data is within acceptable freshness."""
    if df.empty:
        logger.warning("data_freshness_empty", source=source)
        return False

    latest = pd.to_datetime(df["timestamp"]).max()
    if latest.tzinfo is None:
        from datetime import timezone

        latest = latest.replace(tzinfo=timezone.utc)

    now = datetime.now(tz=latest.tzinfo)
    gap_hours = (now - latest).total_seconds() / 3600

    if gap_hours > max_gap_hours:
        logger.warning(
            "stale_data", gap_hours=gap_hours, max=max_gap_hours, source=source
        )
        return False
    return True


def check_gaps(
    df: pd.DataFrame, expected_freq_hours: int = 1
) -> list[tuple[datetime, datetime]]:
    """Find gaps in hourly data. Returns list of (gap_start, gap_end) tuples."""
    if len(df) < 2:
        return []

    ts = pd.to_datetime(df["timestamp"]).sort_values()
    diffs = ts.diff().dt.total_seconds() / 3600
    gap_mask = diffs > expected_freq_hours * 1.5

    gaps = []
    for idx in diffs[gap_mask].index:
        gap_start = ts.iloc[ts.index.get_loc(idx) - 1]
        gap_end = ts.iloc[ts.index.get_loc(idx)]
        gaps.append((gap_start, gap_end))

    if gaps:
        logger.info("data_gaps_found", count=len(gaps))
    return gaps

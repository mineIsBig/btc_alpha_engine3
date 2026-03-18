"""Resample and align all data to hourly UTC grid."""

from __future__ import annotations

import pandas as pd


def align_to_hourly(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Align DataFrame to hourly UTC timestamps.

    Floors timestamps to hour boundaries and aggregates duplicates.
    """
    if df.empty:
        return df

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df[timestamp_col] = df[timestamp_col].dt.floor("h")

    # If duplicates exist after flooring, keep last
    df = df.drop_duplicates(subset=[timestamp_col], keep="last")
    return df.sort_values(timestamp_col).reset_index(drop=True)


def resample_ohlc_to_hourly(
    df: pd.DataFrame, timestamp_col: str = "timestamp"
) -> pd.DataFrame:
    """Resample sub-hourly OHLC data to 1h bars."""
    if df.empty:
        return df

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df = df.set_index(timestamp_col).sort_index()

    resampled = (
        df.resample("1h")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open"])
    )

    return resampled.reset_index()


def fill_hourly_gaps(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    fill_method: str = "ffill",
    value_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Fill gaps in hourly data with forward fill or specified method.

    Creates a complete hourly index from min to max timestamp.
    """
    if df.empty:
        return df

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)

    start = df[timestamp_col].min()
    end = df[timestamp_col].max()
    full_index = pd.date_range(start=start, end=end, freq="1h", tz="UTC")

    df = df.set_index(timestamp_col)
    df = df.reindex(full_index)
    df.index.name = timestamp_col

    if fill_method == "ffill":
        if value_cols:
            df[value_cols] = df[value_cols].ffill()
        else:
            df = df.ffill()
    elif fill_method == "zero":
        df = df.fillna(0)

    return df.reset_index()


def merge_dataframes_on_timestamp(
    dfs: list[pd.DataFrame],
    timestamp_col: str = "timestamp",
    how: str = "inner",
) -> pd.DataFrame:
    """Merge multiple DataFrames on timestamp column."""
    if not dfs:
        return pd.DataFrame()

    result = dfs[0].copy()
    result[timestamp_col] = pd.to_datetime(result[timestamp_col], utc=True)

    for df in dfs[1:]:
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
        # Deduplicate column names with suffixes
        overlapping = set(result.columns) & set(df.columns) - {timestamp_col}
        if overlapping:
            result = result.merge(df, on=timestamp_col, how=how, suffixes=("", "_dup"))
            # Drop duplicate columns
            dup_cols = [c for c in result.columns if c.endswith("_dup")]
            result = result.drop(columns=dup_cols)
        else:
            result = result.merge(df, on=timestamp_col, how=how)

    return result.sort_values(timestamp_col).reset_index(drop=True)

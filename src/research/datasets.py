"""Dataset preparation for research: feature + label alignment, train/test splits."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.common.logging import get_logger
from src.features.feature_pipeline import build_features
from src.labels.labels import build_labels

logger = get_logger(__name__)


def prepare_dataset(
    symbol: str = "BTC",
    start: datetime | None = None,
    end: datetime | None = None,
    horizons: list[int] | None = None,
    raw_data: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build aligned feature + label dataset.

    Returns DataFrame with timestamp, all features, and labels for each horizon.
    """
    if horizons is None:
        horizons = [1, 4, 8, 12, 24]

    features_df = build_features(symbol=symbol, start=start, end=end, raw_data=raw_data)
    if features_df.empty:
        return pd.DataFrame()

    # Build labels from price data
    if raw_data and "price" in raw_data and not raw_data["price"].empty:
        price_df = raw_data["price"][["timestamp", "close", "high", "low"]].copy()
    else:
        # Extract price columns from features data
        from src.storage.database import get_session
        from src.storage.models import PriceBar1h

        session = get_session()
        q = session.query(PriceBar1h).filter(PriceBar1h.symbol == symbol)
        if start:
            q = q.filter(PriceBar1h.timestamp >= start)
        if end:
            q = q.filter(PriceBar1h.timestamp <= end)
        rows = q.order_by(PriceBar1h.timestamp).all()
        session.close()

        if not rows:
            logger.warning("no_price_data_for_labels")
            return features_df

        price_df = pd.DataFrame(
            [
                {
                    "timestamp": r.timestamp,
                    "close": r.close,
                    "high": r.high,
                    "low": r.low,
                }
                for r in rows
            ]
        )

    price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], utc=True)
    labels_df = build_labels(price_df, horizons=horizons)
    labels_df["timestamp"] = pd.to_datetime(labels_df["timestamp"], utc=True)

    # Merge features and labels
    features_df["timestamp"] = pd.to_datetime(features_df["timestamp"], utc=True)
    dataset = features_df.merge(labels_df, on="timestamp", how="inner")

    # Drop rows with all-NaN features
    feat_cols = [
        c
        for c in dataset.columns
        if c not in ["timestamp", "regime_label"]
        and not c.startswith("fwd_ret_")
        and not c.startswith("label_")
        and not c.startswith("mfe_")
        and not c.startswith("mae_")
    ]
    dataset = dataset.dropna(subset=feat_cols, how="all")

    logger.info(
        "dataset_prepared",
        rows=len(dataset),
        features=len(feat_cols),
        horizons=horizons,
    )
    return dataset


def get_feature_columns(dataset: pd.DataFrame) -> list[str]:
    """Extract feature column names from a prepared dataset."""
    skip_prefixes = ("fwd_ret_", "label_", "mfe_", "mae_", "barrier_")
    skip_exact = {"timestamp", "regime_label", "symbol", "id"}

    return [
        c
        for c in dataset.columns
        if c not in skip_exact and not any(c.startswith(p) for p in skip_prefixes)
    ]


def get_label_column(horizon: int) -> str:
    """Get label column name for a horizon."""
    return f"label_{horizon}h"

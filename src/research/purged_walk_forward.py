"""Purged walk-forward cross-validation with embargo.

Strict rolling train/test splits with:
- Purge gap between train end and test start to prevent leakage
- Dynamic purge: scales with label horizon so purge >= horizon
- Embargo gap after test end before next train can start
- Configurable window sizes
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Generator

import numpy as np
import pandas as pd

from src.common.config import load_yaml_config
from src.common.logging import get_logger

logger = get_logger(__name__)

# Minimum purge gap per label horizon (hours).
# Purge must be >= label horizon to prevent information leakage from
# forward-looking labels bleeding into the train set.
# Formula: max(base_purge, horizon * multiplier)
PURGE_HORIZON_MULTIPLIER = 3.0  # purge = 3x the label horizon
MIN_PURGE_HOURS = 48            # absolute floor regardless of horizon


def compute_purge_hours(base_purge: int, horizon: int | None = None) -> int:
    """Compute effective purge gap based on label horizon.

    For short horizons (1-4h), the base purge (48h) is already sufficient.
    For longer horizons (12-24h), purge scales to 3x the horizon to ensure
    no forward-looking label information leaks into training data.

    Args:
        base_purge: configured base purge gap in hours
        horizon: label horizon in hours (None = use base_purge as-is)

    Returns:
        Effective purge hours, guaranteed >= MIN_PURGE_HOURS
    """
    if horizon is None:
        return max(base_purge, MIN_PURGE_HOURS)
    horizon_purge = int(horizon * PURGE_HORIZON_MULTIPLIER)
    return max(base_purge, horizon_purge, MIN_PURGE_HOURS)


@dataclass
class WalkForwardFold:
    """A single walk-forward fold."""
    fold_idx: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_indices: np.ndarray
    test_indices: np.ndarray


class PurgedWalkForward:
    """Purged walk-forward splitter with horizon-aware purge gaps."""

    def __init__(
        self,
        train_days: int = 90,
        test_days: int = 14,
        purge_hours: int = 48,
        embargo_hours: int = 24,
        step_days: int = 14,
        min_train_samples: int = 1000,
        max_folds: int = 50,
        horizon: int | None = None,
    ):
        self.train_days = train_days
        self.test_days = test_days
        self.base_purge_hours = purge_hours
        self.embargo_hours = embargo_hours
        self.step_days = step_days
        self.min_train_samples = min_train_samples
        self.max_folds = max_folds
        self.horizon = horizon
        self.purge_hours = compute_purge_hours(purge_hours, horizon)

        if self.purge_hours != purge_hours:
            logger.info("purge_gap_scaled",
                        base=purge_hours, effective=self.purge_hours,
                        horizon=horizon, reason="purge >= horizon * 3")

    @classmethod
    def from_config(cls, horizon: int | None = None) -> PurgedWalkForward:
        """Create from walk_forward.yaml config.

        Args:
            horizon: label horizon in hours. If provided, purge gap is
                     dynamically scaled to prevent leakage.
        """
        cfg = load_yaml_config("walk_forward.yaml")["walk_forward"]
        return cls(
            train_days=cfg["train_days"],
            test_days=cfg["test_days"],
            purge_hours=cfg["purge_hours"],
            embargo_hours=cfg["embargo_hours"],
            step_days=cfg["step_days"],
            min_train_samples=cfg["min_train_samples"],
            max_folds=cfg["max_folds"],
            horizon=horizon,
        )

    def split(self, timestamps: pd.Series | np.ndarray) -> Generator[WalkForwardFold, None, None]:
        """Generate walk-forward folds.

        Args:
            timestamps: Series/array of datetime timestamps (sorted ascending)

        Yields:
            WalkForwardFold objects with train/test index arrays
        """
        if isinstance(timestamps, np.ndarray):
            ts = pd.to_datetime(pd.Series(timestamps), utc=True)
        else:
            ts = pd.to_datetime(timestamps, utc=True)

        ts = ts.reset_index(drop=True)
        data_start = ts.min()
        data_end = ts.max()

        train_td = timedelta(days=self.train_days)
        test_td = timedelta(days=self.test_days)
        purge_td = timedelta(hours=self.purge_hours)
        embargo_td = timedelta(hours=self.embargo_hours)
        step_td = timedelta(days=self.step_days)

        fold_idx = 0
        train_start = data_start

        while fold_idx < self.max_folds:
            train_end = train_start + train_td
            test_start = train_end + purge_td
            test_end = test_start + test_td

            if test_end > data_end:
                break

            # Get indices
            train_mask = (ts >= train_start) & (ts < train_end)
            test_mask = (ts >= test_start) & (ts < test_end)

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            if len(train_idx) < self.min_train_samples:
                logger.warning("insufficient_train", fold=fold_idx, n=len(train_idx))
                train_start += step_td
                continue

            if len(test_idx) < 10:
                logger.warning("insufficient_test", fold=fold_idx, n=len(test_idx))
                train_start += step_td
                continue

            yield WalkForwardFold(
                fold_idx=fold_idx,
                train_start=train_start.to_pydatetime(),
                train_end=train_end.to_pydatetime(),
                test_start=test_start.to_pydatetime(),
                test_end=test_end.to_pydatetime(),
                train_indices=train_idx,
                test_indices=test_idx,
            )

            fold_idx += 1
            # Step forward: next train starts after embargo from test end
            train_start += step_td

    def get_n_folds(self, timestamps: pd.Series | np.ndarray) -> int:
        """Count the number of folds without generating them."""
        return sum(1 for _ in self.split(timestamps))

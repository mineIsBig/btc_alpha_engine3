"""Purged walk-forward cross-validation with embargo.

Strict rolling train/test splits with:
- Purge gap between train end and test start to prevent leakage
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
    """Purged walk-forward splitter with dynamic purge gaps.
    
    Purge gap is dynamically adjusted based on prediction horizon to prevent
    information leakage. Longer horizons require larger purge gaps.
    """

    def __init__(
        self,
        train_days: int = 90,
        test_days: int = 14,
        purge_hours: int | None = None,
        embargo_hours: int = 24,
        step_days: int = 14,
        min_train_samples: int = 1000,
        max_folds: int = 50,
        horizon: int | None = None,
        purge_horizon_multiplier: float = 2.0,
        min_purge_hours: int = 48,
    ):
        """Initialize purged walk-forward splitter.
        
        Args:
            train_days: Training window size in days
            test_days: Test window size in days
            purge_hours: Fixed purge gap in hours (overrides dynamic if set)
            embargo_hours: Gap after test before next train starts
            step_days: How many days to step forward each fold
            min_train_samples: Minimum samples required in training set
            max_folds: Maximum number of folds to generate
            horizon: Prediction horizon in hours (for dynamic purge calculation)
            purge_horizon_multiplier: Multiplier for horizon to compute purge gap
            min_purge_hours: Minimum purge gap regardless of horizon
        """
        self.train_days = train_days
        self.test_days = test_days
        self.embargo_hours = embargo_hours
        self.step_days = step_days
        self.min_train_samples = min_train_samples
        self.max_folds = max_folds
        self.horizon = horizon
        self.purge_horizon_multiplier = purge_horizon_multiplier
        self.min_purge_hours = min_purge_hours
        
        # Compute effective purge hours
        if purge_hours is not None:
            self.purge_hours = purge_hours
        elif horizon is not None:
            self.purge_hours = max(min_purge_hours, int(horizon * purge_horizon_multiplier))
        else:
            self.purge_hours = min_purge_hours

    @classmethod
    def from_config(cls, horizon: int | None = None) -> PurgedWalkForward:
        """Create from walk_forward.yaml config.
        
        Args:
            horizon: Optional prediction horizon for dynamic purge calculation
        """
        cfg = load_yaml_config("walk_forward.yaml")["walk_forward"]
        return cls(
            train_days=cfg["train_days"],
            test_days=cfg["test_days"],
            purge_hours=cfg.get("purge_hours"),  # None for dynamic
            embargo_hours=cfg["embargo_hours"],
            step_days=cfg["step_days"],
            min_train_samples=cfg["min_train_samples"],
            max_folds=cfg["max_folds"],
            horizon=horizon,
            purge_horizon_multiplier=cfg.get("purge_horizon_multiplier", 2.0),
            min_purge_hours=cfg.get("min_purge_hours", 48),
        )
    
    def with_horizon(self, horizon: int) -> PurgedWalkForward:
        """Return a new splitter configured for a specific horizon.
        
        Args:
            horizon: Prediction horizon in hours
            
        Returns:
            New PurgedWalkForward instance with horizon-appropriate purge gap
        """
        return PurgedWalkForward(
            train_days=self.train_days,
            test_days=self.test_days,
            purge_hours=None,  # Will be computed dynamically
            embargo_hours=self.embargo_hours,
            step_days=self.step_days,
            min_train_samples=self.min_train_samples,
            max_folds=self.max_folds,
            horizon=horizon,
            purge_horizon_multiplier=self.purge_horizon_multiplier,
            min_purge_hours=self.min_purge_hours,
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

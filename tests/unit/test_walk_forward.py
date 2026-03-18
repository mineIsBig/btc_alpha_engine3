"""Tests for purged walk-forward split correctness."""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone


class TestPurgedWalkForward:
    def test_no_overlap(self):
        """Train and test sets must not overlap."""
        from src.research.purged_walk_forward import PurgedWalkForward

        n = 2000
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        timestamps = pd.Series([start + timedelta(hours=i) for i in range(n)])

        splitter = PurgedWalkForward(
            train_days=30, test_days=7, purge_hours=24,
            embargo_hours=12, step_days=7, min_train_samples=100,
        )

        for fold in splitter.split(timestamps):
            train_set = set(fold.train_indices)
            test_set = set(fold.test_indices)
            overlap = train_set & test_set
            assert len(overlap) == 0, f"Fold {fold.fold_idx}: train/test overlap"

    def test_purge_gap_exists(self):
        """Purge gap between train end and test start."""
        from src.research.purged_walk_forward import PurgedWalkForward

        n = 2000
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        timestamps = pd.Series([start + timedelta(hours=i) for i in range(n)])

        splitter = PurgedWalkForward(
            train_days=30, test_days=7, purge_hours=48,
            embargo_hours=12, step_days=7, min_train_samples=100,
        )

        for fold in splitter.split(timestamps):
            train_end_ts = timestamps.iloc[fold.train_indices[-1]]
            test_start_ts = timestamps.iloc[fold.test_indices[0]]
            gap = (test_start_ts - train_end_ts).total_seconds() / 3600
            assert gap >= 48, f"Fold {fold.fold_idx}: purge gap {gap}h < 48h"

    def test_chronological_order(self):
        """Train must come before test."""
        from src.research.purged_walk_forward import PurgedWalkForward

        n = 2000
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        timestamps = pd.Series([start + timedelta(hours=i) for i in range(n)])

        splitter = PurgedWalkForward(
            train_days=30, test_days=7, purge_hours=24,
            embargo_hours=12, step_days=7, min_train_samples=100,
        )

        for fold in splitter.split(timestamps):
            assert fold.train_indices.max() < fold.test_indices.min()
            assert fold.train_end < fold.test_start

    def test_generates_multiple_folds(self):
        """Should produce multiple folds for sufficient data."""
        from src.research.purged_walk_forward import PurgedWalkForward

        n = 5000
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        timestamps = pd.Series([start + timedelta(hours=i) for i in range(n)])

        splitter = PurgedWalkForward(
            train_days=30, test_days=7, purge_hours=24,
            embargo_hours=12, step_days=7, min_train_samples=100,
        )

        n_folds = splitter.get_n_folds(timestamps)
        assert n_folds > 3, f"Expected >3 folds, got {n_folds}"

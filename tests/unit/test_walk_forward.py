"""Tests for purged walk-forward split correctness, model selection,
and regime-dependent slippage.

Covers:
- Base walk-forward fold generation (no overlap, purge gap, chronological)
- Dynamic purge gap scaling per label horizon (our 3x multiplier)
- with_horizon() API for reconfiguring splitters
- Bonferroni correction for both Sharpe AND accuracy thresholds
- Consecutive window requirement and catastrophic fold rejection
- Regime-dependent slippage with vol multiplier, liquidation adder,
  commission scaling, and cost breakdown
"""

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
            train_days=30,
            test_days=7,
            purge_hours=24,
            embargo_hours=12,
            step_days=7,
            min_train_samples=100,
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
            train_days=30,
            test_days=7,
            purge_hours=48,
            embargo_hours=12,
            step_days=7,
            min_train_samples=100,
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
            train_days=30,
            test_days=7,
            purge_hours=24,
            embargo_hours=12,
            step_days=7,
            min_train_samples=100,
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
            train_days=30,
            test_days=7,
            purge_hours=24,
            embargo_hours=12,
            step_days=7,
            min_train_samples=100,
        )

        n_folds = splitter.get_n_folds(timestamps)
        assert n_folds > 3, f"Expected >3 folds, got {n_folds}"


class TestDynamicPurge:
    """Tests for horizon-aware dynamic purge gap scaling."""

    def test_short_horizon_uses_base_purge(self):
        """1h and 4h horizons should use base 48h purge (already sufficient)."""
        from src.research.purged_walk_forward import compute_purge_hours

        assert compute_purge_hours(48, horizon=1) == 48  # 1*3=3 < 48
        assert compute_purge_hours(48, horizon=4) == 48  # 4*3=12 < 48
        assert compute_purge_hours(48, horizon=8) == 48  # 8*3=24 < 48

    def test_long_horizon_scales_purge(self):
        """24h horizon should get purge = 72h (24 * 3)."""
        from src.research.purged_walk_forward import compute_purge_hours

        assert compute_purge_hours(48, horizon=24) == 72  # 24*3=72 > 48
        assert compute_purge_hours(48, horizon=20) == 60  # 20*3=60 > 48

    def test_medium_horizon_transition(self):
        """12h horizon: 12*3=36 < 48, so base purge applies."""
        from src.research.purged_walk_forward import compute_purge_hours

        assert compute_purge_hours(48, horizon=12) == 48  # 12*3=36 < 48

    def test_purge_never_below_minimum(self):
        """Even with small base, purge never goes below MIN_PURGE_HOURS."""
        from src.research.purged_walk_forward import (
            compute_purge_hours,
            MIN_PURGE_HOURS,
        )

        assert compute_purge_hours(10, horizon=1) == MIN_PURGE_HOURS
        assert compute_purge_hours(0, horizon=None) == MIN_PURGE_HOURS

    def test_none_horizon_uses_base(self):
        """No horizon specified = use base purge."""
        from src.research.purged_walk_forward import compute_purge_hours

        assert compute_purge_hours(48, horizon=None) == 48
        assert compute_purge_hours(72, horizon=None) == 72

    def test_splitter_with_horizon(self):
        """PurgedWalkForward with horizon=24 should have wider purge gap."""
        from src.research.purged_walk_forward import PurgedWalkForward

        splitter_short = PurgedWalkForward(purge_hours=48, horizon=1)
        splitter_long = PurgedWalkForward(purge_hours=48, horizon=24)

        assert splitter_short.purge_hours == 48
        assert splitter_long.purge_hours == 72
        assert splitter_long.purge_hours > splitter_short.purge_hours

    def test_with_horizon_method(self):
        """with_horizon() should return new splitter with correct purge."""
        from src.research.purged_walk_forward import PurgedWalkForward

        base = PurgedWalkForward(purge_hours=48, horizon=1)
        assert base.purge_hours == 48

        reconfigured = base.with_horizon(24)
        assert reconfigured.purge_hours == 72
        assert reconfigured.horizon == 24
        # Original unchanged
        assert base.purge_hours == 48
        assert base.horizon == 1
        # Other settings preserved
        assert reconfigured.train_days == base.train_days
        assert reconfigured.embargo_hours == base.embargo_hours

    def test_with_horizon_preserves_base_purge(self):
        """with_horizon() should use original base_purge_hours, not computed."""
        from src.research.purged_walk_forward import PurgedWalkForward

        base = PurgedWalkForward(purge_hours=48, horizon=24)
        assert base.purge_hours == 72  # scaled

        # Reconfigure to short horizon — should go back to 48, not 72
        short = base.with_horizon(1)
        assert short.purge_hours == 48

    def test_dynamic_purge_prevents_leakage(self):
        """With 24h horizon, purge gap in actual folds must be >= 72h."""
        from src.research.purged_walk_forward import PurgedWalkForward

        n = 5000
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        timestamps = pd.Series([start + timedelta(hours=i) for i in range(n)])

        splitter = PurgedWalkForward(
            train_days=30,
            test_days=7,
            purge_hours=48,
            embargo_hours=12,
            step_days=7,
            min_train_samples=100,
            horizon=24,
        )

        folds = list(splitter.split(timestamps))
        assert len(folds) > 0, "Should generate at least one fold"

        for fold in folds:
            train_end_ts = timestamps.iloc[fold.train_indices[-1]]
            test_start_ts = timestamps.iloc[fold.test_indices[0]]
            gap = (test_start_ts - train_end_ts).total_seconds() / 3600
            assert (
                gap >= 72
            ), f"Fold {fold.fold_idx}: purge gap {gap}h < 72h for 24h horizon"


class TestSelectionMultipleComparisons:
    """Tests for Bonferroni-adjusted thresholds and consecutive windows."""

    def test_bonferroni_increases_sharpe_threshold(self):
        """More comparisons should raise the Sharpe threshold."""
        from src.research.selection import _bonferroni_sharpe

        base = 0.5
        assert _bonferroni_sharpe(base, 1) == base
        assert _bonferroni_sharpe(base, 5) > base
        assert _bonferroni_sharpe(base, 20) > _bonferroni_sharpe(base, 5)

    def test_bonferroni_increases_accuracy_threshold(self):
        """More comparisons should also raise the accuracy threshold."""
        from src.research.selection import _bonferroni_accuracy

        base = 0.52
        assert _bonferroni_accuracy(base, 1) == base
        assert _bonferroni_accuracy(base, 5) > base
        assert _bonferroni_accuracy(base, 20) > _bonferroni_accuracy(base, 5)
        # Should never exceed 1.0
        assert _bonferroni_accuracy(base, 1000) < 1.0

    def test_consecutive_windows_pass(self):
        """Model with consistent folds should pass."""
        from src.research.selection import _check_consecutive_windows

        folds = [{"sharpe_ratio": 1.0, "accuracy": 0.6} for _ in range(5)]
        passes, streak = _check_consecutive_windows(folds, 0.5, 0.52, min_consecutive=3)
        assert passes
        assert streak == 5

    def test_consecutive_windows_fail(self):
        """Model alternating good/bad should fail."""
        from src.research.selection import _check_consecutive_windows

        folds = [
            {"sharpe_ratio": 1.0, "accuracy": 0.6},
            {"sharpe_ratio": -0.5, "accuracy": 0.3},
            {"sharpe_ratio": 1.0, "accuracy": 0.6},
            {"sharpe_ratio": -0.5, "accuracy": 0.3},
            {"sharpe_ratio": 1.0, "accuracy": 0.6},
        ]
        passes, streak = _check_consecutive_windows(folds, 0.5, 0.52, min_consecutive=3)
        assert not passes
        assert streak == 1

    def test_catastrophic_fold_rejection(self):
        """Model with one -20% drawdown fold should be flagged."""
        from src.research.selection import _check_no_catastrophic_folds

        folds = [
            {"max_drawdown": -0.03, "sharpe_ratio": 1.0},
            {"max_drawdown": -0.25, "sharpe_ratio": 0.5},  # catastrophic
            {"max_drawdown": -0.02, "sharpe_ratio": 1.2},
        ]
        passes, bad = _check_no_catastrophic_folds(folds)
        assert not passes
        assert 1 in bad

    def test_select_candidates_applies_sharpe_correction(self):
        """select_candidates with many models should raise Sharpe threshold."""
        from src.research.selection import select_candidates

        # Create 20 models all with marginal Sharpe 0.55
        results = []
        for i in range(20):
            results.append(
                {
                    "model_id": f"model_{i}",
                    "folds": [
                        {
                            "sharpe_ratio": 0.55,
                            "accuracy": 0.55,
                            "breach_rate": 0.0,
                            "max_drawdown": -0.05,
                        }
                        for _ in range(5)
                    ],
                }
            )

        # With correction: adjusted threshold ≈ 0.5 + 0.1*ln(20) ≈ 0.80
        # So 0.55 avg Sharpe should NOT pass
        candidates = select_candidates(results, apply_multiple_comparisons=True)
        assert len(candidates) == 0

        # Without correction: 0.55 > 0.5 base, should pass
        candidates = select_candidates(results, apply_multiple_comparisons=False)
        assert len(candidates) == 20

    def test_select_candidates_applies_accuracy_correction(self):
        """Accuracy threshold should also tighten with more comparisons."""
        from src.research.selection import select_candidates

        # Models with high Sharpe but marginal accuracy 0.53
        results = []
        for i in range(20):
            results.append(
                {
                    "model_id": f"model_{i}",
                    "folds": [
                        {
                            "sharpe_ratio": 2.0,
                            "accuracy": 0.53,
                            "breach_rate": 0.0,
                            "max_drawdown": -0.02,
                        }
                        for _ in range(5)
                    ],
                }
            )

        # With correction: accuracy threshold ≈ 0.52 + (0.48)*0.05*ln(20) ≈ 0.592
        # 0.53 < 0.592, should fail
        candidates = select_candidates(results, apply_multiple_comparisons=True)
        assert len(candidates) == 0

        # Without correction: 0.53 > 0.52, should pass
        candidates = select_candidates(results, apply_multiple_comparisons=False)
        assert len(candidates) == 20


class TestRegimeSlippage:
    """Tests for regime-dependent slippage model."""

    def test_calm_market_baseline(self):
        """Low vol should give multiplier near 1.0."""
        from src.execution.slippage_model import compute_volatility_multiplier

        calm_returns = np.random.normal(0, 0.001, 24)  # 0.1% hourly vol
        mult = compute_volatility_multiplier(calm_returns, baseline_vol=0.002)
        assert mult <= 1.5, f"Calm market multiplier too high: {mult}"

    def test_volatile_market_scales_up(self):
        """High vol should give multiplier > 2."""
        from src.execution.slippage_model import compute_volatility_multiplier

        volatile_returns = np.random.normal(0, 0.006, 24)  # 0.6% hourly vol
        mult = compute_volatility_multiplier(volatile_returns, baseline_vol=0.002)
        assert mult >= 2.0, f"Volatile market multiplier too low: {mult}"

    def test_multiplier_capped(self):
        """Multiplier should never exceed MAX_SLIPPAGE_MULTIPLIER."""
        from src.execution.slippage_model import (
            compute_volatility_multiplier,
            MAX_SLIPPAGE_MULTIPLIER,
        )

        extreme_returns = np.random.normal(0, 0.05, 24)  # 5% hourly vol
        mult = compute_volatility_multiplier(extreme_returns, baseline_vol=0.002)
        assert mult <= MAX_SLIPPAGE_MULTIPLIER

    def test_liquidation_adder(self):
        """Liquidation volume should add to slippage."""
        from src.execution.slippage_model import compute_liquidation_adder

        assert compute_liquidation_adder(0) == 0.0
        assert compute_liquidation_adder(1_000_000) == pytest.approx(0.5)
        assert compute_liquidation_adder(5_000_000) == pytest.approx(2.5)

    def test_liquidation_adder_capped(self):
        """Liquidation adder should be capped."""
        from src.execution.slippage_model import compute_liquidation_adder

        assert compute_liquidation_adder(100_000_000) == 10.0  # capped

    def test_regime_slippage_integration(self):
        """Full regime slippage should combine vol + liquidations."""
        from src.execution.slippage_model import compute_regime_slippage

        # Calm market, no liquidations
        calm = compute_regime_slippage(
            base_slippage_bps=5.0,
            recent_returns=np.random.normal(0, 0.001, 24),
            liquidation_volume_usd=0,
        )
        assert 3.0 <= calm <= 8.0  # near baseline

        # Volatile + heavy liquidations
        stressed = compute_regime_slippage(
            base_slippage_bps=5.0,
            recent_returns=np.random.normal(0, 0.008, 24),
            liquidation_volume_usd=10_000_000,
        )
        assert stressed > calm * 2  # significantly higher

    def test_commission_scales_with_sqrt_vol(self):
        """Commission should scale with sqrt of vol multiplier (dampened)."""
        from src.execution.slippage_model import compute_regime_costs

        # Calm market
        _, calm_comm, _ = compute_regime_costs(
            base_commission_bps=2.0,
            recent_returns=np.random.normal(0, 0.001, 24),
        )

        # Very volatile (vol mult should be ~4x)
        _, vol_comm, breakdown = compute_regime_costs(
            base_commission_bps=2.0,
            recent_returns=np.random.normal(0, 0.008, 24),
        )

        # Commission should increase but less than slippage
        assert vol_comm > calm_comm
        # sqrt scaling means if vol_mult=4, commission_mult=2
        assert breakdown["commission_multiplier"] < breakdown["vol_multiplier"]

    def test_compute_regime_costs_returns_breakdown(self):
        """compute_regime_costs should return detailed breakdown dict."""
        from src.execution.slippage_model import compute_regime_costs

        slip, comm, breakdown = compute_regime_costs(
            base_slippage_bps=5.0,
            base_commission_bps=2.0,
            recent_returns=np.random.normal(0, 0.004, 24),
            liquidation_volume_usd=2_000_000,
        )

        assert "vol_multiplier" in breakdown
        assert "commission_multiplier" in breakdown
        assert "liq_adder_bps" in breakdown
        assert breakdown["liq_adder_bps"] == pytest.approx(1.0)
        # Breakdown values are rounded to 3 decimals, so use abs tolerance
        assert slip == pytest.approx(breakdown["effective_slippage_bps"], abs=0.01)
        assert comm == pytest.approx(breakdown["effective_commission_bps"], abs=0.01)

    def test_apply_regime_slippage(self):
        """Convenience function should give different fills in different regimes."""
        from src.execution.slippage_model import apply_regime_slippage

        mid = 50000.0
        calm_fill = apply_regime_slippage(
            mid,
            is_buy=True,
            recent_returns=np.random.normal(0, 0.001, 24),
        )
        volatile_fill = apply_regime_slippage(
            mid,
            is_buy=True,
            recent_returns=np.random.normal(0, 0.01, 24),
        )
        # Both above mid (buy), but volatile fill should be higher (worse)
        assert calm_fill > mid
        assert volatile_fill > calm_fill

    def test_backward_compatible_apply_slippage(self):
        """Original apply_slippage should still work unchanged."""
        from src.execution.slippage_model import apply_slippage

        mid = 50000.0
        buy_fill = apply_slippage(mid, is_buy=True, slippage_bps=5.0)
        sell_fill = apply_slippage(mid, is_buy=False, slippage_bps=5.0)
        assert buy_fill == pytest.approx(50025.0)
        assert sell_fill == pytest.approx(49975.0)

    def test_regime_cost_multipliers_exist(self):
        """Named regime presets should be available for callers with regime detectors."""
        from src.execution.slippage_model import REGIME_COST_MULTIPLIERS

        assert REGIME_COST_MULTIPLIERS["panic_flush"] == 3.0
        assert REGIME_COST_MULTIPLIERS["normal"] == 1.0
        assert "squeeze" in REGIME_COST_MULTIPLIERS

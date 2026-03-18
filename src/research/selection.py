"""Model selection and promotion logic.

Addresses multiple-comparisons risk when testing many model-horizon
combinations by:
1. Bonferroni-style threshold adjustment based on number of candidates
2. Requiring models to pass on consecutive walk-forward windows
3. Per-fold consistency checks (no single fold should be catastrophic)
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.common.config import load_yaml_config
from src.common.logging import get_logger

logger = get_logger(__name__)


def _bonferroni_sharpe(base_sharpe: float, n_comparisons: int) -> float:
    """Adjust Sharpe threshold for multiple comparisons.

    Uses a conservative Bonferroni-inspired correction: as the number of
    model-horizon combinations grows, the bar to clear gets higher.
    This isn't a p-value correction per se, but a practical inflation
    of the minimum Sharpe to reduce false discoveries.

    Formula: adjusted = base + 0.1 * ln(n_comparisons)
    - 1 comparison:  0.5 + 0.0  = 0.50
    - 5 comparisons: 0.5 + 0.16 = 0.66
    - 20 comparisons: 0.5 + 0.30 = 0.80
    - 50 comparisons: 0.5 + 0.39 = 0.89
    """
    if n_comparisons <= 1:
        return base_sharpe
    adjustment = 0.1 * np.log(n_comparisons)
    adjusted = base_sharpe + adjustment
    logger.debug("sharpe_threshold_adjusted",
                 base=base_sharpe, n_comparisons=n_comparisons,
                 adjusted=round(adjusted, 3))
    return adjusted


def _check_consecutive_windows(
    folds: list[dict[str, Any]],
    min_sharpe: float,
    min_accuracy: float,
    min_consecutive: int = 3,
) -> tuple[bool, int]:
    """Check if a model passes thresholds on consecutive walk-forward windows.

    Rather than only checking the aggregate average across all folds,
    this requires models to demonstrate sustained performance over
    min_consecutive adjacent windows. A model that crushes 2 folds
    but bombs 3 won't pass even if its average looks OK.

    Args:
        folds: list of per-fold metric dicts
        min_sharpe: minimum Sharpe per fold
        min_accuracy: minimum accuracy per fold
        min_consecutive: required number of consecutive passing folds

    Returns:
        (passes, longest_streak) tuple
    """
    if len(folds) < min_consecutive:
        return False, 0

    streak = 0
    longest = 0

    for f in folds:
        sharpe = f.get("sharpe_ratio", 0)
        acc = f.get("accuracy", 0)
        # Per-fold thresholds are relaxed vs aggregate (0.8x)
        # to allow some variance while still catching disasters
        if sharpe >= min_sharpe * 0.8 and acc >= min_accuracy * 0.8:
            streak += 1
            longest = max(longest, streak)
        else:
            streak = 0

    return longest >= min_consecutive, longest


def _check_no_catastrophic_folds(
    folds: list[dict[str, Any]],
    max_allowed_dd: float = -0.15,
    max_allowed_negative_sharpe: float = -1.0,
) -> tuple[bool, list[int]]:
    """Reject models with any catastrophic individual folds.

    Even if averages look fine, a single fold with -20% drawdown or
    deeply negative Sharpe suggests the model is fragile.

    Returns:
        (passes, list_of_catastrophic_fold_indices)
    """
    catastrophic = []
    for i, f in enumerate(folds):
        dd = f.get("max_drawdown", 0)
        sharpe = f.get("sharpe_ratio", 0)
        if dd < max_allowed_dd or sharpe < max_allowed_negative_sharpe:
            catastrophic.append(i)
    return len(catastrophic) == 0, catastrophic


def select_candidates(
    fold_results: list[dict[str, Any]],
    min_oos_sharpe: float | None = None,
    min_oos_accuracy: float | None = None,
    max_breach_rate: float | None = None,
    min_folds: int | None = None,
    min_consecutive_windows: int | None = None,
    apply_multiple_comparisons: bool = True,
) -> list[dict[str, Any]]:
    """Select candidate models that pass minimum thresholds.

    Applies three layers of filtering:
    1. Aggregate metrics (Sharpe, accuracy, breach rate) with optional
       Bonferroni-adjusted thresholds
    2. Consecutive window requirement - must pass on N adjacent folds
    3. No catastrophic folds - no single fold with extreme drawdown

    Args:
        fold_results: list of dicts with model_id, fold metrics
        min_oos_sharpe: minimum average OOS Sharpe (before adjustment)
        min_oos_accuracy: minimum average OOS accuracy
        max_breach_rate: maximum allowable breach rate
        min_folds: minimum number of folds evaluated
        min_consecutive_windows: required consecutive passing windows
        apply_multiple_comparisons: if True, adjust thresholds for n_comparisons

    Returns:
        Filtered list of model results that meet all criteria
    """
    cfg = load_yaml_config("model_registry.yaml").get("promotion", {})
    min_oos_sharpe = min_oos_sharpe or cfg.get("min_oos_sharpe", 0.5)
    min_oos_accuracy = min_oos_accuracy or cfg.get("min_oos_accuracy", 0.52)
    max_breach_rate = max_breach_rate if max_breach_rate is not None else cfg.get("max_breach_rate", 0.0)
    min_folds = min_folds or cfg.get("min_folds", 3)
    min_consecutive_windows = min_consecutive_windows or cfg.get("min_consecutive_windows", 3)

    # Adjust thresholds for multiple comparisons
    n_comparisons = len(fold_results)
    if apply_multiple_comparisons and n_comparisons > 1:
        adjusted_sharpe = _bonferroni_sharpe(min_oos_sharpe, n_comparisons)
    else:
        adjusted_sharpe = min_oos_sharpe

    candidates = []
    for result in fold_results:
        model_id = result.get("model_id", "unknown")
        folds = result.get("folds", [])

        if len(folds) < min_folds:
            logger.debug("skipped_insufficient_folds", model_id=model_id, n_folds=len(folds))
            continue

        avg_sharpe = np.mean([f.get("sharpe_ratio", 0) for f in folds])
        avg_accuracy = np.mean([f.get("accuracy", 0) for f in folds])
        avg_breach_rate = np.mean([f.get("breach_rate", 0) for f in folds])

        # Layer 1: Aggregate threshold check (with adjusted Sharpe)
        if avg_sharpe < adjusted_sharpe:
            logger.debug("skipped_low_sharpe", model_id=model_id,
                        sharpe=avg_sharpe, threshold=adjusted_sharpe)
            continue
        if avg_accuracy < min_oos_accuracy:
            logger.debug("skipped_low_accuracy", model_id=model_id, acc=avg_accuracy)
            continue
        if avg_breach_rate > max_breach_rate:
            logger.debug("skipped_high_breach", model_id=model_id, breach=avg_breach_rate)
            continue

        # Layer 2: Consecutive window requirement
        passes_consecutive, longest_streak = _check_consecutive_windows(
            folds, min_oos_sharpe, min_oos_accuracy, min_consecutive_windows,
        )
        if not passes_consecutive:
            logger.debug("skipped_no_consecutive_windows", model_id=model_id,
                        longest_streak=longest_streak, required=min_consecutive_windows)
            continue

        # Layer 3: No catastrophic folds
        passes_catastrophic, bad_folds = _check_no_catastrophic_folds(folds)
        if not passes_catastrophic:
            logger.debug("skipped_catastrophic_folds", model_id=model_id,
                        catastrophic_fold_indices=bad_folds)
            continue

        result["avg_sharpe"] = avg_sharpe
        result["avg_accuracy"] = avg_accuracy
        result["avg_breach_rate"] = avg_breach_rate
        result["consecutive_passing_streak"] = longest_streak
        result["adjusted_sharpe_threshold"] = adjusted_sharpe
        candidates.append(result)

    candidates.sort(key=lambda x: x.get("avg_sharpe", 0), reverse=True)
    logger.info("candidates_selected",
                total=len(fold_results), passed=len(candidates),
                sharpe_threshold=round(adjusted_sharpe, 3),
                n_comparisons=n_comparisons)
    return candidates


def rank_models(candidates: list[dict[str, Any]], metric: str = "avg_sharpe") -> list[dict[str, Any]]:
    """Rank candidate models by a metric."""
    return sorted(candidates, key=lambda x: x.get(metric, 0), reverse=True)

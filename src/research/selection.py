"""Model selection and promotion logic."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.common.config import load_yaml_config
from src.common.logging import get_logger

logger = get_logger(__name__)


def select_candidates(
    fold_results: list[dict[str, Any]],
    min_oos_sharpe: float | None = None,
    min_oos_accuracy: float | None = None,
    max_breach_rate: float | None = None,
    min_folds: int | None = None,
    min_consecutive_folds: int | None = None,
    require_positive_sharpe_each_fold: bool | None = None,
    n_combinations_tested: int | None = None,
    bonferroni_correction: bool | None = None,
) -> list[dict[str, Any]]:
    """Select candidate models that pass minimum thresholds.

    Implements stricter criteria to address multiple comparisons problem:
    - Requires consecutive fold passes (not just average)
    - Bonferroni correction for multiple model-horizon combinations
    - Ensures no single fold has negative Sharpe

    Args:
        fold_results: list of dicts with model_id, fold metrics
        min_oos_sharpe: minimum average OOS Sharpe
        min_oos_accuracy: minimum average OOS accuracy
        max_breach_rate: maximum allowable breach rate
        min_folds: minimum number of folds evaluated
        min_consecutive_folds: minimum consecutive folds passing thresholds
        require_positive_sharpe_each_fold: if True, every fold must have positive Sharpe
        n_combinations_tested: number of model-horizon combinations (for Bonferroni)
        bonferroni_correction: if True, apply Bonferroni correction to thresholds

    Returns:
        Filtered list of model results that meet criteria
    """
    cfg = load_yaml_config("model_registry.yaml").get("promotion", {})
    min_oos_sharpe = min_oos_sharpe or cfg.get("min_oos_sharpe", 0.5)
    min_oos_accuracy = min_oos_accuracy or cfg.get("min_oos_accuracy", 0.52)
    max_breach_rate = max_breach_rate if max_breach_rate is not None else cfg.get("max_breach_rate", 0.0)
    min_folds = min_folds or cfg.get("min_folds", 3)
    min_consecutive_folds = min_consecutive_folds or cfg.get("min_consecutive_folds", 2)
    require_positive_sharpe_each_fold = require_positive_sharpe_each_fold if require_positive_sharpe_each_fold is not None else cfg.get("require_positive_sharpe_each_fold", True)
    n_combinations_tested = n_combinations_tested or cfg.get("n_combinations_tested", 20)
    bonferroni_correction = bonferroni_correction if bonferroni_correction is not None else cfg.get("bonferroni_correction", True)

    # Apply Bonferroni correction to reduce false positives from multiple comparisons
    if bonferroni_correction and n_combinations_tested > 1:
        # Sharpe threshold increases with more combinations tested
        # Using Bonferroni: effective alpha = alpha / n, so we need higher thresholds
        correction_factor = 1.0 + 0.1 * np.log(n_combinations_tested)
        adjusted_min_sharpe = min_oos_sharpe * correction_factor
        adjusted_min_accuracy = min_oos_accuracy + (1 - min_oos_accuracy) * 0.05 * np.log(n_combinations_tested)
        logger.info("bonferroni_correction_applied", 
                   n_combinations=n_combinations_tested,
                   original_sharpe=min_oos_sharpe,
                   adjusted_sharpe=adjusted_min_sharpe,
                   original_accuracy=min_oos_accuracy,
                   adjusted_accuracy=adjusted_min_accuracy)
        min_oos_sharpe = adjusted_min_sharpe
        min_oos_accuracy = adjusted_min_accuracy

    candidates = []
    for result in fold_results:
        folds = result.get("folds", [])
        if len(folds) < min_folds:
            logger.debug("skipped_insufficient_folds", model_id=result.get("model_id"), n_folds=len(folds))
            continue

        # Check for consecutive passing folds
        if min_consecutive_folds > 1:
            consecutive_passes = 0
            max_consecutive = 0
            for fold in folds:
                sharpe = fold.get("sharpe_ratio", 0)
                acc = fold.get("accuracy", 0)
                if sharpe >= min_oos_sharpe and acc >= min_oos_accuracy:
                    consecutive_passes += 1
                    max_consecutive = max(max_consecutive, consecutive_passes)
                else:
                    consecutive_passes = 0
            
            if max_consecutive < min_consecutive_folds:
                logger.debug("skipped_insufficient_consecutive_folds", 
                            model_id=result.get("model_id"), 
                            max_consecutive=max_consecutive,
                            required=min_consecutive_folds)
                continue

        # Check that every fold has positive Sharpe (no single fold failure)
        if require_positive_sharpe_each_fold:
            negative_sharpe_folds = [f for f in folds if f.get("sharpe_ratio", 0) <= 0]
            if negative_sharpe_folds:
                logger.debug("skipped_negative_sharpe_fold", 
                            model_id=result.get("model_id"),
                            n_negative=len(negative_sharpe_folds))
                continue

        avg_sharpe = np.mean([f.get("sharpe_ratio", 0) for f in folds])
        avg_accuracy = np.mean([f.get("accuracy", 0) for f in folds])
        avg_breach_rate = np.mean([f.get("breach_rate", 0) for f in folds])

        if avg_sharpe < min_oos_sharpe:
            logger.debug("skipped_low_sharpe", model_id=result.get("model_id"), sharpe=avg_sharpe)
            continue
        if avg_accuracy < min_oos_accuracy:
            logger.debug("skipped_low_accuracy", model_id=result.get("model_id"), acc=avg_accuracy)
            continue
        if avg_breach_rate > max_breach_rate:
            logger.debug("skipped_high_breach", model_id=result.get("model_id"), breach=avg_breach_rate)
            continue

        result["avg_sharpe"] = avg_sharpe
        result["avg_accuracy"] = avg_accuracy
        result["avg_breach_rate"] = avg_breach_rate
        candidates.append(result)

    candidates.sort(key=lambda x: x.get("avg_sharpe", 0), reverse=True)
    logger.info("candidates_selected", total=len(fold_results), passed=len(candidates))
    return candidates


def rank_models(candidates: list[dict[str, Any]], metric: str = "avg_sharpe") -> list[dict[str, Any]]:
    """Rank candidate models by a metric."""
    return sorted(candidates, key=lambda x: x.get(metric, 0), reverse=True)

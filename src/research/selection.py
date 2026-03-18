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
) -> list[dict[str, Any]]:
    """Select candidate models that pass minimum thresholds.

    Args:
        fold_results: list of dicts with model_id, fold metrics
        min_oos_sharpe: minimum average OOS Sharpe
        min_oos_accuracy: minimum average OOS accuracy
        max_breach_rate: maximum allowable breach rate
        min_folds: minimum number of folds evaluated

    Returns:
        Filtered list of model results that meet criteria
    """
    cfg = load_yaml_config("model_registry.yaml").get("promotion", {})
    min_oos_sharpe = min_oos_sharpe or cfg.get("min_oos_sharpe", 0.5)
    min_oos_accuracy = min_oos_accuracy or cfg.get("min_oos_accuracy", 0.52)
    max_breach_rate = max_breach_rate if max_breach_rate is not None else cfg.get("max_breach_rate", 0.0)
    min_folds = min_folds or cfg.get("min_folds", 3)

    candidates = []
    for result in fold_results:
        folds = result.get("folds", [])
        if len(folds) < min_folds:
            logger.debug("skipped_insufficient_folds", model_id=result.get("model_id"), n_folds=len(folds))
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

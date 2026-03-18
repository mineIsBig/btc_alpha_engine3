"""Research cycle: data refresh -> feature build -> walk-forward -> model selection.

Supports dynamic hyperparameter overrides from EvolutionConfig, allowing the
autonomous agent to tune model parameters between retraining cycles.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.common.logging import get_logger
from src.data.ingest_jobs import incremental_refresh
from src.features.feature_pipeline import build_features, get_feature_names
from src.labels.labels import build_labels
from src.models.baseline import LogisticRegressionModel, RandomForestModel
from src.models.gradient_boost import LightGBMModel, XGBoostModel
from src.models.registry import ModelArtifactRegistry
from src.research.datasets import prepare_dataset, get_feature_columns, get_label_column
from src.research.purged_walk_forward import PurgedWalkForward
from src.research.scoring import compute_fold_metrics
from src.research.selection import select_candidates
from src.research.reports import save_fold_report, generate_summary_report

logger = get_logger(__name__)

DEFAULT_MODEL_CONFIGS = [
    ("lr", LogisticRegressionModel, {"C": 1.0, "penalty": "l2"}),
    ("rf", RandomForestModel, {"n_estimators": 200, "max_depth": 10}),
    ("lgbm", LightGBMModel, {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05}),
    ("xgb", XGBoostModel, {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05}),
]

# Keep backward compat alias
MODEL_CONFIGS = DEFAULT_MODEL_CONFIGS

HORIZONS = [1, 4, 8, 12, 24]


def _get_evolved_model_configs() -> list[tuple[str, type, dict[str, Any]]]:
    """Build model configs with dynamic hyperparameter overrides from EvolutionConfig."""
    try:
        from src.agent.evolution_config import load_evolution_config
        config = load_evolution_config()
    except Exception:
        return list(DEFAULT_MODEL_CONFIGS)

    evolved = []
    for name, cls, default_params in DEFAULT_MODEL_CONFIGS:
        override = config.hyperparam_overrides.get(name)
        if override and override.params:
            merged = {**default_params, **override.params}
            logger.info("hyperparam_override_applied", model=name, overrides=override.params)
            evolved.append((name, cls, merged))
        else:
            evolved.append((name, cls, dict(default_params)))
    return evolved


def run_research_cycle(
    dataset: pd.DataFrame | None = None,
    horizons: list[int] | None = None,
    use_evolved_configs: bool = False,
) -> list[dict[str, Any]]:
    """Run a complete research cycle.

    1. Prepare dataset
    2. For each horizon x model type, run walk-forward
    3. Score and select candidates
    4. Return results

    Args:
        use_evolved_configs: If True, apply hyperparameter overrides from EvolutionConfig.
    """
    horizons = horizons or HORIZONS

    if dataset is None:
        logger.info("preparing_dataset")
        dataset = prepare_dataset(horizons=horizons)

    if dataset.empty:
        logger.error("empty_dataset")
        return []

    feature_cols = get_feature_columns(dataset)
    registry = ModelArtifactRegistry()
    model_configs = _get_evolved_model_configs() if use_evolved_configs else DEFAULT_MODEL_CONFIGS

    all_results = []

    for horizon in horizons:
        label_col = get_label_column(horizon)
        fwd_ret_col = f"fwd_ret_{horizon}h"

        if label_col not in dataset.columns:
            logger.warning("missing_label", horizon=horizon)
            continue

        # Create a horizon-specific splitter so purge gap scales with
        # label horizon (e.g. 72h purge for 24h labels instead of 48h)
        splitter = PurgedWalkForward.from_config(horizon=horizon)

        for model_name, model_cls, default_params in model_configs:
            model_id = f"{model_name}_h{horizon}"
            logger.info("walk_forward_start", model_id=model_id,
                        purge_hours=splitter.purge_hours)

            fold_metrics = []
            for fold in splitter.split(dataset["timestamp"]):
                try:
                    X_train = dataset.iloc[fold.train_indices][feature_cols].fillna(0)
                    y_train = dataset.iloc[fold.train_indices][label_col].fillna(0).astype(int).values
                    X_test = dataset.iloc[fold.test_indices][feature_cols].fillna(0)
                    y_test = dataset.iloc[fold.test_indices][label_col].fillna(0).astype(int).values
                    fwd_ret = dataset.iloc[fold.test_indices][fwd_ret_col].fillna(0).values
                    test_ts = dataset.iloc[fold.test_indices]["timestamp"].values

                    if len(np.unique(y_train)) < 2:
                        continue

                    model = model_cls(horizon=horizon, params=default_params, model_id=model_id)
                    model.fit(X_train, y_train, feature_names=feature_cols)
                    y_pred = model.predict(X_test)

                    metrics = compute_fold_metrics(
                        y_true=y_test, y_pred=y_pred, fwd_returns=fwd_ret,
                        timestamps=test_ts,
                    )
                    fold_metrics.append(metrics)

                    save_fold_report(
                        model_id=model_id,
                        fold_idx=fold.fold_idx,
                        train_start=fold.train_start,
                        train_end=fold.train_end,
                        test_start=fold.test_start,
                        test_end=fold.test_end,
                        n_train=len(fold.train_indices),
                        n_test=len(fold.test_indices),
                        metrics=metrics,
                    )

                except Exception as e:
                    logger.error("fold_error", model_id=model_id, fold=fold.fold_idx, error=str(e))

            if fold_metrics:
                avg_sharpe = np.mean([m["sharpe_ratio"] for m in fold_metrics])
                avg_acc = np.mean([m["accuracy"] for m in fold_metrics])
                logger.info("walk_forward_done", model_id=model_id,
                           avg_sharpe=avg_sharpe, avg_acc=avg_acc, n_folds=len(fold_metrics))

                all_results.append({
                    "model_id": model_id,
                    "model_type": model_name,
                    "horizon": horizon,
                    "params": default_params,
                    "folds": fold_metrics,
                    "feature_names": feature_cols,
                })

                # Save last-fold model
                try:
                    last_model = model_cls(horizon=horizon, params=default_params, model_id=model_id)
                    X_full = dataset[feature_cols].fillna(0)
                    y_full = dataset[label_col].fillna(0).astype(int).values
                    last_model.fit(X_full, y_full, feature_names=feature_cols)
                    registry.save_model(last_model, metrics={
                        "sharpe": avg_sharpe, "accuracy": avg_acc,
                        "breach_rate": np.mean([m.get("breach_rate", 0) for m in fold_metrics]),
                    })
                except Exception as e:
                    logger.error("save_model_error", model_id=model_id, error=str(e))

    # Select candidates
    candidates = select_candidates(all_results)
    logger.info("research_cycle_complete", total_models=len(all_results), candidates=len(candidates))

    return candidates


def auto_retrain_and_promote(iteration: int = 0) -> dict[str, Any]:
    """Autonomous retrain: run research cycle with evolved configs, promote best models.

    Called by the agent loop when retrain triggers fire. Uses EvolutionConfig
    hyperparameter overrides and automatically promotes winning candidates.

    Returns summary of what happened.
    """
    from src.agent.evolution_config import load_evolution_config, save_evolution_config

    logger.info("auto_retrain_start", iteration=iteration)
    config = load_evolution_config()
    config.retrain.retrain_in_progress = True
    save_evolution_config(config)

    result: dict[str, Any] = {
        "iteration": iteration,
        "status": "failed",
        "candidates": 0,
        "promoted": 0,
        "retired": 0,
        "errors": [],
    }

    try:
        candidates = run_research_cycle(use_evolved_configs=True)
        result["candidates"] = len(candidates)

        if candidates:
            registry = ModelArtifactRegistry()

            # Promote candidates that pass selection
            promoted = 0
            for c in candidates:
                model_id = c.get("model_id", "")
                try:
                    registry.promote_model(model_id)
                    promoted += 1
                except Exception as e:
                    result["errors"].append(f"promote {model_id}: {e}")

            result["promoted"] = promoted

            # Retire underperforming models
            retired = 0
            lifecycle = config.model_lifecycle
            all_promoted = registry.get_promoted_models()
            for model_info in all_promoted:
                sharpe = model_info.get("oos_sharpe", 0) or 0
                if sharpe < lifecycle.retire_sharpe_threshold:
                    try:
                        registry.retire_model(model_info["model_id"])
                        retired += 1
                    except Exception as e:
                        result["errors"].append(f"retire {model_info['model_id']}: {e}")

            result["retired"] = retired
            result["status"] = "success"

        else:
            result["status"] = "no_candidates"

    except Exception as e:
        logger.error("auto_retrain_failed", error=str(e))
        result["errors"].append(str(e))

    finally:
        config = load_evolution_config()
        config.retrain.retrain_in_progress = False
        config.retrain.last_retrain_iteration = iteration
        save_evolution_config(config)
        logger.info("auto_retrain_complete", **result)

    return result

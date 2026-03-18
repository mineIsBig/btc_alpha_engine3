"""Training cycle: retrain models on latest data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logging import get_logger
from src.models.registry import ModelArtifactRegistry, MODEL_CLASSES
from src.research.datasets import prepare_dataset, get_feature_columns, get_label_column
from src.storage.database import session_scope
from src.storage.models import ModelRegistry

logger = get_logger(__name__)


def retrain_promoted_models(dataset: pd.DataFrame | None = None) -> int:
    """Retrain all promoted models on latest data.

    Returns number of models retrained.
    """
    registry = ModelArtifactRegistry()

    if dataset is None:
        dataset = prepare_dataset()

    if dataset.empty:
        logger.warning("empty_dataset_for_retrain")
        return 0

    feature_cols = get_feature_columns(dataset)
    count = 0

    with session_scope() as session:
        promoted = session.query(ModelRegistry).filter_by(status="promoted").all()
        model_infos = [
            {
                "model_id": m.model_id,
                "model_type": m.model_type,
                "horizon": m.horizon,
                "params_json": m.params_json,
            }
            for m in promoted
        ]

    for info in model_infos:
        try:
            import json

            model_cls = MODEL_CLASSES.get(info["model_type"])
            if model_cls is None:
                continue

            params = json.loads(info["params_json"]) if info["params_json"] else {}
            horizon = info["horizon"]
            label_col = get_label_column(horizon)

            if label_col not in dataset.columns:
                continue

            model = model_cls(horizon=horizon, params=params, model_id=info["model_id"])
            X = dataset[feature_cols].fillna(0)
            y = dataset[label_col].fillna(0).astype(int).values

            if len(np.unique(y)) < 2:
                continue

            model.fit(X, y, feature_names=feature_cols)
            registry.save_model(model)
            count += 1
            logger.info("model_retrained", model_id=info["model_id"])

        except Exception as e:
            logger.error("retrain_error", model_id=info["model_id"], error=str(e))

    logger.info("retrain_complete", count=count)
    return count

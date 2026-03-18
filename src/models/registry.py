"""Model artifact registry: save, load, promote, retire models."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.common.logging import get_logger
from src.models.base import BaseAlphaModel
from src.models.baseline import LogisticRegressionModel, RandomForestModel
from src.models.gradient_boost import LightGBMModel, XGBoostModel
from src.storage.database import session_scope
from src.storage.models import ModelRegistry

logger = get_logger(__name__)

MODEL_CLASSES: dict[str, type[BaseAlphaModel]] = {
    "LogisticRegressionModel": LogisticRegressionModel,
    "RandomForestModel": RandomForestModel,
    "LightGBMModel": LightGBMModel,
    "XGBoostModel": XGBoostModel,
}

ARTIFACT_DIR = Path("artifacts/models")


class ModelArtifactRegistry:
    """Manages model artifacts and DB registry."""

    def __init__(self, artifact_dir: Path | None = None):
        self.artifact_dir = artifact_dir or ARTIFACT_DIR
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model: BaseAlphaModel,
        metrics: dict[str, float] | None = None,
        train_start: datetime | None = None,
        train_end: datetime | None = None,
    ) -> str:
        """Save model artifact and register in DB."""
        model_id = model.model_id
        artifact_path = self.artifact_dir / f"{model_id}.joblib"
        model.save(artifact_path)

        metrics = metrics or {}

        with session_scope() as session:
            existing = session.query(ModelRegistry).filter_by(model_id=model_id).first()
            if existing:
                existing.oos_sharpe = metrics.get("sharpe")
                existing.oos_accuracy = metrics.get("accuracy")
                existing.oos_profit_factor = metrics.get("profit_factor")
                existing.breach_rate = metrics.get("breach_rate", 0.0)
                existing.artifact_path = str(artifact_path)
                existing.train_start = train_start
                existing.train_end = train_end
            else:
                reg = ModelRegistry(
                    model_id=model_id,
                    model_type=model.__class__.__name__,
                    horizon=model.horizon,
                    features_json=json.dumps(model.feature_names),
                    params_json=json.dumps(model.params),
                    artifact_path=str(artifact_path),
                    train_start=train_start,
                    train_end=train_end,
                    oos_sharpe=metrics.get("sharpe"),
                    oos_accuracy=metrics.get("accuracy"),
                    oos_profit_factor=metrics.get("profit_factor"),
                    breach_rate=metrics.get("breach_rate", 0.0),
                    status="candidate",
                )
                session.add(reg)

        logger.info("model_registered", model_id=model_id)
        return model_id

    def load_model(self, model_id: str) -> BaseAlphaModel:
        """Load a model by its ID."""
        with session_scope() as session:
            reg = session.query(ModelRegistry).filter_by(model_id=model_id).first()
            if reg is None:
                raise ValueError(f"Model not found: {model_id}")

            model_cls = MODEL_CLASSES.get(reg.model_type)
            if model_cls is None:
                raise ValueError(f"Unknown model type: {reg.model_type}")

            params = json.loads(reg.params_json) if reg.params_json else {}
            model = model_cls(horizon=reg.horizon, params=params, model_id=model_id)
            model.load(reg.artifact_path)
            return model

    def promote_model(self, model_id: str) -> None:
        """Promote a candidate model to active status."""
        with session_scope() as session:
            reg = session.query(ModelRegistry).filter_by(model_id=model_id).first()
            if reg is None:
                raise ValueError(f"Model not found: {model_id}")
            reg.status = "promoted"
            reg.promoted_at = datetime.utcnow()
            logger.info("model_promoted", model_id=model_id)

    def retire_model(self, model_id: str) -> None:
        """Retire a model."""
        with session_scope() as session:
            reg = session.query(ModelRegistry).filter_by(model_id=model_id).first()
            if reg:
                reg.status = "retired"
                logger.info("model_retired", model_id=model_id)

    def get_promoted_models(self, horizon: int | None = None) -> list[dict[str, Any]]:
        """Get all promoted models, optionally filtered by horizon."""
        with session_scope() as session:
            q = session.query(ModelRegistry).filter_by(status="promoted")
            if horizon is not None:
                q = q.filter_by(horizon=horizon)
            results = []
            for reg in q.all():
                results.append(
                    {
                        "model_id": reg.model_id,
                        "model_type": reg.model_type,
                        "horizon": reg.horizon,
                        "oos_sharpe": reg.oos_sharpe,
                        "oos_accuracy": reg.oos_accuracy,
                        "artifact_path": reg.artifact_path,
                    }
                )
            return results

    def get_best_model_per_horizon(self) -> dict[int, str]:
        """Get the best promoted model for each horizon by OOS Sharpe."""
        with session_scope() as session:
            promoted = session.query(ModelRegistry).filter_by(status="promoted").all()
            best: dict[int, tuple[float, str]] = {}
            for reg in promoted:
                sharpe = reg.oos_sharpe or 0.0
                if reg.horizon not in best or sharpe > best[reg.horizon][0]:
                    best[reg.horizon] = (sharpe, reg.model_id)
            return {h: mid for h, (_, mid) in best.items()}

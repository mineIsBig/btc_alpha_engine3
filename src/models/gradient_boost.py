"""Gradient boosting models: LightGBM and XGBoost."""
from __future__ import annotations

from typing import Any

from src.models.base import BaseAlphaModel
from src.common.logging import get_logger

logger = get_logger(__name__)


class LightGBMModel(BaseAlphaModel):
    """LightGBM classifier."""

    def _build_model(self) -> Any:
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm not installed. Install with: pip install lightgbm")

        return lgb.LGBMClassifier(
            n_estimators=self.params.get("n_estimators", 300),
            max_depth=self.params.get("max_depth", 5),
            learning_rate=self.params.get("learning_rate", 0.05),
            num_leaves=self.params.get("num_leaves", 31),
            min_child_samples=self.params.get("min_child_samples", 20),
            subsample=self.params.get("subsample", 0.8),
            colsample_bytree=self.params.get("colsample_bytree", 0.8),
            reg_alpha=self.params.get("reg_alpha", 0.1),
            reg_lambda=self.params.get("reg_lambda", 0.1),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )


class XGBoostModel(BaseAlphaModel):
    """XGBoost classifier."""

    def _build_model(self) -> Any:
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost not installed. Install with: pip install xgboost")

        return xgb.XGBClassifier(
            n_estimators=self.params.get("n_estimators", 300),
            max_depth=self.params.get("max_depth", 5),
            learning_rate=self.params.get("learning_rate", 0.05),
            subsample=self.params.get("subsample", 0.8),
            colsample_bytree=self.params.get("colsample_bytree", 0.8),
            reg_alpha=self.params.get("reg_alpha", 0.1),
            reg_lambda=self.params.get("reg_lambda", 0.1),
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    def fit(self, X, y, feature_names=None):
        """Override to remap labels for XGBoost (requires 0-indexed classes)."""
        import numpy as np
        # XGBoost needs labels starting from 0
        y_mapped = y.copy()
        unique_labels = np.unique(y)
        self._label_map = {orig: i for i, orig in enumerate(sorted(unique_labels))}
        self._label_inv_map = {i: orig for orig, i in self._label_map.items()}
        y_mapped = np.array([self._label_map[v] for v in y])
        super().fit(X, y_mapped, feature_names)

    def predict(self, X):
        import numpy as np
        raw = super().predict(X)
        if hasattr(self, "_label_inv_map"):
            return np.array([self._label_inv_map.get(int(v), v) for v in raw])
        return raw

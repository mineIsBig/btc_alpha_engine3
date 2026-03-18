"""Base model interface for all alpha models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib

from src.common.logging import get_logger

logger = get_logger(__name__)


class BaseAlphaModel(ABC):
    """Unified interface for all alpha signal models."""

    def __init__(self, horizon: int, params: dict[str, Any] | None = None, model_id: str = ""):
        self.horizon = horizon
        self.params = params or {}
        self.model_id = model_id
        self.feature_names: list[str] = []
        self.is_fitted = False
        self._model: Any = None
        self._calibrator: Any = None

    @abstractmethod
    def _build_model(self) -> Any:
        """Build the underlying sklearn/lgb/xgb model object."""
        ...

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> None:
        """Train the model."""
        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)

        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        self._model = self._build_model()
        self._model.fit(X_arr, y)
        self.is_fitted = True
        logger.info("model_fitted", model_id=self.model_id, horizon=self.horizon, n_samples=len(y))

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class labels (-1, 0, 1)."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return self._model.predict(X_arr)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities. Returns (n_samples, n_classes) array."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(X_arr)
            if self._calibrator is not None:
                proba = self._calibrator.predict_proba(X_arr)
            return proba
        # Fallback: use decision function or raw predictions
        preds = self._model.predict(X_arr)
        n_classes = len(np.unique(preds))
        proba = np.zeros((len(X_arr), max(n_classes, 3)))
        for i, p in enumerate(preds):
            idx = int(p) + 1  # map -1->0, 0->1, 1->2
            idx = max(0, min(idx, proba.shape[1] - 1))
            proba[i, idx] = 1.0
        return proba

    def calibrate(self, X_val: pd.DataFrame | np.ndarray, y_val: np.ndarray, method: str = "isotonic") -> None:
        """Calibrate predicted probabilities using validation data."""
        from sklearn.calibration import CalibratedClassifierCV
        X_arr = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
        self._calibrator = CalibratedClassifierCV(self._model, method=method, cv="prefit")
        self._calibrator.fit(X_arr, y_val)
        logger.info("model_calibrated", model_id=self.model_id, method=method)

    def save(self, path: str | Path) -> None:
        """Save model artifact to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": self._model,
            "calibrator": self._calibrator,
            "horizon": self.horizon,
            "params": self.params,
            "model_id": self.model_id,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
            "model_class": self.__class__.__name__,
        }
        joblib.dump(artifact, path)
        logger.info("model_saved", path=str(path))

    def load(self, path: str | Path) -> None:
        """Load model artifact from disk."""
        artifact = joblib.load(path)
        self._model = artifact["model"]
        self._calibrator = artifact.get("calibrator")
        self.horizon = artifact["horizon"]
        self.params = artifact["params"]
        self.model_id = artifact["model_id"]
        self.feature_names = artifact["feature_names"]
        self.is_fitted = artifact["is_fitted"]
        logger.info("model_loaded", path=str(path), model_id=self.model_id)

    def get_signal(self, X: pd.DataFrame | np.ndarray) -> dict[str, Any]:
        """Generate a trading signal from the latest row of features.

        Returns dict with: side (-1,0,1), probability, confidence
        """
        if isinstance(X, pd.DataFrame):
            X_row = X.iloc[[-1]]
        else:
            X_row = X[[-1]]

        proba = self.predict_proba(X_row)[0]

        # Map probabilities to side
        # Assumes classes are [-1, 0, 1] mapped to indices [0, 1, 2]
        if len(proba) == 3:
            side_probs = {-1: proba[0], 0: proba[1], 1: proba[2]}
        elif len(proba) == 2:
            # Binary: short vs long
            side_probs = {-1: proba[0], 0: 0.0, 1: proba[1]}
        else:
            pred = self.predict(X_row)[0]
            side_probs = {-1: 0.0, 0: 0.0, 1: 0.0}
            side_probs[int(pred)] = 1.0

        best_side = max(side_probs, key=side_probs.get)
        best_prob = side_probs[best_side]
        confidence = best_prob - max(v for k, v in side_probs.items() if k != best_side)

        return {
            "side": best_side,
            "probability": best_prob,
            "confidence": max(0.0, confidence),
            "proba_dist": side_probs,
        }

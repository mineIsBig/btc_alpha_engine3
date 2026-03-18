"""Regime gate model: adjusts signals based on detected market regime."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from src.models.base import BaseAlphaModel
from src.common.logging import get_logger

logger = get_logger(__name__)

# Regime-specific signal adjustments
REGIME_SIGNAL_MULTIPLIERS = {
    "trend_up": {"long": 1.2, "short": 0.5, "flat": 0.8},
    "trend_down": {"long": 0.5, "short": 1.2, "flat": 0.8},
    "mean_revert": {"long": 0.8, "short": 0.8, "flat": 1.2},
    "crowded_long": {"long": 0.3, "short": 1.5, "flat": 1.0},
    "crowded_short": {"long": 1.5, "short": 0.3, "flat": 1.0},
    "panic_flush": {"long": 0.2, "short": 0.5, "flat": 1.5},
    "squeeze": {"long": 1.3, "short": 0.2, "flat": 0.8},
    "neutral": {"long": 1.0, "short": 1.0, "flat": 1.0},
}


class RegimeGateModel(BaseAlphaModel):
    """Random forest model for regime classification."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._label_encoder = LabelEncoder()

    def _build_model(self) -> Any:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=self.params.get("n_estimators", 200),
                max_depth=self.params.get("max_depth", 8),
                min_samples_leaf=self.params.get("min_samples_leaf", 30),
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ])

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> None:
        """Fit regime classifier. y should be string regime labels."""
        y_encoded = self._label_encoder.fit_transform(y)
        super().fit(X, y_encoded, feature_names)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict regime labels (strings)."""
        encoded = super().predict(X)
        return self._label_encoder.inverse_transform(encoded.astype(int))

    def predict_regime(self, X: pd.DataFrame | np.ndarray) -> str:
        """Predict regime for the latest row."""
        preds = self.predict(X)
        return str(preds[-1])

    @staticmethod
    def get_signal_multiplier(regime: str, side: int) -> float:
        """Get the signal strength multiplier for a given regime and side.

        Args:
            regime: detected regime label
            side: -1 (short), 0 (flat), 1 (long)

        Returns:
            multiplier in [0, 2] range
        """
        multipliers = REGIME_SIGNAL_MULTIPLIERS.get(regime, REGIME_SIGNAL_MULTIPLIERS["neutral"])
        if side == 1:
            return multipliers["long"]
        elif side == -1:
            return multipliers["short"]
        else:
            return multipliers["flat"]

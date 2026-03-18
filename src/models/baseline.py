"""Baseline models: Logistic Regression and Random Forest."""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.models.base import BaseAlphaModel


class LogisticRegressionModel(BaseAlphaModel):
    """Logistic regression with elastic net regularization."""

    def _build_model(self) -> Any:
        C = self.params.get("C", 1.0)
        penalty = self.params.get("penalty", "l2")
        solver = self.params.get("solver", "saga")
        max_iter = self.params.get("max_iter", 2000)

        lr_params: dict[str, Any] = {
            "C": C,
            "penalty": penalty,
            "solver": solver,
            "max_iter": max_iter,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }

        if penalty == "elasticnet":
            lr_params["l1_ratio"] = self.params.get("l1_ratio", 0.5)

        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**lr_params)),
        ])


class RandomForestModel(BaseAlphaModel):
    """Random Forest classifier."""

    def _build_model(self) -> Any:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=self.params.get("n_estimators", 200),
                max_depth=self.params.get("max_depth", 10),
                min_samples_leaf=self.params.get("min_samples_leaf", 20),
                max_features=self.params.get("max_features", "sqrt"),
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ])

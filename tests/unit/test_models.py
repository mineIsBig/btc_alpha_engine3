"""Tests for model training and prediction."""
import numpy as np
import pytest


class TestBaselineModels:
    def test_logistic_regression_fit_predict(self, synthetic_features_and_labels):
        from src.models.baseline import LogisticRegressionModel
        from src.research.datasets import get_feature_columns

        ds = synthetic_features_and_labels
        feat_cols = get_feature_columns(ds)
        X = ds[feat_cols].fillna(0).iloc[:300]
        y = ds["label_4h"].fillna(0).astype(int).values[:300]

        model = LogisticRegressionModel(horizon=4, params={"C": 1.0, "penalty": "l2"}, model_id="test_lr")
        model.fit(X, y, feature_names=feat_cols)
        assert model.is_fitted

        preds = model.predict(X.iloc[:10])
        assert len(preds) == 10
        assert set(preds).issubset({-1, 0, 1})

    def test_random_forest_fit_predict(self, synthetic_features_and_labels):
        from src.models.baseline import RandomForestModel
        from src.research.datasets import get_feature_columns

        ds = synthetic_features_and_labels
        feat_cols = get_feature_columns(ds)
        X = ds[feat_cols].fillna(0).iloc[:300]
        y = ds["label_4h"].fillna(0).astype(int).values[:300]

        model = RandomForestModel(horizon=4, params={"n_estimators": 50}, model_id="test_rf")
        model.fit(X, y, feature_names=feat_cols)
        assert model.is_fitted

        proba = model.predict_proba(X.iloc[:10])
        assert proba.shape[0] == 10
        # Probabilities should sum to ~1
        assert np.allclose(proba.sum(axis=1), 1.0, atol=0.01)

    def test_model_save_load(self, synthetic_features_and_labels, tmp_path):
        from src.models.baseline import LogisticRegressionModel
        from src.research.datasets import get_feature_columns

        ds = synthetic_features_and_labels
        feat_cols = get_feature_columns(ds)
        X = ds[feat_cols].fillna(0).iloc[:300]
        y = ds["label_4h"].fillna(0).astype(int).values[:300]

        model = LogisticRegressionModel(horizon=4, params={"C": 1.0}, model_id="test_save")
        model.fit(X, y, feature_names=feat_cols)

        path = tmp_path / "model.joblib"
        model.save(path)

        loaded = LogisticRegressionModel(horizon=4, model_id="test_save")
        loaded.load(path)
        assert loaded.is_fitted
        assert loaded.feature_names == feat_cols

    def test_get_signal(self, synthetic_features_and_labels):
        from src.models.baseline import RandomForestModel
        from src.research.datasets import get_feature_columns

        ds = synthetic_features_and_labels
        feat_cols = get_feature_columns(ds)
        X = ds[feat_cols].fillna(0).iloc[:300]
        y = ds["label_4h"].fillna(0).astype(int).values[:300]

        model = RandomForestModel(horizon=4, params={"n_estimators": 50}, model_id="test_sig")
        model.fit(X, y, feature_names=feat_cols)

        sig = model.get_signal(X.iloc[:10])
        assert "side" in sig
        assert sig["side"] in (-1, 0, 1)
        assert 0 <= sig["probability"] <= 1
        assert sig["confidence"] >= 0

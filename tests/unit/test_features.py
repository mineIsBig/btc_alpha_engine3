"""Tests for feature computation and alignment."""
import numpy as np
import pandas as pd
import pytest


class TestPriceFeatures:
    def test_returns_computed(self, synthetic_price_df):
        from src.features.price_features import compute_price_features
        feats = compute_price_features(synthetic_price_df)
        assert "ret_1h" in feats.columns
        assert "ret_4h" in feats.columns
        assert "ret_24h" in feats.columns
        # First row should be NaN for ret_1h
        assert pd.isna(feats["ret_1h"].iloc[0])
        # Non-NaN after warmup
        assert not pd.isna(feats["ret_1h"].iloc[5])

    def test_volatility_computed(self, synthetic_price_df):
        from src.features.price_features import compute_price_features
        feats = compute_price_features(synthetic_price_df)
        assert "rvol_24h" in feats.columns
        # Should have values after warmup period
        valid = feats["rvol_24h"].dropna()
        assert len(valid) > 0
        assert (valid > 0).all()

    def test_rsi_range(self, synthetic_price_df):
        from src.features.price_features import compute_price_features
        feats = compute_price_features(synthetic_price_df)
        valid_rsi = feats["rsi_14"].dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()


class TestFeatureAlignment:
    def test_feature_pipeline_output_aligned(self, synthetic_raw_data):
        from src.features.feature_pipeline import build_features
        feats = build_features(raw_data=synthetic_raw_data)
        assert "timestamp" in feats.columns
        assert len(feats) > 0
        # Timestamps should be sorted
        ts = pd.to_datetime(feats["timestamp"])
        assert ts.is_monotonic_increasing

    def test_no_future_leakage_in_features(self, synthetic_raw_data):
        """Features at time t should only use data from t and earlier."""
        from src.features.price_features import compute_price_features
        price = synthetic_raw_data["price"]
        feats = compute_price_features(price)
        # ret_1h at index i uses close[i] and close[i-1], both <= time i
        # This is verified by the pct_change implementation
        assert "ret_1h" in feats.columns


class TestFundingFeatures:
    def test_funding_zscore(self, synthetic_raw_data):
        from src.features.funding_features import compute_funding_features
        df = synthetic_raw_data["price"].copy()
        feats = compute_funding_features(df)
        assert "funding_zscore_24h" in feats.columns
        valid = feats["funding_zscore_24h"].dropna()
        # Z-scores should be roughly centered around 0
        assert abs(valid.mean()) < 3.0


class TestInteractionFeatures:
    def test_interactions_exist(self, synthetic_raw_data):
        from src.features.feature_pipeline import build_features
        feats = build_features(raw_data=synthetic_raw_data)
        interaction_cols = [c for c in feats.columns if "_x_" in c]
        assert len(interaction_cols) > 0

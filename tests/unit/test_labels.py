"""Tests for label correctness."""

import pandas as pd


class TestLabels:
    def test_forward_return_correctness(self, synthetic_price_df):
        from src.labels.labels import build_labels

        labels = build_labels(synthetic_price_df, horizons=[1, 4])

        # Check fwd_ret_1h: should equal close[i+1]/close[i] - 1
        close = synthetic_price_df["close"].values
        for i in range(10, 20):
            expected = close[i + 1] / close[i] - 1.0
            actual = labels["fwd_ret_1h"].iloc[i]
            assert abs(actual - expected) < 1e-10, f"Mismatch at index {i}"

    def test_label_ternary(self, synthetic_price_df):
        from src.labels.labels import build_labels

        labels = build_labels(synthetic_price_df, horizons=[4])
        valid = labels["label_4h"].dropna()
        unique = set(valid.unique())
        assert unique.issubset({-1, 0, 1})

    def test_mfe_mae_valid(self, synthetic_price_df):
        from src.labels.labels import build_labels

        labels = build_labels(synthetic_price_df, horizons=[4])
        mfe = labels["mfe_4h"].dropna()
        mae = labels["mae_4h"].dropna()
        # MFE should be >= 0 (max upside from entry)
        assert (mfe >= -0.001).all(), "MFE should be non-negative or very close to 0"
        # MAE should be <= 0 (max downside from entry) or close to 0
        assert (mae <= 0.001).all(), "MAE should be non-positive or very close to 0"

    def test_no_label_for_last_rows(self, synthetic_price_df):
        from src.labels.labels import build_labels

        labels = build_labels(synthetic_price_df, horizons=[24])
        # Last 24 rows should have NaN forward returns
        assert pd.isna(labels["fwd_ret_24h"].iloc[-1])
        assert pd.isna(labels["fwd_ret_24h"].iloc[-24])

    def test_triple_barrier(self, synthetic_price_df):
        from src.labels.labels import build_triple_barrier_labels

        tb = build_triple_barrier_labels(synthetic_price_df, horizon=24)
        valid = tb["barrier_label"].dropna()
        assert set(valid.unique()).issubset({-1, 0, 1})
        assert "barrier_type" in tb.columns

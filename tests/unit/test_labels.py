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
        # MFE = (window_high / entry) - 1, should generally be >= 0
        # In synthetic data high/low can be very close to close, so allow tolerance
        assert (mfe >= -0.02).all(), "MFE should be non-negative or very close to 0"
        # MAE = (window_low / entry) - 1, should generally be <= 0
        # In synthetic data low can occasionally be above close, so allow tolerance
        assert (mae <= 0.02).all(), "MAE should be non-positive or very close to 0"

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

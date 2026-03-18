"""Pytest fixtures with synthetic datasets for testing."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Force SQLite for tests
os.environ["DATABASE_URL"] = "sqlite:///./test_btc_alpha.db"
os.environ["COINALYZE_API_KEY"] = "test_key"
os.environ["LIVE_TRADING_ENABLED"] = "false"
os.environ["PAPER_MODE"] = "true"


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    """Create test database."""
    from src.storage.database import init_db, get_engine

    init_db()
    yield
    # Cleanup
    engine = get_engine()
    engine.dispose()
    import pathlib

    db_file = pathlib.Path("./test_btc_alpha.db")
    if db_file.exists():
        db_file.unlink()


@pytest.fixture
def synthetic_price_df() -> pd.DataFrame:
    """Generate synthetic hourly BTC price data."""
    np.random.seed(42)
    n = 500
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(hours=i) for i in range(n)]

    # Random walk for price
    returns = np.random.normal(0.0001, 0.005, n)
    close = 40000 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.003, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.003, n)))
    open_ = close * (1 + np.random.normal(0, 0.001, n))
    volume = np.abs(np.random.normal(1e6, 3e5, n))

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def synthetic_funding_df(synthetic_price_df) -> pd.DataFrame:
    """Generate synthetic funding rate data."""
    np.random.seed(43)
    n = len(synthetic_price_df)
    return pd.DataFrame(
        {
            "timestamp": synthetic_price_df["timestamp"],
            "funding_close": np.random.normal(0.0001, 0.0003, n),
        }
    )


@pytest.fixture
def synthetic_oi_df(synthetic_price_df) -> pd.DataFrame:
    """Generate synthetic OI data."""
    np.random.seed(44)
    n = len(synthetic_price_df)
    oi = 1e9 + np.cumsum(np.random.normal(0, 1e7, n))
    return pd.DataFrame(
        {
            "timestamp": synthetic_price_df["timestamp"],
            "oi_close": np.abs(oi),
        }
    )


@pytest.fixture
def synthetic_raw_data(
    synthetic_price_df, synthetic_funding_df, synthetic_oi_df
) -> dict[str, pd.DataFrame]:
    """Build a complete raw data dict for feature pipeline."""
    np.random.seed(45)
    n = len(synthetic_price_df)
    ts = synthetic_price_df["timestamp"]

    liq_df = pd.DataFrame(
        {
            "timestamp": ts,
            "long_liquidations_usd": np.abs(np.random.normal(1e6, 5e5, n)),
            "short_liquidations_usd": np.abs(np.random.normal(1e6, 5e5, n)),
            "total_liquidations_usd": np.abs(np.random.normal(2e6, 1e6, n)),
            "count": np.random.randint(0, 100, n),
        }
    )

    ls_df = pd.DataFrame(
        {
            "timestamp": ts,
            "long_ratio": np.random.uniform(0.4, 0.6, n),
            "short_ratio": np.random.uniform(0.4, 0.6, n),
            "long_short_ratio": np.random.uniform(0.8, 1.2, n),
        }
    )

    tf_df = pd.DataFrame(
        {
            "timestamp": ts,
            "buy_volume": np.abs(np.random.normal(5e5, 2e5, n)),
            "sell_volume": np.abs(np.random.normal(5e5, 2e5, n)),
            "buy_sell_ratio": np.random.uniform(0.8, 1.2, n),
        }
    )

    # Merge funding and oi columns into price for the feature pipeline
    price_with_extras = synthetic_price_df.copy()
    price_with_extras["funding_close"] = synthetic_funding_df["funding_close"].values
    price_with_extras["oi_close"] = synthetic_oi_df["oi_close"].values

    return {
        "price": price_with_extras,
        "funding": pd.DataFrame(
            {
                "timestamp": ts,
                "close": synthetic_funding_df["funding_close"].values,
                "open": synthetic_funding_df["funding_close"].values * 0.99,
                "symbol": "BTC",
                "exchange": "Binance",
            }
        ),
        "oi": pd.DataFrame(
            {
                "timestamp": ts,
                "close": synthetic_oi_df["oi_close"].values,
                "open": synthetic_oi_df["oi_close"].values * 0.99,
                "symbol": "BTC",
                "exchange": "Binance",
            }
        ),
        "liquidations": liq_df,
        "long_short": ls_df,
        "taker_flow": tf_df,
    }


@pytest.fixture
def synthetic_features_and_labels(synthetic_raw_data) -> pd.DataFrame:
    """Build complete feature + label dataset from synthetic data."""
    from src.features.feature_pipeline import build_features
    from src.labels.labels import build_labels

    features = build_features(raw_data=synthetic_raw_data)

    price_df = synthetic_raw_data["price"][["timestamp", "close", "high", "low"]].copy()
    labels = build_labels(price_df, horizons=[1, 4, 8, 12, 24])

    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], utc=True)

    dataset = features.merge(labels, on="timestamp", how="inner")
    return dataset.dropna(how="all").reset_index(drop=True)

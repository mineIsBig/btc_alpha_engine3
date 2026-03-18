"""Feature pipeline: orchestrates all feature computations from raw data.

Supports dynamic feature evolution via EvolutionConfig:
- Features can be disabled by the agent (feature_toggles)
- Custom interaction features can be added at runtime (custom_interactions)
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select

from src.common.logging import get_logger
from src.data.resampler import merge_dataframes_on_timestamp, fill_hourly_gaps
from src.features.price_features import compute_price_features
from src.features.funding_features import compute_funding_features
from src.features.oi_features import compute_oi_features
from src.features.liquidation_features import compute_liquidation_features
from src.features.flow_features import compute_flow_features
from src.features.regime_features import compute_regime_features
from src.features.temporal_interaction_features import compute_temporal_interaction_features
from src.storage.database import get_session
from src.storage.models import (
    PriceBar1h, CGFunding1h, CGOI1h, CGLiquidations1h,
    CGLongShort1h, CGTakerFlow1h,
)

logger = get_logger(__name__)


def _apply_evolution_config(result: pd.DataFrame) -> pd.DataFrame:
    """Apply dynamic feature evolution: disable features and add custom interactions.

    Reads EvolutionConfig and:
    1. Drops columns that have been disabled by the agent
    2. Adds custom interaction features defined by the agent
    """
    try:
        from src.agent.evolution_config import load_evolution_config
        config = load_evolution_config()
    except Exception:
        return result  # no config = no changes

    # 1. Disable toggled-off features
    disabled = [name for name, toggle in config.feature_toggles.items()
                if not toggle.enabled and name in result.columns]
    if disabled:
        result = result.drop(columns=disabled)
        logger.info("features_disabled_by_evolution", count=len(disabled), features=disabled[:10])

    # 2. Add custom interaction features
    ops = {
        "multiply": lambda a, b: a * b,
        "divide": lambda a, b: a / b.replace(0, np.nan),
        "add": lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
    }
    added = []
    for ci in config.custom_interactions:
        if not ci.enabled:
            continue
        if ci.feature_a in result.columns and ci.feature_b in result.columns:
            op_fn = ops.get(ci.operation)
            if op_fn is not None:
                result[ci.name] = op_fn(result[ci.feature_a], result[ci.feature_b])
                added.append(ci.name)
    if added:
        logger.info("custom_interactions_added", count=len(added), features=added)

    return result


def load_raw_data(
    symbol: str = "BTC",
    start: datetime | None = None,
    end: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    """Load all raw data from database into DataFrames."""
    session = get_session()

    def _query_to_df(model, extra_filters=None):
        q = session.query(model).filter(model.symbol == symbol)
        if start:
            q = q.filter(model.timestamp >= start)
        if end:
            q = q.filter(model.timestamp <= end)
        if extra_filters:
            for f in extra_filters:
                q = q.filter(f)
        q = q.order_by(model.timestamp)
        rows = q.all()
        if not rows:
            return pd.DataFrame()
        data = [{c.name: getattr(r, c.name) for c in model.__table__.columns} for r in rows]
        return pd.DataFrame(data)

    dfs = {
        "price": _query_to_df(PriceBar1h),
        "funding": _query_to_df(CGFunding1h),
        "oi": _query_to_df(CGOI1h),
        "liquidations": _query_to_df(CGLiquidations1h),
        "long_short": _query_to_df(CGLongShort1h),
        "taker_flow": _query_to_df(CGTakerFlow1h),
    }

    session.close()
    return dfs


def build_features(
    symbol: str = "BTC",
    start: datetime | None = None,
    end: datetime | None = None,
    raw_data: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build complete feature matrix from raw data.

    Returns a DataFrame indexed by timestamp with all features.
    """
    if raw_data is None:
        raw_data = load_raw_data(symbol, start, end)

    price_df = raw_data.get("price", pd.DataFrame())
    if price_df.empty:
        logger.warning("no_price_data_for_features")
        return pd.DataFrame()

    # Ensure timestamp is datetime
    price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], utc=True)

    # Build base DataFrame from price
    base = price_df[["timestamp", "open", "high", "low", "close"]].copy()
    if "volume" in price_df.columns:
        base["volume"] = price_df["volume"]

    # Merge in other data sources
    for key, rename_map in [
        ("funding", {"close": "funding_close", "open": "funding_open"}),
        ("oi", {"close": "oi_close", "open": "oi_open"}),
    ]:
        df = raw_data.get(key, pd.DataFrame())
        if not df.empty:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            cols_to_keep = ["timestamp"] + [c for c in rename_map.keys() if c in df.columns]
            df = df[cols_to_keep].drop_duplicates(subset=["timestamp"], keep="last")
            df = df.rename(columns=rename_map)
            base = base.merge(df, on="timestamp", how="left")

    # Merge liquidations
    liq_df = raw_data.get("liquidations", pd.DataFrame())
    if not liq_df.empty:
        liq_df = liq_df.copy()
        liq_df["timestamp"] = pd.to_datetime(liq_df["timestamp"], utc=True)
        liq_cols = ["timestamp", "long_liquidations_usd", "short_liquidations_usd", "total_liquidations_usd"]
        liq_cols = [c for c in liq_cols if c in liq_df.columns]
        liq_df = liq_df[liq_cols].drop_duplicates(subset=["timestamp"], keep="last")
        base = base.merge(liq_df, on="timestamp", how="left")

    # Merge long/short
    ls_df = raw_data.get("long_short", pd.DataFrame())
    if not ls_df.empty:
        ls_df = ls_df.copy()
        ls_df["timestamp"] = pd.to_datetime(ls_df["timestamp"], utc=True)
        ls_cols = ["timestamp", "long_ratio", "short_ratio", "long_short_ratio"]
        ls_cols = [c for c in ls_cols if c in ls_df.columns]
        ls_df = ls_df[ls_cols].drop_duplicates(subset=["timestamp"], keep="last")
        base = base.merge(ls_df, on="timestamp", how="left")

    # Merge taker flow
    tf_df = raw_data.get("taker_flow", pd.DataFrame())
    if not tf_df.empty:
        tf_df = tf_df.copy()
        tf_df["timestamp"] = pd.to_datetime(tf_df["timestamp"], utc=True)
        tf_cols = ["timestamp", "buy_volume", "sell_volume", "buy_sell_ratio"]
        tf_cols = [c for c in tf_cols if c in tf_df.columns]
        tf_df = tf_df[tf_cols].drop_duplicates(subset=["timestamp"], keep="last")
        base = base.merge(tf_df, on="timestamp", how="left")

    base = base.sort_values("timestamp").reset_index(drop=True)

    # ── Compute feature groups ───────────────────────────────
    logger.info("computing_features", rows=len(base))

    price_feats = compute_price_features(base)
    funding_feats = compute_funding_features(base)
    oi_feats = compute_oi_features(base)
    liq_feats = compute_liquidation_features(base)
    flow_feats = compute_flow_features(base)
    regime_feats = compute_regime_features(base)

    # ── Static interaction features ──────────────────────────
    interactions = pd.DataFrame(index=base.index)

    # Funding x OI
    if "funding_rate" in funding_feats.columns and "oi_change_1h" in oi_feats.columns:
        interactions["funding_x_oi_change"] = funding_feats["funding_rate"] * oi_feats["oi_change_1h"]

    # Liquidation x Taker flow
    if "liq_imbalance" in liq_feats.columns and "taker_net_flow" in flow_feats.columns:
        interactions["liq_x_taker_flow"] = liq_feats["liq_imbalance"] * flow_feats["taker_net_flow"]

    # Funding x LS ratio
    if "funding_rate" in funding_feats.columns and "ls_ratio" in flow_feats.columns:
        interactions["funding_x_ls_ratio"] = funding_feats["funding_rate"] * flow_feats["ls_ratio"]

    # OI change x Price momentum
    if "oi_change_24h" in oi_feats.columns and "ret_24h" in price_feats.columns:
        interactions["oi_change_x_ret_24h"] = oi_feats["oi_change_24h"] * price_feats["ret_24h"]

    # ── Temporal interaction features (second-order cross-features) ──
    temporal_feats = compute_temporal_interaction_features(base)

    # ── Combine all features ─────────────────────────────────
    result = pd.concat([
        base[["timestamp"]],
        price_feats,
        funding_feats,
        oi_feats,
        liq_feats,
        flow_feats,
        regime_feats,
        interactions,
        temporal_feats,
    ], axis=1)

    # Drop regime_label from features (it's a label, not a feature for the model)
    regime_label = result.pop("regime_label") if "regime_label" in result.columns else None

    # Replace inf with nan
    result = result.replace([np.inf, -np.inf], np.nan)

    logger.info("features_built", n_features=len(result.columns) - 1, n_rows=len(result))

    # Apply dynamic evolution config (disable features, add custom interactions)
    result = _apply_evolution_config(result)

    # Re-attach regime_label for use by labels module
    if regime_label is not None:
        result["regime_label"] = regime_label

    return result


def get_feature_names(exclude: list[str] | None = None) -> list[str]:
    """Get list of all feature column names (excluding timestamp and regime_label)."""
    # Build from a small dummy to discover column names
    dummy = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC"),
        "open": np.random.randn(200).cumsum() + 50000,
        "high": np.random.randn(200).cumsum() + 50100,
        "low": np.random.randn(200).cumsum() + 49900,
        "close": np.random.randn(200).cumsum() + 50000,
        "volume": np.abs(np.random.randn(200)) * 1000,
        "funding_close": np.random.randn(200) * 0.001,
        "oi_close": np.random.randn(200).cumsum() + 1e9,
        "long_liquidations_usd": np.abs(np.random.randn(200)) * 1e6,
        "short_liquidations_usd": np.abs(np.random.randn(200)) * 1e6,
        "total_liquidations_usd": np.abs(np.random.randn(200)) * 2e6,
        "long_short_ratio": np.random.randn(200) * 0.1 + 1.0,
        "buy_volume": np.abs(np.random.randn(200)) * 1e6,
        "sell_volume": np.abs(np.random.randn(200)) * 1e6,
        "buy_sell_ratio": np.random.randn(200) * 0.1 + 1.0,
    })

    raw_data = {"price": dummy, "funding": pd.DataFrame(), "oi": pd.DataFrame(),
                "liquidations": pd.DataFrame(), "long_short": pd.DataFrame(), "taker_flow": pd.DataFrame()}
    feats = build_features(raw_data=raw_data)

    skip = {"timestamp", "regime_label"} | set(exclude or [])
    return [c for c in feats.columns if c not in skip]

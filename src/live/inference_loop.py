"""Inference loop: runs model predictions on a schedule."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.common.logging import get_logger
from src.common.time_utils import utc_now
from src.features.feature_pipeline import build_features
from src.models.registry import ModelArtifactRegistry
from src.models.regime import RegimeGateModel
from src.portfolio.signal_schema import ModelSignal
from src.storage.database import session_scope
from src.storage.models import SignalRecord

logger = get_logger(__name__)

HORIZONS = [1, 4, 8, 12, 24]


class InferenceLoop:
    """Run model inference to generate signals."""

    def __init__(self):
        self.registry = ModelArtifactRegistry()

    def run_inference(
        self,
        features_df: pd.DataFrame | None = None,
        timestamp: datetime | None = None,
    ) -> dict[int, list[ModelSignal]]:
        """Run all promoted models and return signals grouped by horizon.

        Returns: {horizon: [ModelSignal, ...]}
        """
        ts = timestamp or utc_now()

        if features_df is None:
            # Load latest features from DB
            from src.features.feature_pipeline import load_raw_data
            from datetime import timedelta
            raw = load_raw_data(start=ts - timedelta(days=10), end=ts)
            features_df = build_features(raw_data=raw)

        if features_df.empty:
            logger.warning("no_features_for_inference")
            return {}

        # Get latest row
        features_df = features_df.sort_values("timestamp")
        latest_row = features_df.iloc[[-1]]

        # Detect regime
        regime = "neutral"
        if "regime_label" in features_df.columns:
            regime = str(features_df["regime_label"].iloc[-1])

        signals_by_horizon: dict[int, list[ModelSignal]] = {}

        for horizon in HORIZONS:
            models = self.registry.get_promoted_models(horizon=horizon)
            if not models:
                continue

            horizon_signals = []
            for model_info in models:
                try:
                    model = self.registry.load_model(model_info["model_id"])
                    feat_cols = model.feature_names
                    available = [c for c in feat_cols if c in latest_row.columns]

                    if len(available) < len(feat_cols) * 0.5:
                        logger.warning("insufficient_features", model_id=model_info["model_id"])
                        continue

                    X = latest_row[available].fillna(0)
                    sig = model.get_signal(X)

                    # Apply regime gating
                    regime_mult = RegimeGateModel.get_signal_multiplier(regime, sig["side"])
                    adjusted_conf = sig["confidence"] * regime_mult

                    signal = ModelSignal(
                        model_id=model_info["model_id"],
                        horizon=horizon,
                        side=sig["side"],
                        probability=sig["probability"],
                        confidence=adjusted_conf,
                        regime=regime,
                        oos_sharpe=model_info.get("oos_sharpe", 0.0),
                    )
                    horizon_signals.append(signal)

                    # Persist signal
                    with session_scope() as session:
                        session.add(SignalRecord(
                            timestamp=ts,
                            symbol="BTC",
                            horizon=horizon,
                            model_id=model_info["model_id"],
                            side=sig["side"],
                            probability=sig["probability"],
                            confidence=adjusted_conf,
                            regime=regime,
                        ))

                except Exception as e:
                    logger.error("inference_error", model_id=model_info["model_id"], error=str(e))

            if horizon_signals:
                signals_by_horizon[horizon] = horizon_signals

        total = sum(len(v) for v in signals_by_horizon.values())
        logger.info("inference_complete", n_signals=total, n_horizons=len(signals_by_horizon))
        return signals_by_horizon

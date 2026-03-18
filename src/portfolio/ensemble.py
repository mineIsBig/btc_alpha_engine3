"""Ensemble signal aggregation across models and horizons.

Supports dynamic parameter evolution via EvolutionConfig. The agent can
adjust consensus thresholds, weighting power, and confidence gates at runtime.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from src.common.logging import get_logger
from src.common.time_utils import utc_now
from src.portfolio.signal_schema import ModelSignal, AggregatedSignal

logger = get_logger(__name__)


def _load_ensemble_params() -> dict[str, float | int]:
    """Load ensemble parameters from EvolutionConfig, with defaults."""
    defaults = {
        "min_consensus_pct": 0.5,
        "min_avg_confidence": 0.1,
        "sharpe_weight_power": 1.5,
    }
    try:
        from src.agent.evolution_config import load_evolution_config
        config = load_evolution_config()
        return {
            "min_consensus_pct": config.ensemble.min_consensus_pct,
            "min_avg_confidence": config.ensemble.min_avg_confidence,
            "sharpe_weight_power": config.ensemble.sharpe_weight_power,
        }
    except Exception:
        return defaults


class EnsembleAggregator:
    """Aggregate signals from multiple models using Sharpe-weighted consensus.

    Parameters can be overridden at construction time, or loaded dynamically
    from EvolutionConfig when use_evolution_config=True.
    """

    def __init__(
        self,
        min_consensus_pct: float | None = None,
        min_avg_confidence: float | None = None,
        sharpe_weight_power: float | None = None,
        use_evolution_config: bool = True,
    ):
        if use_evolution_config and all(v is None for v in [min_consensus_pct, min_avg_confidence, sharpe_weight_power]):
            params = _load_ensemble_params()
            self.min_consensus_pct = params["min_consensus_pct"]
            self.min_avg_confidence = params["min_avg_confidence"]
            self.sharpe_weight_power = params["sharpe_weight_power"]
        else:
            self.min_consensus_pct = min_consensus_pct if min_consensus_pct is not None else 0.5
            self.min_avg_confidence = min_avg_confidence if min_avg_confidence is not None else 0.1
            self.sharpe_weight_power = sharpe_weight_power if sharpe_weight_power is not None else 1.5

    def aggregate(
        self,
        signals: list[ModelSignal],
        timestamp: datetime | None = None,
    ) -> AggregatedSignal:
        """Aggregate multiple model signals into a single decision.

        Weighting: OOS Sharpe ^ power * calibrated probability
        Consensus gate: requires min_consensus_pct of models to agree on direction
        """
        if timestamp is None:
            timestamp = utc_now()

        if not signals:
            return AggregatedSignal(
                timestamp=timestamp,
                target_side=0,
                raw_score=0.0,
                consensus_pct=0.0,
                avg_probability=0.0,
                avg_confidence=0.0,
                reason="no_signals",
            )

        # Compute weights from OOS Sharpe
        sharpes = np.array([max(s.oos_sharpe, 0.01) for s in signals])
        weights = sharpes ** self.sharpe_weight_power
        weights = weights / weights.sum()

        # Weighted vote
        weighted_scores = []
        for s, w in zip(signals, weights):
            score = s.side * s.probability * w
            if s.calibrated:
                score *= 1.1  # slight bonus for calibrated models
            weighted_scores.append(score)

        raw_score = sum(weighted_scores)

        # Consensus check
        directional_signals = [s for s in signals if s.side != 0]
        if directional_signals:
            long_count = sum(1 for s in directional_signals if s.side == 1)
            short_count = sum(1 for s in directional_signals if s.side == -1)
            total_dir = len(directional_signals)

            if long_count > short_count:
                majority_side = 1
                consensus_pct = long_count / total_dir
            elif short_count > long_count:
                majority_side = -1
                consensus_pct = short_count / total_dir
            else:
                majority_side = 0
                consensus_pct = 0.0
        else:
            majority_side = 0
            consensus_pct = 0.0

        avg_prob = np.mean([s.probability for s in signals])
        avg_conf = np.mean([s.confidence for s in signals])

        # Gating
        if consensus_pct < self.min_consensus_pct:
            target_side = 0
            reason = f"consensus_too_low ({consensus_pct:.2f} < {self.min_consensus_pct})"
        elif avg_conf < self.min_avg_confidence:
            target_side = 0
            reason = f"confidence_too_low ({avg_conf:.2f} < {self.min_avg_confidence})"
        else:
            target_side = majority_side
            reason = f"consensus={consensus_pct:.2f}, score={raw_score:.4f}"

        # Determine regime from majority
        regime_counts: dict[str, int] = {}
        for s in signals:
            regime_counts[s.regime] = regime_counts.get(s.regime, 0) + 1
        regime = max(regime_counts, key=regime_counts.get) if regime_counts else "neutral"

        return AggregatedSignal(
            timestamp=timestamp,
            target_side=target_side,
            raw_score=float(raw_score),
            consensus_pct=float(consensus_pct),
            avg_probability=float(avg_prob),
            avg_confidence=float(avg_conf),
            regime=regime,
            component_signals=signals,
            reason=reason,
        )

"""Tests for ensemble and consensus logic."""
import pytest
from datetime import datetime, timezone
from src.portfolio.signal_schema import ModelSignal


class TestEnsemble:
    def test_flat_on_no_signals(self):
        from src.portfolio.ensemble import EnsembleAggregator
        agg = EnsembleAggregator()
        result = agg.aggregate([])
        assert result.target_side == 0

    def test_long_consensus(self):
        from src.portfolio.ensemble import EnsembleAggregator
        agg = EnsembleAggregator(min_consensus_pct=0.5)
        signals = [
            ModelSignal(model_id="m1", horizon=4, side=1, probability=0.7, confidence=0.4, oos_sharpe=1.0),
            ModelSignal(model_id="m2", horizon=4, side=1, probability=0.65, confidence=0.3, oos_sharpe=0.8),
            ModelSignal(model_id="m3", horizon=4, side=-1, probability=0.55, confidence=0.1, oos_sharpe=0.3),
        ]
        result = agg.aggregate(signals)
        assert result.target_side == 1
        assert result.consensus_pct > 0.5

    def test_flat_on_low_consensus(self):
        from src.portfolio.ensemble import EnsembleAggregator
        agg = EnsembleAggregator(min_consensus_pct=0.8)
        signals = [
            ModelSignal(model_id="m1", horizon=4, side=1, probability=0.6, confidence=0.2, oos_sharpe=1.0),
            ModelSignal(model_id="m2", horizon=4, side=-1, probability=0.6, confidence=0.2, oos_sharpe=1.0),
        ]
        result = agg.aggregate(signals)
        assert result.target_side == 0


class TestConsensusGate:
    def test_agreement_across_horizons(self):
        from src.portfolio.consensus import ConsensusGate
        from src.portfolio.signal_schema import AggregatedSignal
        gate = ConsensusGate(min_horizon_agreement=2)

        horizon_signals = {
            1: AggregatedSignal(timestamp=datetime.now(timezone.utc), target_side=1,
                               raw_score=0.5, consensus_pct=0.8, avg_probability=0.7, avg_confidence=0.5),
            4: AggregatedSignal(timestamp=datetime.now(timezone.utc), target_side=1,
                               raw_score=0.4, consensus_pct=0.7, avg_probability=0.6, avg_confidence=0.4),
            8: AggregatedSignal(timestamp=datetime.now(timezone.utc), target_side=-1,
                               raw_score=-0.3, consensus_pct=0.6, avg_probability=0.55, avg_confidence=0.2),
        }
        side, reason = gate.check(horizon_signals)
        assert side == 1

    def test_conflicting_signals(self):
        from src.portfolio.consensus import ConsensusGate
        from src.portfolio.signal_schema import AggregatedSignal
        gate = ConsensusGate(min_horizon_agreement=2)

        horizon_signals = {
            1: AggregatedSignal(timestamp=datetime.now(timezone.utc), target_side=1,
                               raw_score=0.5, consensus_pct=0.8, avg_probability=0.7, avg_confidence=0.5),
            4: AggregatedSignal(timestamp=datetime.now(timezone.utc), target_side=1,
                               raw_score=0.4, consensus_pct=0.7, avg_probability=0.6, avg_confidence=0.4),
            8: AggregatedSignal(timestamp=datetime.now(timezone.utc), target_side=-1,
                               raw_score=-0.5, consensus_pct=0.8, avg_probability=0.7, avg_confidence=0.5),
            12: AggregatedSignal(timestamp=datetime.now(timezone.utc), target_side=-1,
                                raw_score=-0.4, consensus_pct=0.7, avg_probability=0.6, avg_confidence=0.4),
        }
        side, reason = gate.check(horizon_signals)
        assert side == 0  # conflicting

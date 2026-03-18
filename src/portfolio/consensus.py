"""Cross-horizon consensus gating."""
from __future__ import annotations

from src.common.logging import get_logger
from src.portfolio.signal_schema import AggregatedSignal

logger = get_logger(__name__)


class ConsensusGate:
    """Gate that checks agreement across multiple time horizons."""

    def __init__(self, min_horizon_agreement: int = 2, horizons: list[int] | None = None):
        self.min_horizon_agreement = min_horizon_agreement
        self.horizons = horizons or [1, 4, 8, 12, 24]

    def check(self, horizon_signals: dict[int, AggregatedSignal]) -> tuple[int, str]:
        """Check cross-horizon consensus.

        Args:
            horizon_signals: {horizon: AggregatedSignal} for each horizon

        Returns:
            (side, reason) - the consensus side and explanation
        """
        sides = {}
        for h, sig in horizon_signals.items():
            if sig.target_side != 0:
                sides[h] = sig.target_side

        if not sides:
            return 0, "no_directional_signals"

        long_horizons = [h for h, s in sides.items() if s == 1]
        short_horizons = [h for h, s in sides.items() if s == -1]

        if len(long_horizons) >= self.min_horizon_agreement and len(short_horizons) < self.min_horizon_agreement:
            return 1, f"long_consensus_horizons={long_horizons}"
        elif len(short_horizons) >= self.min_horizon_agreement and len(long_horizons) < self.min_horizon_agreement:
            return -1, f"short_consensus_horizons={short_horizons}"
        elif len(long_horizons) >= self.min_horizon_agreement and len(short_horizons) >= self.min_horizon_agreement:
            return 0, f"conflicting_consensus long={long_horizons} short={short_horizons}"
        else:
            return 0, f"insufficient_agreement long={len(long_horizons)} short={len(short_horizons)}"

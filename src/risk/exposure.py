"""Exposure tracking and limits."""
from __future__ import annotations

from src.common.logging import get_logger

logger = get_logger(__name__)


class ExposureTracker:
    """Track and limit portfolio exposure."""

    def __init__(self, max_gross_pct: float = 1.0, max_net_pct: float = 0.5):
        self.max_gross_pct = max_gross_pct
        self.max_net_pct = max_net_pct

    def check(self, gross_exposure: float, net_exposure: float, equity: float) -> tuple[bool, str]:
        """Check if exposure is within limits."""
        if equity <= 0:
            return False, "zero_equity"

        gross_pct = gross_exposure / equity
        net_pct = abs(net_exposure) / equity

        if gross_pct > self.max_gross_pct:
            return False, f"gross_exposure={gross_pct:.2f} > {self.max_gross_pct}"
        if net_pct > self.max_net_pct:
            return False, f"net_exposure={net_pct:.2f} > {self.max_net_pct}"

        return True, "ok"

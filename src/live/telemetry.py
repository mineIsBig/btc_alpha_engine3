"""Telemetry and metrics collection."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from src.common.logging import get_logger
from src.common.time_utils import utc_now

logger = get_logger(__name__)


class Telemetry:
    """Collect and log system telemetry."""

    def __init__(self):
        self._metrics: dict[str, list[tuple[datetime, Any]]] = {}

    def record(self, metric: str, value: Any) -> None:
        if metric not in self._metrics:
            self._metrics[metric] = []
        self._metrics[metric].append((utc_now(), value))
        # Keep only last 1000 entries per metric
        if len(self._metrics[metric]) > 1000:
            self._metrics[metric] = self._metrics[metric][-500:]

    def get_latest(self, metric: str) -> Any | None:
        entries = self._metrics.get(metric, [])
        return entries[-1][1] if entries else None

    def log_summary(self) -> None:
        summary = {k: len(v) for k, v in self._metrics.items()}
        logger.info("telemetry_summary", metrics=summary)

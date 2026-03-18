"""Health check monitoring for live trading."""
from __future__ import annotations

from datetime import datetime, timedelta

from src.common.logging import get_logger
from src.common.time_utils import utc_now
from src.storage.database import get_session
from src.storage.models import AccountSnapshot, SignalRecord

logger = get_logger(__name__)


class HealthChecker:
    """Monitor system health during live trading."""

    def __init__(
        self,
        stale_data_threshold_seconds: int = 600,
        max_missed_decisions: int = 3,
    ):
        self.stale_threshold = timedelta(seconds=stale_data_threshold_seconds)
        self.max_missed = max_missed_decisions
        self._last_decision_time: datetime | None = None
        self._missed_count = 0

    def record_decision(self) -> None:
        self._last_decision_time = utc_now()
        self._missed_count = 0

    def record_miss(self) -> None:
        self._missed_count += 1

    def check_all(self) -> dict[str, bool]:
        """Run all health checks. Returns {check_name: passed}."""
        results = {}

        # Data freshness
        session = get_session()
        latest_snap = session.query(AccountSnapshot).order_by(
            AccountSnapshot.timestamp.desc()
        ).first()
        session.close()

        if latest_snap:
            age = utc_now() - latest_snap.timestamp.replace(tzinfo=utc_now().tzinfo)
            results["data_fresh"] = age < self.stale_threshold
        else:
            results["data_fresh"] = False

        # Decision recency
        if self._last_decision_time:
            since_last = utc_now() - self._last_decision_time
            results["decision_recent"] = since_last < timedelta(hours=2)
        else:
            results["decision_recent"] = False

        # Missed decisions
        results["missed_ok"] = self._missed_count < self.max_missed

        all_ok = all(results.values())
        if not all_ok:
            logger.warning("health_check_failed", results=results)

        return results

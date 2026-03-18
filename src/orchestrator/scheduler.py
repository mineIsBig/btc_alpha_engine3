"""Scheduler for orchestrating recurring jobs."""

from __future__ import annotations

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.common.logging import get_logger

logger = get_logger(__name__)


class Scheduler:
    """APScheduler-based job scheduler."""

    def __init__(self):
        self._scheduler = BlockingScheduler()

    def add_hourly_job(self, func, name: str = "hourly_job") -> None:
        self._scheduler.add_job(
            func, CronTrigger(minute=5), id=name, replace_existing=True
        )

    def add_interval_job(self, func, seconds: int, name: str = "interval_job") -> None:
        self._scheduler.add_job(
            func, IntervalTrigger(seconds=seconds), id=name, replace_existing=True
        )

    def add_daily_reset_job(self, func) -> None:
        self._scheduler.add_job(
            func, CronTrigger(hour=0, minute=0), id="daily_reset", replace_existing=True
        )

    def start(self) -> None:
        logger.info("scheduler_starting")
        try:
            self._scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("scheduler_shutdown")

    def shutdown(self) -> None:
        self._scheduler.shutdown(wait=False)

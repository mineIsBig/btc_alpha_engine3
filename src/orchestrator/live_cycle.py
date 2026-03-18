"""Live cycle: hourly trading loop with risk monitoring."""
from __future__ import annotations

from src.common.logging import get_logger, setup_logging
from src.common.config import get_settings
from src.data.ingest_jobs import incremental_refresh
from src.live.trade_loop import TradeLoop
from src.live.health_checks import HealthChecker
from src.live.telemetry import Telemetry
from src.orchestrator.scheduler import Scheduler

logger = get_logger(__name__)


def run_live_cycle(initial_equity: float = 100000.0) -> None:
    """Start the live/paper trading loop."""
    settings = get_settings()
    setup_logging(level=settings.log_level, fmt=settings.log_format)

    trade_loop = TradeLoop(initial_equity=initial_equity)
    health = HealthChecker()
    telemetry = Telemetry()
    scheduler = Scheduler()

    # Restore state
    trade_loop.risk.restore_state()

    def hourly_tick() -> None:
        """Run hourly: refresh data + inference + trade."""
        try:
            # Refresh data
            incremental_refresh()

            # Run trade decision
            result = trade_loop.tick()
            telemetry.record("trade_result", result)

            if result.get("action") == "trade":
                health.record_decision()
            else:
                health.record_miss()

            logger.info("hourly_tick", result=result)

        except Exception as e:
            logger.error("hourly_tick_error", error=str(e))
            health.record_miss()

    def equity_check() -> None:
        """Frequent equity/risk check."""
        try:
            price = trade_loop.last_price
            if price > 0:
                equity = trade_loop.router.paper_broker.get_equity({"BTC": price}, trade_loop.cash)
                trade_loop.risk.update_equity(equity, timestamp=None)

                if not trade_loop.risk.can_trade():
                    if settings.flatten_on_breach:
                        trade_loop.emergency.cancel_and_flatten(
                            {"BTC": price},
                            reason=trade_loop.risk.account.breach_reason or "equity_check_breach",
                        )
                    trade_loop.risk.persist_state()

                telemetry.record("equity", equity)

        except Exception as e:
            logger.error("equity_check_error", error=str(e))

    def daily_reset() -> None:
        """00:00 UTC reset."""
        logger.info("daily_reset_trigger")
        trade_loop.risk.persist_state()

    def health_check() -> None:
        results = health.check_all()
        telemetry.record("health", results)

    # Schedule jobs
    scheduler.add_hourly_job(hourly_tick, "hourly_tick")
    scheduler.add_interval_job(equity_check, seconds=60, name="equity_check")
    scheduler.add_daily_reset_job(daily_reset)
    scheduler.add_interval_job(health_check, seconds=300, name="health_check")

    logger.info("live_cycle_starting", paper_mode=settings.paper_mode, live=settings.live_trading_enabled)
    scheduler.start()

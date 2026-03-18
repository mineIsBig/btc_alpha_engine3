"""Perpetual agent loop (Arbos-inspired ralph-loop).

Runs the AlphaAgent.iterate() on a configurable delay, forever.
Each iteration: design → run → measure → reflect → improve → signal output.
"""
from __future__ import annotations

import time
import signal as sig_module
from datetime import datetime

from src.common.config import get_settings, load_yaml_config
from src.common.logging import get_logger, setup_logging
from src.agent.alpha_agent import AlphaAgent
from src.agent.signal_output import SignalOutput
from src.monitoring.prometheus_metrics import MetricsCollector

logger = get_logger(__name__)


class AgentLoop:
    """Perpetual ralph-loop for the autonomous alpha agent.

    Runs indefinitely with configurable delay between iterations.
    """

    def __init__(
        self,
        delay_seconds: int = 3600,  # 1 hour default
        equity: float = 100000.0,
    ):
        self.delay_seconds = delay_seconds
        self.equity = equity
        self.agent = AlphaAgent()
        self.metrics = MetricsCollector()
        self._running = True
        self._signal_history: list[SignalOutput] = []

        # Handle graceful shutdown
        sig_module.signal(sig_module.SIGINT, self._shutdown)
        sig_module.signal(sig_module.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame) -> None:
        logger.info("agent_loop_shutdown_requested")
        self._running = False

    def run(self) -> None:
        """Run the perpetual loop. Blocks until shutdown."""
        logger.info("agent_loop_starting", delay=self.delay_seconds, equity=self.equity)

        while self._running:
            iteration_start = time.monotonic()

            try:
                # Get current price
                current_price = self._get_price()

                # Run one iteration
                signal = self.agent.iterate(
                    current_price=current_price,
                    equity=self.equity,
                )

                # Record signal
                self._signal_history.append(signal)
                if len(self._signal_history) > 1000:
                    self._signal_history = self._signal_history[-500:]

                # Update Prometheus metrics
                self.metrics.record_signal(signal)
                self.metrics.record_iteration(
                    iteration=self.agent.state.iteration,
                    duration=time.monotonic() - iteration_start,
                    sharpe=self.agent.state.rolling_sharpe,
                    drawdown=self.agent.state.max_drawdown,
                )

                # Log signal
                logger.info("signal_generated", summary=signal.to_summary())
                print("\n" + "=" * 60)
                print(signal.to_summary())
                print("=" * 60 + "\n")

            except Exception as e:
                logger.error("agent_loop_iteration_error", error=str(e))
                self.metrics.record_error()

            # Sleep until next iteration
            elapsed = time.monotonic() - iteration_start
            sleep_time = max(0, self.delay_seconds - elapsed)
            logger.info("agent_loop_sleeping", seconds=sleep_time)

            # Interruptible sleep
            sleep_start = time.monotonic()
            while self._running and (time.monotonic() - sleep_start) < sleep_time:
                time.sleep(min(5.0, sleep_time))

        logger.info("agent_loop_stopped", total_iterations=self.agent.state.iteration)

    def _get_price(self) -> float:
        """Get current BTC price."""
        try:
            from src.data.hyperliquid_client import HyperliquidClient
            client = HyperliquidClient()
            price = client.get_mid_price("BTC")
            client.close()
            return price or 0.0
        except Exception:
            return 0.0

    @property
    def latest_signal(self) -> SignalOutput | None:
        return self._signal_history[-1] if self._signal_history else None


def run_agent_loop(delay: int = 3600, equity: float = 100000.0) -> None:
    """Entry point for running the agent loop."""
    settings = get_settings()
    setup_logging(level=settings.log_level, fmt=settings.log_format)

    loop = AgentLoop(delay_seconds=delay, equity=equity)
    loop.run()

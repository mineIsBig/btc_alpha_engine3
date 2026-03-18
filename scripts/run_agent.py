#!/usr/bin/env python3
"""Run the autonomous alpha agent (Arbos-inspired ralph-loop).

The agent runs perpetually:
  design → run → measure → reflect → improve → signal output
  
Output: long/short with position size, expected returns, TP/SL levels.
No live trades are executed.

Prometheus metrics exposed on port 9090 for Grafana dashboards.
"""
import sys
sys.path.insert(0, ".")

import click
from dotenv import load_dotenv
load_dotenv()

from src.common.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


@click.command()
@click.option("--delay", default=3600, help="Seconds between iterations (default: 3600 = 1h)")
@click.option("--equity", default=100000.0, help="Starting equity USD for sizing")
@click.option("--metrics-port", default=9090, help="Prometheus metrics port")
@click.option("--once", is_flag=True, help="Run one iteration and exit")
def main(delay: int, equity: float, metrics_port: int, once: bool) -> None:
    """Start the autonomous alpha agent."""

    logger.info("agent_starting", delay=delay, equity=equity, metrics_port=metrics_port)

    # Start Prometheus metrics server
    from src.monitoring.prometheus_metrics import start_metrics_server
    metrics = start_metrics_server(port=metrics_port)
    logger.info("prometheus_metrics_server_started", port=metrics_port)

    if once:
        # Single iteration mode (useful for testing)
        from src.agent.alpha_agent import AlphaAgent
        agent = AlphaAgent()
        signal = agent.iterate(current_price=0.0, equity=equity)
        metrics.record_signal(signal)
        print("\n" + "=" * 60)
        print(signal.to_summary())
        print("=" * 60)
        print("\nFull signal JSON:")
        print(signal.model_dump_json(indent=2))
        return

    # Run perpetual loop
    from src.agent.loop import run_agent_loop
    run_agent_loop(delay=delay, equity=equity)


if __name__ == "__main__":
    main()

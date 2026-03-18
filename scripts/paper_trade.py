#!/usr/bin/env python3
"""Start paper trading loop."""

import sys

sys.path.insert(0, ".")

import click
from dotenv import load_dotenv

load_dotenv()

from src.common.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


@click.command()
@click.option("--equity", default=100000.0, help="Starting equity USD")
def main(equity: float) -> None:
    logger.info("paper_trading_starting", equity=equity)

    # Force paper mode
    import os

    os.environ["PAPER_MODE"] = "true"
    os.environ["LIVE_TRADING_ENABLED"] = "false"

    from src.orchestrator.live_cycle import run_live_cycle

    run_live_cycle(initial_equity=equity)


if __name__ == "__main__":
    main()

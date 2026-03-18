#!/usr/bin/env python3
"""Shadow/live trading mode (live trading must be explicitly enabled)."""

import sys

sys.path.insert(0, ".")

import click
from dotenv import load_dotenv

load_dotenv()

from src.common.logging import setup_logging, get_logger
from src.common.config import get_settings

setup_logging()
logger = get_logger(__name__)


@click.command()
@click.option("--equity", default=100000.0, help="Starting equity USD")
@click.option("--confirm", is_flag=True, help="Confirm live trading intent")
def main(equity: float, confirm: bool) -> None:
    settings = get_settings()

    if settings.live_trading_enabled and not settings.paper_mode:
        if not confirm:
            logger.error(
                "live_trading_requires_confirmation",
                msg="Pass --confirm to start live trading",
            )
            sys.exit(1)
        logger.warning(
            "LIVE_TRADING_MODE", msg="Live orders will be submitted to exchange"
        )
    else:
        logger.info("shadow_mode", msg="Running in shadow mode (paper broker)")

    from src.orchestrator.live_cycle import run_live_cycle

    run_live_cycle(initial_equity=equity)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Backfill historical data from Coinalyze (formerly CoinGlass)."""

import sys

sys.path.insert(0, ".")

import click
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

from src.common.logging import setup_logging, get_logger
from src.data.ingest_jobs import backfill_all

setup_logging()
logger = get_logger(__name__)


@click.command()
@click.option("--symbol", default="BTC", help="Symbol to backfill")
@click.option("--start", default="2023-01-01", help="Start date YYYY-MM-DD")
@click.option("--end", default=None, help="End date YYYY-MM-DD (default: now)")
def main(symbol: str, start: str, end: str | None) -> None:
    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = (
        datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc) if end else None
    )

    logger.info("backfill_starting", symbol=symbol, start=start, end=end)
    results = backfill_all(symbol=symbol, start=start_dt, end=end_dt)

    for dataset, count in results.items():
        status = "OK" if count >= 0 else "FAILED"
        logger.info("backfill_result", dataset=dataset, rows=count, status=status)

    logger.info("backfill_complete")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Promote candidate models that pass selection criteria."""
import sys
sys.path.insert(0, ".")

import click
from dotenv import load_dotenv
load_dotenv()

from src.common.logging import setup_logging, get_logger
from src.models.registry import ModelArtifactRegistry
from src.research.reports import generate_summary_report
from src.storage.database import session_scope
from src.storage.models import ModelRegistry

setup_logging()
logger = get_logger(__name__)


@click.command()
@click.option("--min-sharpe", default=0.5, help="Minimum OOS Sharpe for promotion")
@click.option("--model-id", default=None, help="Promote a specific model by ID")
@click.option("--dry-run", is_flag=True, help="Show what would be promoted")
def main(min_sharpe: float, model_id: str | None, dry_run: bool) -> None:
    registry = ModelArtifactRegistry()

    if model_id:
        if dry_run:
            report = generate_summary_report(model_id)
            logger.info("would_promote", model_id=model_id, report=report)
        else:
            registry.promote_model(model_id)
            logger.info("promoted", model_id=model_id)
        return

    # Auto-promote candidates meeting threshold
    with session_scope() as session:
        candidates = session.query(ModelRegistry).filter_by(status="candidate").all()
        for c in candidates:
            sharpe = c.oos_sharpe or 0.0
            breach = c.breach_rate or 0.0
            if sharpe >= min_sharpe and breach == 0.0:
                if dry_run:
                    logger.info("would_promote", model_id=c.model_id, sharpe=sharpe)
                else:
                    c.status = "promoted"
                    from datetime import datetime
                    c.promoted_at = datetime.utcnow()
                    logger.info("promoted", model_id=c.model_id, sharpe=sharpe)

    logger.info("promotion_complete")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run walk-forward training and model selection."""
import sys
sys.path.insert(0, ".")

import click
from dotenv import load_dotenv
load_dotenv()

from src.common.logging import setup_logging, get_logger
from src.orchestrator.research_cycle import run_research_cycle
from src.research.reports import generate_summary_report

setup_logging()
logger = get_logger(__name__)


@click.command()
@click.option("--horizons", default="1,4,8,12,24", help="Comma-separated horizons")
@click.option("--evo-search/--no-evo-search", default=False, help="Run evolutionary search")
def main(horizons: str, evo_search: bool) -> None:
    horizon_list = [int(h) for h in horizons.split(",")]

    logger.info("training_starting", horizons=horizon_list, evo=evo_search)

    candidates = run_research_cycle(horizons=horizon_list)

    for c in candidates:
        model_id = c["model_id"]
        report = generate_summary_report(model_id)
        logger.info("candidate", model_id=model_id,
                    avg_sharpe=report.get("avg_sharpe"),
                    avg_accuracy=report.get("avg_accuracy"))

    if evo_search:
        from src.research.datasets import prepare_dataset, get_feature_columns
        from src.research.evolutionary_search import EvolutionarySearch
        from src.research.purged_walk_forward import PurgedWalkForward

        dataset = prepare_dataset(horizons=horizon_list)
        if not dataset.empty:
            feature_cols = get_feature_columns(dataset)
            splitter = PurgedWalkForward.from_config()

            for h in horizon_list:
                logger.info("evo_search_start", horizon=h)
                evo = EvolutionarySearch(all_features=feature_cols)
                results = evo.run(dataset, horizon=h, splitter=splitter)
                if results:
                    best = results[0]
                    logger.info("evo_best", horizon=h, fitness=best.fitness,
                               model_type=best.model_type, n_features=len(best.feature_subset))

    logger.info("training_complete", n_candidates=len(candidates))


if __name__ == "__main__":
    main()

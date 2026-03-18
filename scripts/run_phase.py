#!/usr/bin/env python3
"""Run individual agent phases independently.

Instead of running the full monolithic iterate(), this script lets you
run any single phase (or subset) for debugging and development.

Examples:
    # Run just the measure phase with cached context
    python scripts/run_phase.py measure --price 65000

    # Run score + measure (no LLM calls) for rapid iteration
    python scripts/run_phase.py score measure --price 65000

    # Run the full pipeline (equivalent to run_agent.py --once)
    python scripts/run_phase.py all --price 65000

    # Re-run reflect with a saved context from a previous run
    python scripts/run_phase.py reflect --context artifacts/phase_context.json

    # Run design + reflect only (just the LLM phases) on cached data
    python scripts/run_phase.py design reflect --context artifacts/phase_context.json
"""

import sys

sys.path.insert(0, ".")

import click
from dotenv import load_dotenv

load_dotenv()

from src.common.logging import setup_logging, get_logger  # noqa: E402

setup_logging()
logger = get_logger(__name__)

PHASES = ["score", "design", "run", "measure", "reflect", "improve", "signal"]


@click.command()
@click.argument("phases", nargs=-1, required=True)
@click.option(
    "--price", default=0.0, help="Current BTC price for scoring/signal generation"
)
@click.option("--equity", default=100000.0, help="Equity USD for position sizing")
@click.option(
    "--context",
    default=None,
    help="Path to saved PhaseContext JSON (for replaying phases)",
)
@click.option("--save/--no-save", default=True, help="Save context after each phase")
def main(
    phases: tuple[str, ...],
    price: float,
    equity: float,
    context: str | None,
    save: bool,
) -> None:
    """Run one or more agent phases independently.

    PHASES can be: score, design, run, measure, reflect, improve, signal, or 'all'.
    """
    from src.agent.phases import (
        PhaseContext,
        phase_score,
        phase_design,
        phase_run,
        phase_measure,
        phase_reflect,
        phase_improve,
        phase_signal,
        run_full_pipeline,
    )

    # Handle 'all' shortcut
    if "all" in phases:
        logger.info("running_full_pipeline", price=price, equity=equity)
        ctx = run_full_pipeline(price=price, equity=equity, save_context=save)
        _print_summary(ctx)
        return

    # Validate phase names
    for p in phases:
        if p not in PHASES:
            click.echo(f"Unknown phase '{p}'. Valid phases: {', '.join(PHASES)}")
            sys.exit(1)

    # Load or create context
    if context:
        logger.info("loading_saved_context", path=context)
        ctx = PhaseContext.load(context)
    else:
        ctx = PhaseContext()

    # Phase dispatch
    dispatch = {
        "score": lambda c: phase_score(price=price, ctx=c),
        "design": phase_design,
        "run": lambda c: phase_run(c),
        "measure": lambda c: phase_measure(c, price=price),
        "reflect": phase_reflect,
        "improve": phase_improve,
        "signal": lambda c: phase_signal(c, price=price, equity=equity),
    }

    for phase_name in phases:
        logger.info("running_phase", phase=phase_name)
        ctx = dispatch[phase_name](ctx)

        if ctx.errors and ctx.errors[-1].startswith(f"{phase_name}:"):
            click.echo(f"  Phase '{phase_name}' had errors: {ctx.errors[-1]}")

    if save:
        ctx.save()

    _print_summary(ctx)


def _print_summary(ctx) -> None:
    """Print human-readable summary of phase results."""
    click.echo("\n" + "=" * 60)
    click.echo(f"  Iteration: {ctx.iteration}")
    click.echo(f"  Phases completed: {' → '.join(ctx.phases_completed)}")

    if ctx.scorecard_metrics:
        sc = ctx.scorecard_metrics
        click.echo(f"  Sharpe: {sc.get('signal_sharpe', 0):.3f}")
        click.echo(f"  Accuracy: {sc.get('signal_accuracy', 0):.1%}")
        click.echo(f"  PnL: {sc.get('cumulative_pnl_pct', 0):+.2%}")

    if ctx.signal:
        click.echo(f"  Signal: {ctx.signal.get('direction', 'N/A')}")
        click.echo(f"  Confidence: {ctx.signal.get('confidence', 0):.2f}")

    if ctx.improvements:
        click.echo(f"  Changes applied: {len(ctx.improvements)}")
        for imp in ctx.improvements:
            click.echo(f"    • {imp}")

    if ctx.errors:
        click.echo(f"  Errors: {len(ctx.errors)}")
        for err in ctx.errors:
            click.echo(f"    ⚠ {err}")

    click.echo("=" * 60)


if __name__ == "__main__":
    main()

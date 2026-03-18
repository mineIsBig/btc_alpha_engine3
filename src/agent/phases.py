"""Decomposed agent phases — independently runnable steps of the ralph-loop.

Instead of one monolithic iterate() that chains design → run → measure →
reflect → improve → signal, each phase is a standalone function that reads
from and writes to a shared PhaseContext (serializable JSON).

This enables:
1. Re-running a single phase (e.g., just "measure") without LLM calls
2. Debugging individual phases in isolation
3. Swapping in mock data for any upstream phase
4. Parallel execution of independent phases (design + run)
5. Scheduling phases as separate jobs (Celery, cron, etc.)

Usage:
    # Run full loop (equivalent to iterate()):
    ctx = phase_score(price=65000.0)
    ctx = phase_design(ctx)
    ctx = phase_run(ctx)
    ctx = phase_measure(ctx, price=65000.0)
    ctx = phase_reflect(ctx)
    ctx = phase_improve(ctx)
    signal = phase_signal(ctx, price=65000.0, equity=100000.0)

    # Re-run just measure with cached context:
    ctx = PhaseContext.load("artifacts/phase_context.json")
    ctx = phase_measure(ctx, price=66000.0)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.common.logging import get_logger, set_log_context

logger = get_logger(__name__)

CONTEXT_PATH = Path("artifacts/phase_context.json")


class PhaseContext(BaseModel):
    """Serializable state passed between phases.

    Each phase reads what it needs and writes its results here.
    The full context can be saved/loaded for debugging.
    """

    # Identity
    iteration: int = 0
    timestamp: str = ""

    # phase_score outputs
    scorecard_metrics: dict[str, Any] = Field(default_factory=dict)
    is_profitable: bool = False
    profitability_verdict: str = ""
    lifecycle_result: dict[str, Any] = Field(default_factory=dict)

    # phase_design outputs
    design_response: dict[str, Any] = Field(default_factory=dict)

    # phase_run outputs
    run_results: dict[str, Any] = Field(default_factory=dict)

    # phase_measure outputs
    performance: dict[str, Any] = Field(default_factory=dict)

    # phase_reflect outputs
    reflect_response: dict[str, Any] = Field(default_factory=dict)

    # phase_improve outputs
    improvements: list[str] = Field(default_factory=list)

    # phase_signal outputs
    signal: dict[str, Any] = Field(default_factory=dict)

    # Metadata
    phases_completed: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def save(self, path: Path | str | None = None) -> None:
        """Persist context to JSON for debugging and replay."""
        p = Path(path) if path else CONTEXT_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2))
        logger.info("phase_context_saved", path=str(p))

    @classmethod
    def load(cls, path: Path | str | None = None) -> PhaseContext:
        """Load context from a previous run."""
        p = Path(path) if path else CONTEXT_PATH
        if not p.exists():
            raise FileNotFoundError(f"No phase context at {p}")
        return cls.model_validate_json(p.read_text())


_agent_instance: Any = None


def _get_agent() -> Any:
    """Lazy-load agent singleton to avoid circular imports."""
    global _agent_instance
    if _agent_instance is None:
        from src.agent.alpha_agent import AlphaAgent

        _agent_instance = AlphaAgent()
    return _agent_instance


# ── Phase 1: Score previous signals ─────────────────────────


def phase_score(
    price: float = 0.0,
    ctx: PhaseContext | None = None,
) -> PhaseContext:
    """Score previous signals against current price. No LLM calls.

    This is the ground-truth step — checks if signals hit TP/SL/expired.
    """
    agent = _get_agent()
    ctx = ctx or PhaseContext()
    ctx.iteration = agent.state.iteration + 1
    ctx.timestamp = datetime.now(timezone.utc).isoformat()

    set_log_context(iteration=ctx.iteration)
    logger.info("phase_score_start", price=price)

    try:
        if price > 0:
            just_closed = agent.scorecard.score_signals(price)
            for sig in just_closed:
                logger.info(
                    "signal_outcome",
                    id=sig.signal_id,
                    status=sig.status,
                    pnl=f"{sig.pnl_pct:+.2%}",
                )

        is_profitable, verdict = agent.scorecard.is_profitable(min_signals=10)
        ctx.is_profitable = is_profitable
        ctx.profitability_verdict = verdict
        ctx.scorecard_metrics = agent.scorecard.compute_metrics()

        # Model lifecycle
        ctx.lifecycle_result = agent.manage_model_lifecycle(ctx.scorecard_metrics)

        ctx.phases_completed.append("score")
        logger.info("phase_score_complete", profitable=is_profitable)
    except Exception as e:
        ctx.errors.append(f"score: {e}")
        logger.error("phase_score_failed", error=str(e))

    return ctx


# ── Phase 2: Design (LLM call) ──────────────────────────────


def phase_design(ctx: PhaseContext) -> PhaseContext:
    """LLM analyzes performance and proposes changes. Requires score phase."""
    agent = _get_agent()

    set_log_context(iteration=ctx.iteration)
    logger.info("phase_design_start")

    try:
        prev_perf = {
            "scorecard_metrics": ctx.scorecard_metrics,
            "is_profitable": ctx.is_profitable,
            "verdict": ctx.profitability_verdict,
            "iteration": ctx.iteration,
        }
        design = agent.design_or_modify(prev_perf)
        ctx.design_response = design.model_dump()
        ctx.phases_completed.append("design")
        logger.info(
            "phase_design_complete",
            n_changes=len(design.proposed_changes),
        )
    except Exception as e:
        ctx.errors.append(f"design: {e}")
        logger.error("phase_design_failed", error=str(e))

    return ctx


# ── Phase 3: Run system (model inference) ────────────────────


def phase_run(
    ctx: PhaseContext,
    raw_data: dict | None = None,
) -> PhaseContext:
    """Run promoted models, ensemble, consensus. No LLM calls."""
    agent = _get_agent()

    set_log_context(iteration=ctx.iteration)
    logger.info("phase_run_start")

    try:
        ctx.run_results = agent.run_system(raw_data=raw_data)
        ctx.phases_completed.append("run")
        n_models = ctx.run_results.get("metrics", {}).get("n_models", 0)
        set_log_context(active_models=n_models)
        logger.info("phase_run_complete", n_models=n_models)
    except Exception as e:
        ctx.errors.append(f"run: {e}")
        logger.error("phase_run_failed", error=str(e))

    return ctx


# ── Phase 4: Measure (ground truth metrics) ──────────────────


def phase_measure(
    ctx: PhaseContext,
    price: float = 0.0,
) -> PhaseContext:
    """Compute metrics from run results + scorecard. No LLM calls.

    This is the most commonly re-run phase for debugging.
    """
    agent = _get_agent()

    set_log_context(iteration=ctx.iteration)
    logger.info("phase_measure_start")

    try:
        ctx.performance = agent.measure(ctx.run_results, price)
        ctx.phases_completed.append("measure")

        sharpe = ctx.performance.get("scorecard_metrics", {}).get("signal_sharpe", 0)
        set_log_context(
            regime=str(
                ctx.performance.get("scorecard_metrics", {}).get(
                    "dominant_regime", "unknown"
                )
            )
        )
        logger.info("phase_measure_complete", sharpe=sharpe)
    except Exception as e:
        ctx.errors.append(f"measure: {e}")
        logger.error("phase_measure_failed", error=str(e))

    return ctx


# ── Phase 5: Reflect (LLM call) ─────────────────────────────


def phase_reflect(ctx: PhaseContext) -> PhaseContext:
    """LLM reflects on performance. Requires measure phase."""
    agent = _get_agent()

    set_log_context(iteration=ctx.iteration)
    logger.info("phase_reflect_start")

    try:
        reflection = agent.reflect(ctx.performance)
        ctx.reflect_response = reflection.model_dump()
        ctx.phases_completed.append("reflect")
        logger.info(
            "phase_reflect_complete",
            n_weaknesses=len(reflection.weaknesses),
        )
    except Exception as e:
        ctx.errors.append(f"reflect: {e}")
        logger.error("phase_reflect_failed", error=str(e))

    return ctx


# ── Phase 6: Improve (execute changes) ──────────────────────


def phase_improve(ctx: PhaseContext) -> PhaseContext:
    """Apply approved changes with guardrails. No LLM calls.

    Requires design + reflect + measure phases.
    """
    agent = _get_agent()

    set_log_context(iteration=ctx.iteration)
    logger.info("phase_improve_start")

    try:
        from src.agent.guardrails import DesignResponse, ReflectResponse

        design = DesignResponse.model_validate(ctx.design_response)
        reflection = ReflectResponse.model_validate(ctx.reflect_response)
        scorecard_metrics = ctx.performance.get("scorecard_metrics", {})

        ctx.improvements = agent.improve(design, reflection, scorecard_metrics)
        ctx.phases_completed.append("improve")
        logger.info("phase_improve_complete", n_applied=len(ctx.improvements))
    except Exception as e:
        ctx.errors.append(f"improve: {e}")
        logger.error("phase_improve_failed", error=str(e))

    return ctx


# ── Phase 7: Signal output ───────────────────────────────────


def phase_signal(
    ctx: PhaseContext,
    price: float = 0.0,
    equity: float = 100000.0,
) -> PhaseContext:
    """Generate final signal. No LLM calls."""
    agent = _get_agent()

    set_log_context(iteration=ctx.iteration)
    logger.info("phase_signal_start")

    try:
        # Increment iteration (this is where the agent officially "ticks")
        agent.state.iteration = ctx.iteration

        signal = agent.generate_signal(ctx.run_results, ctx.performance, price, equity)
        agent.state.total_signals += 1
        agent.state.last_signal = signal
        agent.scorecard.record_signal(signal)
        agent.scorecard.save()
        agent._save_state()

        ctx.signal = signal.model_dump(mode="json")
        ctx.phases_completed.append("signal")
        logger.info(
            "phase_signal_complete",
            direction=signal.direction,
            confidence=signal.confidence,
        )
    except Exception as e:
        ctx.errors.append(f"signal: {e}")
        logger.error("phase_signal_failed", error=str(e))

    return ctx


# ── Full pipeline (equivalent to iterate()) ──────────────────


def run_full_pipeline(
    price: float = 0.0,
    equity: float = 100000.0,
    raw_data: dict | None = None,
    save_context: bool = True,
) -> PhaseContext:
    """Run all phases sequentially. Equivalent to AlphaAgent.iterate().

    Returns PhaseContext with full results, also saved for debugging.
    """
    ctx = phase_score(price=price)
    ctx = phase_design(ctx)
    ctx = phase_run(ctx, raw_data=raw_data)
    ctx = phase_measure(ctx, price=price)
    ctx = phase_reflect(ctx)
    ctx = phase_improve(ctx)
    ctx = phase_signal(ctx, price=price, equity=equity)

    if save_context:
        ctx.save()

    return ctx

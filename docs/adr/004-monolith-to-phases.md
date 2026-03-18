# ADR-004: Decomposing the Monolithic Agent Loop into Phases

**Status**: Accepted
**Date**: 2026-03-18
**Context**: The single `iterate()` method chains 7 steps; debugging any step requires running the full loop including LLM calls.

## Decision

Extract each step of the ralph-loop into a standalone function in `src/agent/phases.py`, passing state via a serializable `PhaseContext` (Pydantic model, saved to JSON).

Phases:
1. `phase_score` — Score previous signals (no LLM)
2. `phase_design` — LLM proposes changes
3. `phase_run` — Model inference + ensemble (no LLM)
4. `phase_measure` — Compute metrics (no LLM)
5. `phase_reflect` — LLM reflects on performance
6. `phase_improve` — Execute approved changes (no LLM)
7. `phase_signal` — Generate output signal (no LLM)

A CLI script (`scripts/run_phase.py`) runs any subset of phases.

## Rationale

- **Debuggability**: The "measure" phase is the most frequently tweaked. With the monolith, every measure debug run incurs two LLM API calls ($0.01-0.10 each). Now you can re-run `python scripts/run_phase.py measure --price 65000` in <1 second.
- **Testability**: Unit tests can test each phase in isolation by injecting mock PhaseContext.
- **Observability**: The saved PhaseContext JSON provides a complete audit trail of what each phase produced.
- **Future orchestration**: Phases can be dispatched to Celery workers, run as separate Kubernetes jobs, or triggered by cron without architectural changes.

## Alternatives Considered

- **Celery task queue**: Too heavy for the current deployment. The phase decomposition is the prerequisite; Celery can be added later by wrapping each phase function as a task.
- **Event-driven (pub/sub)**: Phases have strict ordering dependencies. Event-driven adds complexity without benefit when the flow is linear.
- **Separate processes with IPC**: Overhead of serialization between processes. The PhaseContext JSON approach gives the same debuggability without IPC complexity.

## Consequences

- `AlphaAgent.iterate()` remains as the backward-compatible entry point (it calls phases internally via `run_full_pipeline`).
- The `PhaseContext` JSON file at `artifacts/phase_context.json` grows to ~5-20KB per iteration.
- Developers can now contribute changes to individual phases without understanding the full loop.

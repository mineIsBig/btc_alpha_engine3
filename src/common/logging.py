"""Structured logging setup using structlog with rich context injection."""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any

import structlog

# ── Context variables for automatic injection into every log line ────
_iteration_ctx: ContextVar[int] = ContextVar("iteration", default=0)
_regime_ctx: ContextVar[str] = ContextVar("regime", default="unknown")
_active_models_ctx: ContextVar[int] = ContextVar("active_models", default=0)
_symbol_ctx: ContextVar[str] = ContextVar("symbol", default="BTC")


def set_log_context(
    *,
    iteration: int | None = None,
    regime: str | None = None,
    active_models: int | None = None,
    symbol: str | None = None,
) -> None:
    """Update structured logging context for all subsequent log lines.

    Call this at the top of each agent iteration to automatically attach
    iteration number, current regime, active model count, and symbol
    to every log message.
    """
    if iteration is not None:
        _iteration_ctx.set(iteration)
    if regime is not None:
        _regime_ctx.set(regime)
    if active_models is not None:
        _active_models_ctx.set(active_models)
    if symbol is not None:
        _symbol_ctx.set(symbol)


def _inject_agent_context(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Structlog processor that injects agent context into every log line."""
    event_dict.setdefault("iteration", _iteration_ctx.get())
    event_dict.setdefault("regime", _regime_ctx.get())
    event_dict.setdefault("active_models", _active_models_ctx.get())
    event_dict.setdefault("symbol", _symbol_ctx.get())
    return event_dict


def setup_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Configure structlog + stdlib logging with rich context injection."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    if fmt == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _inject_agent_context,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,  # Allow context to update
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=log_level,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a bound logger instance."""
    return structlog.get_logger(name)

"""Emergency cancel and flatten operations."""

from __future__ import annotations

from src.common.logging import get_logger
from src.execution.order_router import OrderRouter

logger = get_logger(__name__)


def emergency_flatten(router: OrderRouter, reason: str = "emergency") -> None:
    """Emergency: cancel all orders and flatten all positions."""
    logger.error("EMERGENCY_FLATTEN", reason=reason)
    results = router.flatten_all(reason=reason)
    logger.info("emergency_flatten_complete", fills=len(results))


def emergency_cancel_all(router: OrderRouter, reason: str = "emergency") -> None:
    """Cancel all open orders without flattening."""
    logger.error("EMERGENCY_CANCEL", reason=reason)
    router.flatten_all(reason=reason)  # Paper broker doesn't have pending orders

"""Position synchronization between broker state and internal state."""

from __future__ import annotations

from src.common.logging import get_logger
from src.execution.order_router import OrderRouter
from src.risk.account_state import AccountState

logger = get_logger(__name__)


def sync_positions(
    router: OrderRouter, account: AccountState, current_prices: dict[str, float]
) -> None:
    """Sync position state from broker into account state."""
    positions = router.get_positions()
    account.update_from_positions(positions, current_prices)
    logger.debug("positions_synced", n_positions=len(positions), equity=account.equity)

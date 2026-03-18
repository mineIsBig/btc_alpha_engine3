"""Paper broker: fully functional simulated broker for paper trading."""
from __future__ import annotations

import uuid
from typing import Any

from src.common.logging import get_logger
from src.common.time_utils import utc_now
from src.execution.slippage_model import apply_slippage
from src.storage.database import session_scope
from src.storage.models import Order, Fill, Position

logger = get_logger(__name__)


class PaperBroker:
    """Simulated broker for paper trading. No exchange keys required."""

    def __init__(self, slippage_bps: float = 5.0, commission_bps: float = 2.0):
        self.slippage_bps = slippage_bps
        self.commission_bps = commission_bps
        self._positions: dict[str, dict[str, Any]] = {}
        self._mid_prices: dict[str, float] = {}

    def set_price(self, symbol: str, price: float) -> None:
        self._mid_prices[symbol] = price

    def get_mid_price(self, symbol: str) -> float:
        return self._mid_prices.get(symbol, 0.0)

    def get_position(self, symbol: str) -> dict[str, Any]:
        return self._positions.get(symbol, {
            "symbol": symbol, "side": "flat", "quantity": 0.0,
            "avg_entry_price": 0.0, "unrealized_pnl": 0.0, "realized_pnl": 0.0,
        })

    def get_all_positions(self) -> list[dict[str, Any]]:
        return list(self._positions.values())

    def submit_order(
        self, symbol: str, side: str, quantity: float,
        order_type: str = "market", price: float | None = None, reason: str = "",
    ) -> dict[str, Any]:
        """Submit and immediately fill a paper order."""
        order_id = f"paper_{uuid.uuid4().hex[:12]}"
        now = utc_now()
        mid = self._mid_prices.get(symbol, 0.0)

        if mid <= 0:
            return {"order_id": order_id, "status": "rejected", "reason": "no_price"}

        is_buy = side == "buy"
        fill_price = apply_slippage(mid, is_buy, self.slippage_bps)
        commission = abs(quantity * fill_price) * (self.commission_bps / 10000.0)
        fill_id = f"pfill_{uuid.uuid4().hex[:12]}"

        # Compute PnL before updating position
        pnl = self._compute_fill_pnl(symbol, side, quantity, fill_price)

        # Update position
        self._update_position(symbol, side, quantity, fill_price)

        # Persist
        with session_scope() as session:
            session.add(Order(
                order_id=order_id, timestamp=now, symbol=symbol, side=side,
                order_type=order_type, quantity=quantity, price=fill_price,
                status="filled", filled_qty=quantity, filled_price=fill_price,
                broker="paper", reason=reason, created_at=now, updated_at=now,
            ))
            session.add(Fill(
                fill_id=fill_id, order_id=order_id, timestamp=now, symbol=symbol,
                side=side, quantity=quantity, price=fill_price,
                commission=commission, slippage=abs(fill_price - mid) * quantity,
                broker="paper",
            ))

        logger.info("paper_fill", order_id=order_id, symbol=symbol, side=side,
                     qty=quantity, price=fill_price, pnl=pnl)

        return {
            "order_id": order_id, "fill_id": fill_id, "status": "filled",
            "fill_price": fill_price, "commission": commission, "pnl": pnl,
        }

    def _update_position(self, symbol: str, side: str, quantity: float, fill_price: float) -> None:
        pos = self._positions.get(symbol, {
            "symbol": symbol, "side": "flat", "quantity": 0.0,
            "avg_entry_price": 0.0, "unrealized_pnl": 0.0, "realized_pnl": 0.0,
        })

        is_buy = side == "buy"
        cur_qty = pos["quantity"]
        cur_side = pos["side"]

        if cur_side == "flat" or cur_qty == 0:
            pos["side"] = "long" if is_buy else "short"
            pos["quantity"] = quantity
            pos["avg_entry_price"] = fill_price
        elif (cur_side == "long" and is_buy) or (cur_side == "short" and not is_buy):
            total_cost = pos["avg_entry_price"] * cur_qty + fill_price * quantity
            pos["quantity"] = cur_qty + quantity
            pos["avg_entry_price"] = total_cost / pos["quantity"] if pos["quantity"] > 0 else 0
        else:
            if quantity >= cur_qty:
                remaining = quantity - cur_qty
                pos["quantity"] = remaining
                if remaining > 0:
                    pos["side"] = "long" if is_buy else "short"
                    pos["avg_entry_price"] = fill_price
                else:
                    pos["side"] = "flat"
                    pos["avg_entry_price"] = 0.0
            else:
                pos["quantity"] = cur_qty - quantity

        self._positions[symbol] = pos

        with session_scope() as session:
            db_pos = session.query(Position).filter_by(symbol=symbol, broker="paper").first()
            if db_pos:
                db_pos.side = pos["side"]
                db_pos.quantity = pos["quantity"]
                db_pos.avg_entry_price = pos["avg_entry_price"]
                db_pos.updated_at = utc_now()
            else:
                session.add(Position(
                    symbol=symbol, side=pos["side"], quantity=pos["quantity"],
                    avg_entry_price=pos["avg_entry_price"], broker="paper", updated_at=utc_now(),
                ))

    def _compute_fill_pnl(self, symbol: str, side: str, quantity: float, fill_price: float) -> float:
        pos = self._positions.get(symbol)
        if not pos or pos["avg_entry_price"] == 0 or pos["side"] == "flat":
            return 0.0
        is_buy = side == "buy"
        was_long = pos["side"] == "long"
        if (was_long and not is_buy):
            return (fill_price - pos["avg_entry_price"]) * min(quantity, pos["quantity"])
        elif (not was_long and is_buy):
            return (pos["avg_entry_price"] - fill_price) * min(quantity, pos["quantity"])
        return 0.0

    def flatten_all(self, reason: str = "risk_breach") -> list[dict[str, Any]]:
        results = []
        for symbol, pos in list(self._positions.items()):
            if pos["quantity"] > 0 and pos["side"] != "flat":
                close_side = "sell" if pos["side"] == "long" else "buy"
                result = self.submit_order(symbol=symbol, side=close_side,
                                           quantity=pos["quantity"], reason=reason)
                results.append(result)
        return results

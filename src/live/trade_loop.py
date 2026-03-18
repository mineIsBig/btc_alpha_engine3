"""Trade loop: orchestrates inference -> ensemble -> sizing -> execution."""

from __future__ import annotations


from src.common.config import get_settings
from src.common.logging import get_logger
from src.common.time_utils import utc_now
from src.execution.order_router import OrderRouter
from src.execution.emergency_cancel import EmergencyCancel
from src.live.inference_loop import InferenceLoop
from src.portfolio.ensemble import EnsembleAggregator
from src.portfolio.consensus import ConsensusGate
from src.portfolio.sizing import VolatilitySizer
from src.portfolio.constraints import PortfolioConstraints
from src.risk.risk_manager import RiskManager

logger = get_logger(__name__)


class TradeLoop:
    """Main trading loop: inference -> decision -> execution."""

    def __init__(self, initial_equity: float = 100000.0):
        self.settings = get_settings()
        self.inference = InferenceLoop()
        self.ensemble = EnsembleAggregator()
        self.consensus = ConsensusGate()
        self.sizer = VolatilitySizer()
        self.constraints = PortfolioConstraints()
        self.risk = RiskManager(initial_equity=initial_equity)
        self.router = OrderRouter()
        self.emergency = EmergencyCancel(self.router)

        self.cash = initial_equity
        self.last_price: float = 0.0

    def tick(self, current_price: float | None = None) -> dict:
        """Execute one trading cycle.

        Returns dict with decision details.
        """
        ts = utc_now()

        # Get current price
        if current_price is None:
            from src.data.hyperliquid_client import HyperliquidClient

            try:
                client = HyperliquidClient()
                current_price = client.get_mid_price("BTC") or self.last_price
                client.close()
            except Exception:
                current_price = self.last_price
        self.last_price = current_price

        if current_price <= 0:
            return {"action": "skip", "reason": "no_price"}

        # Update equity
        equity = self.router.paper_broker.get_equity({"BTC": current_price}, self.cash)
        can_continue = self.risk.update_equity(equity, cash=self.cash, timestamp=ts)

        if not can_continue:
            if self.settings.flatten_on_breach:
                self.emergency.cancel_and_flatten(
                    {"BTC": current_price},
                    reason=self.risk.account.breach_reason or "breach",
                )
            self.risk.persist_state()
            return {"action": "breach", "reason": self.risk.account.breach_reason}

        if not self.risk.can_trade():
            return {"action": "skip", "reason": "trading_disabled"}

        # Run inference
        signals_by_horizon = self.inference.run_inference(timestamp=ts)

        if not signals_by_horizon:
            return {"action": "skip", "reason": "no_signals"}

        # Aggregate per horizon
        horizon_agg = {}
        for h, sigs in signals_by_horizon.items():
            horizon_agg[h] = self.ensemble.aggregate(sigs, timestamp=ts)

        # Cross-horizon consensus
        final_side, consensus_reason = self.consensus.check(horizon_agg)

        if final_side == 0:
            return {"action": "flat", "reason": consensus_reason}

        # Size the position
        rvol = 0.5  # placeholder - should come from features
        headroom = self.risk.get_headroom()
        sizing = self.sizer.compute_size(
            equity=equity,
            realized_vol=rvol,
            signal_strength=max(s.avg_confidence for s in horizon_agg.values()),
            headroom_to_breach=headroom,
            current_price=current_price,
        )

        target_usd = sizing["target_size_usd"]
        if target_usd == 0:
            return {"action": "skip", "reason": sizing["reason"]}

        # Apply constraints
        adj_usd, approved, c_reason = self.constraints.validate_order(
            target_size_usd=target_usd * final_side,
            equity=equity,
            current_gross_exposure=self.risk.account.gross_exposure,
            can_trade=self.risk.can_trade(),
        )

        if not approved:
            return {"action": "skip", "reason": c_reason}

        # Pre-trade risk check
        ok, risk_reason = self.risk.pre_trade_check(abs(adj_usd))
        if not ok:
            return {"action": "skip", "reason": risk_reason}

        # Execute
        side_str = "buy" if final_side == 1 else "sell"
        qty = abs(adj_usd) / current_price

        result = self.router.submit_order(
            symbol="BTC",
            side=side_str,
            quantity=qty,
            price=current_price,
            order_type="market",
            reason=consensus_reason,
        )

        # Post-fill update
        fill_price = result.get("fill_price", current_price)
        commission = result.get("commission", 0)
        self.cash -= commission
        pnl = 0.0  # will be realized on close
        self.risk.post_fill(pnl)
        self.risk.persist_state()

        logger.info(
            "trade_executed", side=side_str, qty=qty, price=fill_price, result=result
        )

        return {
            "action": "trade",
            "side": side_str,
            "quantity": qty,
            "price": fill_price,
            "order_id": result.get("order_id"),
            "reason": consensus_reason,
        }

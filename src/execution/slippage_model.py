"""Slippage model for realistic fill simulation."""
from __future__ import annotations


def apply_slippage(mid_price: float, is_buy: bool, slippage_bps: float = 5.0) -> float:
    """Apply slippage to a mid price.

    Buys fill above mid, sells fill below mid.
    """
    slip = mid_price * (slippage_bps / 10000.0)
    if is_buy:
        return mid_price + slip
    else:
        return mid_price - slip

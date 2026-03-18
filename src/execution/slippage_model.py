"""Regime-aware slippage model for realistic fill simulation."""
from __future__ import annotations

import numpy as np


def apply_slippage(mid_price: float, is_buy: bool, slippage_bps: float = 5.0) -> float:
    """Apply slippage to a mid price.

    Buys fill above mid, sells fill below mid.
    """
    slip = mid_price * (slippage_bps / 10000.0)
    if is_buy:
        return mid_price + slip
    else:
        return mid_price - slip


class RegimeAwareCostModel:
    """Dynamic transaction cost model that scales with market regime.
    
    Increases costs during volatile regimes when liquidation cascades occur,
    making performance estimates more realistic for BTC derivatives.
    """
    
    def __init__(
        self,
        base_slippage_bps: float = 5.0,
        base_commission_bps: float = 2.0,
        volatility_multiplier: float = 2.0,
        liquidation_multiplier: float = 1.5,
        regime_multipliers: dict[str, float] | None = None,
    ):
        """Initialize regime-aware cost model.
        
        Args:
            base_slippage_bps: Base one-way slippage in basis points
            base_commission_bps: Base one-way commission in basis points
            volatility_multiplier: Cost multiplier per unit of volatility
            liquidation_multiplier: Cost multiplier per unit of liquidation intensity
            regime_multipliers: Additional multipliers for specific regimes
        """
        self.base_slippage_bps = base_slippage_bps
        self.base_commission_bps = base_commission_bps
        self.volatility_multiplier = volatility_multiplier
        self.liquidation_multiplier = liquidation_multiplier
        self.regime_multipliers = regime_multipliers or {
            "panic_flush": 3.0,
            "squeeze": 2.0,
            "crowded_long": 1.5,
            "crowded_short": 1.5,
        }
    
    def compute_dynamic_costs(
        self,
        regime: str | None = None,
        volatility_zscore: float = 0.0,
        liquidation_zscore: float = 0.0,
    ) -> tuple[float, float]:
        """Compute dynamic slippage and commission based on market conditions.
        
        Args:
            regime: Current market regime (e.g., "panic_flush", "squeeze")
            volatility_zscore: Realized volatility z-score over lookback window
            liquidation_zscore: Liquidation intensity z-score
            
        Returns:
            Tuple of (slippage_bps, commission_bps)
        """
        # Start with base costs
        slippage = self.base_slippage_bps
        commission = self.base_commission_bps
        
        # Scale costs with volatility (more volatile = higher costs)
        if volatility_zscore > 0:
            slippage += self.volatility_multiplier * volatility_zscore
        
        # Scale costs with liquidation intensity (more liqs = higher costs)
        if liquidation_zscore > 0:
            slippage += self.liquidation_multiplier * liquidation_zscore
        
        # Apply regime-specific multipliers
        if regime and regime in self.regime_multipliers:
            multiplier = self.regime_multipliers[regime]
            slippage *= multiplier
            commission *= multiplier
            
        return slippage, commission
    
    def apply_costs(
        self,
        mid_price: float,
        is_buy: bool,
        regime: str | None = None,
        volatility_zscore: float = 0.0,
        liquidation_zscore: float = 0.0,
    ) -> float:
        """Apply regime-aware costs to a mid price.
        
        Args:
            mid_price: Mid-market price
            is_buy: True if buying, False if selling
            regime: Current market regime
            volatility_zscore: Realized volatility z-score
            liquidation_zscore: Liquidation intensity z-score
            
        Returns:
            Fill price after applying dynamic costs
        """
        slippage_bps, _ = self.compute_dynamic_costs(regime, volatility_zscore, liquidation_zscore)
        return apply_slippage(mid_price, is_buy, slippage_bps)

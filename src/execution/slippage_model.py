"""Regime-dependent slippage model for realistic fill simulation.

The base 5bps slippage assumption is optimistic during volatile regimes --
exactly when liquidation cascade signals fire and spreads widen.
This module scales slippage dynamically based on:
1. Realized volatility (rolling std of returns)
2. Liquidation intensity (volume of recent liquidations)

During calm markets, slippage stays near the baseline. During stress,
it can scale up to 5x the baseline to reflect real-world spread widening.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logging import get_logger

logger = get_logger(__name__)

# Regime slippage configuration
BASELINE_SLIPPAGE_BPS = 5.0
BASELINE_VOLATILITY = 0.002       # ~0.2% hourly vol = "normal" BTC
MAX_SLIPPAGE_MULTIPLIER = 5.0     # cap at 5x baseline (25 bps)
MIN_SLIPPAGE_MULTIPLIER = 0.8     # floor at 0.8x baseline (4 bps)
VOLATILITY_LOOKBACK = 24          # hours for rolling vol calculation
LIQUIDATION_SCALING_FACTOR = 0.5  # additional bps per 1M USD liquidation volume


def apply_slippage(mid_price: float, is_buy: bool, slippage_bps: float = 5.0) -> float:
    """Apply slippage to a mid price.

    Buys fill above mid, sells fill below mid.
    """
    slip = mid_price * (slippage_bps / 10000.0)
    if is_buy:
        return mid_price + slip
    else:
        return mid_price - slip


def compute_volatility_multiplier(
    recent_returns: np.ndarray | pd.Series,
    baseline_vol: float = BASELINE_VOLATILITY,
) -> float:
    """Compute slippage multiplier from realized volatility.

    When vol is at baseline (0.2% hourly), multiplier = 1.0.
    When vol is 2x baseline, multiplier ≈ 2.0.
    Capped at MAX_SLIPPAGE_MULTIPLIER, floored at MIN_SLIPPAGE_MULTIPLIER.

    Args:
        recent_returns: array of recent hourly returns
        baseline_vol: "normal" hourly volatility level

    Returns:
        multiplier >= MIN_SLIPPAGE_MULTIPLIER, <= MAX_SLIPPAGE_MULTIPLIER
    """
    if len(recent_returns) < 2:
        return 1.0

    realized_vol = float(np.std(recent_returns))
    if realized_vol <= 0 or baseline_vol <= 0:
        return 1.0

    ratio = realized_vol / baseline_vol
    multiplier = max(MIN_SLIPPAGE_MULTIPLIER, min(ratio, MAX_SLIPPAGE_MULTIPLIER))
    return multiplier


def compute_liquidation_adder(
    liquidation_volume_usd: float,
    scaling_factor: float = LIQUIDATION_SCALING_FACTOR,
    cap_bps: float = 10.0,
) -> float:
    """Compute additional slippage bps from liquidation intensity.

    During liquidation cascades, order book depth evaporates and market
    impact spikes. This adds slippage proportional to recent liquidation
    volume.

    Args:
        liquidation_volume_usd: total liquidation volume in USD over lookback
        scaling_factor: bps to add per 1M USD of liquidations
        cap_bps: maximum additional bps from liquidations

    Returns:
        Additional slippage in bps (0 to cap_bps)
    """
    if liquidation_volume_usd <= 0:
        return 0.0
    adder = (liquidation_volume_usd / 1_000_000.0) * scaling_factor
    return min(adder, cap_bps)


def compute_regime_slippage(
    base_slippage_bps: float = BASELINE_SLIPPAGE_BPS,
    recent_returns: np.ndarray | pd.Series | None = None,
    liquidation_volume_usd: float = 0.0,
    baseline_vol: float = BASELINE_VOLATILITY,
) -> float:
    """Compute regime-adjusted slippage in bps.

    Combines volatility-based multiplier with liquidation-based adder:
        effective_slippage = base * vol_multiplier + liquidation_adder

    Examples:
        Normal market:  5 * 1.0 + 0 = 5.0 bps
        2x vol:         5 * 2.0 + 0 = 10.0 bps
        3x vol + 5M liquidations: 5 * 3.0 + 2.5 = 17.5 bps
        5x vol + 20M liquidations: 5 * 5.0 + 10.0 = 35.0 bps (capped)

    Args:
        base_slippage_bps: baseline slippage assumption
        recent_returns: recent hourly returns for vol calculation
        liquidation_volume_usd: recent liquidation volume in USD
        baseline_vol: "normal" hourly volatility

    Returns:
        Effective slippage in bps
    """
    if recent_returns is not None and len(recent_returns) >= 2:
        vol_mult = compute_volatility_multiplier(recent_returns, baseline_vol)
    else:
        vol_mult = 1.0

    liq_adder = compute_liquidation_adder(liquidation_volume_usd)
    effective = base_slippage_bps * vol_mult + liq_adder

    if vol_mult > 1.5 or liq_adder > 1.0:
        logger.debug("regime_slippage_elevated",
                     base=base_slippage_bps, vol_mult=round(vol_mult, 2),
                     liq_adder=round(liq_adder, 2), effective=round(effective, 2))

    return effective


def apply_regime_slippage(
    mid_price: float,
    is_buy: bool,
    recent_returns: np.ndarray | pd.Series | None = None,
    liquidation_volume_usd: float = 0.0,
    base_slippage_bps: float = BASELINE_SLIPPAGE_BPS,
) -> float:
    """Apply regime-adjusted slippage to a mid price.

    Convenience wrapper combining regime slippage calculation with
    fill price computation.

    Args:
        mid_price: current mid price
        is_buy: True for buy orders, False for sells
        recent_returns: recent hourly returns for vol calculation
        liquidation_volume_usd: recent liquidation volume in USD
        base_slippage_bps: baseline slippage assumption

    Returns:
        Fill price adjusted for regime-dependent slippage
    """
    effective_bps = compute_regime_slippage(
        base_slippage_bps=base_slippage_bps,
        recent_returns=recent_returns,
        liquidation_volume_usd=liquidation_volume_usd,
    )
    return apply_slippage(mid_price, is_buy, effective_bps)

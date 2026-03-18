"""Temporal interaction features: second-order cross-features that capture
dynamics between different market microstructure signals over time.

Static interactions (funding × OI) miss how the *change* in one signal
relates to the *acceleration* of another. These temporal interactions
capture momentum divergences, regime shifts, and feedback loops.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logging import get_logger

logger = get_logger(__name__)


def compute_temporal_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute second-order temporal cross-features.

    Expects the merged base DataFrame with columns from price, funding, and OI
    feature computations already available (or the raw columns to derive from).

    Returns DataFrame with temporal interaction columns only.
    """
    out = pd.DataFrame(index=df.index)

    # ── Helper: safe column access ────────────────────────────
    def _col(name: str) -> pd.Series:
        if name in df.columns:
            return df[name].astype(float)
        return pd.Series(np.nan, index=df.index)

    close = _col("close")
    funding = _col("funding_close")
    oi = _col("oi_close")

    # Pre-compute base changes if raw columns exist
    price_ret_1h = close.pct_change(1)
    price_ret_4h = close.pct_change(4)
    price_accel_4h = price_ret_1h.diff(4)

    funding_chg_1h = funding.diff(1)
    funding_chg_4h = funding.diff(4)
    funding_accel_4h = funding_chg_1h.diff(4)

    oi_chg_1h = oi.pct_change(1)
    oi_chg_4h = oi.pct_change(4)
    oi_accel_4h = oi_chg_1h.diff(4)

    # ══════════════════════════════════════════════════════════
    # 1. Funding change × OI acceleration
    #    Captures: rising funding + accelerating OI = crowded leverage
    # ══════════════════════════════════════════════════════════
    out["funding_chg_x_oi_accel_4h"] = funding_chg_4h * oi_accel_4h

    # ══════════════════════════════════════════════════════════
    # 2. OI change × Funding acceleration
    #    Captures: OI building while funding rate is accelerating
    # ══════════════════════════════════════════════════════════
    out["oi_chg_x_funding_accel_4h"] = oi_chg_4h * funding_accel_4h

    # ══════════════════════════════════════════════════════════
    # 3. Price acceleration × OI acceleration (momentum divergence)
    #    Captures: price slowing while OI still accelerating = top signal
    # ══════════════════════════════════════════════════════════
    out["price_accel_x_oi_accel_4h"] = price_accel_4h * oi_accel_4h

    # ══════════════════════════════════════════════════════════
    # 4. Funding momentum divergence
    #    Rolling correlation between Δfunding and Δprice (sign flips = divergence)
    # ══════════════════════════════════════════════════════════
    out["funding_price_corr_chg_24h"] = funding_chg_1h.rolling(24).corr(price_ret_1h)
    out["funding_price_corr_chg_72h"] = funding_chg_1h.rolling(72).corr(price_ret_1h)

    # ══════════════════════════════════════════════════════════
    # 5. OI-Funding velocity ratio
    #    Normalized: how fast OI grows relative to funding change
    # ══════════════════════════════════════════════════════════
    funding_velocity = funding.diff(4).rolling(8).mean()
    oi_velocity = oi.pct_change(4).rolling(8).mean()
    denom = funding_velocity.abs().replace(0, np.nan)
    out["oi_funding_velocity_ratio"] = oi_velocity / denom

    # ══════════════════════════════════════════════════════════
    # 6. Lagged cross-features (funding leads price by N hours)
    #    Tests if funding predicts near-term returns
    # ══════════════════════════════════════════════════════════
    for lag in [1, 4, 8]:
        out[f"funding_lag{lag}_x_ret_4h"] = funding_chg_1h.shift(lag) * price_ret_4h

    # ══════════════════════════════════════════════════════════
    # 7. OI surge × Volatility compression
    #    Big OI build during low vol = coiled spring
    # ══════════════════════════════════════════════════════════
    rvol_24h = price_ret_1h.rolling(24).std() * np.sqrt(24)
    rvol_168h = price_ret_1h.rolling(168).std() * np.sqrt(168)
    vol_compression = rvol_24h / rvol_168h.replace(0, np.nan)
    oi_surge_24h = oi.pct_change(24)
    out["oi_surge_x_vol_compression"] = oi_surge_24h * (1 - vol_compression)

    # ══════════════════════════════════════════════════════════
    # 8. Liquidation cascade indicator
    #    Uses available liquidation columns if present
    # ══════════════════════════════════════════════════════════
    if "total_liquidations_usd" in df.columns:
        liqs = _col("total_liquidations_usd")
        liq_accel = liqs.diff(1).diff(1)  # second derivative
        out["liq_accel_x_oi_chg"] = liq_accel * oi_chg_1h
        out["liq_accel_x_price_ret"] = liq_accel * price_ret_1h

    # ══════════════════════════════════════════════════════════
    # 9. Taker flow momentum × funding
    # ══════════════════════════════════════════════════════════
    if "buy_sell_ratio" in df.columns:
        bsr = _col("buy_sell_ratio")
        bsr_chg = bsr.diff(4)
        out["taker_momentum_x_funding_chg"] = bsr_chg * funding_chg_4h

    # ══════════════════════════════════════════════════════════
    # 10. Regime-transition features: rolling covariance shifts
    # ══════════════════════════════════════════════════════════
    cov_short = price_ret_1h.rolling(24).cov(oi_chg_1h)
    cov_long = price_ret_1h.rolling(168).cov(oi_chg_1h)
    out["price_oi_cov_regime_shift"] = cov_short - cov_long

    # Clean infinities
    out = out.replace([np.inf, -np.inf], np.nan)

    logger.info(
        "temporal_interactions_computed",
        n_features=len(out.columns),
        n_rows=len(out),
    )
    return out

"""Open Interest derived features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_oi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute OI features.

    Expects: oi_close (OI level), and optionally close (price) for divergence.
    """
    out = pd.DataFrame(index=df.index)

    oi = (
        df["oi_close"].astype(float)
        if "oi_close" in df.columns
        else pd.Series(0.0, index=df.index)
    )

    # ── Level and change ─────────────────────────────────────
    out["oi_level"] = oi
    out["oi_change_1h"] = oi.pct_change(1)
    out["oi_change_4h"] = oi.pct_change(4)
    out["oi_change_24h"] = oi.pct_change(24)

    # ── Acceleration ─────────────────────────────────────────
    oi_chg = oi.pct_change(1)
    out["oi_acceleration_4h"] = oi_chg - oi_chg.shift(4)
    out["oi_acceleration_24h"] = oi_chg - oi_chg.shift(24)

    # ── Z-score ──────────────────────────────────────────────
    for w in [24, 72, 168]:
        mu = oi.rolling(w).mean()
        sigma = oi.rolling(w).std().replace(0, np.nan)
        out[f"oi_zscore_{w}h"] = (oi - mu) / sigma

    # ── Divergence vs price ──────────────────────────────────
    if "close" in df.columns:
        price = df["close"].astype(float)
        price_ret_24 = price.pct_change(24)
        oi_ret_24 = oi.pct_change(24)
        out["oi_price_divergence_24h"] = oi_ret_24 - price_ret_24

        # Rolling correlation
        out["oi_price_corr_24h"] = oi.rolling(24).corr(price)
        out["oi_price_corr_72h"] = oi.rolling(72).corr(price)

    # ── OI relative to moving average ────────────────────────
    for w in [24, 72, 168]:
        ma = oi.rolling(w).mean().replace(0, np.nan)
        out[f"oi_vs_ma{w}"] = (oi - ma) / ma

    return out

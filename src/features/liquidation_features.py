"""Liquidation-derived features."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_liquidation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute liquidation features.

    Expects: long_liquidations_usd, short_liquidations_usd, total_liquidations_usd
    """
    out = pd.DataFrame(index=df.index)

    long_liq = df.get("long_liquidations_usd", pd.Series(0.0, index=df.index)).astype(float)
    short_liq = df.get("short_liquidations_usd", pd.Series(0.0, index=df.index)).astype(float)
    total_liq = df.get("total_liquidations_usd", long_liq + short_liq).astype(float)

    # ── Imbalance ────────────────────────────────────────────
    denom = (long_liq + short_liq).replace(0, np.nan)
    out["liq_imbalance"] = (long_liq - short_liq) / denom
    out["liq_long_ratio"] = long_liq / denom

    # ── Magnitude ────────────────────────────────────────────
    out["liq_total_usd"] = total_liq
    out["liq_long_usd"] = long_liq
    out["liq_short_usd"] = short_liq

    # ── Rolling sums ─────────────────────────────────────────
    for w in [4, 12, 24]:
        out[f"liq_total_{w}h_sum"] = total_liq.rolling(w).sum()
        out[f"liq_imbalance_{w}h_mean"] = out["liq_imbalance"].rolling(w).mean()

    # ── Shock detection: z-score of liquidation volume ───────
    for w in [24, 72, 168]:
        mu = total_liq.rolling(w).mean()
        sigma = total_liq.rolling(w).std().replace(0, np.nan)
        out[f"liq_shock_zscore_{w}h"] = (total_liq - mu) / sigma

    # ── Cascade: sustained high liquidation ──────────────────
    shock_24 = out["liq_shock_zscore_24h"]
    out["liq_cascade_flag"] = (shock_24 > 2.0).rolling(3).sum()  # 3+ hours of elevated liqs

    # ── Liquidation momentum ─────────────────────────────────
    out["liq_momentum_4h"] = total_liq.rolling(4).sum() / total_liq.rolling(24).sum().replace(0, np.nan)

    return out

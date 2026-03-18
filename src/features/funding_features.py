"""Funding rate derived features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_funding_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute funding rate features.

    Expects columns from cg_funding_1h: close (funding rate close)
    """
    out = pd.DataFrame(index=df.index)

    fr = (
        df["funding_close"].astype(float)
        if "funding_close" in df.columns
        else df.get("close", pd.Series(dtype=float))
    )

    # ── Level ────────────────────────────────────────────────
    out["funding_rate"] = fr

    # ── Change ───────────────────────────────────────────────
    out["funding_change_1h"] = fr.diff(1)
    out["funding_change_8h"] = fr.diff(8)
    out["funding_change_24h"] = fr.diff(24)

    # ── Z-score (rolling windows) ────────────────────────────
    for w in [24, 72, 168]:
        mu = fr.rolling(w).mean()
        sigma = fr.rolling(w).std().replace(0, np.nan)
        out[f"funding_zscore_{w}h"] = (fr - mu) / sigma

    # ── Moving averages ──────────────────────────────────────
    out["funding_ma8"] = fr.rolling(8).mean()
    out["funding_ma24"] = fr.rolling(24).mean()
    out["funding_ma168"] = fr.rolling(168).mean()

    # ── Persistence: how many consecutive hours same sign ────
    sign = np.sign(fr)
    groups = (sign != sign.shift()).cumsum()
    out["funding_persistence"] = sign.groupby(groups).cumcount() + 1
    out["funding_persistence"] = (
        out["funding_persistence"] * sign
    )  # negative if negative funding

    # ── Extreme flags ────────────────────────────────────────
    q95 = fr.rolling(168).quantile(0.95)
    q05 = fr.rolling(168).quantile(0.05)
    out["funding_extreme_high"] = (fr > q95).astype(float)
    out["funding_extreme_low"] = (fr < q05).astype(float)

    # ── Cumulative funding (rolling sums) ────────────────────
    out["funding_cum_8h"] = fr.rolling(8).sum()
    out["funding_cum_24h"] = fr.rolling(24).sum()

    return out

"""Regime detection features and labels."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute regime classification features and flags.

    Uses price, funding, OI, liquidation, and flow features to identify:
    - trend_up, trend_down, mean_revert
    - crowded_long, crowded_short
    - panic_flush, squeeze
    """
    out = pd.DataFrame(index=df.index)

    close = (
        df["close"].astype(float)
        if "close" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    # ── Trend detection ──────────────────────────────────────
    ma24 = close.rolling(24).mean()
    ma168 = close.rolling(168).mean()
    ret_24 = close.pct_change(24)

    out["regime_trend_up"] = ((close > ma24) & (ma24 > ma168) & (ret_24 > 0.02)).astype(
        float
    )
    out["regime_trend_down"] = (
        (close < ma24) & (ma24 < ma168) & (ret_24 < -0.02)
    ).astype(float)

    # Mean reversion: price far from MA but reverting
    price_vs_ma24 = (close - ma24) / ma24.replace(0, np.nan)
    out["regime_mean_revert"] = (price_vs_ma24.abs() > 0.03).astype(float) * (
        1 - out["regime_trend_up"] - out["regime_trend_down"]
    ).clip(0, 1)

    # ── Crowding detection ───────────────────────────────────
    ls_ratio = df.get("long_short_ratio", pd.Series(1.0, index=df.index)).astype(float)
    funding = df.get(
        "funding_close", df.get("funding_rate", pd.Series(0.0, index=df.index))
    ).astype(float)

    ls_high = ls_ratio.rolling(168).quantile(0.9)
    ls_low = ls_ratio.rolling(168).quantile(0.1)
    funding_high = funding.rolling(168).quantile(0.9)

    out["regime_crowded_long"] = (
        (ls_ratio > ls_high) & (funding > funding_high)
    ).astype(float)

    out["regime_crowded_short"] = (
        (ls_ratio < ls_low) & (funding < funding.rolling(168).quantile(0.1))
    ).astype(float)

    # ── Panic flush ──────────────────────────────────────────
    total_liq = df.get("total_liquidations_usd", pd.Series(0.0, index=df.index)).astype(
        float
    )
    liq_shock = total_liq / total_liq.rolling(168).mean().replace(0, np.nan)

    out["regime_panic_flush"] = ((ret_24 < -0.05) & (liq_shock > 3.0)).astype(float)

    # ── Squeeze detection ────────────────────────────────────
    # Short squeeze: price up + high short liquidations + funding going positive
    long_liq = df.get("long_liquidations_usd", pd.Series(0.0, index=df.index)).astype(
        float
    )
    short_liq = df.get("short_liquidations_usd", pd.Series(0.0, index=df.index)).astype(
        float
    )
    liq_denom = (long_liq + short_liq).replace(0, np.nan)
    short_liq_pct = short_liq / liq_denom

    out["regime_squeeze"] = (
        (ret_24 > 0.03) & (short_liq_pct > 0.7) & (funding.diff(8) > 0)
    ).astype(float)

    # ── Composite regime label ───────────────────────────────
    # Priority: panic_flush > squeeze > crowded > trend > mean_revert > neutral
    def _assign_regime(row: pd.Series) -> str:
        if row.get("regime_panic_flush", 0) > 0.5:
            return "panic_flush"
        if row.get("regime_squeeze", 0) > 0.5:
            return "squeeze"
        if row.get("regime_crowded_long", 0) > 0.5:
            return "crowded_long"
        if row.get("regime_crowded_short", 0) > 0.5:
            return "crowded_short"
        if row.get("regime_trend_up", 0) > 0.5:
            return "trend_up"
        if row.get("regime_trend_down", 0) > 0.5:
            return "trend_down"
        if row.get("regime_mean_revert", 0) > 0.5:
            return "mean_revert"
        return "neutral"

    out["regime_label"] = out.apply(_assign_regime, axis=1)

    return out

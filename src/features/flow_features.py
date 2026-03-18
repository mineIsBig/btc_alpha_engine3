"""Flow features: long/short ratios, taker buy/sell, top trader ratios."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute flow-based features.

    Expects columns from merged data:
    - long_ratio, short_ratio, long_short_ratio  (from long/short table)
    - buy_volume, sell_volume, buy_sell_ratio  (from taker flow table)
    """
    out = pd.DataFrame(index=df.index)

    # ── Long/Short Ratio Features ────────────────────────────
    ls_ratio = df.get("long_short_ratio", pd.Series(1.0, index=df.index)).astype(float)

    out["ls_ratio"] = ls_ratio
    out["ls_ratio_change_1h"] = ls_ratio.diff(1)
    out["ls_ratio_change_4h"] = ls_ratio.diff(4)
    out["ls_ratio_change_24h"] = ls_ratio.diff(24)

    for w in [24, 72, 168]:
        mu = ls_ratio.rolling(w).mean()
        sigma = ls_ratio.rolling(w).std().replace(0, np.nan)
        out[f"ls_zscore_{w}h"] = (ls_ratio - mu) / sigma

    out["ls_extreme_long"] = (ls_ratio > ls_ratio.rolling(168).quantile(0.95)).astype(float)
    out["ls_extreme_short"] = (ls_ratio < ls_ratio.rolling(168).quantile(0.05)).astype(float)

    # ── Taker Buy/Sell Features ──────────────────────────────
    buy_vol = df.get("buy_volume", pd.Series(0.0, index=df.index)).astype(float)
    sell_vol = df.get("sell_volume", pd.Series(0.0, index=df.index)).astype(float)
    bs_ratio = df.get("buy_sell_ratio", pd.Series(1.0, index=df.index)).astype(float)

    out["taker_bs_ratio"] = bs_ratio
    out["taker_bs_change_1h"] = bs_ratio.diff(1)
    out["taker_bs_change_4h"] = bs_ratio.diff(4)

    # Net taker flow
    total = (buy_vol + sell_vol).replace(0, np.nan)
    out["taker_net_flow"] = (buy_vol - sell_vol) / total

    for w in [4, 12, 24]:
        out[f"taker_net_flow_{w}h_ma"] = out["taker_net_flow"].rolling(w).mean()

    # Z-score
    for w in [24, 72, 168]:
        mu = bs_ratio.rolling(w).mean()
        sigma = bs_ratio.rolling(w).std().replace(0, np.nan)
        out[f"taker_bs_zscore_{w}h"] = (bs_ratio - mu) / sigma

    # ── Divergence: taker flow vs price ──────────────────────
    if "close" in df.columns:
        price_ret = df["close"].astype(float).pct_change(4)
        out["taker_price_divergence_4h"] = out["taker_net_flow"].rolling(4).mean() - price_ret.clip(-1, 1)

    return out

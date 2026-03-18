"""Label generation for supervised learning.

Builds forward-return labels, ternary side labels, MFE/MAE,
and optional triple-barrier meta-labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logging import get_logger

logger = get_logger(__name__)

HORIZONS = [1, 4, 8, 12, 24]


def build_labels(
    price_df: pd.DataFrame,
    horizons: list[int] | None = None,
    threshold_bps: float = 10.0,
    cost_bps: float = 7.0,
) -> pd.DataFrame:
    """Build forward-return labels for all horizons.

    Args:
        price_df: DataFrame with 'timestamp', 'close', 'high', 'low' columns
        horizons: list of forward-looking horizons in hours
        threshold_bps: minimum return in bps to assign +1/-1 (else 0)
        cost_bps: round-trip cost in bps to subtract from returns

    Returns:
        DataFrame with columns for each horizon:
        - fwd_ret_{h}h: forward return
        - label_{h}h: ternary label (-1, 0, 1)
        - mfe_{h}h: maximum favorable excursion
        - mae_{h}h: maximum adverse excursion
    """
    if horizons is None:
        horizons = HORIZONS

    price_df = price_df.copy()
    price_df = price_df.sort_values("timestamp").reset_index(drop=True)
    close = price_df["close"].astype(float)
    high = price_df["high"].astype(float)
    low = price_df["low"].astype(float)

    result = price_df[["timestamp"]].copy()
    threshold = threshold_bps / 10000.0
    cost = cost_bps / 10000.0

    for h in horizons:
        # Forward return
        fwd_ret = close.shift(-h) / close - 1.0
        result[f"fwd_ret_{h}h"] = fwd_ret

        # Net return after costs
        net_ret = fwd_ret - cost

        # Ternary label
        labels = pd.Series(0, index=price_df.index, dtype=int)
        labels[net_ret > threshold] = 1
        labels[net_ret < -threshold] = -1
        result[f"label_{h}h"] = labels

        # MFE/MAE: max favorable/adverse excursion within horizon
        mfe = pd.Series(np.nan, index=price_df.index)
        mae = pd.Series(np.nan, index=price_df.index)

        for i in range(len(price_df) - h):
            window_high = high.iloc[i + 1:i + h + 1].max()
            window_low = low.iloc[i + 1:i + h + 1].min()
            entry = close.iloc[i]

            mfe.iloc[i] = (window_high / entry - 1.0)
            mae.iloc[i] = (window_low / entry - 1.0)

        result[f"mfe_{h}h"] = mfe
        result[f"mae_{h}h"] = mae

    logger.info("labels_built", horizons=horizons, rows=len(result))
    return result


def build_triple_barrier_labels(
    price_df: pd.DataFrame,
    horizon: int = 24,
    take_profit_bps: float = 100.0,
    stop_loss_bps: float = 50.0,
    cost_bps: float = 7.0,
) -> pd.DataFrame:
    """Build triple-barrier labels.

    Three barriers:
    1. Take profit: price moves up by take_profit_bps
    2. Stop loss: price moves down by stop_loss_bps
    3. Time expiry: horizon reached

    Returns:
        DataFrame with timestamp, barrier_label (-1, 0, 1), barrier_type, barrier_time
    """
    price_df = price_df.copy().sort_values("timestamp").reset_index(drop=True)
    close = price_df["close"].astype(float).values
    high = price_df["high"].astype(float).values
    low = price_df["low"].astype(float).values

    tp = take_profit_bps / 10000.0
    sl = stop_loss_bps / 10000.0

    labels = []
    for i in range(len(price_df) - horizon):
        entry = close[i]
        tp_price = entry * (1 + tp)
        sl_price = entry * (1 - sl)

        barrier_label = 0
        barrier_type = "time"
        barrier_bar = horizon

        for j in range(1, horizon + 1):
            if high[i + j] >= tp_price:
                barrier_label = 1
                barrier_type = "take_profit"
                barrier_bar = j
                break
            if low[i + j] <= sl_price:
                barrier_label = -1
                barrier_type = "stop_loss"
                barrier_bar = j
                break

        # If time barrier hit, label by sign of final return
        if barrier_type == "time":
            final_ret = close[i + horizon] / entry - 1.0 - cost_bps / 10000.0
            barrier_label = int(np.sign(final_ret)) if abs(final_ret) > cost_bps / 10000.0 else 0

        labels.append({
            "timestamp": price_df["timestamp"].iloc[i],
            "barrier_label": barrier_label,
            "barrier_type": barrier_type,
            "barrier_bar": barrier_bar,
        })

    # Pad remaining rows with NaN
    for i in range(len(price_df) - horizon, len(price_df)):
        labels.append({
            "timestamp": price_df["timestamp"].iloc[i],
            "barrier_label": np.nan,
            "barrier_type": None,
            "barrier_bar": np.nan,
        })

    return pd.DataFrame(labels)

"""Price-derived features: returns, volatility, trend indicators."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute price-based features from OHLC data.

    Expects columns: close, high, low, open, volume
    Returns DataFrame with feature columns (no timestamp).
    """
    out = pd.DataFrame(index=df.index)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # ── Returns at multiple horizons ─────────────────────────
    for h in [1, 4, 8, 12, 24]:
        out[f"ret_{h}h"] = close.pct_change(h)

    # ── Log returns ──────────────────────────────────────────
    out["log_ret_1h"] = np.log(close / close.shift(1))

    # ── Realized volatility ──────────────────────────────────
    log_ret = out["log_ret_1h"]
    for w in [6, 12, 24, 48, 168]:
        out[f"rvol_{w}h"] = log_ret.rolling(w).std() * np.sqrt(w)

    # ── High-low range ───────────────────────────────────────
    out["hl_range"] = (high - low) / close
    out["hl_range_ma24"] = out["hl_range"].rolling(24).mean()

    # ── Price momentum / trend ───────────────────────────────
    for w in [12, 24, 48, 168]:
        ma = close.rolling(w).mean()
        out[f"price_vs_ma{w}"] = (close - ma) / ma

    # ── RSI (14-period) ──────────────────────────────────────
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # ── Price acceleration ───────────────────────────────────
    out["ret_accel_4h"] = out["ret_4h"] - out["ret_4h"].shift(4)

    # ── Volume features ──────────────────────────────────────
    if "volume" in df.columns:
        vol = df["volume"].astype(float)
        out["volume_ma24_ratio"] = vol / vol.rolling(24).mean().replace(0, np.nan)
        out["volume_ma168_ratio"] = vol / vol.rolling(168).mean().replace(0, np.nan)

    return out

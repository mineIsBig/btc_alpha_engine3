#!/usr/bin/env python3
"""Download historical BTC derivatives data from Coinalyze API.

Fetches all 6 data types needed for backtesting:
  1. Funding Rate (OHLC)
  2. Open Interest (OHLC)
  3. Liquidations
  4. Long/Short Ratio
  5. Taker Buy/Sell Volume
  6. Price OHLC (from Binance spot via public API)

Usage:
  # First, get your free API key from https://coinalyze.net (sign up -> account settings)
  # Then set it as an environment variable:
  set COINALYZE_API_KEY=your_key_here

  python scripts/download_coinalyze_data.py
  python scripts/download_coinalyze_data.py --interval 1hour --output-dir data/historical
  python scripts/download_coinalyze_data.py --interval daily   # daily has unlimited history

Notes:
  - Coinalyze keeps only ~1500-2000 datapoints for intraday (1min-12hour).
    For 1hour that is roughly 60-80 days of data.
  - Daily interval has NO data deletion -- unlimited history.
  - Rate limit: 40 API calls/minute (handled automatically).
  - Price data is fetched from Binance public API (no key needed).
"""
from __future__ import annotations

import os
import sys
import time
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx
import pandas as pd

# ---------------------------------------------------------------------------
# Coinalyze API config
# ---------------------------------------------------------------------------
COINALYZE_BASE = "https://api.coinalyze.net/v1"

# BTC perpetual symbols on major exchanges (Coinalyze format)
# .A = Binance, .6 = Bybit, .7 = OKX, .B = dYdX, .D = Bitget
BTC_SYMBOLS = {
    "binance": "BTCUSDT_PERP.A",
    "bybit": "BTCUSDT_PERP.6",
    "okx": "BTCUSD_PERP.7",
}

# Default: aggregate across Binance + Bybit + OKX
DEFAULT_SYMBOLS = ",".join(BTC_SYMBOLS.values())

COINALYZE_ENDPOINTS = {
    "funding_rate": "/funding-rate-history",
    "open_interest": "/open-interest-history",
    "liquidation": "/liquidation-history",
    "long_short_ratio": "/long-short-ratio-history",
    # predicted funding rate can also be useful
    "predicted_funding": "/predicted-funding-rate-history",
}

# Binance public klines endpoint (no API key needed)
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

INTERVAL_MAP = {
    "1hour": "1hour",
    "4hour": "4hour",
    "daily": "daily",
    "1h": "1hour",
    "4h": "4hour",
    "1d": "daily",
}

BINANCE_INTERVAL_MAP = {
    "1hour": "1h",
    "4hour": "4h",
    "daily": "1d",
}


def get_api_key() -> str:
    key = os.environ.get("COINALYZE_API_KEY", "")
    if not key:
        print("ERROR: COINALYZE_API_KEY environment variable not set.")
        print("  Get your free key at https://coinalyze.net (sign up -> account settings)")
        print("  Then: set COINALYZE_API_KEY=your_key_here")
        sys.exit(1)
    return key


class RateLimiter:
    """Simple sliding-window rate limiter (40 calls/min)."""

    def __init__(self, max_calls: int = 38, window: float = 60.0):
        self.max_calls = max_calls
        self.window = window
        self._times: list[float] = []

    def wait(self) -> None:
        now = time.monotonic()
        self._times = [t for t in self._times if now - t < self.window]
        if len(self._times) >= self.max_calls:
            sleep_for = self.window - (now - self._times[0]) + 1.0
            print(f"  Rate limit reached, sleeping {sleep_for:.1f}s...")
            time.sleep(sleep_for)
        self._times.append(time.monotonic())


rate_limiter = RateLimiter()


# ---------------------------------------------------------------------------
# Coinalyze fetchers
# ---------------------------------------------------------------------------
def fetch_coinalyze(
    client: httpx.Client,
    endpoint: str,
    symbols: str,
    interval: str,
    from_ts: int | None = None,
    to_ts: int | None = None,
) -> list[dict]:
    """Fetch data from a Coinalyze API endpoint."""
    rate_limiter.wait()

    params: dict = {
        "symbols": symbols,
        "interval": interval,
    }
    if from_ts:
        params["from"] = from_ts
    if to_ts:
        params["to"] = to_ts

    url = f"{COINALYZE_BASE}{endpoint}"
    resp = client.get(url, params=params)

    if resp.status_code == 429:
        retry_after = int(resp.headers.get("Retry-After", "60"))
        print(f"  429 Rate limited. Waiting {retry_after}s...")
        time.sleep(retry_after + 1)
        rate_limiter.wait()
        resp = client.get(url, params=params)

    if resp.status_code == 401:
        print("ERROR: Invalid API key. Check your COINALYZE_API_KEY.")
        sys.exit(1)

    resp.raise_for_status()
    return resp.json()


def parse_coinalyze_ohlc(data: list[dict], label: str) -> pd.DataFrame:
    """Parse OHLC-style response (funding rate, open interest, predicted funding)."""
    rows = []
    for item in data:
        symbol = item.get("symbol", "unknown")
        exchange = symbol.split(".")[-1] if "." in symbol else "agg"
        for h in item.get("history", []):
            rows.append({
                "timestamp": pd.to_datetime(h["t"], unit="s", utc=True),
                "symbol": symbol,
                "exchange": exchange,
                f"{label}_open": h.get("o", 0),
                f"{label}_high": h.get("h", 0),
                f"{label}_low": h.get("l", 0),
                f"{label}_close": h.get("c", 0),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def parse_coinalyze_volume(data: list[dict], label: str) -> pd.DataFrame:
    """Parse volume-style response (liquidations, long/short ratio)."""
    rows = []
    for item in data:
        symbol = item.get("symbol", "unknown")
        exchange = symbol.split(".")[-1] if "." in symbol else "agg"
        for h in item.get("history", []):
            row = {
                "timestamp": pd.to_datetime(h["t"], unit="s", utc=True),
                "symbol": symbol,
                "exchange": exchange,
            }
            # Different endpoints return different fields
            for k, v in h.items():
                if k != "t":
                    row[f"{label}_{k}"] = v
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def download_funding_rate(client: httpx.Client, symbols: str, interval: str) -> pd.DataFrame:
    print("  [1/6] Downloading funding rate history...")
    data = fetch_coinalyze(client, COINALYZE_ENDPOINTS["funding_rate"], symbols, interval)
    return parse_coinalyze_ohlc(data, "funding_rate")


def download_open_interest(client: httpx.Client, symbols: str, interval: str) -> pd.DataFrame:
    print("  [2/6] Downloading open interest history...")
    data = fetch_coinalyze(client, COINALYZE_ENDPOINTS["open_interest"], symbols, interval)
    return parse_coinalyze_ohlc(data, "oi")


def download_liquidations(client: httpx.Client, symbols: str, interval: str) -> pd.DataFrame:
    print("  [3/6] Downloading liquidation history...")
    data = fetch_coinalyze(client, COINALYZE_ENDPOINTS["liquidation"], symbols, interval)
    return parse_coinalyze_volume(data, "liq")


def download_long_short_ratio(client: httpx.Client, symbols: str, interval: str) -> pd.DataFrame:
    print("  [4/6] Downloading long/short ratio history...")
    data = fetch_coinalyze(client, COINALYZE_ENDPOINTS["long_short_ratio"], symbols, interval)
    return parse_coinalyze_volume(data, "lsr")


def download_predicted_funding(client: httpx.Client, symbols: str, interval: str) -> pd.DataFrame:
    print("  [5/6] Downloading predicted funding rate history...")
    data = fetch_coinalyze(client, COINALYZE_ENDPOINTS["predicted_funding"], symbols, interval)
    return parse_coinalyze_ohlc(data, "pred_funding")


# ---------------------------------------------------------------------------
# Binance price data (no API key needed)
# ---------------------------------------------------------------------------
def download_binance_price(interval: str, days_back: int = 90) -> pd.DataFrame:
    """Download BTC/USDT price OHLCV from Binance public API."""
    print("  [6/6] Downloading BTC/USDT price OHLCV from Binance...")
    binance_interval = BINANCE_INTERVAL_MAP.get(interval, "1h")

    all_rows = []
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp() * 1000)

    current_start = start_ms
    with httpx.Client(timeout=30.0) as client:
        while current_start < end_ms:
            params = {
                "symbol": "BTCUSDT",
                "interval": binance_interval,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": 1000,
            }
            resp = client.get(BINANCE_KLINES, params=params)
            resp.raise_for_status()
            klines = resp.json()

            if not klines:
                break

            for k in klines:
                all_rows.append({
                    "timestamp": pd.to_datetime(int(k[0]), unit="ms", utc=True),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "quote_volume": float(k[7]),
                    "trades": int(k[8]),
                    "taker_buy_base": float(k[9]),
                    "taker_buy_quote": float(k[10]),
                })

            # Move start forward past last candle
            last_ts = int(klines[-1][0])
            if last_ts <= current_start:
                break
            current_start = last_ts + 1

            # Binance rate limit is generous, but be polite
            time.sleep(0.2)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    print(f"    -> {len(df)} price candles downloaded")
    return df


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def aggregate_by_timestamp(df: pd.DataFrame, value_cols: list[str], method: str = "mean") -> pd.DataFrame:
    """Aggregate multi-exchange data to single time series."""
    if df.empty:
        return df

    agg_funcs = {}
    for col in value_cols:
        if col in df.columns:
            agg_funcs[col] = method

    if not agg_funcs:
        return df

    agg = df.groupby("timestamp").agg(agg_funcs).reset_index()
    agg = agg.sort_values("timestamp").reset_index(drop=True)
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Download BTC historical data from Coinalyze + Binance")
    parser.add_argument("--interval", default="1hour", choices=list(INTERVAL_MAP.keys()),
                        help="Data interval (default: 1hour)")
    parser.add_argument("--output-dir", default="data/historical",
                        help="Output directory for CSV files (default: data/historical)")
    parser.add_argument("--symbols", default=None,
                        help=f"Coinalyze symbols (default: {DEFAULT_SYMBOLS})")
    parser.add_argument("--price-days", type=int, default=90,
                        help="Days of price history to download from Binance (default: 90)")
    parser.add_argument("--skip-price", action="store_true",
                        help="Skip Binance price download")
    parser.add_argument("--aggregate", action="store_true", default=True,
                        help="Also save aggregated (cross-exchange mean) CSVs")
    args = parser.parse_args()

    interval = INTERVAL_MAP[args.interval]
    symbols = args.symbols or DEFAULT_SYMBOLS
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = get_api_key()

    print(f"=== BTC Historical Data Download ===")
    print(f"  Interval:  {interval}")
    print(f"  Symbols:   {symbols}")
    print(f"  Output:    {out_dir.resolve()}")
    print(f"  Note: Intraday data limited to ~1500-2000 datapoints (~60-80 days for 1h)")
    print()

    client = httpx.Client(
        timeout=30.0,
        headers={
            "accept": "application/json",
            "api_key": api_key,
        },
    )

    results = {}

    try:
        # 1. Funding Rate
        df_funding = download_funding_rate(client, symbols, interval)
        if not df_funding.empty:
            path = out_dir / "btc_funding_rate_raw.csv"
            df_funding.to_csv(path, index=False)
            print(f"    -> {len(df_funding)} rows saved to {path}")
            results["funding_rate"] = len(df_funding)

            if args.aggregate:
                fr_cols = [c for c in df_funding.columns if c.startswith("funding_rate_")]
                agg = aggregate_by_timestamp(df_funding, fr_cols)
                agg_path = out_dir / "btc_funding_rate.csv"
                agg.to_csv(agg_path, index=False)
                print(f"    -> Aggregated: {len(agg)} rows to {agg_path}")
        else:
            print("    -> No funding rate data returned")
            results["funding_rate"] = 0

        # 2. Open Interest
        df_oi = download_open_interest(client, symbols, interval)
        if not df_oi.empty:
            path = out_dir / "btc_open_interest_raw.csv"
            df_oi.to_csv(path, index=False)
            print(f"    -> {len(df_oi)} rows saved to {path}")
            results["open_interest"] = len(df_oi)

            if args.aggregate:
                oi_cols = [c for c in df_oi.columns if c.startswith("oi_")]
                # For OI, sum across exchanges instead of mean
                agg = aggregate_by_timestamp(df_oi, oi_cols, method="sum")
                agg_path = out_dir / "btc_open_interest.csv"
                agg.to_csv(agg_path, index=False)
                print(f"    -> Aggregated: {len(agg)} rows to {agg_path}")
        else:
            print("    -> No open interest data returned")
            results["open_interest"] = 0

        # 3. Liquidations
        df_liq = download_liquidations(client, symbols, interval)
        if not df_liq.empty:
            path = out_dir / "btc_liquidations_raw.csv"
            df_liq.to_csv(path, index=False)
            print(f"    -> {len(df_liq)} rows saved to {path}")
            results["liquidations"] = len(df_liq)

            if args.aggregate:
                liq_cols = [c for c in df_liq.columns if c.startswith("liq_")]
                agg = aggregate_by_timestamp(df_liq, liq_cols, method="sum")
                agg_path = out_dir / "btc_liquidations.csv"
                agg.to_csv(agg_path, index=False)
                print(f"    -> Aggregated: {len(agg)} rows to {agg_path}")
        else:
            print("    -> No liquidation data returned")
            results["liquidations"] = 0

        # 4. Long/Short Ratio
        df_lsr = download_long_short_ratio(client, symbols, interval)
        if not df_lsr.empty:
            path = out_dir / "btc_long_short_ratio_raw.csv"
            df_lsr.to_csv(path, index=False)
            print(f"    -> {len(df_lsr)} rows saved to {path}")
            results["long_short_ratio"] = len(df_lsr)

            if args.aggregate:
                lsr_cols = [c for c in df_lsr.columns if c.startswith("lsr_")]
                agg = aggregate_by_timestamp(df_lsr, lsr_cols)
                agg_path = out_dir / "btc_long_short_ratio.csv"
                agg.to_csv(agg_path, index=False)
                print(f"    -> Aggregated: {len(agg)} rows to {agg_path}")
        else:
            print("    -> No long/short ratio data returned")
            results["long_short_ratio"] = 0

        # 5. Predicted Funding Rate
        df_pred = download_predicted_funding(client, symbols, interval)
        if not df_pred.empty:
            path = out_dir / "btc_predicted_funding_raw.csv"
            df_pred.to_csv(path, index=False)
            print(f"    -> {len(df_pred)} rows saved to {path}")
            results["predicted_funding"] = len(df_pred)
        else:
            print("    -> No predicted funding data returned")
            results["predicted_funding"] = 0

    finally:
        client.close()

    # 6. Price OHLCV from Binance (no API key needed)
    if not args.skip_price:
        df_price = download_binance_price(interval, days_back=args.price_days)
        if not df_price.empty:
            path = out_dir / "btc_price_ohlcv.csv"
            df_price.to_csv(path, index=False)
            print(f"    -> {len(df_price)} price rows saved to {path}")
            results["price_ohlcv"] = len(df_price)
        else:
            print("    -> No price data returned")
            results["price_ohlcv"] = 0

    # Summary
    print()
    print("=== Download Summary ===")
    total = 0
    for name, count in results.items():
        status = "OK" if count > 0 else "EMPTY"
        print(f"  {name:25s} {count:>6} rows  [{status}]")
        total += count
    print(f"  {'TOTAL':25s} {total:>6} rows")
    print()
    print(f"Files saved to: {out_dir.resolve()}")

    if interval in ("1hour", "4hour"):
        print()
        print("NOTE: Coinalyze keeps only ~1500-2000 intraday datapoints.")
        print("  For 1h data, that is roughly 60-80 days.")
        print("  For longer history, use --interval daily (unlimited retention).")
        print("  You can also supplement with Binance price data (--price-days 365).")


if __name__ == "__main__":
    main()

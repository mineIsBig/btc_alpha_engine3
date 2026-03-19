#!/usr/bin/env python3
"""Download 2+ years of BTC historical data for backtesting.

Uses Binance public APIs (NO API key needed) to fetch:
  1. Price OHLCV (1h) -- Binance Futures /fapi/v1/klines -- unlimited history
  2. Funding Rate    -- Binance Futures /fapi/v1/fundingRate -- unlimited history (8h intervals)
  3. Open Interest   -- Binance /futures/data/openInterestHist -- 30 days only (1h)
  4. Long/Short Ratio-- Binance /futures/data/globalLongShortAccountRatio -- 30 days (1h)
  5. Taker Buy/Sell  -- Binance /futures/data/takerlongshortRatio -- 30 days (1h)
  6. Daily Metrics   -- data.binance.vision ZIP archives -- unlimited history (daily)
     (includes OI, L/S ratios, taker volume -- forward-filled to 1h)

For derivatives data older than 30 days, daily metrics from data.binance.vision
are downloaded and forward-filled to 1h resolution.

Usage:
  python scripts/download_historical_data.py
  python scripts/download_historical_data.py --days 730 --output-dir data/historical
  python scripts/download_historical_data.py --days 365 --skip-daily-metrics

No API keys required. All endpoints are public.
"""
from __future__ import annotations

import io
import sys
import time
import zipfile
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx
import pandas as pd

# ---------------------------------------------------------------------------
# Binance endpoints (all public, no API key needed)
# ---------------------------------------------------------------------------
BINANCE_FAPI = "https://fapi.binance.com"
BINANCE_DATA = "https://data.binance.vision"

ENDPOINTS = {
    "klines":       f"{BINANCE_FAPI}/fapi/v1/klines",
    "funding_rate": f"{BINANCE_FAPI}/fapi/v1/fundingRate",
    "oi_hist":      f"{BINANCE_FAPI}/futures/data/openInterestHist",
    "ls_ratio":     f"{BINANCE_FAPI}/futures/data/globalLongShortAccountRatio",
    "taker_ratio":  f"{BINANCE_FAPI}/futures/data/takerlongshortRatio",
}

SYMBOL = "BTCUSDT"


class ProgressTracker:
    """Simple progress display."""

    def __init__(self, label: str, total: int | None = None):
        self.label = label
        self.total = total
        self.count = 0
        self.start_time = time.monotonic()

    def update(self, n: int = 1) -> None:
        self.count += n
        elapsed = time.monotonic() - self.start_time
        if self.total:
            pct = self.count / self.total * 100
            print(f"\r  {self.label}: {self.count}/{self.total} ({pct:.0f}%) [{elapsed:.0f}s]", end="", flush=True)
        else:
            print(f"\r  {self.label}: {self.count} records [{elapsed:.0f}s]", end="", flush=True)

    def done(self) -> None:
        elapsed = time.monotonic() - self.start_time
        print(f"\r  {self.label}: {self.count} records [{elapsed:.1f}s] -- DONE")


# ---------------------------------------------------------------------------
# 1. Price OHLCV (1h klines from Binance Futures)
# ---------------------------------------------------------------------------
def download_price_klines(client: httpx.Client, days: int) -> pd.DataFrame:
    """Download 1h BTCUSDT futures klines. Unlimited history via pagination."""
    print("[1/6] Price OHLCV (1h klines)...")
    progress = ProgressTracker("Klines", total=days * 24)

    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    current = start_ms
    rows = []

    while current < end_ms:
        resp = client.get(ENDPOINTS["klines"], params={
            "symbol": SYMBOL,
            "interval": "1h",
            "startTime": current,
            "endTime": end_ms,
            "limit": 1500,
        })
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break

        for k in data:
            rows.append({
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

        progress.update(len(data))
        last_ts = int(data[-1][0])
        if last_ts <= current:
            break
        current = last_ts + 1
        time.sleep(0.15)

    progress.done()
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. Funding Rate (8h intervals, unlimited history)
# ---------------------------------------------------------------------------
def download_funding_rate(client: httpx.Client, days: int) -> pd.DataFrame:
    """Download funding rate history. Binance funding is every 8h.
    Returns raw 8h data -- will be resampled to 1h later.
    """
    print("[2/6] Funding Rate (8h intervals, full history)...")
    expected_records = days * 3  # 3 funding events per day
    progress = ProgressTracker("Funding", total=expected_records)

    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    current = start_ms
    rows = []

    while current < end_ms:
        resp = client.get(ENDPOINTS["funding_rate"], params={
            "symbol": SYMBOL,
            "startTime": current,
            "endTime": end_ms,
            "limit": 1000,
        })
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break

        for item in data:
            rows.append({
                "timestamp": pd.to_datetime(int(item["fundingTime"]), unit="ms", utc=True),
                "funding_rate": float(item["fundingRate"]),
                "mark_price": float(item.get("markPrice", 0)),
            })

        progress.update(len(data))
        last_ts = int(data[-1]["fundingTime"])
        if last_ts <= current:
            break
        current = last_ts + 1
        time.sleep(0.15)

    progress.done()
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 3-5. Derivatives data (30 days only from Binance API)
# ---------------------------------------------------------------------------
def download_derivatives_30d(client: httpx.Client, endpoint: str, label: str,
                               step_num: int, parse_fn) -> pd.DataFrame:
    """Download recent 30 days of hourly derivatives data from Binance.

    These endpoints only keep ~30 days of data and may return 400
    for timestamps too far in the past.
    """
    print(f"[{step_num}/6] {label} (recent 30 days, 1h)...")
    progress = ProgressTracker(label, total=30 * 24)

    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=29)).timestamp() * 1000)
    current = start_ms
    rows = []

    while current < end_ms:
        params = {
            "symbol": SYMBOL,
            "period": "1h",
            "startTime": current,
            "limit": 500,
        }
        try:
            resp = client.get(endpoint, params=params)

            if resp.status_code == 429:
                print("\n  Rate limited, waiting 60s...")
                time.sleep(60)
                resp = client.get(endpoint, params=params)

            if resp.status_code == 400:
                # Try without startTime (just get latest data)
                if current == start_ms:
                    print(f"\n  Got 400 with timestamps, trying without startTime...")
                    resp2 = client.get(endpoint, params={
                        "symbol": SYMBOL, "period": "1h", "limit": 500
                    })
                    if resp2.status_code == 200:
                        data = resp2.json()
                        for item in data:
                            rows.append(parse_fn(item))
                        progress.update(len(data))
                    else:
                        print(f"\n  Endpoint returned {resp2.status_code}, skipping...")
                break

            if resp.status_code != 200:
                print(f"\n  HTTP {resp.status_code}, skipping remaining...")
                break

            data = resp.json()
            if not data:
                break

            for item in data:
                rows.append(parse_fn(item))

            progress.update(len(data))
            last_ts = int(data[-1].get("timestamp", data[-1].get("t", 0)))
            if last_ts <= current:
                break
            current = last_ts + 1
            time.sleep(0.3)

        except Exception as e:
            print(f"\n  Error: {e}, skipping...")
            break

    progress.done()
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


def parse_oi(item: dict) -> dict:
    return {
        "timestamp": pd.to_datetime(int(item["timestamp"]), unit="ms", utc=True),
        "open_interest": float(item.get("sumOpenInterest", 0)),
        "open_interest_value": float(item.get("sumOpenInterestValue", 0)),
    }


def parse_ls_ratio(item: dict) -> dict:
    return {
        "timestamp": pd.to_datetime(int(item["timestamp"]), unit="ms", utc=True),
        "long_account": float(item.get("longAccount", 0.5)),
        "short_account": float(item.get("shortAccount", 0.5)),
        "long_short_ratio": float(item.get("longShortRatio", 1.0)),
    }


def parse_taker(item: dict) -> dict:
    return {
        "timestamp": pd.to_datetime(int(item["timestamp"]), unit="ms", utc=True),
        "buy_sell_ratio": float(item.get("buySellRatio", 1.0)),
        "buy_vol": float(item.get("buyVol", 0)),
        "sell_vol": float(item.get("sellVol", 0)),
    }


# ---------------------------------------------------------------------------
# 6. Daily metrics from data.binance.vision (unlimited history)
# ---------------------------------------------------------------------------
def download_daily_metrics(days: int) -> pd.DataFrame:
    """Download daily BTCUSDT metrics ZIPs from data.binance.vision.

    Each ZIP contains a CSV with columns:
      create_time, symbol, sum_open_interest, sum_open_interest_value,
      count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio,
      count_long_short_ratio, sum_taker_long_short_vol_ratio
    """
    print("[6/6] Daily metrics from data.binance.vision (OI, L/S, taker)...")

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days)

    # Generate list of dates
    dates = []
    d = start_date
    while d <= end_date:
        dates.append(d)
        d += timedelta(days=1)

    progress = ProgressTracker("Daily metrics", total=len(dates))
    rows = []
    failed = 0

    with httpx.Client(timeout=30.0) as client:
        for date in dates:
            date_str = date.strftime("%Y-%m-%d")
            url = f"{BINANCE_DATA}/data/futures/um/daily/metrics/{SYMBOL}/{SYMBOL}-metrics-{date_str}.zip"

            try:
                resp = client.get(url)
                if resp.status_code == 200:
                    # Extract CSV from ZIP
                    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                        for name in zf.namelist():
                            if name.endswith(".csv"):
                                csv_data = zf.read(name).decode("utf-8")
                                df_day = pd.read_csv(io.StringIO(csv_data))
                                for _, row in df_day.iterrows():
                                    rows.append({
                                        "timestamp": pd.to_datetime(date_str, utc=True),
                                        "daily_open_interest": float(row.get("sum_open_interest", 0) or 0),
                                        "daily_oi_value": float(row.get("sum_open_interest_value", 0) or 0),
                                        "daily_top_ls_ratio": float(row.get("sum_toptrader_long_short_ratio", 0) or 0),
                                        "daily_ls_ratio": float(row.get("count_long_short_ratio", 0) or 0),
                                        "daily_taker_ls_ratio": float(row.get("sum_taker_long_short_vol_ratio", 0) or 0),
                                    })
                elif resp.status_code == 404:
                    failed += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

            progress.update()
            time.sleep(0.05)  # Be polite to the data server

    progress.done()
    if failed > 0:
        print(f"  ({failed} dates had no data or failed)")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Merge and forward-fill to 1h
# ---------------------------------------------------------------------------
def merge_all(
    df_price: pd.DataFrame,
    df_funding: pd.DataFrame,
    df_oi_30d: pd.DataFrame,
    df_ls_30d: pd.DataFrame,
    df_taker_30d: pd.DataFrame,
    df_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all datasets onto the 1h price index."""
    print("\nMerging all datasets...")
    df = df_price.copy()

    # --- Funding rate: resample 8h -> 1h via forward fill ---
    if not df_funding.empty:
        df = df.merge(df_funding[["timestamp", "funding_rate", "mark_price"]],
                      on="timestamp", how="left")
        df["funding_rate"] = df["funding_rate"].ffill()
        df["mark_price"] = df["mark_price"].ffill()
        print(f"  Merged funding rate: {df_funding['timestamp'].min()} to {df_funding['timestamp'].max()}")

    # --- Recent 30d OI (hourly) ---
    if not df_oi_30d.empty:
        df = df.merge(df_oi_30d[["timestamp", "open_interest", "open_interest_value"]],
                      on="timestamp", how="left")
        print(f"  Merged 30d OI: {df_oi_30d['timestamp'].min()} to {df_oi_30d['timestamp'].max()}")

    # --- Recent 30d L/S ratio (hourly) ---
    if not df_ls_30d.empty:
        df = df.merge(df_ls_30d[["timestamp", "long_account", "short_account", "long_short_ratio"]],
                      on="timestamp", how="left")
        print(f"  Merged 30d L/S: {df_ls_30d['timestamp'].min()} to {df_ls_30d['timestamp'].max()}")

    # --- Recent 30d taker (hourly) ---
    if not df_taker_30d.empty:
        df = df.merge(df_taker_30d[["timestamp", "buy_sell_ratio", "buy_vol", "sell_vol"]],
                      on="timestamp", how="left")
        print(f"  Merged 30d taker: {df_taker_30d['timestamp'].min()} to {df_taker_30d['timestamp'].max()}")

    # --- Daily metrics: forward-fill to hourly for older data ---
    if not df_daily.empty:
        # Create hourly index from daily data
        df_daily_expanded = df_daily.set_index("timestamp").resample("1h").ffill().reset_index()
        # Only fill where we don't already have hourly data
        for col in ["daily_open_interest", "daily_oi_value", "daily_top_ls_ratio",
                     "daily_ls_ratio", "daily_taker_ls_ratio"]:
            if col in df_daily_expanded.columns:
                df = df.merge(df_daily_expanded[["timestamp", col]], on="timestamp", how="left")

        # Use daily OI where hourly OI is missing
        if "open_interest" in df.columns and "daily_open_interest" in df.columns:
            df["open_interest"] = df["open_interest"].fillna(df["daily_open_interest"])
        elif "daily_open_interest" in df.columns:
            df["open_interest"] = df["daily_open_interest"]

        # Use daily L/S ratio where hourly is missing
        if "long_short_ratio" in df.columns and "daily_ls_ratio" in df.columns:
            df["long_short_ratio"] = df["long_short_ratio"].fillna(df["daily_ls_ratio"])
        elif "daily_ls_ratio" in df.columns:
            df["long_short_ratio"] = df["daily_ls_ratio"]

        # Use daily taker ratio where hourly is missing
        if "buy_sell_ratio" in df.columns and "daily_taker_ls_ratio" in df.columns:
            df["buy_sell_ratio"] = df["buy_sell_ratio"].fillna(df["daily_taker_ls_ratio"])
        elif "daily_taker_ls_ratio" in df.columns:
            df["buy_sell_ratio"] = df["daily_taker_ls_ratio"]

        print(f"  Merged daily metrics: {df_daily['timestamp'].min()} to {df_daily['timestamp'].max()}")

    # Forward-fill remaining gaps
    fill_cols = [c for c in df.columns if c not in ("timestamp",)]
    df[fill_cols] = df[fill_cols].ffill()

    # Drop daily helper columns
    drop_cols = [c for c in df.columns if c.startswith("daily_")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


# ---------------------------------------------------------------------------
# Synthetic liquidation estimate (no historical source available)
# ---------------------------------------------------------------------------
def estimate_liquidations(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate liquidation intensity from price volatility.

    Since Binance only provides 7 days of liquidation history,
    we create a proxy based on:
      - Absolute hourly returns (large moves = more liquidations)
      - Volume spikes
      - OI changes (large OI drops often coincide with liquidation cascades)

    This is a rough proxy, not actual liquidation data.
    """
    if "close" not in df.columns:
        return df

    print("  Estimating liquidation proxy from volatility...")
    df = df.copy()

    # Hourly absolute return
    df["abs_return"] = df["close"].pct_change().abs()

    # Volatility z-score (rolling 24h)
    rolling_mean = df["abs_return"].rolling(24, min_periods=1).mean()
    rolling_std = df["abs_return"].rolling(24, min_periods=1).std().replace(0, 1e-8)
    df["vol_zscore"] = (df["abs_return"] - rolling_mean) / rolling_std

    # Liquidation proxy: scaled by volume and volatility
    if "volume" in df.columns:
        vol_ratio = df["volume"] / df["volume"].rolling(24, min_periods=1).mean().replace(0, 1)
    else:
        vol_ratio = 1.0

    # High vol z-score + high volume = more liquidations
    df["liquidation_intensity"] = (df["vol_zscore"].clip(lower=0) * vol_ratio).fillna(0)

    # Normalize to 0-1 range
    li_max = df["liquidation_intensity"].quantile(0.99)
    if li_max > 0:
        df["liquidation_intensity"] = (df["liquidation_intensity"] / li_max).clip(0, 1)

    # Estimate long vs short liquidations based on price direction
    price_change = df["close"].pct_change().fillna(0)
    df["est_long_liq"] = df["liquidation_intensity"] * (price_change < 0).astype(float)
    df["est_short_liq"] = df["liquidation_intensity"] * (price_change > 0).astype(float)

    # Clean up intermediate columns
    df = df.drop(columns=["abs_return", "vol_zscore"], errors="ignore")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download 2+ years of BTC historical data for backtesting (no API key needed)"
    )
    parser.add_argument("--days", type=int, default=730,
                        help="Days of history to download (default: 730 = 2 years)")
    parser.add_argument("--output-dir", default="data/historical",
                        help="Output directory (default: data/historical)")
    parser.add_argument("--skip-daily-metrics", action="store_true",
                        help="Skip downloading daily metrics ZIPs from data.binance.vision")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip merging -- only download individual CSVs")
    parser.add_argument("--no-liquidation-proxy", action="store_true",
                        help="Don't generate synthetic liquidation estimates")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    days = args.days

    print("=" * 60)
    print("  BTC Historical Data Downloader")
    print("  No API keys required (Binance public endpoints)")
    print("=" * 60)
    print(f"  Period:  {days} days ({days/365:.1f} years)")
    print(f"  Output:  {out_dir.resolve()}")
    print(f"  Sources:")
    print(f"    - Price + Funding: Binance Futures API (full history)")
    print(f"    - OI/LS/Taker:    Binance API (30d) + data.binance.vision (daily, full history)")
    print(f"    - Liquidations:   Synthetic proxy from volatility")
    print()

    client = httpx.Client(timeout=30.0, headers={"accept": "application/json"})
    results = {}

    try:
        # 1. Price OHLCV
        df_price = download_price_klines(client, days)
        df_price.to_csv(out_dir / "btc_price_1h.csv", index=False)
        results["price_ohlcv"] = len(df_price)
        print(f"  Saved: {len(df_price)} rows -> btc_price_1h.csv")
        print()

        # 2. Funding Rate
        df_funding = download_funding_rate(client, days)
        df_funding.to_csv(out_dir / "btc_funding_rate.csv", index=False)
        results["funding_rate"] = len(df_funding)
        print(f"  Saved: {len(df_funding)} rows -> btc_funding_rate.csv")
        print()

        # 3. Open Interest (30 days hourly)
        df_oi = download_derivatives_30d(
            client, ENDPOINTS["oi_hist"], "Open Interest", 3, parse_oi
        )
        df_oi.to_csv(out_dir / "btc_oi_30d.csv", index=False)
        results["oi_30d"] = len(df_oi)
        print(f"  Saved: {len(df_oi)} rows -> btc_oi_30d.csv")
        print()

        # 4. Long/Short Ratio (30 days hourly)
        df_ls = download_derivatives_30d(
            client, ENDPOINTS["ls_ratio"], "Long/Short Ratio", 4, parse_ls_ratio
        )
        df_ls.to_csv(out_dir / "btc_ls_ratio_30d.csv", index=False)
        results["ls_ratio_30d"] = len(df_ls)
        print(f"  Saved: {len(df_ls)} rows -> btc_ls_ratio_30d.csv")
        print()

        # 5. Taker Buy/Sell (30 days hourly)
        df_taker = download_derivatives_30d(
            client, ENDPOINTS["taker_ratio"], "Taker Buy/Sell", 5, parse_taker
        )
        df_taker.to_csv(out_dir / "btc_taker_30d.csv", index=False)
        results["taker_30d"] = len(df_taker)
        print(f"  Saved: {len(df_taker)} rows -> btc_taker_30d.csv")
        print()

    finally:
        client.close()

    # 6. Daily metrics from data.binance.vision
    df_daily = pd.DataFrame()
    if not args.skip_daily_metrics:
        df_daily = download_daily_metrics(days)
        if not df_daily.empty:
            df_daily.to_csv(out_dir / "btc_daily_metrics.csv", index=False)
            results["daily_metrics"] = len(df_daily)
            print(f"  Saved: {len(df_daily)} rows -> btc_daily_metrics.csv")
        else:
            results["daily_metrics"] = 0
            print("  No daily metrics downloaded")
        print()

    # --- Merge everything ---
    if not args.skip_merge and not df_price.empty:
        df_merged = merge_all(df_price, df_funding, df_oi, df_ls, df_taker, df_daily)

        # Add liquidation proxy
        if not args.no_liquidation_proxy:
            df_merged = estimate_liquidations(df_merged)

        # Save merged dataset
        merged_path = out_dir / "btc_merged_1h.csv"
        df_merged.to_csv(merged_path, index=False)
        results["merged"] = len(df_merged)
        print(f"\n  Merged dataset: {len(df_merged)} rows x {len(df_merged.columns)} columns")
        print(f"  Columns: {list(df_merged.columns)}")
        print(f"  Date range: {df_merged['timestamp'].min()} to {df_merged['timestamp'].max()}")
        print(f"  Saved to: {merged_path}")

        # Report null coverage
        print("\n  Data coverage (non-null %):")
        for col in df_merged.columns:
            if col == "timestamp":
                continue
            pct = (1 - df_merged[col].isna().mean()) * 100
            bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
            print(f"    {col:30s} [{bar}] {pct:.1f}%")

    # Summary
    print()
    print("=" * 60)
    print("  Download Summary")
    print("=" * 60)
    total = 0
    for name, count in results.items():
        status = "OK" if count > 0 else "EMPTY"
        print(f"  {name:25s} {count:>8} rows  [{status}]")
        total += count
    print(f"  {'TOTAL':25s} {total:>8} rows")
    print()
    print(f"  Output directory: {out_dir.resolve()}")
    print()
    print("  Next steps:")
    print("    1. Review data coverage above")
    print("    2. Run backtest: python scripts/backtest_historical.py --data data/historical/btc_merged_1h.csv")
    print()
    print("  Data quality notes:")
    print("    - Price & funding rate: full {}-day coverage".format(days))
    print("    - OI/LS/Taker hourly: last 30 days only (Binance API limit)")
    print("    - OI/LS/Taker older: daily granularity, forward-filled to 1h")
    print("    - Liquidations: synthetic proxy from volatility (no historical source)")


if __name__ == "__main__":
    main()

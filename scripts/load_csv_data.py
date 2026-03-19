#!/usr/bin/env python3
"""Load downloaded CSV historical data and rename columns for the feature pipeline.

Reads the merged CSV from data/historical/btc_merged_1h.csv and renames
columns to match what the feature pipeline expects:

  Downloaded Name          -> Pipeline Expects
  ─────────────────────────────────────────────
  funding_rate             -> funding_close
  open_interest            -> oi_close
  est_long_liq             -> long_liquidations_usd
  est_short_liq            -> short_liquidations_usd
  long_account             -> long_ratio
  short_account            -> short_ratio
  buy_vol                  -> buy_volume
  sell_vol                 -> sell_volume

Usage:
  python scripts/load_csv_data.py
  python scripts/load_csv_data.py --input data/historical/btc_merged_1h.csv --output data/btc_ready.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# Column mapping: downloaded name -> feature pipeline name
COLUMN_RENAME = {
    "funding_rate":   "funding_close",
    "open_interest":  "oi_close",
    "est_long_liq":   "long_liquidations_usd",
    "est_short_liq":  "short_liquidations_usd",
    "long_account":   "long_ratio",
    "short_account":  "short_ratio",
    "buy_vol":        "buy_volume",
    "sell_vol":       "sell_volume",
}

# Derived columns to add if missing
DERIVED_COLUMNS = {
    "funding_open":           "funding_close",   # same as close (8h data ffilled)
    "oi_open":                "oi_close",         # same as close (daily data ffilled)
    "total_liquidations_usd": None,               # computed: long + short
}


def load_and_prepare(
    input_path: str = "data/historical/btc_merged_1h.csv",
    output_path: str | None = None,
) -> pd.DataFrame:
    """Load merged CSV and prepare for the feature pipeline."""
    path = Path(input_path)
    if not path.exists():
        print(f"ERROR: File not found: {path.resolve()}")
        print("  Run: python scripts/download_historical_data.py")
        return pd.DataFrame()

    print(f"Loading {path}...")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    print(f"  Loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # ── Rename columns ─────────────────────────────────────────
    renames_applied = {}
    for old_name, new_name in COLUMN_RENAME.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
            renames_applied[old_name] = new_name
    print(f"\n  Renamed {len(renames_applied)} columns:")
    for old, new in renames_applied.items():
        print(f"    {old:30s} -> {new}")

    # ── Add derived columns ────────────────────────────────────
    # funding_open = funding_close (since it's 8h data forward-filled to 1h)
    if "funding_open" not in df.columns and "funding_close" in df.columns:
        df["funding_open"] = df["funding_close"]
        print(f"    Added: funding_open (copy of funding_close)")

    # oi_open = oi_close (since it's daily data forward-filled to 1h)
    if "oi_open" not in df.columns and "oi_close" in df.columns:
        df["oi_open"] = df["oi_close"]
        print(f"    Added: oi_open (copy of oi_close)")

    # total_liquidations_usd = long + short
    if "total_liquidations_usd" not in df.columns:
        long_liq = df.get("long_liquidations_usd", pd.Series(0.0, index=df.index))
        short_liq = df.get("short_liquidations_usd", pd.Series(0.0, index=df.index))
        df["total_liquidations_usd"] = long_liq + short_liq
        print(f"    Added: total_liquidations_usd (long + short)")

    # ── Validate all required columns ──────────────────────────
    required = {
        "price":       ["timestamp", "open", "high", "low", "close", "volume"],
        "funding":     ["funding_close"],
        "oi":          ["oi_close"],
        "liquidations": ["long_liquidations_usd", "short_liquidations_usd", "total_liquidations_usd"],
        "long_short":  ["long_short_ratio"],
        "taker_flow":  ["buy_sell_ratio"],
    }
    optional = {
        "funding":     ["funding_open"],
        "oi":          ["oi_open"],
        "long_short":  ["long_ratio", "short_ratio"],
        "taker_flow":  ["buy_volume", "sell_volume"],
    }

    print(f"\n  Column validation:")
    all_ok = True
    for group, cols in required.items():
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"    {group:15s} MISSING required: {missing}")
            all_ok = False
        else:
            opt_cols = optional.get(group, [])
            opt_present = [c for c in opt_cols if c in df.columns]
            opt_missing = [c for c in opt_cols if c not in df.columns]
            status = "OK"
            if opt_missing:
                status += f" (optional missing: {opt_missing})"
            print(f"    {group:15s} {status}")

    # ── Coverage report ────────────────────────────────────────
    print(f"\n  Data coverage (non-null %):")
    key_cols = ["close", "volume", "funding_close", "oi_close",
                "long_liquidations_usd", "short_liquidations_usd",
                "long_short_ratio", "buy_sell_ratio",
                "buy_volume", "sell_volume", "long_ratio", "short_ratio"]
    for col in key_cols:
        if col in df.columns:
            pct = (1 - df[col].isna().mean()) * 100
            bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
            print(f"    {col:30s} [{bar}] {pct:.1f}%")

    # ── Save ───────────────────────────────────────────────────
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\n  Saved pipeline-ready data to: {out.resolve()}")

    if all_ok:
        print(f"\n  All required columns present. Data is ready for backtesting!")
    else:
        print(f"\n  WARNING: Some required columns are missing. Backtest may fail.")

    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare downloaded data for the feature pipeline")
    parser.add_argument("--input", default="data/historical/btc_merged_1h.csv",
                        help="Input merged CSV")
    parser.add_argument("--output", default="data/btc_ready.csv",
                        help="Output pipeline-ready CSV")
    args = parser.parse_args()

    df = load_and_prepare(args.input, args.output)
    if df.empty:
        return

    print(f"\n  Next step: python scripts/backtest_historical.py --data {args.output}")


if __name__ == "__main__":
    main()

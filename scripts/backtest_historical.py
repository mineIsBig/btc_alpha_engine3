#!/usr/bin/env python3
"""End-to-end historical backtest.

Replays the full signal pipeline on historical data:
  synthetic data -> features -> labels -> walk-forward train ->
  ensemble -> consensus -> sizing -> paper execution -> performance report

Usage:
  python scripts/backtest_historical.py --hours 2000 --equity 100000
  python scripts/backtest_historical.py --hours 5000 --equity 50000 --horizons 4,8,24
"""
import sys
import os

sys.path.insert(0, ".")
os.environ["DATABASE_URL"] = "sqlite:///./backtest.db"
os.environ["COINGLASS_API_KEY"] = "backtest"
os.environ["LIVE_TRADING_ENABLED"] = "false"
os.environ["PAPER_MODE"] = "true"

import click
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.common.logging import setup_logging, get_logger

setup_logging(level="WARNING", fmt="console")
logger = get_logger(__name__)


# --- Synthetic data generation --------------------------------------
def generate_synthetic_data(n_hours: int = 2000, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Generate realistic synthetic BTC market data with regime changes."""
    np.random.seed(seed)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(hours=i) for i in range(n_hours)]

    # Price: random walk with regime-dependent drift/vol
    regime_len = n_hours // 5
    regimes = []
    for _ in range(5):
        regime = np.random.choice(["bull", "bear", "chop"])
        regimes.extend([regime] * regime_len)
    regimes = regimes[:n_hours]

    returns = np.zeros(n_hours)
    for i, r in enumerate(regimes):
        if r == "bull":
            returns[i] = np.random.normal(0.0005, 0.004)
        elif r == "bear":
            returns[i] = np.random.normal(-0.0004, 0.005)
        else:
            returns[i] = np.random.normal(0.0, 0.003)

    close = 42000 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.003, n_hours)))
    low = close * (1 - np.abs(np.random.normal(0, 0.003, n_hours)))
    open_ = close * (1 + np.random.normal(0, 0.001, n_hours))
    volume = np.abs(np.random.normal(1e6, 3e5, n_hours))

    price_df = pd.DataFrame({
        "timestamp": timestamps, "open": open_, "high": high,
        "low": low, "close": close, "volume": volume, "symbol": "BTC",
    })

    # Funding: correlated with price momentum
    funding_base = np.random.normal(0.0001, 0.0003, n_hours)
    for i in range(24, n_hours):
        momentum = (close[i] - close[i - 24]) / close[i - 24]
        funding_base[i] += momentum * 0.005  # funding tracks momentum
    funding_df = pd.DataFrame({
        "timestamp": timestamps,
        "close": funding_base,
        "open": funding_base * 0.99,
        "symbol": "BTC", "exchange": "Binance",
    })

    # OI: trends with price
    oi = 1e9 + np.cumsum(np.random.normal(0, 1e7, n_hours))
    for i in range(1, n_hours):
        oi[i] += returns[i] * 5e8  # OI grows when price moves
    oi = np.abs(oi)
    oi_df = pd.DataFrame({
        "timestamp": timestamps,
        "close": oi, "open": oi * 0.99,
        "symbol": "BTC", "exchange": "Binance",
    })

    # Liquidations: spike on big moves
    long_liq = np.abs(np.random.normal(1e6, 5e5, n_hours))
    short_liq = np.abs(np.random.normal(1e6, 5e5, n_hours))
    for i in range(n_hours):
        if returns[i] < -0.01:
            long_liq[i] *= 5  # big down -> long liquidations
        if returns[i] > 0.01:
            short_liq[i] *= 5  # big up -> short liquidations
    liq_df = pd.DataFrame({
        "timestamp": timestamps,
        "long_liquidations_usd": long_liq,
        "short_liquidations_usd": short_liq,
        "total_liquidations_usd": long_liq + short_liq,
        "count": np.random.randint(0, 100, n_hours),
    })

    # Long/Short ratios
    ls_df = pd.DataFrame({
        "timestamp": timestamps,
        "long_ratio": np.random.uniform(0.4, 0.6, n_hours),
        "short_ratio": np.random.uniform(0.4, 0.6, n_hours),
        "long_short_ratio": np.random.uniform(0.8, 1.2, n_hours),
    })

    # Taker flow
    tf_df = pd.DataFrame({
        "timestamp": timestamps,
        "buy_volume": np.abs(np.random.normal(5e5, 2e5, n_hours)),
        "sell_volume": np.abs(np.random.normal(5e5, 2e5, n_hours)),
        "buy_sell_ratio": np.random.uniform(0.8, 1.2, n_hours),
    })

    # Price df for feature pipeline needs funding/oi merged in
    price_with_extras = price_df.copy()
    price_with_extras["funding_close"] = funding_base
    price_with_extras["oi_close"] = oi

    return {
        "price": price_with_extras,
        "funding": funding_df,
        "oi": oi_df,
        "liquidations": liq_df,
        "long_short": ls_df,
        "taker_flow": tf_df,
    }


# --- Backtest engine ------------------------------------------------
class HistoricalBacktester:
    """Walk-forward train + out-of-sample signal replay backtest."""

    def __init__(
        self,
        raw_data: dict[str, pd.DataFrame],
        horizons: list[int],
        initial_equity: float = 100_000.0,
        slippage_bps: float = 5.0,
        commission_bps: float = 2.0,
        daily_loss_limit: float = 0.05,
        eod_trailing_limit: float = 0.05,
    ):
        self.raw_data = raw_data
        self.horizons = horizons
        self.initial_equity = initial_equity
        self.slippage_bps = slippage_bps
        self.commission_bps = commission_bps
        self.daily_loss_limit = daily_loss_limit
        self.eod_trailing_limit = eod_trailing_limit

        self.cost_per_trade = (slippage_bps + commission_bps) / 10_000.0

    def run(self) -> dict:
        """Run the full backtest. Returns performance report dict."""
        from src.features.feature_pipeline import build_features
        from src.labels.labels import build_labels
        from src.research.datasets import get_feature_columns, get_label_column
        from src.research.purged_walk_forward import PurgedWalkForward
        from src.models.baseline import LogisticRegressionModel, RandomForestModel
        from src.models.gradient_boost import LightGBMModel, XGBoostModel
        from src.portfolio.signal_schema import ModelSignal, AggregatedSignal
        from src.portfolio.ensemble import EnsembleAggregator
        from src.portfolio.consensus import ConsensusGate

        print("=" * 70)
        print("  BTC ALPHA ENGINE - HISTORICAL BACKTEST")
        print("=" * 70)

        # -- Step 1: Build features + labels ----------------------
        print("\n[1/5] Building features and labels...")
        features_df = build_features(raw_data=self.raw_data)
        price_df = self.raw_data["price"][["timestamp", "close", "high", "low"]].copy()
        labels_df = build_labels(price_df, horizons=self.horizons)

        features_df["timestamp"] = pd.to_datetime(features_df["timestamp"], utc=True)
        labels_df["timestamp"] = pd.to_datetime(labels_df["timestamp"], utc=True)
        dataset = features_df.merge(labels_df, on="timestamp", how="inner")
        dataset = dataset.dropna(how="all").reset_index(drop=True)

        feature_cols = get_feature_columns(dataset)
        print(f"  -> {len(dataset)} rows, {len(feature_cols)} features, horizons={self.horizons}")

        # -- Step 2: Walk-forward model training ------------------
        print("\n[2/5] Walk-forward model training...")
        model_configs = [
            ("lr", LogisticRegressionModel, {"C": 1.0, "penalty": "l2"}),
            ("rf", RandomForestModel, {"n_estimators": 100, "max_depth": 8}),
            ("lgbm", LightGBMModel, {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05}),
            ("xgb", XGBoostModel, {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05}),
        ]

        # Use shorter windows for synthetic data to get enough folds
        n_hours = len(dataset)
        train_days = min(60, n_hours // (24 * 4))  # at most 1/4 of data
        test_days = min(14, train_days // 4)
        train_days = max(14, train_days)
        test_days = max(3, test_days)

        splitter = PurgedWalkForward(
            train_days=train_days,
            test_days=test_days,
            purge_hours=24,
            embargo_hours=12,
            step_days=test_days,
            min_train_samples=100,
            max_folds=30,
        )

        n_folds = splitter.get_n_folds(dataset["timestamp"])
        print(f"  -> train={train_days}d, test={test_days}d, purge=24h, embargo=12h")
        print(f"  -> {n_folds} folds detected")

        if n_folds == 0:
            print("  ERROR: Not enough data for walk-forward splits.")
            print(f"  Need at least {train_days + test_days + 2} days = {(train_days + test_days + 2) * 24} hours")
            return {"error": "insufficient_data"}

        # Collect out-of-sample predictions per fold, per model, per horizon
        # Key insight: we aggregate predictions on the TEST set only (true OOS)
        oos_records = []  # List of {timestamp, horizon, model_id, side, proba, conf, sharpe}
        model_sharpes = {}  # model_id -> avg OOS sharpe

        total_combos = len(self.horizons) * len(model_configs)
        combo_idx = 0

        for horizon in self.horizons:
            label_col = get_label_column(horizon)
            fwd_ret_col = f"fwd_ret_{horizon}h"

            if label_col not in dataset.columns:
                continue

            for model_name, model_cls, default_params in model_configs:
                combo_idx += 1
                model_id = f"{model_name}_h{horizon}"
                fold_sharpes = []

                for fold in splitter.split(dataset["timestamp"]):
                    X_train = dataset.iloc[fold.train_indices][feature_cols].fillna(0)
                    y_train = dataset.iloc[fold.train_indices][label_col].fillna(0).astype(int).values
                    X_test = dataset.iloc[fold.test_indices][feature_cols].fillna(0)
                    y_test = dataset.iloc[fold.test_indices][label_col].fillna(0).astype(int).values
                    fwd_ret = dataset.iloc[fold.test_indices][fwd_ret_col].fillna(0).values
                    test_ts = dataset.iloc[fold.test_indices]["timestamp"].values
                    test_close = self.raw_data["price"].set_index("timestamp").reindex(
                        pd.to_datetime(test_ts, utc=True)
                    )["close"].values

                    if len(np.unique(y_train)) < 2:
                        continue

                    try:
                        model = model_cls(horizon=horizon, params=default_params, model_id=model_id)
                        model.fit(X_train, y_train, feature_names=feature_cols)
                        y_pred = model.predict(X_test)

                        # Compute fold-level Sharpe for weighting
                        fold_pnl = []
                        for i in range(len(y_test)):
                            sig = int(y_pred[i])
                            if sig != 0:
                                ret = fwd_ret[i] if not np.isnan(fwd_ret[i]) else 0.0
                                pnl = sig * ret - abs(sig) * self.cost_per_trade
                            else:
                                pnl = 0.0
                            fold_pnl.append(pnl)

                        fold_pnl = np.array(fold_pnl)
                        if fold_pnl.std() > 0:
                            fold_sharpe = float(fold_pnl.mean() / fold_pnl.std() * np.sqrt(8760 / max(len(fold_pnl), 1)))
                        else:
                            fold_sharpe = 0.0
                        fold_sharpes.append(fold_sharpe)

                        # Get prediction probabilities for signal generation
                        try:
                            probas = model.predict_proba(X_test)
                        except Exception:
                            probas = None

                        for i in range(len(y_test)):
                            side = int(y_pred[i])
                            if probas is not None and len(probas[i]) == 3:
                                prob = float(max(probas[i]))
                                conf = float(prob - sorted(probas[i])[-2])
                            else:
                                prob = 0.6
                                conf = 0.2

                            oos_records.append({
                                "timestamp": pd.Timestamp(test_ts[i]),
                                "horizon": horizon,
                                "model_id": model_id,
                                "side": side,
                                "probability": prob,
                                "confidence": max(0.0, conf),
                                "close": test_close[i] if not np.isnan(test_close[i]) else 0.0,
                                "fwd_ret": fwd_ret[i],
                            })
                    except Exception as e:
                        print(f"  ! {model_id} fold {fold.fold_idx} error: {e}")

                avg_sharpe = np.mean(fold_sharpes) if fold_sharpes else 0.0
                model_sharpes[model_id] = avg_sharpe

                status = "OK" if avg_sharpe > 0 else "FAIL"
                print(f"  [{combo_idx}/{total_combos}] {model_id:12s}  sharpe={avg_sharpe:+.3f} ({len(fold_sharpes)} folds)  {status}")

        if not oos_records:
            print("\n  ERROR: No OOS predictions generated.")
            return {"error": "no_predictions"}

        # -- Step 3: Replay OOS signals through ensemble + consensus --
        print("\n[3/5] Replaying signals through ensemble + consensus...")
        oos_df = pd.DataFrame(oos_records)
        all_timestamps = sorted(oos_df["timestamp"].unique())
        print(f"  -> {len(all_timestamps)} unique OOS timestamps, {len(oos_df)} total predictions")

        ensemble = EnsembleAggregator(min_consensus_pct=0.5, min_avg_confidence=0.05)
        consensus_gate = ConsensusGate(min_horizon_agreement=2, horizons=self.horizons)

        # -- Step 4: Simulate execution with risk management ------
        print("\n[4/5] Simulating execution with risk management...")
        equity = self.initial_equity
        equity_curve = [equity]
        equity_timestamps = [all_timestamps[0] - pd.Timedelta(hours=1)]
        trades = []

        # Risk state
        day_opening_equity = equity
        eod_hwm = equity
        current_day = None
        can_trade = True
        breach_count = 0
        consecutive_losses = 0
        max_consec_losses = 5

        for ts in all_timestamps:
            ts_data = oos_df[oos_df["timestamp"] == ts]
            day = ts.date()

            # Day reset
            if current_day is None or day != current_day:
                if current_day is not None:
                    eod_hwm = max(eod_hwm, equity)
                day_opening_equity = equity
                can_trade = True
                consecutive_losses = 0
                current_day = day

            # Check risk limits
            daily_floor = day_opening_equity * (1 - self.daily_loss_limit)
            eod_floor = eod_hwm * (1 - self.eod_trailing_limit)

            if equity < daily_floor:
                can_trade = False
                breach_count += 1
            if equity < eod_floor:
                can_trade = False
                breach_count += 1

            # Kill switch
            if consecutive_losses >= max_consec_losses:
                can_trade = False

            if not can_trade:
                equity_curve.append(equity)
                equity_timestamps.append(ts)
                continue

            # Build model signals per horizon
            horizon_signals = {}
            for horizon in self.horizons:
                h_data = ts_data[ts_data["horizon"] == horizon]
                if h_data.empty:
                    continue

                model_signals = []
                for _, row in h_data.iterrows():
                    model_signals.append(ModelSignal(
                        model_id=row["model_id"],
                        horizon=horizon,
                        side=int(row["side"]),
                        probability=float(row["probability"]),
                        confidence=float(row["confidence"]),
                        oos_sharpe=max(0.01, model_sharpes.get(row["model_id"], 0.01)),
                    ))

                if model_signals:
                    agg = ensemble.aggregate(model_signals, timestamp=ts)
                    horizon_signals[horizon] = agg

            # Cross-horizon consensus
            if horizon_signals:
                final_side, reason = consensus_gate.check(horizon_signals)
            else:
                final_side = 0

            # Execute trade
            if final_side != 0:
                # Use avg close from this timestamp's data
                current_price = ts_data["close"].mean()
                if current_price <= 0 or np.isnan(current_price):
                    equity_curve.append(equity)
                    equity_timestamps.append(ts)
                    continue

                # Use the best-matching horizon's forward return
                best_horizon = min(self.horizons, key=lambda h: abs(h - 4))  # prefer 4h
                h_data = ts_data[ts_data["horizon"] == best_horizon]
                if h_data.empty:
                    h_data = ts_data

                avg_fwd_ret = h_data["fwd_ret"].mean()
                if np.isnan(avg_fwd_ret):
                    avg_fwd_ret = 0.0

                # Position size: simplified vol-targeting
                recent_returns = []
                idx = all_timestamps.index(ts)
                for prev_ts in all_timestamps[max(0, idx - 24):idx]:
                    prev_data = oos_df[oos_df["timestamp"] == prev_ts]
                    if not prev_data.empty:
                        recent_returns.append(prev_data["fwd_ret"].mean())

                if recent_returns and np.std(recent_returns) > 0:
                    realized_vol = np.std(recent_returns) * np.sqrt(8760)
                    vol_size_pct = min(0.30 / realized_vol, 0.25)
                else:
                    vol_size_pct = 0.10

                # Compute headroom
                headroom = min(
                    (equity - daily_floor) / equity if equity > 0 else 0,
                    (equity - eod_floor) / equity if equity > 0 else 0,
                )
                if headroom < 0.02:
                    vol_size_pct *= 0.5  # reduce near breach

                # Confidence-adjusted size
                avg_conf = ts_data["confidence"].mean()
                position_pct = vol_size_pct * min(1.0, avg_conf * 2)
                position_pct = max(0.02, min(position_pct, 0.25))  # [2%, 25%]

                position_usd = equity * position_pct
                pnl = final_side * avg_fwd_ret * position_usd - self.cost_per_trade * position_usd
                equity += pnl

                is_win = pnl > 0
                if is_win:
                    consecutive_losses = 0
                else:
                    consecutive_losses += 1

                trades.append({
                    "timestamp": ts,
                    "side": "LONG" if final_side == 1 else "SHORT",
                    "price": current_price,
                    "position_pct": position_pct,
                    "position_usd": position_usd,
                    "fwd_ret": avg_fwd_ret,
                    "pnl": pnl,
                    "equity": equity,
                    "headroom": headroom,
                })

            equity_curve.append(equity)
            equity_timestamps.append(ts)

        # -- Step 5: Generate report ------------------------------
        print("\n[5/5] Generating performance report...\n")
        return self._generate_report(
            equity_curve=np.array(equity_curve),
            timestamps=equity_timestamps,
            trades=trades,
            model_sharpes=model_sharpes,
            n_folds=n_folds,
            breach_count=breach_count,
        )

    def _generate_report(
        self,
        equity_curve: np.ndarray,
        timestamps: list,
        trades: list[dict],
        model_sharpes: dict[str, float],
        n_folds: int,
        breach_count: int,
    ) -> dict:
        """Generate and print the performance report."""
        final_equity = equity_curve[-1]
        total_return = (final_equity - self.initial_equity) / self.initial_equity

        # Returns
        returns = np.diff(equity_curve) / equity_curve[:-1]
        returns = returns[np.isfinite(returns)]

        if len(returns) > 0 and returns.std() > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(8760 / max(len(returns), 1)))
        else:
            sharpe = 0.0

        # Drawdown
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cummax) / cummax
        max_dd = float(np.min(drawdown))

        # Trade stats
        n_trades = len(trades)
        if n_trades > 0:
            wins = sum(1 for t in trades if t["pnl"] > 0)
            losses = sum(1 for t in trades if t["pnl"] < 0)
            win_rate = wins / n_trades
            gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
            gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
            avg_pnl = np.mean([t["pnl"] for t in trades])
            avg_win = np.mean([t["pnl"] for t in trades if t["pnl"] > 0]) if wins > 0 else 0
            avg_loss = np.mean([t["pnl"] for t in trades if t["pnl"] < 0]) if losses > 0 else 0
            longs = sum(1 for t in trades if t["side"] == "LONG")
            shorts = n_trades - longs
        else:
            wins = losses = longs = shorts = 0
            win_rate = profit_factor = avg_pnl = avg_win = avg_loss = 0

        # Calmar
        if max_dd < 0:
            ann_hours = len(equity_curve)
            ann_ret = total_return * (8760 / max(ann_hours, 1))
            calmar = ann_ret / abs(max_dd)
        else:
            calmar = 0.0

        # Print report
        print("=" * 70)
        print("  BACKTEST RESULTS")
        print("=" * 70)

        print(f"\n  Period:          {timestamps[0]} -> {timestamps[-1]}")
        print(f"  Initial equity:  ${self.initial_equity:,.2f}")
        print(f"  Final equity:    ${final_equity:,.2f}")
        print(f"  Total return:    {total_return:+.2%}")

        print(f"\n  -- Risk-Adjusted Metrics --")
        print(f"  Sharpe ratio:    {sharpe:+.3f}")
        print(f"  Calmar ratio:    {calmar:+.3f}")
        print(f"  Max drawdown:    {max_dd:.2%}")
        print(f"  Risk breaches:   {breach_count}")

        print(f"\n  -- Trade Statistics --")
        print(f"  Total trades:    {n_trades}")
        print(f"  Win rate:        {win_rate:.1%}")
        print(f"  Profit factor:   {profit_factor:.2f}")
        print(f"  Avg PnL/trade:   ${avg_pnl:+,.2f}")
        print(f"  Avg win:         ${avg_win:+,.2f}")
        print(f"  Avg loss:        ${avg_loss:+,.2f}")
        print(f"  Long / Short:    {longs} / {shorts}")

        print(f"\n  -- Model Performance (OOS Sharpe) --")
        for mid, s in sorted(model_sharpes.items(), key=lambda x: x[1], reverse=True):
            bar = "#" * max(0, int(s * 5)) if s > 0 else "." * max(0, int(abs(s) * 5))
            print(f"  {mid:16s} {s:+.3f}  {bar}")

        print(f"\n  -- Walk-Forward --")
        print(f"  Folds:           {n_folds}")
        print(f"  Slippage:        {self.slippage_bps} bps")
        print(f"  Commission:      {self.commission_bps} bps")

        # Save equity curve to CSV
        output_dir = Path("artifacts/backtest")
        output_dir.mkdir(parents=True, exist_ok=True)

        eq_df = pd.DataFrame({"timestamp": timestamps, "equity": equity_curve})
        eq_path = output_dir / "equity_curve.csv"
        eq_df.to_csv(eq_path, index=False)
        print(f"\n  Equity curve saved to:  {eq_path}")

        if trades:
            trades_df = pd.DataFrame(trades)
            trades_path = output_dir / "trade_log.csv"
            trades_df.to_csv(trades_path, index=False)
            print(f"  Trade log saved to:    {trades_path}")

            # Print last 10 trades
            print(f"\n  -- Last 10 Trades --")
            print(f"  {'Timestamp':<22s} {'Side':>5s} {'Price':>10s} {'Size%':>6s} {'PnL':>10s} {'Equity':>12s}")
            print(f"  {'-'*22} {'-'*5} {'-'*10} {'-'*6} {'-'*10} {'-'*12}")
            for t in trades[-10:]:
                print(f"  {str(t['timestamp'])[:19]:<22s} {t['side']:>5s} "
                      f"${t['price']:>9,.0f} {t['position_pct']:>5.1%} "
                      f"${t['pnl']:>+9,.2f} ${t['equity']:>11,.2f}")

        print("\n" + "=" * 70)

        return {
            "initial_equity": self.initial_equity,
            "final_equity": final_equity,
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "calmar_ratio": calmar,
            "max_drawdown": max_dd,
            "n_trades": n_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "breach_count": breach_count,
            "model_sharpes": model_sharpes,
            "equity_curve_path": str(eq_path),
        }


@click.command()
@click.option("--hours", default=2000, help="Hours of synthetic data to generate")
@click.option("--equity", default=100_000.0, help="Starting equity USD")
@click.option("--horizons", default="1,4,8,12,24", help="Comma-separated horizons")
@click.option("--seed", default=42, help="Random seed for reproducibility")
@click.option("--slippage", default=5.0, help="Slippage in bps")
@click.option("--commission", default=2.0, help="Commission in bps")
def main(hours: int, equity: float, horizons: str, seed: int, slippage: float, commission: float) -> None:
    horizon_list = [int(h) for h in horizons.split(",")]

    print(f"\nGenerating {hours} hours of synthetic BTC data (seed={seed})...")
    raw_data = generate_synthetic_data(n_hours=hours, seed=seed)

    backtester = HistoricalBacktester(
        raw_data=raw_data,
        horizons=horizon_list,
        initial_equity=equity,
        slippage_bps=slippage,
        commission_bps=commission,
    )

    results = backtester.run()

    if "error" not in results:
        print(f"\nBacktest completed successfully.")
        print(f"Results saved to artifacts/backtest/")


if __name__ == "__main__":
    main()

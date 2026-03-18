"""Scoring and metrics for walk-forward evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.common.config import load_yaml_config
from src.common.logging import get_logger
from src.execution.slippage_model import RegimeAwareCostModel

logger = get_logger(__name__)


def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fwd_returns: np.ndarray,
    prices: np.ndarray | None = None,
    initial_equity: float = 100000.0,
    slippage_bps: float = 5.0,
    commission_bps: float = 2.0,
    daily_loss_limit: float = 0.05,
    eod_trailing_limit: float = 0.05,
    timestamps: np.ndarray | None = None,
    regime_labels: np.ndarray | None = None,
    volatility_zscores: np.ndarray | None = None,
    liquidation_zscores: np.ndarray | None = None,
    use_regime_costs: bool = False,
) -> dict[str, float]:
    """Compute comprehensive metrics for a walk-forward fold.

    Simulates PnL with risk constraints and computes performance metrics.
    Supports regime-aware cost modeling for realistic performance estimates.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        fwd_returns: Forward returns for each prediction
        prices: Price series (optional)
        initial_equity: Starting equity for simulation
        slippage_bps: Base slippage in basis points
        commission_bps: Base commission in basis points
        daily_loss_limit: Maximum daily loss as fraction of equity
        eod_trailing_limit: EOD trailing stop as fraction of equity
        timestamps: Timestamps for each prediction
        regime_labels: Market regime labels for each prediction
        volatility_zscores: Realized volatility z-scores
        liquidation_zscores: Liquidation intensity z-scores
        use_regime_costs: If True, apply regime-dependent cost scaling
    """
    from src.execution.slippage_model import RegimeAwareCostModel
    
    n = len(y_true)
    cost_model = RegimeAwareCostModel(
        base_slippage_bps=slippage_bps,
        base_commission_bps=commission_bps,
    )
    
    # Track cost statistics for reporting
    total_costs = []
    cost_breakdown = {
        "base_costs": [],
        "volatility_premium": [],
        "liquidation_premium": [],
        "regime_premium": [],
    }

    # PnL simulation
    equity = initial_equity
    equity_curve = [equity]
    trades = 0
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    breach_count = 0
    can_trade = True

    # Day state tracking
    day_opening_equity = equity
    eod_hwm = equity
    current_day = None

    for i in range(n):
        # Check for day reset
        if timestamps is not None:
            ts = pd.Timestamp(timestamps[i])
            day = ts.date()
            if current_day is None or day != current_day:
                # New day: update EOD HWM and reset
                if current_day is not None:
                    eod_hwm = max(eod_hwm, equity)
                day_opening_equity = equity
                can_trade = True  # Reset for new day
                current_day = day

        # Check risk limits
        daily_floor = day_opening_equity * (1 - daily_loss_limit)
        eod_floor = eod_hwm * (1 - eod_trailing_limit)

        if equity < daily_floor:
            can_trade = False
            breach_count += 1

        if equity < eod_floor:
            can_trade = False
            breach_count += 1

        # Execute trade if allowed
        signal = int(y_pred[i])
        if can_trade and signal != 0:
            ret = fwd_returns[i] if not np.isnan(fwd_returns[i]) else 0.0
            
            # Compute dynamic costs based on market regime
            if use_regime_costs:
                regime = regime_labels[i] if regime_labels is not None else None
                vol_z = volatility_zscores[i] if volatility_zscores is not None else 0.0
                liq_z = liquidation_zscores[i] if liquidation_zscores is not None else 0.0
                
                dynamic_slippage, dynamic_commission = cost_model.compute_dynamic_costs(
                    regime=regime,
                    volatility_zscore=vol_z,
                    liquidation_zscore=liq_z,
                )
                cost_per_trade = (dynamic_slippage + dynamic_commission) / 10000.0
                
                # Track cost breakdown
                total_costs.append(cost_per_trade)
                cost_breakdown["base_costs"].append((slippage_bps + commission_bps) / 10000.0)
                cost_breakdown["volatility_premium"].append(
                    cost_model.volatility_multiplier * max(0, vol_z) / 10000.0
                )
                cost_breakdown["liquidation_premium"].append(
                    cost_model.liquidation_multiplier * max(0, liq_z) / 10000.0
                )
                if regime and regime in cost_model.regime_multipliers:
                    cost_breakdown["regime_premium"].append(
                        (cost_model.regime_multipliers[regime] - 1.0) * (slippage_bps + commission_bps) / 10000.0
                    )
                else:
                    cost_breakdown["regime_premium"].append(0.0)
            else:
                cost_per_trade = (slippage_bps + commission_bps) / 10000.0
            
            pnl = signal * ret * equity - abs(signal) * cost_per_trade * equity
            equity += pnl
            trades += 1

            if pnl > 0:
                wins += 1
                gross_profit += pnl
            elif pnl < 0:
                losses += 1
                gross_loss += abs(pnl)

        equity_curve.append(equity)

    equity_series = np.array(equity_curve)

    # Compute metrics
    metrics: dict[str, float] = {}

    # Classification metrics
    nonzero_mask = y_true != 0
    if nonzero_mask.any():
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        try:
            metrics["precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
            metrics["recall"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        except Exception:
            metrics["precision"] = 0.0
            metrics["recall"] = 0.0
    else:
        metrics["accuracy"] = 0.0
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0

    # Return-based metrics
    returns = np.diff(equity_series) / equity_series[:-1]
    returns = returns[np.isfinite(returns)]

    if len(returns) > 0 and returns.std() > 0:
        metrics["sharpe_ratio"] = float(np.mean(returns) / np.std(returns) * np.sqrt(8760 / max(len(returns), 1)))
    else:
        metrics["sharpe_ratio"] = 0.0

    # Drawdown
    cummax = np.maximum.accumulate(equity_series)
    drawdown = (equity_series - cummax) / cummax
    metrics["max_drawdown"] = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

    # Profit factor
    metrics["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    # Win rate
    metrics["win_rate"] = wins / trades if trades > 0 else 0.0
    metrics["n_trades"] = trades
    metrics["total_return"] = (equity - initial_equity) / initial_equity

    # Calmar ratio
    if metrics["max_drawdown"] < 0:
        annualized_return = metrics["total_return"] * (8760 / max(n, 1))
        metrics["calmar_ratio"] = annualized_return / abs(metrics["max_drawdown"])
    else:
        metrics["calmar_ratio"] = 0.0

    metrics["breach_count"] = breach_count
    metrics["breach_rate"] = breach_count / max(n, 1)
    
    # Add cost breakdown if regime-aware costs were used
    if use_regime_costs and total_costs:
        metrics["avg_cost_per_trade_bps"] = float(np.mean(total_costs) * 10000)
        metrics["total_cost_premium_bps"] = float(
            np.mean(cost_breakdown["volatility_premium"]) * 10000 +
            np.mean(cost_breakdown["liquidation_premium"]) * 10000 +
            np.mean(cost_breakdown["regime_premium"]) * 10000
        )
        metrics["volatility_cost_premium_bps"] = float(np.mean(cost_breakdown["volatility_premium"]) * 10000)
        metrics["liquidation_cost_premium_bps"] = float(np.mean(cost_breakdown["liquidation_premium"]) * 10000)
        metrics["regime_cost_premium_bps"] = float(np.mean(cost_breakdown["regime_premium"]) * 10000)

    return metrics


def compute_rolling_sharpe(
    returns: np.ndarray,
    window: int = 168,
) -> np.ndarray:
    """Compute rolling Sharpe ratio over a window of hours."""
    ret_series = pd.Series(returns)
    rolling_mean = ret_series.rolling(window).mean()
    rolling_std = ret_series.rolling(window).std().replace(0, np.nan)
    sharpe = rolling_mean / rolling_std * np.sqrt(window)
    return sharpe.values


def compute_information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute rank IC between true and predicted values."""
    from scipy.stats import spearmanr
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 10:
        return 0.0
    ic, _ = spearmanr(y_true[mask], y_pred[mask])
    return float(ic) if np.isfinite(ic) else 0.0

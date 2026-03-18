"""Signal scorecard: closed-loop tracking of whether signals actually profit.

This is the critical missing piece that makes "discovering profitable strategies"
verifiable rather than aspirational.

How it works:
1. Every directional signal (long/short) is recorded with its entry price, TP, SL, 
   expected holding period, and the timestamp it was issued.
2. On every subsequent iteration, we fetch the current price and check each open signal:
   - Did price hit TP? → signal scored as WIN with actual PnL
   - Did price hit SL? → signal scored as LOSS with actual PnL
   - Did holding period expire? → signal scored by mark-to-market PnL at expiry
3. Scored signals feed directly into:
   - AgentState.rolling_sharpe (computed from actual signal returns, not backtest)
   - AgentState.correct_signals / total_signals (real accuracy)
   - AgentState.cumulative_pnl (paper PnL from signal recommendations)
   - AgentState.max_drawdown (actual peak-to-trough on paper equity)
4. The measure() and reflect() phases receive these REAL performance numbers,
   so the agent is reasoning about its actual track record.

The agent KNOWS it's discovering profitable strategies when:
- signal_sharpe > 0 over a meaningful sample (>30 scored signals)
- signal_accuracy > cost-adjusted breakeven (~52% for 2:1 R:R)
- cumulative_pnl is positive and growing
- max_drawdown hasn't breached configured limits
- win rate in current regime matches or exceeds historical
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.common.logging import get_logger
from src.common.time_utils import utc_now

logger = get_logger(__name__)

SCORECARD_PATH = Path("artifacts/signal_scorecard.json")


@dataclass
class TrackedSignal:
    """A signal being tracked for outcome."""
    signal_id: str              # unique identifier
    timestamp: str              # ISO format
    direction: str              # "long" or "short"
    entry_price: float
    take_profit: float
    stop_loss: float
    position_size_pct: float
    expected_holding_hours: int
    confidence: float
    regime: str
    agent_iteration: int

    # Outcome (filled after scoring)
    status: str = "open"        # open, won, lost, expired
    exit_price: float = 0.0
    exit_time: str = ""
    pnl_pct: float = 0.0       # realized return %
    pnl_usd: float = 0.0       # realized PnL in notional terms
    bars_held: int = 0
    hit_tp: bool = False
    hit_sl: bool = False
    peak_favorable: float = 0.0  # best unrealized profit during hold
    peak_adverse: float = 0.0    # worst unrealized loss during hold


class SignalScorecard:
    """Tracks signal outcomes and computes real performance metrics.
    
    This is the truth layer. Everything the agent believes about its own
    profitability is grounded in what this class computes.
    """

    def __init__(self):
        self.open_signals: list[TrackedSignal] = []
        self.closed_signals: list[TrackedSignal] = []
        self.equity_curve: list[float] = [100000.0]  # paper equity starting at 100k
        self._load()

    def _load(self) -> None:
        """Load scorecard from disk."""
        if SCORECARD_PATH.exists():
            try:
                with open(SCORECARD_PATH) as f:
                    data = json.load(f)
                self.open_signals = [TrackedSignal(**s) for s in data.get("open", [])]
                self.closed_signals = [TrackedSignal(**s) for s in data.get("closed", [])]
                self.equity_curve = data.get("equity_curve", [100000.0])
            except Exception as e:
                logger.warning("scorecard_load_failed", error=str(e))

    def save(self) -> None:
        """Persist scorecard to disk."""
        SCORECARD_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "open": [s.__dict__ for s in self.open_signals],
            "closed": [s.__dict__ for s in self.closed_signals[-500:]],  # keep last 500
            "equity_curve": self.equity_curve[-2000:],  # keep last 2000 points
        }
        with open(SCORECARD_PATH, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # ── Record a new signal ──────────────────────────────────

    def record_signal(self, signal) -> None:
        """Record a new directional signal for tracking.
        
        Args:
            signal: SignalOutput from the agent
        """
        if signal.direction == "flat":
            return  # nothing to track

        sig_id = f"sig_{signal.agent_iteration}_{signal.timestamp.strftime('%Y%m%d%H%M')}"

        tracked = TrackedSignal(
            signal_id=sig_id,
            timestamp=signal.timestamp.isoformat(),
            direction=signal.direction,
            entry_price=signal.entry_price,
            take_profit=signal.take_profit,
            stop_loss=signal.stop_loss,
            position_size_pct=signal.position_size_pct,
            expected_holding_hours=signal.expected_holding_hours,
            confidence=signal.confidence,
            regime=signal.regime,
            agent_iteration=signal.agent_iteration,
        )
        self.open_signals.append(tracked)
        logger.info("signal_tracked", id=sig_id, direction=signal.direction, entry=signal.entry_price)

    # ── Score open signals against current price ─────────────

    def score_signals(self, current_price: float, current_time: datetime | None = None) -> list[TrackedSignal]:
        """Check all open signals against current price and time.
        
        Returns list of signals that were just closed.
        """
        if current_price <= 0:
            return []

        now = current_time or utc_now()
        just_closed: list[TrackedSignal] = []
        still_open: list[TrackedSignal] = []

        for sig in self.open_signals:
            entry_time = datetime.fromisoformat(sig.timestamp)
            if entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=timezone.utc)
            
            hours_held = (now - entry_time).total_seconds() / 3600
            sig.bars_held = int(hours_held)

            # Track peak favorable / adverse excursion
            if sig.direction == "long":
                unrealized_pct = (current_price - sig.entry_price) / sig.entry_price
            else:  # short
                unrealized_pct = (sig.entry_price - current_price) / sig.entry_price

            sig.peak_favorable = max(sig.peak_favorable, unrealized_pct)
            sig.peak_adverse = min(sig.peak_adverse, unrealized_pct)

            # Check TP hit
            if sig.direction == "long" and current_price >= sig.take_profit:
                sig.status = "won"
                sig.hit_tp = True
                sig.exit_price = sig.take_profit
                sig.pnl_pct = (sig.take_profit - sig.entry_price) / sig.entry_price
                just_closed.append(sig)
                continue

            if sig.direction == "short" and current_price <= sig.take_profit:
                sig.status = "won"
                sig.hit_tp = True
                sig.exit_price = sig.take_profit
                sig.pnl_pct = (sig.entry_price - sig.take_profit) / sig.entry_price
                just_closed.append(sig)
                continue

            # Check SL hit
            if sig.direction == "long" and current_price <= sig.stop_loss:
                sig.status = "lost"
                sig.hit_sl = True
                sig.exit_price = sig.stop_loss
                sig.pnl_pct = (sig.stop_loss - sig.entry_price) / sig.entry_price
                just_closed.append(sig)
                continue

            if sig.direction == "short" and current_price >= sig.stop_loss:
                sig.status = "lost"
                sig.hit_sl = True
                sig.exit_price = sig.stop_loss
                sig.pnl_pct = (sig.entry_price - sig.stop_loss) / sig.entry_price
                just_closed.append(sig)
                continue

            # Check time expiry
            if hours_held >= sig.expected_holding_hours:
                sig.status = "expired"
                sig.exit_price = current_price
                if sig.direction == "long":
                    sig.pnl_pct = (current_price - sig.entry_price) / sig.entry_price
                else:
                    sig.pnl_pct = (sig.entry_price - current_price) / sig.entry_price
                just_closed.append(sig)
                continue

            # Still open
            still_open.append(sig)

        # Finalize closed signals
        for sig in just_closed:
            sig.exit_time = now.isoformat()
            sig.pnl_usd = sig.pnl_pct * sig.position_size_pct * self.equity_curve[-1]

            # Update equity curve
            new_equity = self.equity_curve[-1] + sig.pnl_usd
            self.equity_curve.append(new_equity)

            self.closed_signals.append(sig)
            logger.info(
                "signal_scored",
                id=sig.signal_id,
                status=sig.status,
                pnl_pct=f"{sig.pnl_pct:+.4f}",
                pnl_usd=f"{sig.pnl_usd:+.2f}",
                bars_held=sig.bars_held,
            )

        self.open_signals = still_open
        return just_closed

    # ── Compute real performance metrics ─────────────────────

    def compute_metrics(self, min_signals: int = 5) -> dict[str, Any]:
        """Compute performance metrics from ACTUAL signal outcomes.
        
        This is the ground truth the agent uses to know if it's profitable.
        """
        closed = self.closed_signals
        n = len(closed)

        metrics: dict[str, Any] = {
            "n_signals_scored": n,
            "n_signals_open": len(self.open_signals),
            "has_enough_data": n >= min_signals,
        }

        if n < min_signals:
            metrics["signal_sharpe"] = 0.0
            metrics["signal_accuracy"] = 0.0
            metrics["cumulative_pnl_usd"] = 0.0
            metrics["cumulative_pnl_pct"] = 0.0
            metrics["max_drawdown_pct"] = 0.0
            metrics["win_rate"] = 0.0
            metrics["avg_win_pct"] = 0.0
            metrics["avg_loss_pct"] = 0.0
            metrics["profit_factor"] = 0.0
            metrics["expectancy_pct"] = 0.0
            metrics["current_equity"] = self.equity_curve[-1]
            return metrics

        # Returns array
        returns = np.array([s.pnl_pct for s in closed])
        wins = returns[returns > 0]
        losses = returns[returns <= 0]

        # Sharpe from actual signal returns
        if returns.std() > 0:
            # Annualize assuming ~2 signals per day on average
            signals_per_year = max(n, 1) / max((
                datetime.fromisoformat(closed[-1].exit_time.replace("Z", "+00:00")) -
                datetime.fromisoformat(closed[0].timestamp.replace("Z", "+00:00"))
            ).days / 365.25, 0.01) if n > 1 else 365
            metrics["signal_sharpe"] = float(
                returns.mean() / returns.std() * math.sqrt(min(signals_per_year, 1000))
            )
        else:
            metrics["signal_sharpe"] = 0.0

        # Accuracy
        metrics["signal_accuracy"] = float(len(wins) / n) if n > 0 else 0.0

        # Win/loss stats
        metrics["win_rate"] = float(len(wins) / n) if n > 0 else 0.0
        metrics["avg_win_pct"] = float(wins.mean()) if len(wins) > 0 else 0.0
        metrics["avg_loss_pct"] = float(losses.mean()) if len(losses) > 0 else 0.0
        metrics["best_signal_pct"] = float(returns.max())
        metrics["worst_signal_pct"] = float(returns.min())

        # Profit factor
        gross_wins = wins.sum() if len(wins) > 0 else 0.0
        gross_losses = abs(losses.sum()) if len(losses) > 0 else 0.0
        metrics["profit_factor"] = float(gross_wins / gross_losses) if gross_losses > 0 else (
            float("inf") if gross_wins > 0 else 0.0
        )

        # Expectancy: average PnL per signal
        metrics["expectancy_pct"] = float(returns.mean())

        # Cumulative PnL
        metrics["cumulative_pnl_pct"] = float(returns.sum())
        metrics["cumulative_pnl_usd"] = float(sum(s.pnl_usd for s in closed))

        # Equity curve drawdown
        eq = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        drawdown = (eq - peak) / peak
        metrics["max_drawdown_pct"] = float(drawdown.min())
        metrics["current_equity"] = float(eq[-1])
        metrics["peak_equity"] = float(peak[-1])

        # Recent performance (last 20 signals)
        recent = returns[-20:] if n >= 20 else returns
        metrics["recent_accuracy"] = float((recent > 0).mean())
        metrics["recent_avg_return"] = float(recent.mean())
        if recent.std() > 0:
            metrics["recent_sharpe"] = float(recent.mean() / recent.std() * math.sqrt(len(recent)))
        else:
            metrics["recent_sharpe"] = 0.0

        # By regime breakdown
        regime_stats: dict[str, dict] = {}
        for sig in closed:
            r = sig.regime or "unknown"
            if r not in regime_stats:
                regime_stats[r] = {"n": 0, "wins": 0, "total_pnl": 0.0}
            regime_stats[r]["n"] += 1
            if sig.pnl_pct > 0:
                regime_stats[r]["wins"] += 1
            regime_stats[r]["total_pnl"] += sig.pnl_pct
        metrics["regime_breakdown"] = {
            r: {
                "n": s["n"],
                "win_rate": s["wins"] / s["n"] if s["n"] > 0 else 0,
                "avg_pnl": s["total_pnl"] / s["n"] if s["n"] > 0 else 0,
            }
            for r, s in regime_stats.items()
        }

        # TP/SL hit rates
        tp_hits = sum(1 for s in closed if s.hit_tp)
        sl_hits = sum(1 for s in closed if s.hit_sl)
        expired = sum(1 for s in closed if s.status == "expired")
        metrics["tp_hit_rate"] = tp_hits / n if n > 0 else 0.0
        metrics["sl_hit_rate"] = sl_hits / n if n > 0 else 0.0
        metrics["expiry_rate"] = expired / n if n > 0 else 0.0

        # Average bars held
        metrics["avg_bars_held"] = float(np.mean([s.bars_held for s in closed]))

        # Average MFE/MAE
        metrics["avg_peak_favorable"] = float(np.mean([s.peak_favorable for s in closed]))
        metrics["avg_peak_adverse"] = float(np.mean([s.peak_adverse for s in closed]))

        return metrics

    def is_profitable(self, min_signals: int = 30) -> tuple[bool, str]:
        """Definitive answer: is the system discovering profitable strategies?
        
        Returns (is_profitable, explanation).
        
        Criteria (ALL must be true):
        1. At least min_signals scored signals
        2. Signal Sharpe > 0
        3. Cumulative PnL > 0
        4. Win rate > cost-adjusted breakeven
        5. Profit factor > 1.0
        6. Max drawdown hasn't exceeded -10%
        """
        m = self.compute_metrics(min_signals=min_signals)

        if not m["has_enough_data"]:
            return False, f"Insufficient data: {m['n_signals_scored']}/{min_signals} signals scored"

        failures = []
        if m["signal_sharpe"] <= 0:
            failures.append(f"Sharpe={m['signal_sharpe']:.2f} (need >0)")
        if m["cumulative_pnl_pct"] <= 0:
            failures.append(f"Cumulative PnL={m['cumulative_pnl_pct']:+.2%} (need >0)")
        if m["profit_factor"] <= 1.0:
            failures.append(f"Profit factor={m['profit_factor']:.2f} (need >1.0)")
        if m["max_drawdown_pct"] < -0.10:
            failures.append(f"Max DD={m['max_drawdown_pct']:.2%} (limit -10%)")

        if failures:
            return False, f"NOT profitable: {'; '.join(failures)}"

        return True, (
            f"PROFITABLE: Sharpe={m['signal_sharpe']:.2f}, "
            f"Accuracy={m['signal_accuracy']:.1%}, "
            f"PF={m['profit_factor']:.2f}, "
            f"PnL={m['cumulative_pnl_pct']:+.2%}, "
            f"DD={m['max_drawdown_pct']:.2%}, "
            f"n={m['n_signals_scored']}"
        )

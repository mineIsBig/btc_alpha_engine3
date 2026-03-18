"""Tests for the signal scorecard — verifies the closed feedback loop."""
import pytest
from datetime import datetime, timedelta, timezone


class TestSignalScorecard:
    def _make_signal(self, direction="long", entry=65000, tp=66300, sl=64350,
                     hours=8, confidence=0.7, iteration=1):
        from src.agent.signal_output import SignalOutput
        return SignalOutput(
            timestamp=datetime.now(timezone.utc),
            direction=direction,
            entry_price=entry,
            take_profit=tp,
            stop_loss=sl,
            position_size_pct=0.1,
            position_size_usd=10000,
            expected_holding_hours=hours,
            confidence=confidence,
            agent_iteration=iteration,
        )

    def test_record_and_score_tp_hit(self):
        """Signal should score as WON when price hits take profit."""
        from src.agent.scorecard import SignalScorecard
        sc = SignalScorecard()
        sc.open_signals = []
        sc.closed_signals = []

        sig = self._make_signal(direction="long", entry=65000, tp=66300, sl=64350)
        sc.record_signal(sig)
        assert len(sc.open_signals) == 1

        # Price hits TP
        closed = sc.score_signals(66500.0)
        assert len(closed) == 1
        assert closed[0].status == "won"
        assert closed[0].hit_tp is True
        assert closed[0].pnl_pct > 0

    def test_record_and_score_sl_hit(self):
        """Signal should score as LOST when price hits stop loss."""
        from src.agent.scorecard import SignalScorecard
        sc = SignalScorecard()
        sc.open_signals = []
        sc.closed_signals = []

        sig = self._make_signal(direction="long", entry=65000, tp=66300, sl=64350)
        sc.record_signal(sig)

        closed = sc.score_signals(64000.0)
        assert len(closed) == 1
        assert closed[0].status == "lost"
        assert closed[0].hit_sl is True
        assert closed[0].pnl_pct < 0

    def test_short_tp_hit(self):
        """Short signal should win when price drops to TP."""
        from src.agent.scorecard import SignalScorecard
        sc = SignalScorecard()
        sc.open_signals = []
        sc.closed_signals = []

        sig = self._make_signal(direction="short", entry=65000, tp=63700, sl=66300)
        sc.record_signal(sig)

        closed = sc.score_signals(63500.0)
        assert len(closed) == 1
        assert closed[0].status == "won"
        assert closed[0].pnl_pct > 0

    def test_expiry_scoring(self):
        """Signal should expire after holding period and score by mark-to-market."""
        from src.agent.scorecard import SignalScorecard, TrackedSignal
        sc = SignalScorecard()
        sc.open_signals = []
        sc.closed_signals = []

        # Create a signal that was issued 10 hours ago with 8h holding period
        sig = self._make_signal(direction="long", entry=65000, tp=66300, sl=64350, hours=8)
        sc.record_signal(sig)

        # Backdate the signal timestamp
        sc.open_signals[0].timestamp = (datetime.now(timezone.utc) - timedelta(hours=10)).isoformat()

        # Price is modestly up but didn't hit TP or SL
        closed = sc.score_signals(65500.0)
        assert len(closed) == 1
        assert closed[0].status == "expired"
        assert closed[0].pnl_pct > 0  # 65500 > 65000

    def test_flat_signal_not_tracked(self):
        """Flat signals should not be recorded."""
        from src.agent.scorecard import SignalScorecard
        from src.agent.signal_output import SignalOutput
        sc = SignalScorecard()
        sc.open_signals = []

        sig = SignalOutput(timestamp=datetime.now(timezone.utc), direction="flat", reasoning="no consensus")
        sc.record_signal(sig)
        assert len(sc.open_signals) == 0

    def test_equity_curve_updates(self):
        """Equity curve should update after scoring."""
        from src.agent.scorecard import SignalScorecard
        sc = SignalScorecard()
        sc.open_signals = []
        sc.closed_signals = []
        sc.equity_curve = [100000.0]

        sig = self._make_signal(direction="long", entry=65000, tp=66300, sl=64350)
        sc.record_signal(sig)

        sc.score_signals(66500.0)  # TP hit
        assert len(sc.equity_curve) == 2
        assert sc.equity_curve[-1] > 100000.0  # profit added

    def test_compute_metrics_with_enough_data(self):
        """Metrics should compute correctly with sufficient scored signals."""
        from src.agent.scorecard import SignalScorecard
        sc = SignalScorecard()
        sc.open_signals = []
        sc.closed_signals = []
        sc.equity_curve = [100000.0]

        # Generate 10 signals, alternate wins and losses
        for i in range(10):
            sig = self._make_signal(direction="long", entry=65000, tp=66300, sl=64350, iteration=i)
            sc.record_signal(sig)
            if i % 3 == 0:
                sc.score_signals(64000.0)  # SL hit (loss)
            else:
                sc.score_signals(66500.0)  # TP hit (win)

        metrics = sc.compute_metrics(min_signals=5)
        assert metrics["has_enough_data"] is True
        assert metrics["n_signals_scored"] == 10
        assert 0 < metrics["win_rate"] < 1
        assert metrics["profit_factor"] > 0
        assert metrics["avg_win_pct"] > 0
        assert metrics["avg_loss_pct"] < 0

    def test_is_profitable_with_insufficient_data(self):
        from src.agent.scorecard import SignalScorecard
        sc = SignalScorecard()
        sc.open_signals = []
        sc.closed_signals = []

        is_prof, reason = sc.is_profitable(min_signals=30)
        assert is_prof is False
        assert "Insufficient" in reason

    def test_peak_favorable_adverse_tracking(self):
        """MFE/MAE should be tracked during signal lifetime."""
        from src.agent.scorecard import SignalScorecard
        sc = SignalScorecard()
        sc.open_signals = []
        sc.closed_signals = []

        sig = self._make_signal(direction="long", entry=65000, tp=67000, sl=63000, hours=24)
        sc.record_signal(sig)

        # Price goes up first
        sc.score_signals(66000.0)  # favorable
        assert sc.open_signals[0].peak_favorable > 0

        # Then dips
        sc.score_signals(64500.0)  # adverse
        assert sc.open_signals[0].peak_adverse < 0

        # Still open (neither TP nor SL hit, not expired)
        assert len(sc.open_signals) == 1

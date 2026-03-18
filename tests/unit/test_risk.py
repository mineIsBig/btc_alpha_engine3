"""Tests for risk rules: daily loss, EOD trailing, kill switch, sizing."""
import pytest
from datetime import datetime, timedelta, timezone


class TestDailyLossRule:
    def test_breach_when_equity_drops_5pct(self):
        from src.risk.drawdown_rules import check_daily_loss
        result = check_daily_loss(equity=94000, opening_equity=100000, limit_pct=0.05)
        assert result.is_breached

    def test_no_breach_at_exactly_5pct(self):
        from src.risk.drawdown_rules import check_daily_loss
        result = check_daily_loss(equity=95000, opening_equity=100000, limit_pct=0.05)
        assert not result.is_breached

    def test_no_breach_above_floor(self):
        from src.risk.drawdown_rules import check_daily_loss
        result = check_daily_loss(equity=96000, opening_equity=100000, limit_pct=0.05)
        assert not result.is_breached

    def test_breach_with_small_equity(self):
        from src.risk.drawdown_rules import check_daily_loss
        result = check_daily_loss(equity=900, opening_equity=1000, limit_pct=0.05)
        assert result.is_breached


class TestEODTrailingRule:
    def test_breach_below_eod_hwm(self):
        from src.risk.drawdown_rules import check_eod_trailing
        # HWM was 110000, floor is 104500
        result = check_eod_trailing(equity=104000, eod_high_water_mark=110000, limit_pct=0.05)
        assert result.is_breached

    def test_no_breach_above_floor(self):
        from src.risk.drawdown_rules import check_eod_trailing
        result = check_eod_trailing(equity=106000, eod_high_water_mark=110000, limit_pct=0.05)
        assert not result.is_breached


class TestAccountStateDayReset:
    def test_day_reset_updates_floors(self):
        from src.risk.account_state import AccountState
        state = AccountState(initial_equity=100000)

        # Simulate profit: equity goes to 105000
        state.equity = 105000

        # Force day reset
        state._reset_day(105000, datetime(2024, 1, 2, tzinfo=timezone.utc))

        assert state.opening_equity == 105000
        assert state.daily_loss_floor == 105000 * 0.95
        assert state.eod_high_water_mark >= 105000
        assert state.can_trade is True

    def test_day_reset_enables_trading(self):
        from src.risk.account_state import AccountState
        state = AccountState(initial_equity=100000)

        # Simulate breach
        state.can_trade = False
        state.breach_reason = "test_breach"

        # New day reset
        state._reset_day(95000, datetime(2024, 1, 2, tzinfo=timezone.utc))
        assert state.can_trade is True
        assert state.breach_reason is None

    def test_eod_hwm_not_updated_intraday(self):
        from src.risk.account_state import AccountState
        state = AccountState(initial_equity=100000)

        # Intraday equity spike
        state.update(equity=110000, timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc))

        # EOD HWM should NOT be 110000 yet (only updates at day boundary)
        assert state.eod_high_water_mark == 100000  # Still initial


class TestKillSwitch:
    def test_triggers_on_consecutive_losses(self):
        from src.risk.kill_switch import KillSwitch
        ks = KillSwitch()
        ks.max_consecutive_losses = 3

        ks.record_fill(is_loss=True)
        ks.record_fill(is_loss=True)
        ok = ks.record_fill(is_loss=True)
        assert not ok
        assert ks.is_killed

    def test_resets_on_win(self):
        from src.risk.kill_switch import KillSwitch
        ks = KillSwitch()
        ks.max_consecutive_losses = 5

        ks.record_fill(is_loss=True)
        ks.record_fill(is_loss=True)
        ks.record_fill(is_loss=False)  # win resets counter
        ks.record_fill(is_loss=True)
        assert not ks.is_killed

    def test_manual_reset(self):
        from src.risk.kill_switch import KillSwitch
        ks = KillSwitch()
        ks._trigger("test")
        assert ks.is_killed
        ks.reset()
        assert not ks.is_killed


class TestSizing:
    def test_zero_on_no_headroom(self):
        from src.portfolio.sizing import VolatilitySizer
        sizer = VolatilitySizer(min_headroom_pct=0.02)
        result = sizer.compute_size(
            equity=100000, realized_vol=0.5, signal_strength=0.8,
            headroom_to_breach=0.01, current_price=40000,
        )
        assert result["target_size_usd"] == 0.0

    def test_respects_max_position(self):
        from src.portfolio.sizing import VolatilitySizer
        sizer = VolatilitySizer(max_position_pct=0.25)
        result = sizer.compute_size(
            equity=100000, realized_vol=0.1, signal_strength=1.0,
            headroom_to_breach=0.05, current_price=40000,
        )
        assert result["target_size_usd"] <= 100000 * 0.25

    def test_scales_with_signal_strength(self):
        from src.portfolio.sizing import VolatilitySizer
        sizer = VolatilitySizer()
        r1 = sizer.compute_size(equity=100000, realized_vol=0.5, signal_strength=0.3,
                                headroom_to_breach=0.05, current_price=40000)
        r2 = sizer.compute_size(equity=100000, realized_vol=0.5, signal_strength=0.9,
                                headroom_to_breach=0.05, current_price=40000)
        assert r2["target_size_usd"] >= r1["target_size_usd"]

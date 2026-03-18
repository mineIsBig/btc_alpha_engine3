"""Tests for risk rules: daily loss, EOD trailing, kill switch, sizing."""

from unittest.mock import patch


class TestDailyLossRule:
    def test_breach_when_equity_drops_5pct(self):
        from src.risk.drawdown_rules import DrawdownRuleEngine

        engine = DrawdownRuleEngine(
            daily_loss_limit_pct=0.05, eod_trailing_loss_limit_pct=0.05
        )
        engine.initialize_day(100000, force=True)

        can_trade, reason = engine.check_intraday(94000)
        assert not can_trade
        assert "DAILY_LOSS" in reason

    def test_no_breach_at_exactly_5pct(self):
        from src.risk.drawdown_rules import DrawdownRuleEngine

        engine = DrawdownRuleEngine(
            daily_loss_limit_pct=0.05, eod_trailing_loss_limit_pct=0.05
        )
        engine.initialize_day(100000, force=True)

        can_trade, reason = engine.check_intraday(95000)
        assert can_trade

    def test_no_breach_above_floor(self):
        from src.risk.drawdown_rules import DrawdownRuleEngine

        engine = DrawdownRuleEngine(
            daily_loss_limit_pct=0.05, eod_trailing_loss_limit_pct=0.05
        )
        engine.initialize_day(100000, force=True)

        can_trade, reason = engine.check_intraday(96000)
        assert can_trade

    def test_breach_with_small_equity(self):
        from src.risk.drawdown_rules import DrawdownRuleEngine

        engine = DrawdownRuleEngine(
            daily_loss_limit_pct=0.05, eod_trailing_loss_limit_pct=0.05
        )
        engine.initialize_day(1000, force=True)

        can_trade, reason = engine.check_intraday(900)
        assert not can_trade


class TestEODTrailingRule:
    def test_breach_below_eod_hwm(self):
        from src.risk.drawdown_rules import DrawdownRuleEngine

        engine = DrawdownRuleEngine(
            daily_loss_limit_pct=0.05, eod_trailing_loss_limit_pct=0.05
        )
        # Initialize at 100k, then update EOD HWM to 110k
        engine.initialize_day(100000, force=True)
        engine.end_of_day_update(110000)
        # Re-initialize for new day so daily floor is based on 110k
        engine.current_trading_date = None
        engine.initialize_day(110000, force=True)

        # EOD floor = 110000 * 0.95 = 104500
        can_trade, reason = engine.check_intraday(104000)
        assert not can_trade
        assert "EOD_TRAILING" in reason or "DAILY_LOSS" in reason

    def test_no_breach_above_floor(self):
        from src.risk.drawdown_rules import DrawdownRuleEngine

        engine = DrawdownRuleEngine(
            daily_loss_limit_pct=0.05, eod_trailing_loss_limit_pct=0.05
        )
        engine.initialize_day(110000, force=True)
        engine.eod_high_water_mark = 110000
        engine.eod_trailing_floor = 110000 * 0.95

        can_trade, reason = engine.check_intraday(106000)
        assert can_trade


class TestAccountState:
    def test_initial_state(self):
        from src.risk.account_state import AccountState

        state = AccountState(initial_equity=100000)
        assert state.equity == 100000
        assert state.cash == 100000
        assert state.unrealized_pnl == 0.0

    def test_record_fill_updates_cash(self):
        from src.risk.account_state import AccountState

        state = AccountState(initial_equity=100000)
        state.record_fill(500.0)  # profit
        assert state.cash == 100500
        assert state.realized_pnl_today == 500.0

    def test_reset_daily_clears_realized_pnl(self):
        from src.risk.account_state import AccountState

        state = AccountState(initial_equity=100000)
        state.record_fill(1000.0)
        assert state.realized_pnl_today == 1000.0

        state.reset_daily()
        assert state.realized_pnl_today == 0.0

    def test_update_from_positions(self):
        from src.risk.account_state import AccountState

        state = AccountState(initial_equity=100000)
        positions = [
            {
                "symbol": "BTC",
                "quantity": 1.0,
                "side": "long",
                "avg_entry_price": 40000,
            }
        ]
        state.update_from_positions(positions, {"BTC": 41000})
        assert state.unrealized_pnl == 1000.0
        assert state.equity == 101000.0


class TestKillSwitch:
    def test_triggers_on_consecutive_losses(self):
        from src.risk.kill_switch import KillSwitch

        ks = KillSwitch(max_consecutive_losses=3, cooldown_minutes=0)

        ks.record_fill(pnl=-100)
        ks.record_fill(pnl=-100)
        ok = ks.record_fill(pnl=-100)
        assert not ok
        assert ks.is_triggered

    def test_resets_on_win(self):
        from src.risk.kill_switch import KillSwitch

        ks = KillSwitch(max_consecutive_losses=5, cooldown_minutes=0)

        ks.record_fill(pnl=-100)
        ks.record_fill(pnl=-100)
        ks.record_fill(pnl=100)  # win resets counter
        ks.record_fill(pnl=-100)
        assert not ks.is_triggered

    def test_manual_reset(self):
        from src.risk.kill_switch import KillSwitch

        ks = KillSwitch(cooldown_minutes=0)
        ks._trigger("test")
        assert ks.is_triggered
        ks.reset()
        assert not ks.is_triggered


class TestSizing:
    def test_zero_on_no_headroom(self):
        from src.portfolio.sizing import VolatilitySizer

        sizer = VolatilitySizer(min_headroom_pct=0.02)
        result = sizer.compute_size(
            equity=100000,
            realized_vol=0.5,
            signal_strength=0.8,
            headroom_to_breach=0.01,
            current_price=40000,
        )
        assert result["target_size_usd"] == 0.0

    def test_respects_max_position(self):
        from src.portfolio.sizing import VolatilitySizer

        sizer = VolatilitySizer(max_position_pct=0.25)
        result = sizer.compute_size(
            equity=100000,
            realized_vol=0.1,
            signal_strength=1.0,
            headroom_to_breach=0.05,
            current_price=40000,
        )
        assert result["target_size_usd"] <= 100000 * 0.25

    def test_scales_with_signal_strength(self):
        from src.portfolio.sizing import VolatilitySizer

        sizer = VolatilitySizer()
        r1 = sizer.compute_size(
            equity=100000,
            realized_vol=0.5,
            signal_strength=0.3,
            headroom_to_breach=0.05,
            current_price=40000,
        )
        r2 = sizer.compute_size(
            equity=100000,
            realized_vol=0.5,
            signal_strength=0.9,
            headroom_to_breach=0.05,
            current_price=40000,
        )
        assert r2["target_size_usd"] >= r1["target_size_usd"]

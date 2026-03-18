"""Integration test stubs for end-to-end workflows."""
import pytest


class TestEndToEndResearch:
    @pytest.mark.slow
    def test_research_cycle_synthetic(self, synthetic_features_and_labels):
        """Run a mini research cycle on synthetic data."""
        from src.orchestrator.research_cycle import run_research_cycle
        candidates = run_research_cycle(
            dataset=synthetic_features_and_labels,
            horizons=[4],
        )
        # Should produce at least some candidates (or none if data too small)
        assert isinstance(candidates, list)


class TestEndToEndPaperTrade:
    def test_paper_broker_trade_cycle(self):
        """Test a single paper trade cycle."""
        from src.execution.paper_broker import PaperBroker
        from src.risk.risk_manager import RiskManager

        broker = PaperBroker()
        risk = RiskManager(initial_equity=100000)

        # Update equity
        can_trade = risk.update_equity(100000)
        assert can_trade

        # Submit order
        result = broker.submit_order("BTC", "buy", 0.1, 40000.0, reason="test")
        assert result["status"] == "filled"

        # Check equity after
        equity = broker.get_equity({"BTC": 40500.0}, cash=96000.0)
        can_trade = risk.update_equity(equity)
        assert can_trade

        # Flatten
        broker.flatten("BTC", 40500.0)
        pos = broker.get_position("BTC")
        assert pos["side"] == "flat"


class TestDataValidation:
    def test_ohlc_validation(self):
        """Test OHLC data validation."""
        import pandas as pd
        from src.data.validators import validate_ohlc

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
            "open": [100] * 10,
            "high": [105] * 10,
            "low": [95] * 10,
            "close": [102] * 10,
        })
        result = validate_ohlc(df, source="test")
        assert len(result) == 10

    def test_ohlc_validation_fixes_high_low(self):
        """Test that high < low gets fixed."""
        import pandas as pd
        from src.data.validators import validate_ohlc

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
            "open": [100, 100, 100],
            "high": [90, 105, 105],   # First row has high < low
            "low": [95, 95, 95],
            "close": [102, 102, 102],
        })
        result = validate_ohlc(df, source="test")
        assert (result["high"] >= result["low"]).all()

"""Tests for paper broker order lifecycle."""
import pytest


class TestPaperBroker:
    def test_submit_order_fills_immediately(self):
        from src.execution.paper_broker import PaperBroker
        broker = PaperBroker(slippage_bps=5.0, commission_bps=2.0)
        result = broker.submit_order(
            symbol="BTC", side="buy", quantity=0.1,
            price=40000.0, order_type="market", reason="test",
        )
        assert result["status"] == "filled"
        assert result["fill_price"] > 40000.0  # slippage on buy
        assert result["commission"] > 0

    def test_position_tracking(self):
        from src.execution.paper_broker import PaperBroker
        broker = PaperBroker()

        broker.submit_order("BTC", "buy", 0.5, 40000.0)
        pos = broker.get_position("BTC")
        assert pos["side"] == "long"
        assert pos["qty"] == pytest.approx(0.5, abs=0.001)

    def test_flatten_closes_position(self):
        from src.execution.paper_broker import PaperBroker
        broker = PaperBroker()

        broker.submit_order("BTC", "buy", 0.5, 40000.0)
        broker.flatten("BTC", 41000.0)
        pos = broker.get_position("BTC")
        assert pos["side"] == "flat"
        assert pos["qty"] == 0.0

    def test_short_position(self):
        from src.execution.paper_broker import PaperBroker
        broker = PaperBroker()

        broker.submit_order("BTC", "sell", 0.3, 40000.0)
        pos = broker.get_position("BTC")
        assert pos["side"] == "short"
        assert pos["qty"] == pytest.approx(0.3, abs=0.001)

    def test_equity_calculation(self):
        from src.execution.paper_broker import PaperBroker
        broker = PaperBroker(slippage_bps=0, commission_bps=0)

        broker.submit_order("BTC", "buy", 1.0, 40000.0)
        equity = broker.get_equity({"BTC": 41000.0}, cash=60000.0)
        # Long 1 BTC at 40000, now 41000 -> +1000 unrealized
        assert equity == pytest.approx(61000.0, abs=10)

    def test_flatten_all(self):
        from src.execution.paper_broker import PaperBroker
        broker = PaperBroker()

        broker.submit_order("BTC", "buy", 0.5, 40000.0)
        results = broker.flatten_all({"BTC": 41000.0})
        assert len(results) == 1
        pos = broker.get_position("BTC")
        assert pos["side"] == "flat"

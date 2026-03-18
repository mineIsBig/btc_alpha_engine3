"""Tests for paper broker order lifecycle."""

import pytest


class TestPaperBroker:
    def test_submit_order_fills_immediately(self):
        from src.execution.paper_broker import PaperBroker

        broker = PaperBroker(slippage_bps=5.0, commission_bps=2.0)
        broker.set_price("BTC", 40000.0)
        result = broker.submit_order(
            symbol="BTC",
            side="buy",
            quantity=0.1,
            order_type="market",
            reason="test",
        )
        assert result["status"] == "filled"
        assert result["fill_price"] > 40000.0  # slippage on buy
        assert result["commission"] > 0

    def test_position_tracking(self):
        from src.execution.paper_broker import PaperBroker

        broker = PaperBroker()
        broker.set_price("BTC", 40000.0)

        broker.submit_order("BTC", "buy", 0.5)
        pos = broker.get_position("BTC")
        assert pos["side"] == "long"
        assert pos["quantity"] == pytest.approx(0.5, abs=0.001)

    def test_flatten_all_closes_position(self):
        from src.execution.paper_broker import PaperBroker

        broker = PaperBroker()
        broker.set_price("BTC", 40000.0)

        broker.submit_order("BTC", "buy", 0.5)
        broker.set_price("BTC", 41000.0)
        results = broker.flatten_all(reason="test")
        assert len(results) >= 1
        pos = broker.get_position("BTC")
        assert pos["side"] == "flat"
        assert pos["quantity"] == 0.0

    def test_short_position(self):
        from src.execution.paper_broker import PaperBroker

        broker = PaperBroker()
        broker.set_price("BTC", 40000.0)

        broker.submit_order("BTC", "sell", 0.3)
        pos = broker.get_position("BTC")
        assert pos["side"] == "short"
        assert pos["quantity"] == pytest.approx(0.3, abs=0.001)

    def test_order_rejected_without_price(self):
        from src.execution.paper_broker import PaperBroker

        broker = PaperBroker(slippage_bps=0, commission_bps=0)
        result = broker.submit_order("BTC", "buy", 1.0)
        assert result["status"] == "rejected"
        assert result["reason"] == "no_price"

    def test_flatten_all(self):
        from src.execution.paper_broker import PaperBroker

        broker = PaperBroker()
        broker.set_price("BTC", 40000.0)

        broker.submit_order("BTC", "buy", 0.5)
        broker.set_price("BTC", 41000.0)
        results = broker.flatten_all()
        assert len(results) == 1
        pos = broker.get_position("BTC")
        assert pos["side"] == "flat"

"""Tests for autonomous agent, compute dispatcher, signal output, and monitoring."""

from datetime import datetime, timezone


class TestSignalOutput:
    def test_flat_signal(self):
        from src.agent.signal_output import SignalOutput

        sig = SignalOutput(
            timestamp=datetime.now(timezone.utc),
            direction="flat",
            reasoning="no consensus",
        )
        assert sig.direction == "flat"
        assert sig.position_size_usd == 0.0
        summary = sig.to_summary()
        assert "FLAT" in summary

    def test_long_signal(self):
        from src.agent.signal_output import SignalOutput

        sig = SignalOutput(
            timestamp=datetime.now(timezone.utc),
            direction="long",
            position_size_pct=0.1,
            position_size_usd=10000.0,
            entry_price=65000.0,
            take_profit=66300.0,
            stop_loss=64350.0,
            expected_return_pct=2.0,
            expected_holding_hours=8,
            risk_reward_ratio=2.0,
            confidence=0.7,
            regime="trend_up",
            consensus_horizons=[4, 8, 12],
            reasoning="strong consensus across horizons",
        )
        assert sig.direction == "long"
        assert sig.position_size_usd == 10000.0
        summary = sig.to_summary()
        assert "LONG" in summary
        assert "65,000" in summary
        assert "TP" in summary

    def test_short_signal(self):
        from src.agent.signal_output import SignalOutput

        sig = SignalOutput(
            timestamp=datetime.now(timezone.utc),
            direction="short",
            position_size_pct=0.08,
            position_size_usd=8000.0,
            entry_price=65000.0,
            take_profit=62400.0,
            stop_loss=66300.0,
            expected_return_pct=-4.0,
            expected_holding_hours=24,
            risk_reward_ratio=2.0,
            confidence=0.6,
        )
        assert sig.direction == "short"
        summary = sig.to_summary()
        assert "SHORT" in summary

    def test_signal_json_serialization(self):
        from src.agent.signal_output import SignalOutput

        sig = SignalOutput(
            timestamp=datetime.now(timezone.utc),
            direction="long",
            entry_price=65000.0,
        )
        json_str = sig.model_dump_json()
        assert "long" in json_str
        assert "65000" in json_str

        # Round-trip
        sig2 = SignalOutput.model_validate_json(json_str)
        assert sig2.direction == "long"


class TestAgentState:
    def test_state_persistence(self, tmp_path):
        from src.agent.signal_output import AgentState

        state = AgentState(
            iteration=5,
            cumulative_pnl=1500.0,
            rolling_sharpe=1.2,
            weaknesses=["model_drift", "regime_change"],
        )
        path = tmp_path / "state.json"
        with open(path, "w") as f:
            f.write(state.model_dump_json(indent=2))

        with open(path) as f:
            loaded = AgentState.model_validate_json(f.read())

        assert loaded.iteration == 5
        assert loaded.rolling_sharpe == 1.2
        assert "model_drift" in loaded.weaknesses


class TestComputeDispatcher:
    def test_dispatcher_init(self):
        """Dispatcher should initialize without crashing even if providers are unavailable."""
        from src.compute.dispatcher import ComputeDispatcher

        d = ComputeDispatcher()
        health = d.health()
        assert isinstance(health, dict)
        assert "targon" in health
        assert "chutes" in health
        assert "lium" in health

    def test_targon_client_init(self):
        from src.compute.targon_client import TargonClient

        client = TargonClient(api_key="test", base_url="http://localhost:9999/v1")
        assert client.api_key == "test"
        assert "localhost" in client.base_url
        client.close()

    def test_chutes_client_init(self):
        from src.compute.chutes_client import ChutesClient

        client = ChutesClient(api_key="test", base_url="http://localhost:9999/v1")
        assert client.api_key == "test"
        client.close()


class TestMetricsCollector:
    def test_record_signal(self):
        from src.monitoring.prometheus_metrics import MetricsCollector
        from src.agent.signal_output import SignalOutput

        collector = MetricsCollector(port=0)  # don't bind
        sig = SignalOutput(
            timestamp=datetime.now(timezone.utc),
            direction="long",
            confidence=0.75,
            position_size_usd=5000.0,
            expected_return_pct=2.5,
            entry_price=65000.0,
            take_profit=66300.0,
            stop_loss=64350.0,
            risk_reward_ratio=2.0,
        )
        # Should not raise
        collector.record_signal(sig)

    def test_record_iteration(self):
        from src.monitoring.prometheus_metrics import MetricsCollector

        collector = MetricsCollector(port=0)
        collector.record_iteration(
            iteration=1, duration=45.2, sharpe=1.1, drawdown=-0.03
        )

    def test_metrics_text_output(self):
        from src.monitoring.prometheus_metrics import MetricsCollector

        collector = MetricsCollector(port=0)
        collector.record_iteration(
            iteration=1, duration=10.0, sharpe=0.8, drawdown=-0.02
        )
        text = collector.get_metrics_text()
        assert isinstance(text, str)
        # Should contain some metric names
        assert len(text) > 0


class TestAlphaAgentUnit:
    def test_agent_creates_flat_signal_without_models(self):
        """Agent should produce a flat signal when no models are promoted."""
        from src.agent.alpha_agent import AlphaAgent

        agent = AlphaAgent()
        # Override compute to avoid LLM calls
        agent.compute = _MockDispatcher()

        signal = agent.generate_signal(
            run_results={"signals": {"final_side": 0, "reason": "no models"}},
            performance={},
            current_price=65000.0,
        )
        assert signal.direction == "flat"

    def test_agent_creates_long_signal(self):
        from src.agent.alpha_agent import AlphaAgent

        agent = AlphaAgent()
        agent.compute = _MockDispatcher()

        signal = agent.generate_signal(
            run_results={
                "signals": {
                    "final_side": 1,
                    "reason": "consensus_long",
                    "aggregated": {
                        4: {"side": 1, "consensus": 0.8},
                        8: {"side": 1, "consensus": 0.7},
                    },
                },
            },
            performance={"avg_oos_sharpe": 1.0},
            current_price=65000.0,
            equity=100000.0,
        )
        assert signal.direction == "long"
        assert signal.entry_price == 65000.0
        assert signal.take_profit > signal.entry_price
        assert signal.stop_loss < signal.entry_price
        assert signal.position_size_usd > 0


class _MockDispatcher:
    """Mock compute dispatcher that doesn't call any APIs."""

    def agent_inference(self, prompt, **kwargs):
        return '{"analysis": "mock", "weaknesses_found": [], "proposed_changes": []}'

    def agent_inference_json(self, prompt, **kwargs):
        return {
            "analysis": "mock",
            "weaknesses_found": [],
            "proposed_changes": [],
            "priority": "test",
            "expected_impact": "none",
            "reflection": "mock",
            "weaknesses": [],
            "regime_assessment": "neutral",
            "alpha_leakage": "none",
            "next_priorities": [],
        }

    def health(self):
        return {"targon": False, "chutes": False, "lium": False}

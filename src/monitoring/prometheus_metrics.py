"""Prometheus metrics for performance monitoring.

Exposes metrics via HTTP endpoint for Prometheus scraping.
Uses the prometheus_client library (or a minimal fallback).

Metrics exported:
- agent_iteration_total: counter of agent iterations
- agent_iteration_duration_seconds: histogram of iteration durations
- system_sharpe_ratio: gauge of rolling system Sharpe
- system_max_drawdown: gauge of maximum drawdown
- signal_direction: gauge (-1 short, 0 flat, 1 long)
- signal_confidence: gauge of latest signal confidence
- signal_position_size_usd: gauge of recommended position size
- signal_expected_return_pct: gauge of expected return
- model_count: gauge of active models
- error_total: counter of errors
- equity: gauge of current equity
- risk_headroom: gauge of headroom to breach
"""

from __future__ import annotations

import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

from src.common.logging import get_logger

logger = get_logger(__name__)

# Try to use prometheus_client, fall back to manual exposition
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        generate_latest,
        start_http_server,
    )

    _HAS_PROM_CLIENT = True
except ImportError:
    _HAS_PROM_CLIENT = False


class MetricsCollector:
    """Collect and expose Prometheus metrics."""

    def __init__(self, port: int = 9090):
        self.port = port
        self._started = False

        if _HAS_PROM_CLIENT:
            self.iteration_counter = Counter(
                "agent_iteration_total", "Total agent iterations"
            )
            self.iteration_duration = Histogram(
                "agent_iteration_duration_seconds",
                "Duration of agent iterations",
                buckets=[1, 5, 10, 30, 60, 120, 300, 600],
            )
            self.sharpe_gauge = Gauge(
                "system_sharpe_ratio", "Rolling system Sharpe ratio"
            )
            self.drawdown_gauge = Gauge("system_max_drawdown", "Maximum drawdown")
            self.signal_direction = Gauge(
                "signal_direction", "Latest signal direction (-1/0/1)"
            )
            self.signal_confidence = Gauge(
                "signal_confidence", "Latest signal confidence"
            )
            self.signal_size_usd = Gauge(
                "signal_position_size_usd", "Recommended position size USD"
            )
            self.signal_expected_ret = Gauge(
                "signal_expected_return_pct", "Expected return %"
            )
            self.signal_entry_price = Gauge("signal_entry_price", "Signal entry price")
            self.signal_tp = Gauge("signal_take_profit", "Take profit level")
            self.signal_sl = Gauge("signal_stop_loss", "Stop loss level")
            self.model_count = Gauge("model_count", "Number of active models")
            self.error_counter = Counter("error_total", "Total errors")
            self.equity_gauge = Gauge("equity_usd", "Current account equity")
            self.risk_headroom = Gauge("risk_headroom", "Headroom to risk breach")
            self.signal_rr_ratio = Gauge(
                "signal_risk_reward_ratio", "Risk/reward ratio"
            )
            self.agent_iteration_id = Gauge(
                "agent_iteration_id", "Current iteration number"
            )
        else:
            # Minimal fallback: store values in dict for manual exposition
            self._metrics: dict[str, float] = {}

    def start_server(self) -> None:
        """Start the Prometheus metrics HTTP server."""
        if self._started:
            return
        if _HAS_PROM_CLIENT:
            try:
                start_http_server(self.port)
                self._started = True
                logger.info("prometheus_server_started", port=self.port)
            except Exception as e:
                logger.warning("prometheus_server_failed", error=str(e))
        else:
            # Start minimal HTTP server
            server = _MinimalMetricsServer(self, port=self.port)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self._started = True
            logger.info("minimal_metrics_server_started", port=self.port)

    def record_signal(self, signal: Any) -> None:
        """Record a signal output to metrics."""
        if _HAS_PROM_CLIENT:
            direction_val = (
                1
                if signal.direction == "long"
                else -1 if signal.direction == "short" else 0
            )
            self.signal_direction.set(direction_val)
            self.signal_confidence.set(signal.confidence)
            self.signal_size_usd.set(signal.position_size_usd)
            self.signal_expected_ret.set(signal.expected_return_pct)
            self.signal_entry_price.set(signal.entry_price)
            self.signal_tp.set(signal.take_profit)
            self.signal_sl.set(signal.stop_loss)
            self.signal_rr_ratio.set(signal.risk_reward_ratio)
        else:
            self._metrics["signal_direction"] = (
                1
                if signal.direction == "long"
                else -1 if signal.direction == "short" else 0
            )
            self._metrics["signal_confidence"] = signal.confidence
            self._metrics["signal_position_size_usd"] = signal.position_size_usd
            self._metrics["signal_expected_return_pct"] = signal.expected_return_pct

    def record_iteration(
        self, iteration: int, duration: float, sharpe: float, drawdown: float
    ) -> None:
        """Record iteration metrics."""
        if _HAS_PROM_CLIENT:
            self.iteration_counter.inc()
            self.iteration_duration.observe(duration)
            self.sharpe_gauge.set(sharpe)
            self.drawdown_gauge.set(drawdown)
            self.agent_iteration_id.set(iteration)
        else:
            self._metrics["iteration"] = iteration
            self._metrics["duration"] = duration
            self._metrics["sharpe"] = sharpe
            self._metrics["drawdown"] = drawdown

    def record_error(self) -> None:
        if _HAS_PROM_CLIENT:
            self.error_counter.inc()
        else:
            self._metrics["errors"] = self._metrics.get("errors", 0) + 1

    def set_equity(self, equity: float) -> None:
        if _HAS_PROM_CLIENT:
            self.equity_gauge.set(equity)
        else:
            self._metrics["equity"] = equity

    def set_model_count(self, count: int) -> None:
        if _HAS_PROM_CLIENT:
            self.model_count.set(count)
        else:
            self._metrics["model_count"] = count

    def set_risk_headroom(self, headroom: float) -> None:
        if _HAS_PROM_CLIENT:
            self.risk_headroom.set(headroom)
        else:
            self._metrics["risk_headroom"] = headroom

    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text exposition format."""
        if _HAS_PROM_CLIENT:
            return generate_latest().decode("utf-8")
        # Manual format
        lines = []
        for key, val in self._metrics.items():
            lines.append(f"# TYPE {key} gauge")
            lines.append(f"{key} {val}")
        return "\n".join(lines)


class _MinimalMetricsHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler for metrics endpoint."""

    collector: MetricsCollector | None = None

    def do_GET(self):
        if self.path == "/metrics":
            body = self.collector.get_metrics_text() if self.collector else ""
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress HTTP logs


class _MinimalMetricsServer:
    def __init__(self, collector: MetricsCollector, port: int = 9090):
        _MinimalMetricsHandler.collector = collector
        self.server = HTTPServer(("0.0.0.0", port), _MinimalMetricsHandler)

    def serve_forever(self):
        self.server.serve_forever()


def start_metrics_server(port: int = 9090) -> MetricsCollector:
    """Create and start a metrics collector with HTTP server."""
    collector = MetricsCollector(port=port)
    collector.start_server()
    return collector

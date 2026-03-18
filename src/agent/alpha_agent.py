"""Autonomous Alpha Agent — Arbos-inspired ralph-loop.

Goal: design and evolve a system S that discovers profitable strategies
in changing environments.

KEY INSIGHT: The agent's profitability is grounded in the SignalScorecard.
Every directional signal is tracked forward in time. The scorecard checks
whether price hit TP, SL, or expired — computing REAL returns, Sharpe, and
drawdown from actual signal outcomes. The agent reasons about THIS data,
not just historical backtest metrics.
"""
from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.logging import get_logger
from src.common.time_utils import utc_now
from src.agent.signal_output import SignalOutput, AgentState
from src.agent.scorecard import SignalScorecard
from src.compute.dispatcher import ComputeDispatcher

logger = get_logger(__name__)

STATE_PATH = Path("artifacts/agent_state.json")

SYSTEM_PROMPT = """You are an autonomous quantitative research agent operating a BTC alpha engine.
Your goal: design and evolve a system S that discovers profitable strategies in changing environments.

CRITICAL: You receive REAL performance data from your own signal track record.
The signal scorecard tracks every signal you issue, checks whether price hit
your TP or SL, and computes actual Sharpe, accuracy, PnL. This is ground truth.
Reason about THIS data, not hypotheticals.

Be proactive. Be specific. Never stop improving."""


class AlphaAgent:
    """Autonomous alpha research agent with Arbos-style ralph-loop."""

    def __init__(self):
        self.compute = ComputeDispatcher()
        self.state = self._load_state()
        self.scorecard = SignalScorecard()

    def _load_state(self) -> AgentState:
        if STATE_PATH.exists():
            try:
                with open(STATE_PATH) as f:
                    return AgentState.model_validate_json(f.read())
            except Exception as e:
                logger.warning("state_load_failed", error=str(e))
        return AgentState()

    def _save_state(self) -> None:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_PATH, "w") as f:
            f.write(self.state.model_dump_json(indent=2))

    def design_or_modify(self, performance: dict[str, Any]) -> dict[str, Any]:
        is_profitable, verdict = self.scorecard.is_profitable(min_signals=10)
        prompt = f"""Iteration {self.state.iteration}. Analyze REAL signal performance and propose improvements.

PROFITABILITY VERDICT: {"YES " + verdict if is_profitable else "NO " + verdict}

REAL SIGNAL TRACK RECORD (ground truth):
{json.dumps(performance.get("scorecard_metrics", {}), indent=2, default=str)}

AGENT STATE:
- Iteration: {self.state.iteration} | Signals: {self.state.total_signals}
- Cumulative PnL: {self.state.cumulative_pnl:+.2f}
- Weaknesses: {json.dumps(self.state.weaknesses[-5:])}

Focus on SCORECARD METRICS (real outcomes) over backtest metrics.
Respond with JSON:
{{
    "analysis": "honest analysis based on REAL data",
    "is_system_profitable": {str(is_profitable).lower()},
    "weaknesses_found": ["specific weaknesses"],
    "proposed_changes": [
        {{"type": "feature|model|hyperparameter|risk|ensemble", "action": "add|remove|modify", "detail": "specific change"}}
    ],
    "priority": "most important fix"
}}"""
        try:
            return self.compute.agent_inference_json(prompt, system=SYSTEM_PROMPT)
        except Exception as e:
            logger.error("design_phase_failed", error=str(e))
            return {"analysis": "LLM unavailable", "weaknesses_found": [], "proposed_changes": [], "priority": "restore inference"}

    def run_system(self, features_df=None, raw_data: dict | None = None) -> dict[str, Any]:
        from src.features.feature_pipeline import build_features
        from src.models.registry import ModelArtifactRegistry
        from src.portfolio.ensemble import EnsembleAggregator
        from src.portfolio.consensus import ConsensusGate
        from src.portfolio.signal_schema import ModelSignal

        results: dict[str, Any] = {"signals": {}, "metrics": {}, "errors": []}
        try:
            if features_df is None:
                features_df = build_features(raw_data=raw_data) if raw_data else build_features()
            if features_df is None or features_df.empty:
                results["errors"].append("No feature data available")
                return results

            registry = ModelArtifactRegistry()
            ensemble = EnsembleAggregator()
            consensus = ConsensusGate()
            horizon_signals: dict[int, list[ModelSignal]] = {}

            for horizon in [1, 4, 8, 12, 24]:
                promoted = registry.get_promoted_models(horizon=horizon)
                if not promoted:
                    continue
                h_signals = []
                for model_info in promoted:
                    try:
                        model = registry.load_model(model_info["model_id"])
                        feat_cols = [c for c in model.feature_names if c in features_df.columns]
                        if len(feat_cols) < 3:
                            continue
                        X = features_df[feat_cols].fillna(0)
                        sig = model.get_signal(X)
                        h_signals.append(ModelSignal(
                            model_id=model_info["model_id"], horizon=horizon,
                            side=sig["side"], probability=sig["probability"],
                            confidence=sig["confidence"],
                            oos_sharpe=model_info.get("oos_sharpe", 0),
                        ))
                    except Exception as e:
                        results["errors"].append(f"Model {model_info['model_id']}: {e}")
                if h_signals:
                    horizon_signals[horizon] = h_signals

            agg_signals = {h: ensemble.aggregate(sigs) for h, sigs in horizon_signals.items()}
            final_side, reason = consensus.check(agg_signals)

            results["signals"] = {
                "final_side": final_side, "reason": reason,
                "horizon_signals": {h: len(s) for h, s in horizon_signals.items()},
                "aggregated": {h: {"side": a.target_side, "consensus": a.consensus_pct} for h, a in agg_signals.items()},
            }
            results["metrics"]["n_models"] = sum(len(s) for s in horizon_signals.values())
        except Exception as e:
            results["errors"].append(f"run_system: {e}")
            logger.error("run_system_failed", error=str(e))
        return results

    def measure(self, run_results: dict[str, Any], current_price: float) -> dict[str, Any]:
        if current_price > 0:
            self.scorecard.score_signals(current_price)

        scorecard_metrics = self.scorecard.compute_metrics(min_signals=5)

        wf_metrics: dict[str, Any] = {}
        try:
            from src.storage.database import get_session
            from src.storage.models import WalkForwardRun
            session = get_session()
            latest_runs = session.query(WalkForwardRun).order_by(WalkForwardRun.created_at.desc()).limit(20).all()
            if latest_runs:
                sharpes = [r.sharpe for r in latest_runs if r.sharpe is not None]
                wf_metrics["avg_oos_sharpe"] = sum(sharpes) / len(sharpes) if sharpes else 0
                drawdowns = [r.max_drawdown for r in latest_runs if r.max_drawdown is not None]
                wf_metrics["avg_max_drawdown"] = sum(drawdowns) / len(drawdowns) if drawdowns else 0
            session.close()
        except Exception:
            pass

        self.state.rolling_sharpe = scorecard_metrics.get("signal_sharpe", 0.0)
        self.state.max_drawdown = scorecard_metrics.get("max_drawdown_pct", 0.0)
        self.state.cumulative_pnl = scorecard_metrics.get("cumulative_pnl_usd", 0.0)
        self.state.correct_signals = int(scorecard_metrics.get("signal_accuracy", 0) * scorecard_metrics.get("n_signals_scored", 0))

        return {"scorecard_metrics": scorecard_metrics, "walk_forward_metrics": wf_metrics,
                "n_models": run_results.get("metrics", {}).get("n_models", 0), **wf_metrics}

    def reflect(self, performance: dict[str, Any]) -> dict[str, Any]:
        sc = performance.get("scorecard_metrics", {})
        prompt = f"""Reflecting on iteration {self.state.iteration}.

REAL PERFORMANCE:
- Sharpe: {sc.get('signal_sharpe', 0):.3f} | Accuracy: {sc.get('signal_accuracy', 0):.1%}
- PnL: {sc.get('cumulative_pnl_pct', 0):+.2%} | PF: {sc.get('profit_factor', 0):.2f}
- DD: {sc.get('max_drawdown_pct', 0):.2%} | TP Rate: {sc.get('tp_hit_rate', 0):.1%} | SL Rate: {sc.get('sl_hit_rate', 0):.1%}
- Recent Sharpe: {sc.get('recent_sharpe', 0):.2f} | Recent Acc: {sc.get('recent_accuracy', 0):.1%}
- Regime breakdown: {json.dumps(sc.get('regime_breakdown', {}), default=str)}

WEAKNESSES: {json.dumps(self.state.weaknesses[-5:])}

Be honest. Respond with JSON:
{{
    "reflection": "analysis grounded in real data",
    "is_actually_profitable": true/false,
    "weaknesses": ["specific weaknesses"],
    "regime_assessment": "which regimes work",
    "tp_sl_assessment": "calibration quality",
    "next_priorities": ["by impact"]
}}"""
        try:
            return self.compute.agent_inference_json(prompt, system=SYSTEM_PROMPT)
        except Exception as e:
            logger.error("reflect_failed", error=str(e))
            return {"reflection": "LLM unavailable", "is_actually_profitable": False,
                    "weaknesses": [], "regime_assessment": "unknown", "tp_sl_assessment": "unknown", "next_priorities": []}

    def improve(self, design: dict, reflection: dict) -> list[str]:
        improvements = []
        for change in design.get("proposed_changes", [])[:3]:
            desc = f"{change.get('type', '')}/{change.get('action', '')}: {change.get('detail', '')}"
            improvements.append(desc)
            logger.info("improvement_applied", change=desc)
        self.state.weaknesses = reflection.get("weaknesses", [])[:10]
        self.state.improvements_applied.extend(improvements)
        if len(self.state.improvements_applied) > 50:
            self.state.improvements_applied = self.state.improvements_applied[-50:]
        self.state.last_reflection = reflection.get("reflection", "")
        return improvements

    def generate_signal(self, run_results: dict[str, Any], performance: dict[str, Any],
                        current_price: float = 0.0, equity: float = 100000.0) -> SignalOutput:
        signals = run_results.get("signals", {})
        final_side = signals.get("final_side", 0)
        reason = signals.get("reason", "no_signals")
        direction = "long" if final_side == 1 else "short" if final_side == -1 else "flat"

        if direction == "flat" or current_price <= 0:
            return SignalOutput(timestamp=utc_now(), direction="flat", reasoning=reason,
                              agent_iteration=self.state.iteration, system_sharpe=self.state.rolling_sharpe,
                              system_drawdown=self.state.max_drawdown)

        atr_pct = 0.02
        if final_side == 1:
            take_profit = current_price * (1 + atr_pct * 2)
            stop_loss = current_price * (1 - atr_pct)
        else:
            take_profit = current_price * (1 - atr_pct * 2)
            stop_loss = current_price * (1 + atr_pct)

        agg = signals.get("aggregated", {})
        consensus_vals = [v.get("consensus", 0) for v in agg.values()] if agg else [0.5]
        confidence = max(consensus_vals) if consensus_vals else 0.5
        size_pct = min(0.15, confidence * 0.2)
        consensus_horizons = [h for h, a in agg.items() if a.get("side", 0) == final_side]

        return SignalOutput(
            timestamp=utc_now(), direction=direction,
            position_size_pct=size_pct, position_size_usd=round(equity * size_pct, 2),
            entry_price=current_price, take_profit=round(take_profit, 2), stop_loss=round(stop_loss, 2),
            expected_return_pct=round(atr_pct * 2 * final_side * 100, 2),
            expected_holding_hours=max(consensus_horizons) if consensus_horizons else 4,
            risk_reward_ratio=round(abs(take_profit - current_price) / max(abs(stop_loss - current_price), 0.01), 2),
            confidence=confidence, consensus_horizons=consensus_horizons, reasoning=reason,
            agent_iteration=self.state.iteration, system_sharpe=self.state.rolling_sharpe,
            system_drawdown=self.state.max_drawdown,
        )

    def iterate(self, current_price: float = 0.0, equity: float = 100000.0,
                raw_data: dict | None = None) -> SignalOutput:
        """Run one full ralph-loop iteration.

        The critical difference: every iteration starts by SCORING previous
        signals against actual price, and the entire reasoning chain operates
        on real performance data from the scorecard.
        """
        self.state.iteration += 1
        logger.info("agent_iteration_start", iteration=self.state.iteration)

        try:
            # ═══ SCORE PREVIOUS SIGNALS FIRST ═══
            if current_price > 0:
                just_closed = self.scorecard.score_signals(current_price)
                for sig in just_closed:
                    logger.info("signal_outcome", id=sig.signal_id, status=sig.status,
                               pnl=f"{sig.pnl_pct:+.2%}", bars=sig.bars_held)

            # Check real profitability
            is_profitable, verdict = self.scorecard.is_profitable(min_signals=10)
            logger.info("profitability_check", profitable=is_profitable, verdict=verdict)

            prev_perf = {"scorecard_metrics": self.scorecard.compute_metrics(),
                        "is_profitable": is_profitable, "verdict": verdict, "iteration": self.state.iteration}

            design = self.design_or_modify(prev_perf)
            run_results = self.run_system(raw_data=raw_data)
            performance = self.measure(run_results, current_price)
            reflection = self.reflect(performance)
            improvements = self.improve(design, reflection)

            signal = self.generate_signal(run_results, performance, current_price, equity)
            self.state.total_signals += 1
            self.state.last_signal = signal

            # Track new signal in scorecard
            self.scorecard.record_signal(signal)
            self.scorecard.save()
            self._save_state()

            logger.info("agent_iteration_complete", iteration=self.state.iteration,
                       direction=signal.direction, profitable=is_profitable)
            return signal

        except Exception as e:
            logger.error("agent_iteration_failed", error=str(e), trace=traceback.format_exc())
            self._save_state()
            return SignalOutput(timestamp=utc_now(), direction="flat",
                              reasoning=f"Agent error: {e}", agent_iteration=self.state.iteration)

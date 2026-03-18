"""Autonomous Alpha Agent — Arbos-inspired ralph-loop with guardrails.

Goal: design and evolve a system S that discovers profitable strategies
in changing environments.

KEY INSIGHT: The agent's profitability is grounded in the SignalScorecard.
Every directional signal is tracked forward in time. The scorecard checks
whether price hit TP, SL, or expired — computing REAL returns, Sharpe, and
drawdown from actual signal outcomes. The agent reasons about THIS data,
not just historical backtest metrics.

EVOLUTION: The agent truly evolves by executing approved changes:
- Features can be enabled/disabled and new interaction features created
- Model hyperparameters are tuned and retraining triggered automatically
- Ensemble consensus thresholds and weighting are adjusted dynamically
- TP/SL bands are calibrated based on real signal outcomes
- Underperforming models are retired and fresh models promoted

GUARDRAILS:
1. Schema validation: every LLM response is validated against Pydantic schemas.
   Malformed responses fail gracefully to "no change".
2. Scope enforcement: the agent cannot modify risk limits or infrastructure.
   Only feature/model/hyperparameter/ensemble changes are allowed.
3. Validation gate: proposed changes are checked against recent performance.
   In emergency states (deep drawdown, crashed Sharpe), only conservative
   modifications are allowed.
4. Changelog: every change is recorded with rollback capability.
5. Executor safety bounds: all numeric parameters are clamped within safe ranges.
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
from src.agent.guardrails import (
    validate_design_response,
    validate_reflect_response,
    filter_changes_by_scope,
    validation_gate,
    AgentChangelog,
    DesignResponse,
    ReflectResponse,
)
from src.agent.executor import ChangeExecutor, should_retrain
from src.agent.evolution_config import load_evolution_config, save_evolution_config
from src.compute.dispatcher import ComputeDispatcher

logger = get_logger(__name__)

STATE_PATH = Path("artifacts/agent_state.json")

SYSTEM_PROMPT = """You are an autonomous quantitative research agent operating a BTC alpha engine.
Your goal: design and evolve a system S that discovers profitable strategies in changing environments.

CRITICAL: You receive REAL performance data from your own signal track record.
The signal scorecard tracks every signal you issue, checks whether price hit
your TP or SL, and computes actual Sharpe, accuracy, PnL. This is ground truth.
Reason about THIS data, not hypotheticals.

YOUR CHANGES ARE EXECUTED AUTOMATICALLY. When you propose changes, they are:
- Feature changes: features are enabled/disabled, new interaction features created
- Hyperparameter changes: model params are updated (n_estimators, learning_rate, max_depth, etc.)
  and retraining is triggered automatically when needed
- Ensemble changes: consensus thresholds, Sharpe weighting power, horizon agreement adjusted
- TP/SL calibration: ATR multipliers, position sizing factors adjusted

Be SPECIFIC with numbers. Instead of "increase learning rate", say "set learning_rate to 0.1 for lgbm".
Instead of "add interaction feature", say "multiply funding_rate and oi_change_1h".
Instead of "widen TP", say "set atr_multiplier_tp to 2.5".

SCOPE CONSTRAINTS:
You may propose changes to: features, models, hyperparameters, ensemble logic.
You may NOT propose changes to: risk limits, daily loss limits, position size caps,
kill switch thresholds, or infrastructure settings. These are operator-controlled.

Be proactive. Be specific. Never stop improving."""


class AlphaAgent:
    """Autonomous alpha research agent with Arbos-style ralph-loop and guardrails."""

    def __init__(self):
        self.compute = ComputeDispatcher()
        self.state = self._load_state()
        self.scorecard = SignalScorecard()
        self.changelog = AgentChangelog()
        self.evolution_config = load_evolution_config()
        self.executor = ChangeExecutor(self.evolution_config)

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

    def design_or_modify(self, performance: dict[str, Any]) -> DesignResponse:
        """LLM designs or modifies the system based on real performance.

        Returns a validated DesignResponse. Malformed LLM output falls back
        to safe defaults (no changes).
        """
        is_profitable, verdict = self.scorecard.is_profitable(min_signals=10)
        prompt = f"""Iteration {self.state.iteration}. Analyze REAL signal performance and propose improvements.

PROFITABILITY VERDICT: {"YES " + verdict if is_profitable else "NO " + verdict}

REAL SIGNAL TRACK RECORD (ground truth):
{json.dumps(performance.get("scorecard_metrics", {}), indent=2, default=str)}

AGENT STATE:
- Iteration: {self.state.iteration} | Signals: {self.state.total_signals}
- Cumulative PnL: {self.state.cumulative_pnl:+.2f}
- Weaknesses: {json.dumps(self.state.weaknesses[-5:])}

SCOPE: You may propose changes to features, models, hyperparameters, or ensemble logic.
You may NOT propose changes to risk limits, position size caps, or infrastructure.

Focus on SCORECARD METRICS (real outcomes) over backtest metrics.
Respond with JSON:
{{
    "analysis": "honest analysis based on REAL data",
    "is_system_profitable": {str(is_profitable).lower()},
    "weaknesses_found": ["specific weaknesses"],
    "proposed_changes": [
        {{"type": "feature|model|hyperparameter|ensemble", "action": "add|remove|modify", "detail": "specific change"}}
    ],
    "priority": "most important fix"
}}"""
        try:
            raw = self.compute.agent_inference_json(prompt, system=SYSTEM_PROMPT)
            validated = validate_design_response(raw)
            if validated.analysis == "LLM response failed validation":
                logger.warning("design_llm_response_invalid", raw_type=type(raw).__name__)
            return validated
        except json.JSONDecodeError as e:
            logger.error("design_json_parse_failed", error=str(e))
            return DesignResponse(analysis="LLM returned invalid JSON")
        except Exception as e:
            logger.error("design_phase_failed", error=str(e))
            return DesignResponse(analysis=f"LLM unavailable: {e}")

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

    def reflect(self, performance: dict[str, Any]) -> ReflectResponse:
        """LLM reflects on performance. Returns validated ReflectResponse."""
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
            raw = self.compute.agent_inference_json(prompt, system=SYSTEM_PROMPT)
            validated = validate_reflect_response(raw)
            if validated.reflection == "LLM response failed validation":
                logger.warning("reflect_llm_response_invalid")
            return validated
        except json.JSONDecodeError as e:
            logger.error("reflect_json_parse_failed", error=str(e))
            return ReflectResponse(reflection="LLM returned invalid JSON")
        except Exception as e:
            logger.error("reflect_failed", error=str(e))
            return ReflectResponse(reflection=f"LLM unavailable: {e}")

    def improve(self, design: DesignResponse, reflection: ReflectResponse,
                scorecard_metrics: dict[str, Any]) -> list[str]:
        """Apply improvements with full guardrails pipeline + REAL EXECUTION.

        Pipeline:
        1. Take top 3 proposed changes from design
        2. Validate each change against scope constraints (block risk changes)
        3. Run through validation gate (check system health)
        4. EXECUTE approved changes via ChangeExecutor (updates EvolutionConfig)
        5. Record results in changelog
        6. Store in agent state for next iteration's context
        """
        proposed = design.proposed_changes[:3]

        if not proposed:
            logger.info("improve_no_changes_proposed")
            return []

        # ── Step 1: Scope enforcement ────────────────────────
        scope_approved, scope_rejections = filter_changes_by_scope(proposed)

        for rej in scope_rejections:
            self.changelog.record(
                iteration=self.state.iteration,
                change=next(c for c in proposed if c.detail[:100] == rej["detail"]),
                status="rejected",
                validation_result=f"Scope violation: {rej['reason']}",
                scorecard_snapshot=scorecard_metrics,
            )

        if not scope_approved:
            logger.info("improve_all_changes_rejected_by_scope", n_rejected=len(scope_rejections))
            self._update_state_from_reflection(reflection)
            return []

        # ── Step 2: Validation gate ──────────────────────────
        gate_approved, gate_rejections = validation_gate(scope_approved, scorecard_metrics)

        for rej in gate_rejections:
            matching = [c for c in scope_approved if c.detail[:100] == rej["detail"]]
            if matching:
                self.changelog.record(
                    iteration=self.state.iteration,
                    change=matching[0],
                    status="failed_validation",
                    validation_result=f"Gate rejection: {rej['reason']}",
                    scorecard_snapshot=scorecard_metrics,
                )

        # ── Step 3: EXECUTE approved changes ─────────────────
        improvements = []
        execution_results = []

        if gate_approved:
            execution_results = self.executor.execute_batch(
                gate_approved, self.state.iteration, scorecard_metrics
            )
            # Reload evolution config after execution
            self.evolution_config = load_evolution_config()

        for change, exec_result in zip(gate_approved, execution_results):
            desc = f"{change.type}/{change.action}: {change.detail}"
            succeeded = exec_result.get("success", False)
            exec_msg = exec_result.get("message", exec_result.get("error", ""))

            if succeeded:
                improvements.append(desc)
                status = "applied"
                validation_msg = f"Executed: {exec_msg}"
            else:
                status = "execution_failed"
                validation_msg = f"Execution failed: {exec_msg}"

            self.changelog.record(
                iteration=self.state.iteration,
                change=change,
                status=status,
                validation_result=validation_msg,
                scorecard_snapshot=scorecard_metrics,
            )
            logger.info("improvement_result", change=desc, success=succeeded, msg=exec_msg)

        self.changelog.save()

        # ── Step 4: Update agent state ───────────────────────
        self._update_state_from_reflection(reflection)
        self.state.improvements_applied.extend(improvements)
        if len(self.state.improvements_applied) > 50:
            self.state.improvements_applied = self.state.improvements_applied[-50:]

        # Track evolution state
        self.state.evolution_version = self.evolution_config.version
        self.state.retrain_pending = should_retrain(
            self.evolution_config, self.state.iteration, scorecard_metrics
        )

        logger.info("improve_summary",
                     proposed=len(proposed),
                     scope_rejected=len(scope_rejections),
                     gate_rejected=len(gate_rejections),
                     executed=len([r for r in execution_results if r.get("success")]),
                     failed=len([r for r in execution_results if not r.get("success")]),
                     applied=len(improvements))

        return improvements

    def _update_state_from_reflection(self, reflection: ReflectResponse) -> None:
        """Update agent state with reflection results."""
        self.state.weaknesses = reflection.weaknesses[:10]
        self.state.last_reflection = reflection.reflection

    def generate_signal(self, run_results: dict[str, Any], performance: dict[str, Any],
                        current_price: float = 0.0, equity: float = 100000.0) -> SignalOutput:
        """Generate signal using dynamic TP/SL calibration from EvolutionConfig."""
        signals = run_results.get("signals", {})
        final_side = signals.get("final_side", 0)
        reason = signals.get("reason", "no_signals")
        direction = "long" if final_side == 1 else "short" if final_side == -1 else "flat"

        if direction == "flat" or current_price <= 0:
            return SignalOutput(timestamp=utc_now(), direction="flat", reasoning=reason,
                              agent_iteration=self.state.iteration, system_sharpe=self.state.rolling_sharpe,
                              system_drawdown=self.state.max_drawdown)

        # Use dynamic TP/SL from evolution config
        tp_sl = self.evolution_config.tp_sl
        atr_pct = tp_sl.atr_pct

        if final_side == 1:
            take_profit = current_price * (1 + atr_pct * tp_sl.atr_multiplier_tp)
            stop_loss = current_price * (1 - atr_pct * tp_sl.atr_multiplier_sl)
        else:
            take_profit = current_price * (1 - atr_pct * tp_sl.atr_multiplier_tp)
            stop_loss = current_price * (1 + atr_pct * tp_sl.atr_multiplier_sl)

        agg = signals.get("aggregated", {})
        consensus_vals = [v.get("consensus", 0) for v in agg.values()] if agg else [0.5]
        confidence = max(consensus_vals) if consensus_vals else 0.5
        size_pct = min(tp_sl.max_position_size_pct, confidence * tp_sl.confidence_sizing_factor)
        consensus_horizons = [h for h, a in agg.items() if a.get("side", 0) == final_side]

        expected_return = atr_pct * tp_sl.atr_multiplier_tp * final_side * 100

        return SignalOutput(
            timestamp=utc_now(), direction=direction,
            position_size_pct=size_pct, position_size_usd=round(equity * size_pct, 2),
            entry_price=current_price, take_profit=round(take_profit, 2), stop_loss=round(stop_loss, 2),
            expected_return_pct=round(expected_return, 2),
            expected_holding_hours=max(consensus_horizons) if consensus_horizons else 4,
            risk_reward_ratio=round(abs(take_profit - current_price) / max(abs(stop_loss - current_price), 0.01), 2),
            confidence=confidence, consensus_horizons=consensus_horizons, reasoning=reason,
            agent_iteration=self.state.iteration, system_sharpe=self.state.rolling_sharpe,
            system_drawdown=self.state.max_drawdown,
        )

    def check_retrain_needed(self, scorecard_metrics: dict[str, Any]) -> bool:
        """Check if model retraining should be triggered."""
        return should_retrain(self.evolution_config, self.state.iteration, scorecard_metrics)

    def run_retrain(self) -> dict[str, Any]:
        """Execute autonomous retraining cycle.

        Uses evolved hyperparameters from EvolutionConfig and automatically
        promotes/retires models based on walk-forward results.
        """
        from src.orchestrator.research_cycle import auto_retrain_and_promote
        result = auto_retrain_and_promote(iteration=self.state.iteration)
        # Reload config after retrain (retrain updates last_retrain_iteration)
        self.evolution_config = load_evolution_config()
        self.state.last_retrain_iteration = self.state.iteration
        self.state.retrain_pending = False
        return result

    def manage_model_lifecycle(self, scorecard_metrics: dict[str, Any]) -> dict[str, Any]:
        """Evaluate and retire underperforming promoted models based on live signal performance.

        Uses per-model signal tracking from scorecard to identify models that are
        dragging down ensemble performance.
        """
        from src.models.registry import ModelArtifactRegistry
        lifecycle = self.evolution_config.model_lifecycle
        registry = ModelArtifactRegistry()
        result = {"retired": [], "kept": []}

        n_signals = scorecard_metrics.get("n_signals_scored", 0)
        if n_signals < lifecycle.min_signals_for_eval:
            return result

        promoted = registry.get_promoted_models()
        for model_info in promoted:
            sharpe = model_info.get("oos_sharpe") or 0
            if sharpe < lifecycle.retire_sharpe_threshold:
                try:
                    registry.retire_model(model_info["model_id"])
                    result["retired"].append(model_info["model_id"])
                    logger.info("model_auto_retired", model_id=model_info["model_id"], sharpe=sharpe)
                except Exception as e:
                    logger.warning("model_retire_failed", model_id=model_info["model_id"], error=str(e))
            else:
                result["kept"].append(model_info["model_id"])

        # Update state
        self.state.model_population_size = len(result["kept"])
        return result

    def iterate(self, current_price: float = 0.0, equity: float = 100000.0,
                raw_data: dict | None = None) -> SignalOutput:
        """Run one full ralph-loop iteration with guardrails and real evolution.

        The critical difference from advisory-only agents:
        1. Every iteration starts by SCORING previous signals against actual price
        2. The entire reasoning chain operates on real performance data
        3. Approved changes are EXECUTED (features disabled, hyperparams tuned,
           ensemble adjusted, TP/SL calibrated) — not just logged
        4. Retraining is triggered automatically when performance degrades
        5. Models are promoted/retired based on live performance

        Guardrails applied:
        - LLM responses validated against Pydantic schemas
        - Proposed changes filtered by scope (risk changes blocked)
        - Validation gate checks system health before allowing changes
        - All changes recorded in changelog with rollback capability
        - Executor clamps all numeric params within safe bounds
        """
        self.state.iteration += 1
        logger.info("agent_iteration_start", iteration=self.state.iteration,
                     evolution_version=self.evolution_config.version)

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

            scorecard_metrics = self.scorecard.compute_metrics()
            prev_perf = {"scorecard_metrics": scorecard_metrics,
                        "is_profitable": is_profitable, "verdict": verdict, "iteration": self.state.iteration}

            # ═══ MODEL LIFECYCLE — retire underperformers ═══
            lifecycle_result = self.manage_model_lifecycle(scorecard_metrics)
            if lifecycle_result["retired"]:
                logger.info("models_retired_this_iteration", retired=lifecycle_result["retired"])

            # ═══ DESIGN (with schema validation) ═══
            design = self.design_or_modify(prev_perf)

            # ═══ RUN SYSTEM (uses evolved features + ensemble params) ═══
            run_results = self.run_system(raw_data=raw_data)

            # ═══ MEASURE ═══
            performance = self.measure(run_results, current_price)

            # ═══ REFLECT (with schema validation) ═══
            reflection = self.reflect(performance)

            # ═══ IMPROVE (with scope + validation gate + EXECUTION) ═══
            improvements = self.improve(design, reflection, performance.get("scorecard_metrics", {}))

            # ═══ GENERATE SIGNAL (uses evolved TP/SL calibration) ═══
            signal = self.generate_signal(run_results, performance, current_price, equity)
            self.state.total_signals += 1
            self.state.last_signal = signal

            # Track new signal in scorecard
            self.scorecard.record_signal(signal)
            self.scorecard.save()
            self._save_state()

            logger.info("agent_iteration_complete", iteration=self.state.iteration,
                       direction=signal.direction, profitable=is_profitable,
                       changes_applied=len(improvements),
                       evolution_version=self.evolution_config.version,
                       retrain_pending=self.state.retrain_pending)
            return signal

        except Exception as e:
            logger.error("agent_iteration_failed", error=str(e), trace=traceback.format_exc())
            self._save_state()
            return SignalOutput(timestamp=utc_now(), direction="flat",
                              reasoning=f"Agent error: {e}", agent_iteration=self.state.iteration)

"""Change executor: translates approved ProposedChanges into real system modifications.

This is the bridge between the LLM's natural-language proposals and actual
config updates. Each change type has a dedicated handler that parses the
detail string and writes structured updates to EvolutionConfig.

SAFETY:
- All changes go through scope enforcement + validation gate BEFORE reaching here
- Every execution is wrapped in try/except — failures are logged, not fatal
- A config snapshot is taken before any batch of changes for rollback
- The executor never modifies code files — only the evolution_config.json
"""

from __future__ import annotations

import re
from typing import Any

from src.common.logging import get_logger
from src.agent.guardrails import ProposedChange
from src.agent.evolution_config import (
    EvolutionConfig,
    FeatureToggle,
    CustomInteraction,
    HyperparamOverride,
    load_evolution_config,
    save_evolution_config,
    snapshot_config,
)

logger = get_logger(__name__)

# Safe bounds for hyperparameter changes
HYPERPARAM_BOUNDS = {
    "n_estimators": (50, 2000),
    "max_depth": (2, 20),
    "learning_rate": (0.001, 0.5),
    "C": (0.001, 100.0),
    "min_child_weight": (1, 20),
    "subsample": (0.3, 1.0),
    "colsample_bytree": (0.3, 1.0),
    "reg_alpha": (0.0, 10.0),
    "reg_lambda": (0.0, 10.0),
    "num_leaves": (8, 256),
    "min_samples_split": (2, 50),
    "min_samples_leaf": (1, 50),
}

ENSEMBLE_BOUNDS = {
    "min_consensus_pct": (0.3, 0.9),
    "min_avg_confidence": (0.0, 0.5),
    "sharpe_weight_power": (0.5, 3.0),
    "min_horizon_agreement": (1, 4),
}

TPSL_BOUNDS = {
    "atr_multiplier_tp": (1.0, 5.0),
    "atr_multiplier_sl": (0.5, 3.0),
    "atr_pct": (0.005, 0.05),
    "max_position_size_pct": (0.02, 0.15),
    "confidence_sizing_factor": (0.05, 0.5),
}


def _extract_number(text: str, keyword: str) -> float | None:
    """Extract a numeric value associated with a keyword from text."""
    patterns = [
        rf"{keyword}\s*[=:to]\s*([0-9]*\.?[0-9]+)",
        rf"{keyword}\s+([0-9]*\.?[0-9]+)",
        rf"([0-9]*\.?[0-9]+)\s*{keyword}",
        rf"set\s+{keyword}\s+(?:to\s+)?([0-9]*\.?[0-9]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def _extract_feature_name(text: str) -> str | None:
    """Extract a feature name from change detail text."""
    patterns = [
        r"feature[:\s]+['\"]?(\w+)['\"]?",
        r"disable\s+['\"]?(\w+)['\"]?",
        r"enable\s+['\"]?(\w+)['\"]?",
        r"remove\s+['\"]?(\w+)['\"]?\s+feature",
        r"add\s+['\"]?(\w+)['\"]?\s+feature",
        r"['\"](\w+_\w+)['\"]",  # snake_case feature names
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _extract_model_type(text: str) -> str | None:
    """Extract model type from text."""
    for mt in [
        "lgbm",
        "xgb",
        "rf",
        "lr",
        "lightgbm",
        "xgboost",
        "random_forest",
        "logistic",
    ]:
        if mt in text.lower():
            mapping = {
                "lightgbm": "lgbm",
                "xgboost": "xgb",
                "random_forest": "rf",
                "logistic": "lr",
            }
            return mapping.get(mt, mt)
    return None


def _extract_interaction(text: str) -> tuple[str, str, str] | None:
    """Extract interaction feature components: (feature_a, feature_b, operation)."""
    # Pattern: "feature_a * feature_b" or "feature_a x feature_b"
    m = re.search(r"(\w+)\s*([*×x/+\-])\s*(\w+)", text)
    if m:
        op_char = m.group(2)
        op_name = {
            "*": "multiply",
            "×": "multiply",
            "x": "multiply",
            "/": "divide",
            "+": "add",
            "-": "subtract",
        }.get(op_char, "multiply")
        return m.group(1), m.group(3), op_name

    # Pattern: "multiply feature_a and feature_b"
    for op_word in ["multiply", "divide", "subtract", "add"]:
        m = re.search(
            rf"{op_word}\s+(\w+)\s+(?:and|with|by)\s+(\w+)", text, re.IGNORECASE
        )
        if m:
            return m.group(1), m.group(2), op_word

    return None


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    """Clamp a value within safe bounds."""
    return max(bounds[0], min(bounds[1], value))


class ChangeExecutor:
    """Executes approved ProposedChanges by updating EvolutionConfig."""

    def __init__(self, config: EvolutionConfig | None = None):
        self.config = config or load_evolution_config()
        self._pre_snapshot: dict[str, Any] | None = None

    def execute_batch(
        self,
        changes: list[ProposedChange],
        iteration: int,
        scorecard_metrics: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Execute a batch of approved changes.

        Takes a config snapshot first for rollback capability.
        Returns list of execution results.
        """
        self._pre_snapshot = snapshot_config(self.config)
        results = []

        for change in changes:
            result = self._execute_one(change, iteration, scorecard_metrics)
            results.append(result)

        # Persist if any succeeded
        if any(r["success"] for r in results):
            self.config.last_updated_iteration = iteration
            self.config.version += 1
            save_evolution_config(self.config)

        return results

    def rollback(self) -> bool:
        """Rollback to pre-execution config snapshot."""
        if self._pre_snapshot is None:
            return False
        try:
            from src.agent.evolution_config import restore_config

            self.config = restore_config(self._pre_snapshot)
            save_evolution_config(self.config)
            logger.info("executor_rollback_success")
            return True
        except Exception as e:
            logger.error("executor_rollback_failed", error=str(e))
            return False

    def _execute_one(
        self,
        change: ProposedChange,
        iteration: int,
        scorecard_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a single change. Returns {success, change, detail, error}."""
        handlers = {
            "feature": self._execute_feature,
            "hyperparameter": self._execute_hyperparameter,
            "ensemble": self._execute_ensemble,
            "model": self._execute_model,
        }

        handler = handlers.get(change.type)
        if handler is None:
            return {
                "success": False,
                "change": f"{change.type}/{change.action}",
                "detail": change.detail,
                "error": f"No handler for type: {change.type}",
            }

        try:
            success, msg = handler(change, iteration, scorecard_metrics)
            level = "info" if success else "warning"
            getattr(logger, level)(
                "change_executed",
                type=change.type,
                action=change.action,
                success=success,
                msg=msg,
            )
            return {
                "success": success,
                "change": f"{change.type}/{change.action}",
                "detail": change.detail,
                "message": msg,
            }
        except Exception as e:
            logger.error("change_execution_failed", type=change.type, error=str(e))
            return {
                "success": False,
                "change": f"{change.type}/{change.action}",
                "detail": change.detail,
                "error": str(e),
            }

    # ── Feature changes ──────────────────────────────────────

    def _execute_feature(
        self, change: ProposedChange, iteration: int, metrics: dict[str, Any]
    ) -> tuple[bool, str]:
        if change.action == "remove":
            feat_name = _extract_feature_name(change.detail)
            if not feat_name:
                return False, f"Could not extract feature name from: {change.detail}"
            self.config.feature_toggles[feat_name] = FeatureToggle(
                name=feat_name,
                enabled=False,
                reason=change.detail,
                iteration_changed=iteration,
            )
            return True, f"Disabled feature: {feat_name}"

        if change.action == "add":
            # Check if it's an interaction feature
            interaction = _extract_interaction(change.detail)
            if interaction:
                fa, fb, op = interaction
                name = f"{fa}_{op}_{fb}"
                ci = CustomInteraction(
                    name=name,
                    feature_a=fa,
                    feature_b=fb,
                    operation=op,
                    enabled=True,
                    iteration_added=iteration,
                )
                # Avoid duplicates
                existing_names = {c.name for c in self.config.custom_interactions}
                if name in existing_names:
                    return False, f"Interaction {name} already exists"
                self.config.custom_interactions.append(ci)
                return True, f"Added interaction feature: {name} = {fa} {op} {fb}"

            # Re-enable a previously disabled feature
            feat_name = _extract_feature_name(change.detail)
            if feat_name and feat_name in self.config.feature_toggles:
                self.config.feature_toggles[feat_name].enabled = True
                self.config.feature_toggles[feat_name].reason = change.detail
                self.config.feature_toggles[feat_name].iteration_changed = iteration
                return True, f"Re-enabled feature: {feat_name}"

            return False, f"Could not parse feature add: {change.detail}"

        if change.action == "modify":
            feat_name = _extract_feature_name(change.detail)
            if feat_name:
                # Toggle: if currently enabled, interpret as context-dependent
                toggle = self.config.feature_toggles.get(feat_name)
                if toggle and not toggle.enabled:
                    toggle.enabled = True
                    toggle.reason = change.detail
                    toggle.iteration_changed = iteration
                    return True, f"Re-enabled modified feature: {feat_name}"
                # Otherwise just log the modification intent
                self.config.feature_toggles[feat_name] = FeatureToggle(
                    name=feat_name,
                    enabled=True,
                    reason=change.detail,
                    iteration_changed=iteration,
                )
                return True, f"Marked feature modified: {feat_name}"

            return False, f"Could not parse feature modify: {change.detail}"

        return False, f"Unknown feature action: {change.action}"

    # ── Hyperparameter changes ───────────────────────────────

    def _execute_hyperparameter(
        self, change: ProposedChange, iteration: int, metrics: dict[str, Any]
    ) -> tuple[bool, str]:
        detail = change.detail.lower()
        model_type = _extract_model_type(detail)

        # Parse individual parameter changes
        updated_params: dict[str, Any] = {}
        for param_name, bounds in HYPERPARAM_BOUNDS.items():
            value = _extract_number(detail, param_name)
            if value is not None:
                clamped = _clamp(value, bounds)
                updated_params[param_name] = (
                    clamped if isinstance(bounds[0], float) else int(clamped)
                )

        if not updated_params and not model_type:
            return False, f"Could not parse hyperparameter change: {change.detail}"

        # If model type specified, apply to that model; otherwise apply to all
        targets = [model_type] if model_type else ["lgbm", "xgb", "rf", "lr"]

        for mt in targets:
            if not updated_params:
                continue
            existing = self.config.hyperparam_overrides.get(mt)
            if existing:
                existing.params.update(updated_params)
                existing.iteration_changed = iteration
                existing.reason = change.detail
            else:
                self.config.hyperparam_overrides[mt] = HyperparamOverride(
                    model_type=mt,
                    params=updated_params,
                    iteration_changed=iteration,
                    reason=change.detail,
                )

        if updated_params:
            return True, f"Updated {targets}: {updated_params}"

        return False, f"No parseable params in: {change.detail}"

    # ── Ensemble changes ─────────────────────────────────────

    def _execute_ensemble(
        self, change: ProposedChange, iteration: int, metrics: dict[str, Any]
    ) -> tuple[bool, str]:
        detail = change.detail.lower()
        updates = []

        for param_name, bounds in ENSEMBLE_BOUNDS.items():
            value = _extract_number(detail, param_name)
            if value is not None:
                clamped = _clamp(value, bounds)
                if param_name == "min_horizon_agreement":
                    clamped = int(clamped)
                setattr(self.config.ensemble, param_name, clamped)
                updates.append(f"{param_name}={clamped}")

        # Also check for TP/SL adjustments mentioned in ensemble context
        for param_name, bounds in TPSL_BOUNDS.items():
            value = _extract_number(detail, param_name)
            if value is not None:
                clamped = _clamp(value, bounds)
                setattr(self.config.tp_sl, param_name, clamped)
                updates.append(f"tp_sl.{param_name}={clamped}")

        # Heuristic: "increase consensus" / "decrease consensus"
        if not updates:
            if "increase" in detail and "consensus" in detail:
                new_val = min(self.config.ensemble.min_consensus_pct + 0.05, 0.9)
                self.config.ensemble.min_consensus_pct = new_val
                updates.append(f"min_consensus_pct={new_val:.2f}")
            elif "decrease" in detail and "consensus" in detail:
                new_val = max(self.config.ensemble.min_consensus_pct - 0.05, 0.3)
                self.config.ensemble.min_consensus_pct = new_val
                updates.append(f"min_consensus_pct={new_val:.2f}")
            elif "increase" in detail and "horizon" in detail:
                new_val = min(self.config.ensemble.min_horizon_agreement + 1, 4)
                self.config.ensemble.min_horizon_agreement = new_val
                updates.append(f"min_horizon_agreement={new_val}")
            elif "decrease" in detail and "horizon" in detail:
                new_val = max(self.config.ensemble.min_horizon_agreement - 1, 1)
                self.config.ensemble.min_horizon_agreement = new_val
                updates.append(f"min_horizon_agreement={new_val}")
            elif "widen" in detail and (
                "tp" in detail or "take_profit" in detail or "take profit" in detail
            ):
                new_val = min(self.config.tp_sl.atr_multiplier_tp + 0.25, 5.0)
                self.config.tp_sl.atr_multiplier_tp = new_val
                updates.append(f"atr_multiplier_tp={new_val:.2f}")
            elif "tighten" in detail and (
                "sl" in detail or "stop_loss" in detail or "stop loss" in detail
            ):
                new_val = max(self.config.tp_sl.atr_multiplier_sl - 0.15, 0.5)
                self.config.tp_sl.atr_multiplier_sl = new_val
                updates.append(f"atr_multiplier_sl={new_val:.2f}")
            elif "widen" in detail and ("sl" in detail or "stop" in detail):
                new_val = min(self.config.tp_sl.atr_multiplier_sl + 0.15, 3.0)
                self.config.tp_sl.atr_multiplier_sl = new_val
                updates.append(f"atr_multiplier_sl={new_val:.2f}")
            elif "tighten" in detail and ("tp" in detail or "take" in detail):
                new_val = max(self.config.tp_sl.atr_multiplier_tp - 0.25, 1.0)
                self.config.tp_sl.atr_multiplier_tp = new_val
                updates.append(f"atr_multiplier_tp={new_val:.2f}")
            elif "reduce" in detail and "size" in detail:
                new_val = max(self.config.tp_sl.max_position_size_pct - 0.02, 0.02)
                self.config.tp_sl.max_position_size_pct = new_val
                updates.append(f"max_position_size_pct={new_val:.2f}")
            elif "increase" in detail and "size" in detail:
                new_val = min(self.config.tp_sl.max_position_size_pct + 0.02, 0.15)
                self.config.tp_sl.max_position_size_pct = new_val
                updates.append(f"max_position_size_pct={new_val:.2f}")

        if updates:
            self.config.ensemble.iteration_changed = iteration
            self.config.tp_sl.iteration_changed = iteration
            return True, f"Ensemble/TPSL updated: {', '.join(updates)}"

        return False, f"Could not parse ensemble change: {change.detail}"

    # ── Model changes ────────────────────────────────────────

    def _execute_model(
        self, change: ProposedChange, iteration: int, metrics: dict[str, Any]
    ) -> tuple[bool, str]:
        """Model changes flag a retrain. Actual training happens in the loop."""
        detail = change.detail.lower()

        if change.action == "add":
            model_type = _extract_model_type(detail)
            if model_type:
                # Ensure there's a hyperparam override entry so it gets included in next retrain
                if model_type not in self.config.hyperparam_overrides:
                    self.config.hyperparam_overrides[model_type] = HyperparamOverride(
                        model_type=model_type,
                        params={},
                        iteration_changed=iteration,
                        reason=change.detail,
                    )
                return True, f"Model type {model_type} flagged for next retrain cycle"

        if change.action == "modify":
            # This is effectively a hyperparameter change + retrain flag
            model_type = _extract_model_type(detail)
            updated_params: dict[str, Any] = {}
            for param_name, bounds in HYPERPARAM_BOUNDS.items():
                value = _extract_number(detail, param_name)
                if value is not None:
                    clamped = _clamp(value, bounds)
                    updated_params[param_name] = (
                        clamped if isinstance(bounds[0], float) else int(clamped)
                    )

            if model_type and updated_params:
                existing = self.config.hyperparam_overrides.get(model_type)
                if existing:
                    existing.params.update(updated_params)
                    existing.iteration_changed = iteration
                else:
                    self.config.hyperparam_overrides[model_type] = HyperparamOverride(
                        model_type=model_type,
                        params=updated_params,
                        iteration_changed=iteration,
                        reason=change.detail,
                    )
                return (
                    True,
                    f"Model {model_type} params updated: {updated_params}, retrain flagged",
                )

            return True, f"Model modification noted: {change.detail}"

        return False, f"Unknown model action: {change.action}"


def should_retrain(
    config: EvolutionConfig, iteration: int, scorecard_metrics: dict[str, Any]
) -> bool:
    """Check if retraining should be triggered based on config + metrics."""
    trigger = config.retrain

    if trigger.retrain_in_progress:
        return False

    iterations_since = iteration - trigger.last_retrain_iteration

    # Forced retrain after max iterations
    if iterations_since >= trigger.max_iterations_without:
        logger.info("retrain_trigger_max_iterations", since=iterations_since)
        return True

    # Minimum cooldown
    if iterations_since < trigger.min_iterations_between:
        return False

    # Performance triggers
    sharpe = scorecard_metrics.get("signal_sharpe", 0.0)
    accuracy = scorecard_metrics.get("signal_accuracy", 0.5)
    drawdown = scorecard_metrics.get("max_drawdown_pct", 0.0)
    n_signals = scorecard_metrics.get("n_signals_scored", 0)

    if n_signals < 10:
        return False

    if sharpe < trigger.sharpe_floor:
        logger.info("retrain_trigger_sharpe", sharpe=sharpe, floor=trigger.sharpe_floor)
        return True

    if accuracy < trigger.accuracy_floor:
        logger.info(
            "retrain_trigger_accuracy", accuracy=accuracy, floor=trigger.accuracy_floor
        )
        return True

    if drawdown < trigger.drawdown_trigger:
        logger.info(
            "retrain_trigger_drawdown",
            drawdown=drawdown,
            trigger=trigger.drawdown_trigger,
        )
        return True

    # Check if hyperparams were updated since last retrain
    for override in config.hyperparam_overrides.values():
        if override.iteration_changed > trigger.last_retrain_iteration:
            logger.info("retrain_trigger_hyperparam_change", model=override.model_type)
            return True

    return False

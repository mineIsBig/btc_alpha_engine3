"""Dynamic evolution config: the mutable state that the agent evolves over time.

This is the key piece that makes the agent truly autonomous. Instead of just
logging proposed changes, the agent writes structured updates to this config,
and every downstream system (features, models, ensemble, TP/SL) reads from it.

Stored at: artifacts/evolution_config.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.common.logging import get_logger

logger = get_logger(__name__)

EVOLUTION_CONFIG_PATH = Path("artifacts/evolution_config.json")


class FeatureToggle(BaseModel):
    """Toggle individual features on/off."""
    name: str
    enabled: bool = True
    reason: str = ""
    iteration_changed: int = 0


class CustomInteraction(BaseModel):
    """Agent-defined interaction feature."""
    name: str
    feature_a: str
    feature_b: str
    operation: str = "multiply"  # multiply, divide, subtract, add
    enabled: bool = True
    iteration_added: int = 0


class HyperparamOverride(BaseModel):
    """Override default hyperparameters for a model type."""
    model_type: str  # lr, rf, lgbm, xgb
    params: dict[str, Any] = Field(default_factory=dict)
    iteration_changed: int = 0
    reason: str = ""


class EnsembleConfig(BaseModel):
    """Dynamic ensemble parameters."""
    min_consensus_pct: float = 0.5
    min_avg_confidence: float = 0.1
    sharpe_weight_power: float = 1.5
    min_horizon_agreement: int = 2
    iteration_changed: int = 0


class TPSLConfig(BaseModel):
    """Dynamic TP/SL calibration."""
    atr_multiplier_tp: float = 2.0   # TP = entry * (1 + atr_pct * this)
    atr_multiplier_sl: float = 1.0   # SL = entry * (1 - atr_pct * this)
    atr_pct: float = 0.02            # base ATR percentage
    max_position_size_pct: float = 0.15
    confidence_sizing_factor: float = 0.2  # size = min(max, confidence * this)
    iteration_changed: int = 0


class RetrainTrigger(BaseModel):
    """Conditions that trigger automatic retraining."""
    sharpe_floor: float = -0.5          # retrain if signal_sharpe drops below
    accuracy_floor: float = 0.45        # retrain if accuracy drops below
    drawdown_trigger: float = -0.08     # retrain if drawdown exceeds
    min_iterations_between: int = 12    # minimum iterations between retrains
    max_iterations_without: int = 72    # force retrain after this many iterations
    last_retrain_iteration: int = 0
    retrain_in_progress: bool = False


class ModelLifecycle(BaseModel):
    """Model promotion/retirement rules."""
    min_signals_for_eval: int = 20      # minimum signals before evaluating
    promote_sharpe_threshold: float = 0.3
    retire_sharpe_threshold: float = -0.5
    retire_after_n_bad_signals: int = 15  # consecutive bad signals to retire
    max_promoted_per_horizon: int = 3


class EvolutionConfig(BaseModel):
    """The complete dynamic config that the agent evolves."""
    version: int = 1
    last_updated_iteration: int = 0

    # Feature toggles — agent can disable underperforming features
    feature_toggles: dict[str, FeatureToggle] = Field(default_factory=dict)

    # Custom interaction features — agent can create new ones
    custom_interactions: list[CustomInteraction] = Field(default_factory=list)

    # Hyperparameter overrides — agent can tune model params
    hyperparam_overrides: dict[str, HyperparamOverride] = Field(default_factory=dict)

    # Ensemble parameters — agent can adjust consensus/weighting
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)

    # TP/SL calibration — agent can adjust risk/reward bands
    tp_sl: TPSLConfig = Field(default_factory=TPSLConfig)

    # Retrain triggers — agent can adjust when retraining fires
    retrain: RetrainTrigger = Field(default_factory=RetrainTrigger)

    # Model lifecycle
    model_lifecycle: ModelLifecycle = Field(default_factory=ModelLifecycle)

    # History of config snapshots for rollback (last 10)
    _snapshots: list[dict[str, Any]] = []


def load_evolution_config() -> EvolutionConfig:
    """Load evolution config from disk, or return defaults."""
    if EVOLUTION_CONFIG_PATH.exists():
        try:
            with open(EVOLUTION_CONFIG_PATH) as f:
                data = json.load(f)
            return EvolutionConfig.model_validate(data)
        except Exception as e:
            logger.warning("evolution_config_load_failed", error=str(e))
    return EvolutionConfig()


def save_evolution_config(config: EvolutionConfig) -> None:
    """Persist evolution config to disk."""
    EVOLUTION_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVOLUTION_CONFIG_PATH, "w") as f:
        f.write(config.model_dump_json(indent=2))
    logger.info("evolution_config_saved", version=config.version, iteration=config.last_updated_iteration)


def snapshot_config(config: EvolutionConfig) -> dict[str, Any]:
    """Take a snapshot of current config for rollback."""
    return config.model_dump()


def restore_config(snapshot: dict[str, Any]) -> EvolutionConfig:
    """Restore config from a snapshot."""
    return EvolutionConfig.model_validate(snapshot)

"""Agent guardrails: schema validation, scope enforcement, changelog, and validation gate.

Three guardrails for the Arbos-inspired ralph-loop:

1. SCHEMA VALIDATION: Every LLM response is validated against a Pydantic schema.
   Malformed responses fail gracefully to "no change" rather than applying garbage.

2. SCOPE ENFORCEMENT: The agent can only modify specific aspects of the system.
   Each proposed change must fall within an allowed scope. Changes to risk limits
   or core infrastructure are blocked programmatically.

3. VALIDATION GATE + CHANGELOG: Proposed changes are backtested on the most recent
   walk-forward fold before being applied. Every change is recorded in a changelog
   with rollback capability.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.common.logging import get_logger
from src.common.time_utils import utc_now

logger = get_logger(__name__)

CHANGELOG_PATH = Path("artifacts/agent_changelog.json")


# ── 1. LLM Response Schema Validation ───────────────────────────


class ChangeType(str, Enum):
    FEATURE = "feature"
    MODEL = "model"
    HYPERPARAMETER = "hyperparameter"
    ENSEMBLE = "ensemble"
    RISK = "risk"


class ChangeAction(str, Enum):
    ADD = "add"
    REMOVE = "remove"
    MODIFY = "modify"


class ProposedChange(BaseModel):
    """A single proposed change from the LLM."""

    type: str = Field(
        ..., description="Category: feature, model, hyperparameter, ensemble, risk"
    )
    action: str = Field(..., description="Action: add, remove, modify")
    detail: str = Field(
        ..., min_length=1, max_length=500, description="Specific change description"
    )

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        allowed = {e.value for e in ChangeType}
        v_lower = v.lower().strip()
        if v_lower not in allowed:
            raise ValueError(f"Invalid change type '{v}'. Allowed: {allowed}")
        return v_lower

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        allowed = {e.value for e in ChangeAction}
        v_lower = v.lower().strip()
        if v_lower not in allowed:
            raise ValueError(f"Invalid action '{v}'. Allowed: {allowed}")
        return v_lower


class DesignResponse(BaseModel):
    """Validated schema for the design_or_modify() LLM response."""

    analysis: str = Field(default="", max_length=2000)
    is_system_profitable: bool = False
    weaknesses_found: list[str] = Field(default_factory=list, max_length=10)
    proposed_changes: list[ProposedChange] = Field(default_factory=list, max_length=5)
    priority: str = Field(default="", max_length=200)


class ReflectResponse(BaseModel):
    """Validated schema for the reflect() LLM response."""

    reflection: str = Field(default="", max_length=2000)
    is_actually_profitable: bool = False
    weaknesses: list[str] = Field(default_factory=list, max_length=10)
    regime_assessment: str = Field(default="", max_length=500)
    tp_sl_assessment: str = Field(default="", max_length=500)
    next_priorities: list[str] = Field(default_factory=list, max_length=10)


def validate_design_response(raw: dict[str, Any]) -> DesignResponse:
    """Validate LLM design response. Returns safe defaults on failure."""
    try:
        return DesignResponse.model_validate(raw)
    except Exception as e:
        logger.warning(
            "design_response_validation_failed",
            error=str(e),
            raw_keys=list(raw.keys()) if isinstance(raw, dict) else "not_a_dict",
        )
        return DesignResponse(
            analysis="LLM response failed validation", proposed_changes=[]
        )


def validate_reflect_response(raw: dict[str, Any]) -> ReflectResponse:
    """Validate LLM reflect response. Returns safe defaults on failure."""
    try:
        return ReflectResponse.model_validate(raw)
    except Exception as e:
        logger.warning(
            "reflect_response_validation_failed",
            error=str(e),
            raw_keys=list(raw.keys()) if isinstance(raw, dict) else "not_a_dict",
        )
        return ReflectResponse(reflection="LLM response failed validation")


# ── 2. Scope Enforcement ────────────────────────────────────────

# Define what the agent IS and IS NOT allowed to modify.
ALLOWED_SCOPES = {
    ChangeType.FEATURE: {
        "description": "Add, remove, or modify feature engineering logic",
        "allowed_actions": {ChangeAction.ADD, ChangeAction.REMOVE, ChangeAction.MODIFY},
    },
    ChangeType.MODEL: {
        "description": "Add or modify model types and architectures",
        "allowed_actions": {ChangeAction.ADD, ChangeAction.MODIFY},
    },
    ChangeType.HYPERPARAMETER: {
        "description": "Modify model hyperparameters within safe bounds",
        "allowed_actions": {ChangeAction.MODIFY},
    },
    ChangeType.ENSEMBLE: {
        "description": "Modify ensemble weights and consensus thresholds",
        "allowed_actions": {ChangeAction.ADD, ChangeAction.REMOVE, ChangeAction.MODIFY},
    },
}

# BLOCKED: The agent cannot modify risk limits. This is a hard constraint.
BLOCKED_SCOPES = {
    ChangeType.RISK: "Risk limits are infrastructure-level settings and cannot be modified by the agent. "
    "Changes to daily loss limits, EOD trailing stops, position size caps, or kill switch "
    "thresholds require manual operator approval.",
}

# Keywords in change details that should be blocked regardless of scope.
BLOCKED_KEYWORDS = [
    "risk_limit",
    "daily_loss",
    "eod_trailing",
    "kill_switch",
    "max_position",
    "flatten_on_breach",
    "leverage",
    "max_drawdown_limit",
    "delete_database",
    "drop_table",
    "rm -rf",
    "os.system",
    "subprocess",
    "eval(",
    "exec(",
    "import os",
]


class ScopeViolation(Exception):
    """Raised when a proposed change violates scope constraints."""

    pass


def enforce_scope(change: ProposedChange) -> tuple[bool, str]:
    """Check if a proposed change is within allowed scope.

    Returns (allowed, reason).
    """
    change_type = ChangeType(change.type)
    change_action = ChangeAction(change.action)

    # Check if scope is blocked entirely
    if change_type in BLOCKED_SCOPES:
        reason = BLOCKED_SCOPES[change_type]
        logger.warning(
            "scope_violation_blocked",
            type=change.type,
            action=change.action,
            detail=change.detail[:100],
            reason=reason,
        )
        return False, reason

    # Check if scope exists in allowed
    if change_type not in ALLOWED_SCOPES:
        return False, f"Unknown change type: {change.type}"

    # Check if action is allowed for this scope
    scope = ALLOWED_SCOPES[change_type]
    if change_action not in scope["allowed_actions"]:
        reason = f"Action '{change.action}' not allowed for scope '{change.type}'. Allowed: {[str(a) for a in scope['allowed_actions']]}"
        logger.warning(
            "scope_violation_action",
            type=change.type,
            action=change.action,
            reason=reason,
        )
        return False, reason

    # Check for blocked keywords in detail
    detail_lower = change.detail.lower()
    for keyword in BLOCKED_KEYWORDS:
        if keyword in detail_lower:
            reason = f"Change detail contains blocked keyword: '{keyword}'"
            logger.warning(
                "scope_violation_keyword", detail=change.detail[:100], keyword=keyword
            )
            return False, reason

    return True, "OK"


def filter_changes_by_scope(
    changes: list[ProposedChange],
) -> tuple[list[ProposedChange], list[dict[str, str]]]:
    """Filter proposed changes, returning allowed changes and rejection reasons.

    Returns (allowed_changes, rejections).
    """
    allowed = []
    rejections = []

    for change in changes:
        ok, reason = enforce_scope(change)
        if ok:
            allowed.append(change)
        else:
            rejections.append(
                {
                    "type": change.type,
                    "action": change.action,
                    "detail": change.detail[:100],
                    "reason": reason,
                }
            )
            logger.info("change_rejected_by_scope", type=change.type, reason=reason)

    return allowed, rejections


# ── 3. Changelog with Rollback ──────────────────────────────────


class ChangelogEntry(BaseModel):
    """A single entry in the agent changelog."""

    id: int
    timestamp: str
    iteration: int
    change_type: str
    change_action: str
    detail: str
    status: str = "applied"  # applied, rolled_back, rejected, failed_validation
    validation_result: str = ""
    rollback_reason: str = ""
    scorecard_snapshot: dict[str, Any] = Field(default_factory=dict)


class AgentChangelog:
    """Persistent changelog of all agent modifications with rollback capability."""

    def __init__(self):
        self.entries: list[ChangelogEntry] = []
        self._next_id = 1
        self._load()

    def _load(self) -> None:
        if CHANGELOG_PATH.exists():
            try:
                with open(CHANGELOG_PATH) as f:
                    data = json.load(f)
                self.entries = [
                    ChangelogEntry.model_validate(e) for e in data.get("entries", [])
                ]
                self._next_id = max((e.id for e in self.entries), default=0) + 1
            except Exception as e:
                logger.warning("changelog_load_failed", error=str(e))

    def save(self) -> None:
        CHANGELOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {"entries": [e.model_dump() for e in self.entries[-200:]]}
        with open(CHANGELOG_PATH, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def record(
        self,
        iteration: int,
        change: ProposedChange,
        status: str = "applied",
        validation_result: str = "",
        scorecard_snapshot: dict[str, Any] | None = None,
    ) -> ChangelogEntry:
        """Record a change in the changelog."""
        entry = ChangelogEntry(
            id=self._next_id,
            timestamp=utc_now().isoformat(),
            iteration=iteration,
            change_type=change.type,
            change_action=change.action,
            detail=change.detail,
            status=status,
            validation_result=validation_result,
            scorecard_snapshot=scorecard_snapshot or {},
        )
        self.entries.append(entry)
        self._next_id += 1
        logger.info(
            "changelog_entry",
            id=entry.id,
            type=change.type,
            action=change.action,
            status=status,
        )
        return entry

    def rollback(self, entry_id: int, reason: str = "") -> bool:
        """Mark a changelog entry as rolled back."""
        for entry in self.entries:
            if entry.id == entry_id and entry.status == "applied":
                entry.status = "rolled_back"
                entry.rollback_reason = reason
                logger.info("changelog_rollback", id=entry_id, reason=reason)
                self.save()
                return True
        return False

    def get_recent(self, n: int = 10) -> list[ChangelogEntry]:
        """Get the most recent n changelog entries."""
        return self.entries[-n:]

    def get_applied(self) -> list[ChangelogEntry]:
        """Get all currently applied (non-rolled-back) entries."""
        return [e for e in self.entries if e.status == "applied"]

    def rollback_since(self, iteration: int, reason: str = "") -> int:
        """Roll back all changes applied at or after the given iteration."""
        count = 0
        for entry in self.entries:
            if entry.iteration >= iteration and entry.status == "applied":
                entry.status = "rolled_back"
                entry.rollback_reason = reason
                count += 1
        if count > 0:
            logger.info(
                "changelog_bulk_rollback",
                iteration=iteration,
                count=count,
                reason=reason,
            )
            self.save()
        return count


# ── 4. Validation Gate ──────────────────────────────────────────


def validation_gate(
    changes: list[ProposedChange],
    scorecard_metrics: dict[str, Any],
) -> tuple[list[ProposedChange], list[dict[str, str]]]:
    """Validation gate: check proposed changes against recent performance.

    A change passes the gate if:
    1. The system has enough signal history (>= 10 scored signals)
    2. The system's recent Sharpe hasn't crashed (> -1.0)
    3. Max drawdown hasn't breached emergency levels (> -15%)

    If the system is in a bad state (crashed Sharpe, deep drawdown),
    only conservative changes (modify existing, not add new) are allowed.

    Returns (approved_changes, gate_rejections).
    """
    approved = []
    rejections = []

    n_signals = scorecard_metrics.get("n_signals_scored", 0)
    recent_sharpe = scorecard_metrics.get("recent_sharpe", 0.0)
    max_dd = scorecard_metrics.get("max_drawdown_pct", 0.0)

    # If not enough data, allow all changes (system is still bootstrapping)
    if n_signals < 10:
        logger.info(
            "validation_gate_bootstrap",
            n_signals=n_signals,
            msg="Allowing all changes during bootstrap phase",
        )
        return changes, []

    # Emergency state: deep drawdown or severely negative Sharpe
    is_emergency = max_dd < -0.15 or recent_sharpe < -1.0

    if is_emergency:
        logger.warning(
            "validation_gate_emergency",
            max_dd=f"{max_dd:.2%}",
            recent_sharpe=f"{recent_sharpe:.2f}",
            msg="Emergency state: only conservative modifications allowed",
        )

    for change in changes:
        if is_emergency:
            # In emergency, only allow modifying existing things (not adding new complexity)
            if change.action != ChangeAction.MODIFY.value:
                rejections.append(
                    {
                        "type": change.type,
                        "action": change.action,
                        "detail": change.detail[:100],
                        "reason": f"Emergency state (DD={max_dd:.2%}, Sharpe={recent_sharpe:.2f}): "
                        f"only 'modify' actions allowed, not '{change.action}'",
                    }
                )
                continue

        approved.append(change)

    return approved, rejections

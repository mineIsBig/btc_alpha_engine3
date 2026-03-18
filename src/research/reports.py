"""Walk-forward fold report generation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.common.logging import get_logger
from src.storage.database import session_scope
from src.storage.models import WalkForwardRun

logger = get_logger(__name__)

REPORTS_DIR = Path("artifacts/reports")


def save_fold_report(
    model_id: str,
    fold_idx: int,
    train_start: datetime,
    train_end: datetime,
    test_start: datetime,
    test_end: datetime,
    n_train: int,
    n_test: int,
    metrics: dict[str, float],
) -> str:
    """Save a walk-forward fold report to the database."""
    run_id = f"{model_id}_fold{fold_idx}"

    with session_scope() as session:
        existing = session.query(WalkForwardRun).filter_by(run_id=run_id).first()
        if existing:
            existing.sharpe = metrics.get("sharpe_ratio")
            existing.accuracy = metrics.get("accuracy")
            existing.profit_factor = metrics.get("profit_factor")
            existing.max_drawdown = metrics.get("max_drawdown")
            existing.n_trades = metrics.get("n_trades")
            existing.breach_count = int(metrics.get("breach_count", 0))
            existing.metrics_json = json.dumps(metrics, default=str)
        else:
            run = WalkForwardRun(
                run_id=run_id,
                model_id=model_id,
                fold_idx=fold_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                n_train_samples=n_train,
                n_test_samples=n_test,
                sharpe=metrics.get("sharpe_ratio"),
                accuracy=metrics.get("accuracy"),
                profit_factor=metrics.get("profit_factor"),
                max_drawdown=metrics.get("max_drawdown"),
                n_trades=metrics.get("n_trades"),
                breach_count=int(metrics.get("breach_count", 0)),
                metrics_json=json.dumps(metrics, default=str),
            )
            session.add(run)

    return run_id


def generate_summary_report(model_id: str) -> dict[str, Any]:
    """Generate a summary report across all folds for a model."""
    with session_scope() as session:
        folds = session.query(WalkForwardRun).filter_by(model_id=model_id).all()

        if not folds:
            return {"model_id": model_id, "error": "no folds found"}

        sharpes = [f.sharpe for f in folds if f.sharpe is not None]
        accuracies = [f.accuracy for f in folds if f.accuracy is not None]
        drawdowns = [f.max_drawdown for f in folds if f.max_drawdown is not None]
        breach_counts = [f.breach_count or 0 for f in folds]
        n_trades = [f.n_trades or 0 for f in folds]

        report = {
            "model_id": model_id,
            "n_folds": len(folds),
            "avg_sharpe": float(np.mean(sharpes)) if sharpes else 0.0,
            "std_sharpe": float(np.std(sharpes)) if sharpes else 0.0,
            "min_sharpe": float(np.min(sharpes)) if sharpes else 0.0,
            "max_sharpe": float(np.max(sharpes)) if sharpes else 0.0,
            "avg_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
            "avg_max_drawdown": float(np.mean(drawdowns)) if drawdowns else 0.0,
            "worst_drawdown": float(np.min(drawdowns)) if drawdowns else 0.0,
            "total_breaches": sum(breach_counts),
            "avg_trades_per_fold": float(np.mean(n_trades)) if n_trades else 0.0,
        }

    # Save to file
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{model_id}_summary.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("summary_report_generated", model_id=model_id, path=str(report_path))
    return report

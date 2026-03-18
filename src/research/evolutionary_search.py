"""Evolutionary search over hyperparameters and feature subsets.

Constrained search that explores:
- Feature subset selection
- Hyperparameter tuning
- Threshold calibration
"""

from __future__ import annotations

import random
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import load_yaml_config
from src.common.logging import get_logger
from src.models.base import BaseAlphaModel
from src.models.baseline import LogisticRegressionModel, RandomForestModel
from src.models.gradient_boost import LightGBMModel, XGBoostModel
from src.research.purged_walk_forward import PurgedWalkForward
from src.research.scoring import compute_fold_metrics
from src.research.datasets import get_label_column

logger = get_logger(__name__)

MODEL_FACTORIES: dict[str, type[BaseAlphaModel]] = {
    "logistic_regression": LogisticRegressionModel,
    "random_forest": RandomForestModel,
    "lightgbm": LightGBMModel,
    "xgboost": XGBoostModel,
}


@dataclass
class Individual:
    """An individual in the evolutionary population."""

    gene_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_type: str = "lightgbm"
    params: dict[str, Any] = field(default_factory=dict)
    feature_subset: list[str] = field(default_factory=list)
    fitness: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)


class EvolutionarySearch:
    """Evolutionary search over model configurations."""

    def __init__(
        self,
        all_features: list[str],
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
        tournament_size: int = 3,
        feature_subset_min: int = 5,
        feature_subset_max: int = 40,
    ):
        cfg = load_yaml_config("model_registry.yaml").get("evolutionary_search", {})
        self.all_features = all_features
        self.population_size = population_size or cfg.get("population_size", 20)
        self.generations = generations or cfg.get("generations", 10)
        self.mutation_rate = mutation_rate or cfg.get("mutation_rate", 0.2)
        self.crossover_rate = crossover_rate or cfg.get("crossover_rate", 0.7)
        self.tournament_size = tournament_size or cfg.get("tournament_size", 3)
        self.feature_min = max(
            feature_subset_min or cfg.get("feature_subset_min", 5), 3
        )
        self.feature_max = min(
            feature_subset_max or cfg.get("feature_subset_max", 40), len(all_features)
        )

        self.model_registry = load_yaml_config("model_registry.yaml").get(
            "baseline_models", {}
        )

    def _random_params(self, model_type: str) -> dict[str, Any]:
        """Sample random hyperparameters for a model type."""
        registry = self.model_registry.get(model_type, {})
        param_space = registry.get("params", {})
        params = {}
        for key, values in param_space.items():
            if isinstance(values, list):
                params[key] = random.choice(values)
            else:
                params[key] = values
        return params

    def _random_features(self) -> list[str]:
        """Sample a random feature subset."""
        n = random.randint(self.feature_min, self.feature_max)
        return sorted(random.sample(self.all_features, min(n, len(self.all_features))))

    def initialize_population(self) -> list[Individual]:
        """Create initial random population."""
        population = []
        model_types = [
            k for k, v in self.model_registry.items() if v.get("enabled", True)
        ]
        if not model_types:
            model_types = ["lightgbm"]

        for _ in range(self.population_size):
            mt = random.choice(model_types)
            individual = Individual(
                model_type=mt,
                params=self._random_params(mt),
                feature_subset=self._random_features(),
            )
            population.append(individual)
        return population

    def evaluate_individual(
        self,
        individual: Individual,
        dataset: pd.DataFrame,
        horizon: int,
        splitter: PurgedWalkForward,
    ) -> Individual:
        """Evaluate an individual using walk-forward validation."""
        label_col = get_label_column(horizon)
        if label_col not in dataset.columns:
            individual.fitness = -999.0
            return individual

        feature_cols = [f for f in individual.feature_subset if f in dataset.columns]
        if len(feature_cols) < self.feature_min:
            individual.fitness = -999.0
            return individual

        timestamps = dataset["timestamp"]
        fwd_ret_col = f"fwd_ret_{horizon}h"

        fold_sharpes = []
        fold_metrics_list = []

        model_cls = MODEL_FACTORIES.get(individual.model_type, LightGBMModel)

        for fold in splitter.split(timestamps):
            try:
                X_train = dataset.iloc[fold.train_indices][feature_cols].fillna(0)
                y_train = (
                    dataset.iloc[fold.train_indices][label_col]
                    .fillna(0)
                    .astype(int)
                    .values
                )
                X_test = dataset.iloc[fold.test_indices][feature_cols].fillna(0)
                y_test = (
                    dataset.iloc[fold.test_indices][label_col]
                    .fillna(0)
                    .astype(int)
                    .values
                )
                fwd_ret = dataset.iloc[fold.test_indices][fwd_ret_col].fillna(0).values
                test_ts = dataset.iloc[fold.test_indices]["timestamp"].values

                if len(np.unique(y_train)) < 2:
                    continue

                model = model_cls(
                    horizon=horizon,
                    params=individual.params,
                    model_id=individual.gene_id,
                )
                model.fit(X_train, y_train, feature_names=feature_cols)
                y_pred = model.predict(X_test)

                metrics = compute_fold_metrics(
                    y_true=y_test,
                    y_pred=y_pred,
                    fwd_returns=fwd_ret,
                    timestamps=test_ts,
                )
                fold_sharpes.append(metrics.get("sharpe_ratio", 0))
                fold_metrics_list.append(metrics)

            except Exception as e:
                logger.debug(
                    "fold_eval_error", gene_id=individual.gene_id, error=str(e)
                )
                continue

        if fold_sharpes:
            individual.fitness = float(np.mean(fold_sharpes))
            individual.metrics = {
                "avg_sharpe": float(np.mean(fold_sharpes)),
                "avg_accuracy": float(
                    np.mean([m.get("accuracy", 0) for m in fold_metrics_list])
                ),
                "avg_breach_rate": float(
                    np.mean([m.get("breach_rate", 0) for m in fold_metrics_list])
                ),
                "n_folds": len(fold_sharpes),
            }
        else:
            individual.fitness = -999.0

        return individual

    def tournament_select(self, population: list[Individual]) -> Individual:
        """Tournament selection."""
        contestants = random.sample(
            population, min(self.tournament_size, len(population))
        )
        return max(contestants, key=lambda x: x.fitness)

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Crossover two parents to produce offspring."""
        child = Individual(
            model_type=random.choice([parent1.model_type, parent2.model_type]),
        )

        # Crossover params
        all_keys = set(parent1.params.keys()) | set(parent2.params.keys())
        for key in all_keys:
            if key in parent1.params and key in parent2.params:
                child.params[key] = random.choice(
                    [parent1.params[key], parent2.params[key]]
                )
            elif key in parent1.params:
                child.params[key] = parent1.params[key]
            else:
                child.params[key] = parent2.params[key]

        # Crossover features (union with random thinning)
        all_feats = list(set(parent1.feature_subset) | set(parent2.feature_subset))
        n = random.randint(self.feature_min, min(self.feature_max, len(all_feats)))
        child.feature_subset = sorted(random.sample(all_feats, n))

        return child

    def mutate(self, individual: Individual) -> Individual:
        """Mutate an individual."""
        ind = deepcopy(individual)
        ind.gene_id = str(uuid.uuid4())[:8]

        # Mutate params
        if random.random() < self.mutation_rate:
            ind.params = self._random_params(ind.model_type)

        # Mutate features: add or remove some
        if random.random() < self.mutation_rate:
            n_change = max(1, int(len(ind.feature_subset) * 0.2))

            # Remove some
            if len(ind.feature_subset) > self.feature_min + n_change:
                to_remove = random.sample(ind.feature_subset, n_change)
                ind.feature_subset = [
                    f for f in ind.feature_subset if f not in to_remove
                ]

            # Add some
            available = [f for f in self.all_features if f not in ind.feature_subset]
            if available and len(ind.feature_subset) < self.feature_max:
                to_add = random.sample(available, min(n_change, len(available)))
                ind.feature_subset = sorted(set(ind.feature_subset) | set(to_add))

        return ind

    def run(
        self,
        dataset: pd.DataFrame,
        horizon: int,
        splitter: PurgedWalkForward | None = None,
    ) -> list[Individual]:
        """Run the evolutionary search.

        Returns sorted list of individuals by fitness (best first).
        """
        if splitter is None:
            splitter = PurgedWalkForward.from_config(horizon=horizon)
        elif splitter.horizon != horizon:
            # Ensure purge gap is correct for this horizon
            splitter = splitter.with_horizon(horizon)

        logger.info(
            "evo_search_start",
            horizon=horizon,
            pop_size=self.population_size,
            gens=self.generations,
            purge_hours=splitter.purge_hours,
        )

        population = self.initialize_population()

        # Evaluate initial population
        for i, ind in enumerate(population):
            population[i] = self.evaluate_individual(ind, dataset, horizon, splitter)
            logger.debug("evo_eval", gen=0, idx=i, fitness=population[i].fitness)

        for gen in range(self.generations):
            new_population = []

            # Elitism: keep top 2
            sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
            new_population.extend(deepcopy(sorted_pop[:2]))

            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    p1 = self.tournament_select(population)
                    p2 = self.tournament_select(population)
                    child = self.crossover(p1, p2)
                else:
                    child = deepcopy(self.tournament_select(population))

                child = self.mutate(child)
                child = self.evaluate_individual(child, dataset, horizon, splitter)
                new_population.append(child)

            population = new_population
            best = max(population, key=lambda x: x.fitness)
            logger.info(
                "evo_generation",
                gen=gen + 1,
                best_fitness=best.fitness,
                best_id=best.gene_id,
            )

        result = sorted(population, key=lambda x: x.fitness, reverse=True)
        logger.info(
            "evo_search_complete", best_fitness=result[0].fitness if result else 0
        )
        return result

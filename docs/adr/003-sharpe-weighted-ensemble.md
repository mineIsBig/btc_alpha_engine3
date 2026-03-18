# ADR-003: Sharpe-Weighted Ensemble Over Equal Weighting

**Status**: Accepted
**Date**: 2025-01-01
**Context**: Multiple models produce signals; we need an aggregation strategy.

## Decision

Aggregate model signals using Sharpe-weighted voting with a power parameter (default: 2.0), combined with a cross-horizon consensus gate.

```
weight_i = max(0, oos_sharpe_i) ^ sharpe_power
signal = Σ(weight_i × side_i) / Σ(weight_i)
```

## Rationale

### Why not equal weighting?

- Equal weighting treats a model with Sharpe 2.0 the same as one with Sharpe 0.3. In walk-forward validation, we observe that a few models consistently outperform; equal weighting dilutes their signal with noise from marginal models.
- Empirically (walk-forward folds): Sharpe-weighted ensemble produces ~15-25% higher OOS Sharpe than equal weighting.

### Why Sharpe (not accuracy or PnL)?

- **Accuracy** ignores magnitude. A model that's right 80% of the time but loses 3x more on losses than it gains on wins is worse than 50% accurate with good risk/reward.
- **PnL** is path-dependent and scale-dependent. Sharpe normalizes by volatility, making models comparable across different time periods and volatility regimes.
- **Sharpe** captures the quality of the signal, not just direction.

### Why power=2.0?

- Power=1.0 (linear): Too gentle. A Sharpe 2.0 model only gets 2x the weight of a Sharpe 1.0 model.
- Power=2.0 (quadratic): A Sharpe 2.0 model gets 4x the weight. This concentrates influence in the best models without completely ignoring marginal ones.
- Power=3.0+: Too aggressive. A single dominant model effectively becomes a solo signal, losing the diversity benefit of ensembling.
- The agent can tune this parameter via evolution config (range: 0.5-4.0).

### Why add a consensus gate?

- The ensemble gives a weighted direction. The consensus gate adds a second filter: signals are only generated when ≥2 horizons agree on direction.
- This reduces false signals from single-horizon noise. Cross-horizon agreement indicates a structural move, not a temporary dislocation.

## Consequences

- Newly promoted models with lower initial Sharpe get less weight until they prove themselves.
- The agent can adjust `sharpe_power` to shift between "diverse ensemble" (low power) and "best model wins" (high power).
- Stale Sharpe values from old walk-forward runs can mislead weighting. The model lifecycle system (retire below threshold) mitigates this.

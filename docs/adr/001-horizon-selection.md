# ADR-001: Horizon Selection (1h, 4h, 8h, 12h, 24h)

**Status**: Accepted
**Date**: 2025-01-01
**Context**: Model training requires choosing prediction horizons.

## Decision

Use five horizons: 1h, 4h, 8h, 12h, 24h.

## Rationale

- **1h**: Captures microstructure signals (funding spikes, liquidation cascades). High noise but valuable for regime detection. Required for position entry timing.
- **4h**: Core prediction horizon. Derivatives data (funding, OI) has natural 8h settlement cycles; 4h captures half-cycle momentum.
- **8h**: Aligns with funding settlement periods on most exchanges. OI divergence signals resolve most reliably at this timescale.
- **12h**: Bridge between intraday and daily. Captures trend continuation after Asian/European session transitions.
- **24h**: Captures full daily cycle. Lower noise, higher conviction. Aligns with daily loss limit resets.

## Alternatives Considered

- **Shorter horizons (15m, 30m)**: Too noisy for hourly feature resolution. Our data is 1h bars; sub-hourly predictions would overfit to noise.
- **Longer horizons (48h, 168h)**: Too few labels per walk-forward fold. With 6-month training windows, 168h horizons yield ~25 labels — insufficient for gradient boosting.
- **Fewer horizons (4h, 24h only)**: Lost diversity in ensemble. The consensus gate requires cross-horizon agreement; more horizons give stronger consensus signals.

## Consequences

- Each horizon adds ~4 model variants (LR, RF, LightGBM, XGBoost) × N walk-forward folds to the training pipeline.
- Ensemble aggregation uses Sharpe-weighted voting across all horizons.
- The consensus gate requires agreement across ≥2 horizons to generate a signal.

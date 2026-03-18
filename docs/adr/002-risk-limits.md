# ADR-002: Risk Limits (5% Daily Loss, 5% EOD Trailing)

**Status**: Accepted
**Date**: 2025-01-01
**Context**: Autonomous system needs hard risk boundaries the agent cannot modify.

## Decision

- **Daily loss limit**: 5% of opening equity (captured at 00:00 UTC)
- **EOD trailing loss limit**: 5% of end-of-day high water mark
- Both enforced at the infrastructure level, outside agent scope

## Rationale

### Why 5%?

- **Survivability**: A 5% daily loss allows ~14 consecutive losing days before a 50% drawdown. This gives the agent enough runway to iterate and self-correct.
- **Recovery math**: After a 5% loss, you need ~5.3% gain to recover. This is achievable within 2-3 good signal cycles at our average TP of 2-4%.
- **Industry standard**: Most prop trading firms use 2-5% daily limits. 5% is the upper bound — aggressive enough for crypto volatility, conservative enough for capital preservation.
- **BTC-specific**: BTC can move 5-10% in a day during volatility events. A 5% limit means the system stops before capitulating into the worst of a move.

### Why EOD (not intraday) trailing?

- Intraday HWM creates excessive whipsaw. A position that's +3% then retraces to +1% would breach an intraday trailing stop at 5%, even though the day is profitable.
- EOD trailing tracks the equity curve's true peak, not noise.

### Why operator-controlled (not agent-adjustable)?

- The agent optimizes for Sharpe and PnL. Without a hard constraint, it would rationally increase risk exposure when confident, creating tail risk.
- Scope enforcement in guardrails blocks any agent proposal that modifies risk limits.

## Consequences

- Agent cannot self-optimize its way out of a losing streak by loosening risk.
- The agent CAN adjust position sizing (within max bounds) and TP/SL calibration as indirect risk management.
- Breaches are logged, timestamped, and persist until next day reset.

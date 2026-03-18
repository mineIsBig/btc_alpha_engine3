# Contributing to BTC Alpha Engine

This guide covers how to add new model types, feature sources, data providers, and other extensions. The codebase is modular by design — most additions follow a clear pattern.

## Development Setup

```bash
# Clone and install
git clone https://github.com/mineIsBig/btc_alpha_engine3.git
cd btc_alpha_engine3
pip install -e ".[postgres]"
pip install -r requirements.lock

# Install dev dependencies
pip install pytest pytest-cov mypy ruff black

# Copy environment and configure
cp .env.example .env

# Bootstrap database (SQLite for dev)
python scripts/bootstrap_db.py

# Run tests
pytest tests/ -v
```

## Code Quality

All PRs must pass CI checks:

```bash
# Format
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/

# Tests
pytest tests/ -v --tb=short
```

## Adding a New Model Type

Models live in `src/models/`. Every model extends `BaseAlphaModel`.

### Step 1: Create the model class

```python
# src/models/my_model.py
from src.models.base import BaseAlphaModel
import numpy as np

class MyModel(BaseAlphaModel):
    model_type = "my_model"

    def __init__(self, horizon: int = 4, **params):
        super().__init__(horizon=horizon)
        self.params = params
        self._model = None  # Your ML model here

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Train your model
        self._model = ...
        self.feature_names = list(range(X.shape[1]))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Return probability array [n_samples, n_classes]
        return self._model.predict_proba(X)

    def get_signal(self, X) -> dict:
        # Return {"side": 1/-1/0, "probability": float, "confidence": float}
        proba = self.predict_proba(X.values if hasattr(X, 'values') else X)
        last = proba[-1]
        side = 1 if last[2] > last[0] else -1 if last[0] > last[2] else 0
        confidence = max(last) - 0.33
        return {"side": side, "probability": max(last), "confidence": confidence}
```

### Step 2: Register in the research cycle

Edit `src/orchestrator/research_cycle.py`:

```python
from src.models.my_model import MyModel

MODEL_CONFIGS = [
    # ... existing models ...
    ("my_model", MyModel, {"n_estimators": 200, "learning_rate": 0.05}),
]
```

### Step 3: Add hyperparameter bounds (for agent tuning)

Edit `src/agent/executor.py`, in the `HYPERPARAM_BOUNDS` dict:

```python
HYPERPARAM_BOUNDS = {
    # ... existing bounds ...
    "my_model": {
        "n_estimators": (50, 1000),
        "learning_rate": (0.001, 0.5),
    },
}
```

### Step 4: Add a test

```python
# tests/unit/test_my_model.py
def test_my_model_fit_predict():
    from src.models.my_model import MyModel
    import numpy as np

    model = MyModel(horizon=4)
    X = np.random.randn(100, 10)
    y = np.random.choice([-1, 0, 1], size=100)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (100, 3)
```

## Adding a New Feature Source

Features live in `src/features/`. Each module computes features from one data domain.

### Step 1: Create the feature module

```python
# src/features/my_features.py
import pandas as pd
import numpy as np

def compute_my_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features from my data source.

    Args:
        df: DataFrame with at minimum a 'timestamp' column and your raw columns.

    Returns:
        DataFrame with computed feature columns (no timestamp).
    """
    out = pd.DataFrame(index=df.index)

    if "my_column" not in df.columns:
        return out  # Graceful no-op if data missing

    raw = df["my_column"].astype(float)
    out["my_feature_level"] = raw
    out["my_feature_change_1h"] = raw.diff(1)
    out["my_feature_zscore_24h"] = (raw - raw.rolling(24).mean()) / raw.rolling(24).std()

    return out
```

### Step 2: Integrate into the feature pipeline

Edit `src/features/feature_pipeline.py`:

```python
from src.features.my_features import compute_my_features

# In build_features(), after the other compute_* calls:
my_feats = compute_my_features(base)

# Add to the concat list:
result = pd.concat([
    # ... existing ...
    my_feats,
], axis=1)
```

### Step 3: Add raw data loading (if from a new table)

If your feature needs new raw data, add the query in `load_raw_data()` in `feature_pipeline.py`.

## Adding a New Data Provider

Data clients live in `src/data/`.

### Step 1: Create the client

```python
# src/data/my_provider_client.py
import httpx
from src.common.config import get_settings

class MyProviderClient:
    def __init__(self):
        settings = get_settings()
        self.api_key = settings.my_provider_api_key
        self.client = httpx.Client(timeout=30)

    def fetch_data(self, symbol: str, start: str, end: str) -> list[dict]:
        resp = self.client.get(
            "https://api.myprovider.com/v1/data",
            params={"symbol": symbol, "start": start, "end": end},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        resp.raise_for_status()
        return resp.json()["data"]

    def close(self):
        self.client.close()
```

### Step 2: Add config

In `src/common/config.py`:
```python
my_provider_api_key: str = Field(default="")
```

In `.env.example`:
```
MY_PROVIDER_API_KEY=
```

### Step 3: Add storage model

In `src/storage/models.py`:
```python
class MyProviderData1h(Base):
    __tablename__ = "my_provider_1h"
    # ... columns
```

### Step 4: Add ingestion job

In `src/data/ingest_jobs.py`, add a function to fetch, validate, and store data.

## Adding a New Agent Phase

See `src/agent/phases.py` for the pattern. Each phase:
1. Takes a `PhaseContext` as input
2. Reads what it needs from the context
3. Writes its results back to the context
4. Returns the updated context

Register the new phase in `scripts/run_phase.py`'s dispatch table.

## Architecture Decision Records

Major design decisions are documented in `docs/adr/`. When making significant architectural choices, add a new ADR:

```
docs/adr/NNN-short-title.md
```

Format: Status, Date, Context, Decision, Rationale, Alternatives Considered, Consequences.

## Commit Messages

- Use imperative mood: "Add feature", not "Added feature"
- Keep first line under 72 characters
- Reference issue numbers when applicable

## Questions?

Open an issue or check the existing ADRs in `docs/adr/` for context on past decisions.

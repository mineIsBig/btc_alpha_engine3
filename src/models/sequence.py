"""Optional small LSTM sequence model (PyTorch).

This module is feature-flagged and only used if torch is available and
sequence_model.enabled is true in config.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseAlphaModel
from src.common.logging import get_logger

logger = get_logger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    pass


if _TORCH_AVAILABLE:

    class SmallLSTMNet(nn.Module):
        """Small LSTM for sequence classification."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            num_layers: int = 2,
            n_classes: int = 3,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, n_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            lstm_out, _ = self.lstm(x)
            last_hidden = lstm_out[:, -1, :]
            return self.fc(last_hidden)


class SmallLSTMModel(BaseAlphaModel):
    """Small LSTM model wrapper. Requires PyTorch."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.lookback = self.params.get("lookback", 24)
        self.hidden_dim = self.params.get("hidden_dim", 64)
        self.n_epochs = self.params.get("n_epochs", 50)
        self.batch_size = self.params.get("batch_size", 64)
        self.lr = self.params.get("lr", 0.001)

    def _build_model(self) -> Any:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Install with: pip install torch")
        return None  # Built during fit

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch required for SmallLSTMModel")

        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)

        X_arr = X.values if isinstance(X, pd.DataFrame) else X

        # Normalize
        self._mean = X_arr.mean(axis=0)
        self._std = X_arr.std(axis=0) + 1e-8
        X_norm = (X_arr - self._mean) / self._std

        # Create sequences
        X_seq, y_seq = self._make_sequences(X_norm, y)
        if len(X_seq) == 0:
            logger.warning("not_enough_data_for_lstm", lookback=self.lookback)
            self.is_fitted = False
            return

        # Map labels to 0-indexed
        unique_labels = sorted(set(y_seq))
        self._label_map = {lab: i for i, lab in enumerate(unique_labels)}
        self._label_inv = {i: lab for lab, i in self._label_map.items()}
        y_mapped = np.array([self._label_map[v] for v in y_seq])

        n_features = X_arr.shape[1]
        n_classes = len(unique_labels)

        self._model = SmallLSTMNet(n_features, self.hidden_dim, n_classes=n_classes)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        X_tensor = torch.FloatTensor(X_seq)
        y_tensor = torch.LongTensor(y_mapped)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self._model.train()
        for epoch in range(self.n_epochs):
            total_loss = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = self._model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        self._model.eval()
        self.is_fitted = True
        logger.info("lstm_fitted", epochs=self.n_epochs, sequences=len(X_seq))

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        X_norm = (X_arr - self._mean) / self._std
        X_seq, _ = self._make_sequences(X_norm, np.zeros(len(X_arr)))

        if len(X_seq) == 0:
            return np.array([[0.33, 0.34, 0.33]])

        X_tensor = torch.FloatTensor(X_seq)
        with torch.no_grad():
            logits = self._model(X_tensor)
            proba = torch.softmax(logits, dim=1).numpy()
        return proba

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        indices = proba.argmax(axis=1)
        if hasattr(self, "_label_inv"):
            return np.array([self._label_inv.get(i, 0) for i in indices])
        return indices

    def _make_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sliding-window sequences for LSTM input."""
        if len(X) <= self.lookback:
            return np.array([]), np.array([])

        sequences = []
        labels = []
        for i in range(self.lookback, len(X)):
            sequences.append(X[i - self.lookback : i])
            labels.append(y[i])
        return np.array(sequences), np.array(labels)

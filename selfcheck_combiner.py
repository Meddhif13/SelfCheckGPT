"""Combine metric scores via logistic regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class SelfCheckCombiner:
    """Simple logistic regression combiner using :mod:`torch`.

    Parameters
    ----------
    lr: float, optional
        Learning rate for SGD.
    epochs: int, optional
        Number of passes over the training data.
    device: str, optional
        Device to run training on (``"cpu"`` or ``"cuda"``).  If ``None`` the
        device is automatically chosen based on availability.
    seed: int, optional
        Random seed for deterministic initialisation.
    """

    lr: float = 0.1
    epochs: int = 100
    device: str | None = None
    seed: int = 0

    def __post_init__(self) -> None:  # pragma: no cover - small wrapper
        import torch

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None

    def fit(self, X: Iterable[Iterable[float]], y: Iterable[int]) -> "SelfCheckCombiner":
        """Fit the combiner on ``X`` and ``y``."""

        import torch
        from torch import nn, optim

        torch.manual_seed(self.seed)

        X_t = torch.tensor(np.array(list(X)), dtype=torch.float32, device=self.device)
        y_t = torch.tensor(np.array(list(y)), dtype=torch.float32, device=self.device).unsqueeze(1)

        n_features = X_t.shape[1]
        model = nn.Linear(n_features, 1).to(self.device)
        opt = optim.SGD(model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        for _ in range(self.epochs):
            opt.zero_grad()
            logits = model(X_t)
            loss = loss_fn(logits, y_t)
            loss.backward()
            opt.step()

        self._model = model
        return self

    def predict(self, X: Iterable[Iterable[float]]) -> list[float]:
        """Return probabilities for ``X``.

        ``fit`` must be called before ``predict``.
        """

        import torch

        if self._model is None:
            raise RuntimeError("Combiner has not been fitted")
        X_t = torch.tensor(np.array(list(X)), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self._model(X_t)).squeeze().cpu().numpy()
        return probs.tolist()

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
    lr:
        Learning rate for SGD.
    epochs:
        Maximum number of passes over the training data.
    l2:
        L2 regularisation strength (implemented via weight decay).
    patience:
        Number of epochs with no improvement on the validation loss before
        stopping early.  ``0`` disables early stopping.
    device:
        Device to run training on.
    seed:
        Random seed for deterministic initialisation.
    """

    lr: float = 0.1
    epochs: int = 100
    l2: float = 0.0
    patience: int = 0
    device: str | None = None
    seed: int = 0

    def __post_init__(self) -> None:  # pragma: no cover - small wrapper
        import torch

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self.trained_epochs = 0

    def fit(
        self,
        X: Iterable[Iterable[float]],
        y: Iterable[int],
        *,
        X_val: Iterable[Iterable[float]] | None = None,
        y_val: Iterable[int] | None = None,
    ) -> "SelfCheckCombiner":
        """Fit the combiner on ``X`` and ``y``.

        ``X_val`` and ``y_val`` optionally provide a validation split for
        early stopping.
        """

        import torch
        from torch import nn, optim

        torch.manual_seed(self.seed)

        X_t = torch.tensor(np.array(list(X)), dtype=torch.float32, device=self.device)
        y_t = torch.tensor(np.array(list(y)), dtype=torch.float32, device=self.device).unsqueeze(1)

        X_val_t = None
        y_val_t = None
        if X_val is not None and y_val is not None and self.patience > 0:
            X_val_t = torch.tensor(np.array(list(X_val)), dtype=torch.float32, device=self.device)
            y_val_t = torch.tensor(np.array(list(y_val)), dtype=torch.float32, device=self.device).unsqueeze(1)

        n_features = X_t.shape[1]
        model = nn.Linear(n_features, 1).to(self.device)
        opt = optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.l2)
        loss_fn = nn.BCEWithLogitsLoss()

        best_state = None
        best_loss = float("inf")
        epochs_run = 0
        patience_left = self.patience

        for epoch in range(self.epochs):
            opt.zero_grad()
            logits = model(X_t)
            loss = loss_fn(logits, y_t)
            loss.backward()
            opt.step()

            epochs_run = epoch + 1

            if X_val_t is not None:
                with torch.no_grad():
                    val_loss = loss_fn(model(X_val_t), y_val_t).item()
                if val_loss < best_loss - 1e-6:
                    best_loss = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_left = self.patience
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break

        self.trained_epochs = epochs_run
        if best_state is not None:
            model.load_state_dict(best_state)

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


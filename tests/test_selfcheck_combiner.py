import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from selfcheck_combiner import SelfCheckCombiner


def test_combiner_deterministic():
    X = np.array(
        [
            [0.1, 0.2],
            [0.9, 0.8],
            [0.2, 0.3],
            [0.8, 0.7],
        ]
    )
    y = np.array([0, 1, 0, 1])

    comb1 = SelfCheckCombiner(epochs=50, lr=0.5, device="cpu", seed=0)
    comb1.fit(X, y)
    preds1 = comb1.predict(X)

    comb2 = SelfCheckCombiner(epochs=50, lr=0.5, device="cpu", seed=0)
    comb2.fit(X, y)
    preds2 = comb2.predict(X)

    assert np.allclose(preds1, preds2)
    assert preds1[0] < preds1[1]


def test_combiner_l2_changes_predictions():
    X = np.array(
        [
            [0.1, 0.2],
            [0.9, 0.8],
            [0.2, 0.3],
            [0.8, 0.7],
        ]
    )
    y = np.array([0, 1, 0, 1])

    comb_no_reg = SelfCheckCombiner(epochs=50, lr=0.5, device="cpu", seed=0, l2=0.0)
    comb_no_reg.fit(X, y)
    preds_no_reg = comb_no_reg.predict(X)

    comb_reg = SelfCheckCombiner(epochs=50, lr=0.5, device="cpu", seed=0, l2=1.0)
    comb_reg.fit(X, y)
    preds_reg = comb_reg.predict(X)

    assert not np.allclose(preds_no_reg, preds_reg)


def test_combiner_early_stopping():
    X = np.zeros((4, 2), dtype=float)
    y = np.array([0, 1, 0, 1])
    X_val = np.zeros((2, 2), dtype=float)
    y_val = np.array([0, 1])

    comb = SelfCheckCombiner(epochs=50, lr=0.5, device="cpu", patience=2)
    comb.fit(X, y, X_val=X_val, y_val=y_val)

    # With constant validation loss early stopping should trigger before all epochs
    assert comb.trained_epochs < 50

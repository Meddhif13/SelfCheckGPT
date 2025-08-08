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

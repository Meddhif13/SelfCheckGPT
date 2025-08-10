import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from selfcheck_metrics import SelfCheckNLI


class _Batch(dict):
    def to(self, device):
        self.device = device
        return self


class _Tok:
    def __call__(self, premise, hypothesis, return_tensors="pt", truncation=True):
        return _Batch({"premise": premise, "hypothesis": hypothesis})


class _Model:
    def to(self, device):
        self.device = device
        return self

    def eval(self):
        pass

    def __call__(self, *, premise, hypothesis, **_):
        if "France" in premise and "France" in hypothesis:
            logits = torch.tensor([[0.0, 0.0, 2.0]])
        else:
            logits = torch.tensor([[2.0, 0.0, 0.0]])
        return types.SimpleNamespace(logits=logits)


def test_nli_stubbed_predict(monkeypatch):
    module = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda name: _Tok()),
        AutoModelForSequenceClassification=types.SimpleNamespace(
            from_pretrained=lambda name: _Model()
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", module)

    metric = SelfCheckNLI(model="stub")
    sents = ["Paris is in France.", "Paris is in Spain."]
    samples = ["Paris is in France. It is a city."]
    scores, logits = metric.predict(sents, samples, return_logits=True)
    assert scores[0] < 0.5
    assert scores[1] > 0.5
    assert logits == [[[0.0, 0.0, 2.0]], [[2.0, 0.0, 0.0]]]

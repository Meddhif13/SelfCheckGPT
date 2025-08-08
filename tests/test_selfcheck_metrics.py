import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from selfcheck_metrics import (
    SelfCheckBERTScore,
    SelfCheckMQAG,
    SelfCheckNgram,
    SelfCheckNLI,
    SelfCheckPrompt,
)


import pytest


def _load_bertscore():
    try:
        return SelfCheckBERTScore()
    except Exception as e:  # pragma: no cover - dependent on external model
        pytest.skip(f"BERTScore model unavailable: {e}")


def _expected(metric: SelfCheckBERTScore, sent: str, samples: list[str]) -> float:
    joined = " ".join(samples)
    _, _, F = metric.scorer.score([sent], [joined])
    return 1 - F.mean().item()


def test_bertscore_identical():
    metric = _load_bertscore()
    sent = ["Alice is a doctor."]
    samples = ["Alice is a doctor."]
    score = metric.predict(sent, samples)[0]
    expected = _expected(metric, sent[0], samples)
    assert score == pytest.approx(expected, abs=1e-6)


def test_bertscore_doctor_lawyer():
    metric = _load_bertscore()
    sent = ["Alice is a doctor."]
    samples = ["Alice is a lawyer."]
    score = metric.predict(sent, samples)[0]
    expected = _expected(metric, sent[0], samples)
    assert score == pytest.approx(expected, abs=1e-6)


def test_ngram_rare_word():
    metric = SelfCheckNgram()
    sents = ["common word", "rare token"]
    samples = ["common word common word"]
    scores = metric.predict(sents, samples)
    assert scores[1] > scores[0]

def test_ngram_bigram():
    samples = ["a b c", "a b d"]
    metric = SelfCheckNgram(n=2)
    scores = metric.predict(["a b c", "a c b"], samples)
    assert scores[1] > scores[0]


def test_ngram_trigram():
    samples = ["a b c d", "a b e d"]
    metric = SelfCheckNgram(n=3)
    scores = metric.predict(["a b c d", "a c b d"], samples)
    assert scores[1] > scores[0]


def test_nli_entailment_and_contradiction():
    def fake_nli(premise: str, hypothesis: str) -> tuple[float, float]:
        if "France" in premise and "France" in hypothesis:
            return 0.01, 0.95  # entailed
        return 0.9, 0.05  # contradiction

    metric = SelfCheckNLI(nli_fn=fake_nli)
    sents = ["Paris is in France.", "Paris is in Spain."]
    samples = ["Paris is in France. It is a city."]
    scores = metric.predict(sents, samples)
    assert scores[0] < 0.1  # entailed
    assert scores[1] > 0.9  # contradiction


def test_mqag_multiple_questions():
    def fake_qg(sentence: str) -> list[str]:
        if "John" in sentence:
            return ["What does John love?", "Who loves pizza?"]
        return ["What does Lucy climb?", "Who climbs trees?"]

    def fake_qa(question: str, context: str) -> str:
        if question == "What does John love?" and "pizza" in context:
            return "pizza"
        if question == "Who loves pizza?" and "John" in context and "pizza" in context:
            return "john"
        if question == "What does Lucy climb?" and "trees" in context:
            return "trees"
        if question == "Who climbs trees?" and "Lucy" in context and "trees" in context:
            return "lucy"
        return ""

    metric = SelfCheckMQAG(qg_fn=fake_qg, qa_fn=fake_qa)
    sents = ["John loves pizza", "Lucy climbs trees"]
    samples = [
        "Yesterday John ate pizza",
        "John still loves pizza",
        "Yesterday Lucy climbed trees",
    ]
    scores = metric.predict(sents, samples)
    assert math.isclose(scores[0], 1 / 3)
    assert math.isclose(scores[1], 2 / 3)
    assert metric.last_unanswerable == [1 / 3, 2 / 3]


def test_mqag_partial_unanswerable():
    def fake_qg(sentence: str) -> list[str]:
        return ["What does John love?", "Where does John live?"]

    def fake_qa(question: str, context: str) -> str:
        if question == "What does John love?" and "pizza" in context:
            return "pizza"
        if question == "Where does John live?" and "Boston" in context:
            return "boston"
        return ""

    metric = SelfCheckMQAG(qg_fn=fake_qg, qa_fn=fake_qa)
    sents = ["John loves pizza and lives in Boston"]
    samples = ["John loves pizza in Boston", "John loves pizza"]
    scores = metric.predict(sents, samples)
    assert math.isclose(scores[0], 0.25)
    assert metric.last_unanswerable == [0.25]


def test_prompt_mapping_yes_no():
    def fake_ask(context: str, sentence: str) -> str:
        return "Yes" if "earth" in context else "No"

    metric = SelfCheckPrompt(ask_fn=fake_ask)
    sents = ["The earth is round."]
    samples = ["Observation shows the earth is round.", "The moon orbits"]
    score = metric.predict(sents, samples)[0]
    assert score == 0.5  # one yes and one no


def test_prompt_openai_yes_no_mapping(monkeypatch):
    import types

    class FakeCompletions:
        def __init__(self):
            self.responses = ["Yes", "No"]
            self.calls = 0

        def create(self, model, messages, temperature):
            resp = self.responses[self.calls]
            self.calls += 1
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=resp)
                    )
                ]
            )

    class FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    module = types.ModuleType("openai")
    instances: list[FakeClient] = []

    def _factory(api_key=None):
        client = FakeClient(api_key)
        instances.append(client)
        return client

    module.OpenAI = _factory
    module.RateLimitError = Exception
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    metric = SelfCheckPrompt()
    sents = ["The earth is round."]
    samples = ["Observation shows the earth is round.", "The moon orbits"]
    score = metric.predict(sents, samples)[0]
    assert score == 0.5
    assert instances[0].api_key == "test"


def test_prompt_caching():
    calls = []

    def fake_ask(context: str, sentence: str) -> str:
        calls.append((context, sentence))
        return "Yes"

    metric = SelfCheckPrompt(ask_fn=fake_ask)
    sents = ["The earth is round."]
    samples = [
        "Observation shows the earth is round.",
        "Observation shows the earth is round.",
    ]
    score = metric.predict(sents, samples)[0]
    assert score == 0.0
    assert len(calls) == 1

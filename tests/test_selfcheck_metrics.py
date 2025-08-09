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
        return SelfCheckBERTScore(model="roberta-base")
    except Exception as e:  # pragma: no cover - dependent on external model
        pytest.skip(f"BERTScore model unavailable: {e}")


def _load_nli():
    try:
        return SelfCheckNLI(model="microsoft/deberta-v3-large-mnli")
    except Exception as e:  # pragma: no cover - dependent on external model
        pytest.skip(f"NLI model unavailable: {e}")


def _expected(metric: SelfCheckBERTScore, sent: str, samples: list[str]) -> float:
    scores: list[float] = []
    for sample in samples:
        _, _, F = metric.scorer.score([sent], [sample])
        scores.append(1 - F.mean().item())
    return sum(scores) / len(scores)


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


def test_bertscore_multiple_samples():
    metric = _load_bertscore()
    sent = ["Alice is a doctor."]
    samples = ["Alice is a doctor.", "Alice is a lawyer."]
    score = metric.predict(sent, samples)[0]
    expected = _expected(metric, sent[0], samples)
    assert score == pytest.approx(expected, abs=1e-6)


def test_ngram_rare_word():
    metric = SelfCheckNgram()
    sents = ["common word", "rare token"]
    samples = ["common word common word"]
    scores = metric.predict(sents, samples)
    assert scores["sentence_scores"][1] > scores["sentence_scores"][0]

def test_ngram_bigram():
    samples = ["a b c", "a b d"]
    metric = SelfCheckNgram(n=2)
    scores = metric.predict(["a b c", "a c b"], samples)
    assert scores["sentence_scores"][1] > scores["sentence_scores"][0]


def test_ngram_trigram():
    samples = ["a b c d", "a b e d"]
    metric = SelfCheckNgram(n=3)
    scores = metric.predict(["a b c d", "a c b d"], samples)
    assert scores["sentence_scores"][1] > scores["sentence_scores"][0]


def test_ngram_document_aggregates():
    samples = ["a a a"]
    metric = SelfCheckNgram()
    result = metric.predict(["a a", "a b"], samples)
    expected_avg = math.log(4) / 4
    expected_max = math.log(4) / 2
    assert result["avg_neg_logprob"] == pytest.approx(expected_avg, abs=1e-6)
    assert result["avg_max_neg_logprob"] == pytest.approx(expected_max, abs=1e-6)


def test_nli_entailment_and_contradiction():
    def fake_nli(premise: str, hypothesis: str) -> list[float]:
        if "France" in premise and "France" in hypothesis:
            return [0.0, 0.0, 2.0]  # entailed
        return [2.0, 0.0, 0.0]  # contradiction

    metric = SelfCheckNLI(nli_fn=fake_nli)
    sents = ["Paris is in France.", "Paris is in Spain."]
    samples = ["Paris is in France. It is a city."]
    scores = metric.predict(sents, samples)
    assert scores[0] < 0.5
    assert scores[1] > 0.5


def test_nli_model_entailment_contradiction():
    metric = _load_nli()
    sents = ["Paris is in France.", "Paris is in Spain."]
    samples = ["Paris is in France. It is a city."]
    scores = metric.predict(sents, samples)
    assert scores[0] < 0.5
    assert scores[1] > 0.5


def test_nli_allows_model_and_device(monkeypatch):
    import sys
    import types
    import torch

    calls: dict[str, str] = {}

    class _Batch(dict):
        def to(self, device):  # pragma: no cover - simple stub
            self["device"] = device
            return self

    def _tok_from_pretrained(name: str):
        calls["tokenizer"] = name

        class Tok:
            def __call__(self, premise, hypothesis, return_tensors, truncation):
                return _Batch({
                    "input_ids": torch.tensor([[0]]),
                    "attention_mask": torch.tensor([[1]]),
                })

        return Tok()

    class FakeModel:
        def __init__(self):
            self.moved_to = None

        def to(self, device):
            self.moved_to = device
            return self

        def eval(self):
            pass

        def __call__(self, **inputs):
            return types.SimpleNamespace(logits=torch.zeros((1, 3)))

    def _model_from_pretrained(name: str):
        calls["model"] = name
        return FakeModel()

    module = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=_tok_from_pretrained),
        AutoModelForSequenceClassification=types.SimpleNamespace(
            from_pretrained=_model_from_pretrained
        ),
    )

    monkeypatch.setitem(sys.modules, "transformers", module)

    metric = SelfCheckNLI(model="deberta-large-mnli", device="cuda")

    assert calls["model"] == "deberta-large-mnli"
    assert calls["tokenizer"] == "deberta-large-mnli"
    assert str(metric.model.moved_to) == "cuda"


def test_nli_temperature_calibration(monkeypatch):
    import sys
    import types
    import torch

    class _Batch(dict):
        def to(self, device):  # pragma: no cover - simple stub
            return self

    def _tok_from_pretrained(name: str):
        class Tok:
            def __call__(self, premise, hypothesis, return_tensors, truncation):
                return _Batch({
                    "input_ids": torch.tensor([[0]]),
                    "attention_mask": torch.tensor([[1]]),
                })

        return Tok()

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            pass

        def __call__(self, **inputs):
            return types.SimpleNamespace(
                logits=torch.tensor([[1.0, 0.0, -1.0]])
            )

    def _model_from_pretrained(name: str):
        return FakeModel()

    module = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=_tok_from_pretrained),
        AutoModelForSequenceClassification=types.SimpleNamespace(
            from_pretrained=_model_from_pretrained
        ),
    )

    monkeypatch.setitem(sys.modules, "transformers", module)

    metric = SelfCheckNLI(model="foo", temperature=2.0)
    score = metric.predict(["h"], ["p"])[0]

    logits = torch.tensor([[1.0, 0.0, -1.0]]) / 2.0
    probs = logits.softmax(dim=-1)[0]
    expected = probs[0]

    assert score == pytest.approx(expected.item())


def test_find_optimal_temperature():
    import torch
    from selfcheck_metrics import find_optimal_temperature

    logits = [[1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]
    labels = [0, 2]

    t = find_optimal_temperature(logits, labels)
    base = torch.nn.functional.cross_entropy(
        torch.tensor(logits), torch.tensor(labels)
    )
    calibrated = torch.nn.functional.cross_entropy(
        torch.tensor(logits) / t, torch.tensor(labels)
    )

    assert t > 0
    assert calibrated <= base


def test_temperature_calibration_reduces_cross_entropy():
    import torch
    from selfcheck_metrics import SelfCheckNLI, find_optimal_temperature

    def fake_nli(premise: str, hypothesis: str) -> list[float]:
        if "accurate" in hypothesis:
            return [0.2, 0.0, 0.0]
        return [0.0, 0.0, 0.2]

    metric = SelfCheckNLI(nli_fn=fake_nli)
    sentences = ["accurate", "inaccurate"]
    samples = ["ctx"]
    _, per_sent_logits = metric.predict(sentences, samples, return_logits=True)
    flat_logits = [per_sent_logits[0][0], per_sent_logits[1][0]]
    labels = [0, 2]

    t = find_optimal_temperature(flat_logits, labels)
    base = torch.nn.functional.cross_entropy(
        torch.tensor(flat_logits), torch.tensor(labels)
    )
    calibrated = torch.nn.functional.cross_entropy(
        torch.tensor(flat_logits) / t, torch.tensor(labels)
    )
    assert calibrated <= base


def test_mqag_allows_model_and_device(monkeypatch):
    import sys
    import types

    calls: list[tuple] = []

    def fake_pipeline(task, model=None, device=None):
        calls.append((task, model, device))

        if task == "text2text-generation":
            def _qg(text, num_return_sequences, num_beams):
                return [{"generated_text": "Q1"} for _ in range(num_return_sequences)]

            return _qg

        def _qa(inputs):
            return {"answer": "ans"}

        return _qa

    module = types.SimpleNamespace(pipeline=fake_pipeline)
    monkeypatch.setitem(sys.modules, "transformers", module)

    metric = SelfCheckMQAG(
        qg_model="t5-large", qa_model="deberta-v3-large", qg_device="cuda:0", qa_device="cuda:1"
    )
    metric.predict(["context"], ["sample"])

    qg_call = [c for c in calls if c[0] == "text2text-generation"][0]
    qa_call = [c for c in calls if c[0] == "question-answering"][0]
    assert qg_call[1:] == ("t5-large", "cuda:0")
    assert qa_call[1:] == ("deberta-v3-large", "cuda:1")


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
    scores, ans_stats = metric.predict(sents, samples)
    assert math.isclose(scores[0], 0.0)
    assert math.isclose(scores[1], 0.0)
    assert all(
        math.isclose(a, b)
        for row, ref in zip(ans_stats, [[2 / 3, 2 / 3], [1 / 3, 1 / 3]])
        for a, b in zip(row, ref)
    )
    assert metric.last_answerability == ans_stats
    assert metric.last_disagreement == scores
    assert math.isclose(metric.avg_disagreement, sum(scores) / 2)
    assert math.isclose(metric.avg_answerability, 0.5)
    assert all(
        math.isclose(a, b)
        for a, b in zip(metric.last_unanswerable, [1 / 3, 2 / 3])
    )
    assert math.isclose(
        metric.avg_unanswerable, sum(metric.last_unanswerable) / 2
    )


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
    scores, ans_stats = metric.predict(sents, samples)
    assert math.isclose(scores[0], 0.0)
    assert ans_stats == [[1.0, 0.5]]
    assert metric.last_answerability == ans_stats
    assert metric.last_disagreement == scores
    assert metric.avg_disagreement == scores[0]
    assert math.isclose(metric.avg_answerability, 0.75)
    assert metric.last_unanswerable == [0.25]
    assert metric.avg_unanswerable == 0.25


def test_mqag_bayes_methods():
    def fake_qg(sentence: str) -> list[str]:
        return ["What does John love?"]

    def fake_qa(question: str, context: str) -> str:
        if question == "What does John love?" and "pizza" in context:
            return "pizza"
        if question == "What does John love?" and "pasta" in context:
            return "pasta"
        return ""

    metric = SelfCheckMQAG(qg_fn=fake_qg, qa_fn=fake_qa)
    sents = ["John loves pizza"]
    samples = ["John loves pizza", "John loves pasta"]
    beta1, beta2 = 0.1, 0.6
    scores_bayes, _ = metric.predict(
        sents,
        samples,
        scoring_method="bayes",
        beta1=beta1,
        beta2=beta2,
    )
    scores_alpha, _ = metric.predict(
        sents,
        samples,
        scoring_method="bayes_with_alpha",
        beta1=beta1,
        beta2=beta2,
    )
    gamma1 = beta2 / (1 - beta1)
    gamma2 = beta1 / (1 - beta2)
    expected = (gamma2 ** 1) / ((gamma1 ** 1) + (gamma2 ** 1))
    assert math.isclose(scores_bayes[0], expected)
    assert scores_bayes == scores_alpha


def test_mqag_partial_match():
    def fake_qg(sentence: str) -> list[str]:
        return ["What does John love?"]

    def fake_qa(question: str, context: str) -> str:
        if "pepperoni pizza" in context:
            return "pepperoni pizza"
        if "pizza" in context:
            return "pizza"
        return ""

    metric = SelfCheckMQAG(qg_fn=fake_qg, qa_fn=fake_qa)
    sents = ["John loves pepperoni pizza"]
    samples = ["John loves pepperoni pizza", "John loves pizza"]
    scores, _ = metric.predict(sents, samples)
    f1 = metric._f1("pizza", "pepperoni pizza")
    expected = (2 - (1 + f1)) / 2
    assert math.isclose(scores[0], expected)


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


def test_prompt_custom_mapping():
    def fake_ask(context: str, sentence: str) -> str:
        # Echo the context so mapping can decide on the value
        return context

    mapping = {"good": 0.0, "bad": 1.0}

    def map_fn(ans: str) -> float:
        return mapping.get(ans, 0.5)

    metric = SelfCheckPrompt(ask_fn=fake_ask, map_fn=map_fn)
    sents = ["The earth is round."]
    samples = ["good", "bad"]
    score = metric.predict(sents, samples)[0]
    assert score == 0.5


def test_prompt_hf_backend(monkeypatch):
    import sys
    import types

    class DummyPipe:
        def __init__(self):
            self.last_prompt = None

        def __call__(self, prompt, **kwargs):
            self.last_prompt = prompt
            return [{"generated_text": "Yes"}]

    pipe = DummyPipe()

    transformers_mod = types.SimpleNamespace(pipeline=lambda *a, **k: pipe)
    torch_mod = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )

    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    metric = SelfCheckPrompt(hf_model="dummy")
    metric.set_prompt_template("C:{context}|S:{sentence}")
    score = metric.predict(["s"], ["c"])[0]
    assert score == 0.0  # 'Yes' maps to 0.0 by default
    assert pipe.last_prompt == "C:c|S:s"

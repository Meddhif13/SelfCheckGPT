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


def _load_mqag():
    try:
        import torch
    except Exception as e:  # pragma: no cover - optional dependency
        pytest.skip(f"PyTorch unavailable: {e}")
    if not torch.cuda.is_available():  # pragma: no cover - requires GPU
        pytest.skip("MQAG models require a GPU")
    try:
        return SelfCheckMQAG(num_questions=5)
    except Exception as e:  # pragma: no cover - dependent on external model
        pytest.skip(f"MQAG models unavailable: {e}")


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

    class FakeModel:
        def __init__(self, name):
            self.name = name
            calls.append(("model", name))

        def to(self, device):
            calls.append(("to", device))
            return self

        def eval(self):
            calls.append(("eval", self.name))
            return self

    class FakeTokenizer:
        def __init__(self, name):
            calls.append(("tokenizer", name))

    def fake_tok_from_pretrained(name):
        return FakeTokenizer(name)

    def fake_model_from_pretrained(name):
        return FakeModel(name)

    module = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=fake_tok_from_pretrained),
        AutoModelForSeq2SeqLM=types.SimpleNamespace(from_pretrained=fake_model_from_pretrained),
        LongformerTokenizer=types.SimpleNamespace(from_pretrained=fake_tok_from_pretrained),
        LongformerForMultipleChoice=types.SimpleNamespace(from_pretrained=fake_model_from_pretrained),
        AutoModelForSequenceClassification=types.SimpleNamespace(
            from_pretrained=fake_model_from_pretrained
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", module)

    SelfCheckMQAG(
        g1_model="g1", g2_model="g2", qa_model="qa", device="cuda:0"
    )

    assert ("tokenizer", "g1") in calls
    assert ("tokenizer", "g2") in calls
    assert ("tokenizer", "qa") in calls
    assert any("potsawee/longformer-large-4096-answerable-squad2" in t[1] for t in calls if t[0]=="tokenizer")
    assert calls.count(("to", "cuda:0")) == 4
    assert ("eval", "g1") in calls and ("eval", "g2") in calls and ("eval", "qa") in calls
    assert any(c[0] == "eval" and c[1].endswith("answerable-squad2") for c in calls)


def test_mqag_probability_distance():
    def fake_qg(sentence: str) -> list[dict]:
        return [
            {
                "question": "What does John love?",
                "options": ["pizza", "pasta", "sushi", "burger"],
            }
        ]

    def fake_qa(question: str, options: list[str], context: str) -> list[float]:
        if "pizza" in context:
            return [0.9, 0.05, 0.03, 0.02]
        if "pasta" in context:
            return [0.05, 0.9, 0.03, 0.02]
        return [0.25, 0.25, 0.25, 0.25]

    metric = SelfCheckMQAG(qg_fn=fake_qg, qa_fn=fake_qa)
    sents = ["John loves pizza"]
    samples = ["John loves pizza", "John loves pasta"]
    scores, ans_stats = metric.predict(
        sents, samples, metric="counting", disagreement_threshold=0.5
    )
    assert math.isclose(scores[0], 0.5)
    assert ans_stats == [[1.0]]
    assert metric.last_answerability == ans_stats
    assert metric.last_disagreement == scores


def test_mqag_answerability_filter():
    def fake_qg(sentence: str) -> list[dict]:
        return [
            {
                "question": "What does John love?",
                "options": ["pizza", "pasta", "sushi", "burger"],
            }
        ]

    def fake_qa(question: str, options: list[str], context: str) -> list[float]:
        if "pizza" in context:
            return [0.9, 0.05, 0.03, 0.02]
        return [0.3, 0.3, 0.2, 0.2]

    metric = SelfCheckMQAG(qg_fn=fake_qg, qa_fn=fake_qa)
    sents = ["John loves pizza"]
    samples = ["Unknown"]
    scores, ans_stats = metric.predict(
        sents, samples, metric="counting", disagreement_threshold=0.5
    )
    assert math.isclose(scores[0], 0.5)
    assert ans_stats == [[0.0]]
    assert metric.last_unanswerable == [1.0]
    assert math.isclose(metric.avg_unanswerable, 1.0)


def test_mqag_parity_with_paper():
    metric = _load_mqag()

    sents = [
        "Michael Alan Weiner (born March 31, 1942) is an American radio host.",
        "He is the host of The Savage Nation.",
    ]
    samples = [
        (
            "Michael Alan Weiner (born March 31, 1942) is an American radio host. "
            "He is the host of The Savage Country."
        ),
        (
            "Michael Alan Weiner (born January 13, 1960) is a Canadian radio host. "
            "He works at The New York Times."
        ),
        (
            "Michael Alan Weiner (born March 31, 1942) is an American radio host. "
            "He obtained his PhD from MIT."
        ),
    ]

    try:  # pragma: no cover - optional dependency
        import torch

        torch.manual_seed(0)
    except Exception:  # pragma: no cover - optional dependency
        pass

    scores, _ = metric.predict(sents, samples, metric="counting")
    expected = [0.30990949, 0.42376232]
    assert scores == pytest.approx(expected, rel=1e-3)


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

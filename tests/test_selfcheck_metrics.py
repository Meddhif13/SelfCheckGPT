import math
import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from selfcheck_metrics import (
    SelfCheckBERTScore,
    SelfCheckMQAG,
    SelfCheckNgram,
    SelfCheckNLI,
    SelfCheckPrompt,
)


def test_bertscore_identical():
    metric = SelfCheckBERTScore()
    sent = ["Alice is a doctor."]
    samples = ["Alice is a doctor."]
    score = metric.predict(sent, samples)[0]
    assert score < 0.2


def test_ngram_rare_word():
    metric = SelfCheckNgram()
    sents = ["common word", "rare token"]
    samples = ["common word common word"]
    scores = metric.predict(sents, samples)
    assert scores[1] > scores[0]


def test_nli_substring():
    metric = SelfCheckNLI()
    sents = ["Paris is in France."]
    samples = ["Paris is in France. It is a city."]
    score = metric.predict(sents, samples)[0]
    assert score == 0.0


def test_mqag_last_token():
    metric = SelfCheckMQAG()
    sents = ["John loves pizza"]
    samples = ["Yesterday John ate pizza"]
    score = metric.predict(sents, samples)[0]
    assert math.isclose(score, 0.0)


def test_prompt_mapping_yes_no():
    def fake_ask(context: str, sentence: str) -> str:
        return "Yes" if "earth" in context else "No"

    metric = SelfCheckPrompt(ask_fn=fake_ask)
    sents = ["The earth is round."]
    samples = ["Observation shows the earth is round.", "The moon orbits"]
    score = metric.predict(sents, samples)[0]
    assert score == 0.5  # one yes and one no


def test_prompt_openai_mocked(monkeypatch):
    calls = iter(["Yes", "No"])

    def fake_create(**kwargs):
        return {"choices": [{"message": {"content": next(calls)}}]}

    error_module = types.SimpleNamespace(OpenAIError=Exception)
    openai_module = types.SimpleNamespace(
        ChatCompletion=types.SimpleNamespace(create=fake_create),
        api_key="",
        error=error_module,
    )
    monkeypatch.setitem(sys.modules, "openai", openai_module)
    monkeypatch.setitem(sys.modules, "openai.error", error_module)

    metric = SelfCheckPrompt(api_key="test", rate_limit=0.0)
    sents = ["The earth is round."]
    samples = ["Observation shows the earth is round.", "The moon orbits"]
    score = metric.predict(sents, samples)[0]
    assert score == 0.5


def test_prompt_openai_error(monkeypatch):
    def fake_create(**kwargs):
        raise Exception("rate limit")

    error_module = types.SimpleNamespace(OpenAIError=Exception)
    openai_module = types.SimpleNamespace(
        ChatCompletion=types.SimpleNamespace(create=fake_create),
        api_key="",
        error=error_module,
    )
    monkeypatch.setitem(sys.modules, "openai", openai_module)
    monkeypatch.setitem(sys.modules, "openai.error", error_module)

    metric = SelfCheckPrompt(api_key="test", rate_limit=0.0, max_retries=1)
    sents = ["The earth is round."]
    samples = ["Observation shows the earth is round."]
    score = metric.predict(sents, samples)[0]
    assert score == 0.5

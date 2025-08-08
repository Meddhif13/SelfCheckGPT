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


def test_nli_substring():
    metric = SelfCheckNLI()
    sents = ["Paris is in France."]
    samples = ["Paris is in France. It is a city."]
    score = metric.predict(sents, samples)[0]
    assert score == 0.0


def test_mqag_answerable_and_unanswerable():
    def fake_qg(sentence: str) -> str:
        if "John" in sentence:
            return "What does John love?"
        return "What does Lucy climb?"

    def fake_qa(question: str, context: str) -> str:
        if "John" in question and "pizza" in context:
            return "pizza"
        if "Lucy" in question and "trees" in context:
            return "trees"
        return ""

    metric = SelfCheckMQAG(qg_fn=fake_qg, qa_fn=fake_qa)
    sents = ["John loves pizza", "Lucy climbs trees"]
    samples = ["Yesterday John ate pizza", "John still loves pizza"]
    scores = metric.predict(sents, samples)
    assert math.isclose(scores[0], 0.0)
    assert math.isclose(scores[1], 1.0)
    assert metric.last_unanswerable == [0.0, 1.0]


def test_mqag_partial_unanswerable():
    def fake_qg(sentence: str) -> str:
        return "What does John love?"

    def fake_qa(question: str, context: str) -> str:
        if "pizza" in context:
            return "pizza"
        return ""

    metric = SelfCheckMQAG(qg_fn=fake_qg, qa_fn=fake_qa)
    sents = ["John loves pizza"]
    samples = ["John loves pizza", "John hates broccoli"]
    scores = metric.predict(sents, samples)
    assert math.isclose(scores[0], 0.5)
    assert metric.last_unanswerable == [0.5]


def test_prompt_mapping_yes_no():
    def fake_ask(context: str, sentence: str) -> str:
        return "Yes" if "earth" in context else "No"

    metric = SelfCheckPrompt(ask_fn=fake_ask)
    sents = ["The earth is round."]
    samples = ["Observation shows the earth is round.", "The moon orbits"]
    score = metric.predict(sents, samples)[0]
    assert score == 0.5  # one yes and one no

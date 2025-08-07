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


def test_mqag_qg_qa_scoring():
    def fake_qg(sentence: str) -> str:
        return "What does John love?"

    def fake_qa(question: str, context: str) -> str:
        return "pizza" if "pizza" in context else "unknown"

    metric = SelfCheckMQAG(qg_fn=fake_qg, qa_fn=fake_qa)
    sents = ["John loves pizza"]
    samples = ["Yesterday John ate pizza", "John prefers pasta"]
    score = metric.predict(sents, samples)[0]
    assert math.isclose(score, 0.5)


def test_prompt_mapping_yes_no():
    def fake_ask(context: str, sentence: str) -> str:
        return "Yes" if "earth" in context else "No"

    metric = SelfCheckPrompt(ask_fn=fake_ask)
    sents = ["The earth is round."]
    samples = ["Observation shows the earth is round.", "The moon orbits"]
    score = metric.predict(sents, samples)[0]
    assert score == 0.5  # one yes and one no

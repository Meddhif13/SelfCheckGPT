import sys
import types
import math
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

import selfcheck_metrics
from selfcheck_metrics import SelfCheckMQAG
from selfcheckgpt import utils as sc_utils


class _Enc:
    def __init__(self, ids=None):
        self.input_ids = torch.tensor([[0]]) if ids is None else ids

    def to(self, device):
        return self


class _G1Tok:
    pad_token = "<pad>"
    eos_token = "</s>"
    sep_token = "<sep>"

    def __call__(self, *args, **kwargs):
        return _Enc()

    def decode(self, ids, skip_special_tokens=False):
        return "What does John love?<sep>pizza"


class _G2Tok(_G1Tok):
    def decode(self, ids, skip_special_tokens=False):
        return "pasta<sep>sushi<sep>burger"


class _GenModel:
    def to(self, device):
        return self

    def eval(self):
        pass

    def generate(self, *args, **kwargs):
        return torch.tensor([[0]])


class _AnsTokenizer:
    bos_token = "<s>"

    def __call__(self, question, context, return_tensors="pt", padding="longest", truncation=True):
        class Enc(dict):
            def __init__(self, ctx):
                super().__init__(
                    {
                        "input_ids": torch.tensor([[0]]),
                        "attention_mask": torch.tensor([[1]]),
                        "context": ctx,
                    }
                )

            def to(self, device):
                return self

        return Enc(context)


class _AnswerModel:
    def to(self, device):
        return self

    def eval(self):
        pass

    def __call__(self, *, input_ids=None, attention_mask=None, context, options):
        if "pizza" in context:
            logits = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
        elif "pasta" in context:
            logits = torch.tensor([[0.0, 2.0, 0.0, 0.0]])
        else:
            logits = torch.zeros((1, 4))
        return types.SimpleNamespace(logits=logits)


class _AnsModel:
    def to(self, device):
        return self

    def eval(self):
        pass

    def __call__(self, *, input_ids=None, attention_mask=None, context):
        logits = torch.tensor([[0.0, 2.1972246]])  # softmax -> [0.1, 0.9]
        return types.SimpleNamespace(logits=logits)


def _prepare_answering_input(tokenizer, question, options, context, *, device=None, max_seq_length=4096):
    return {
        "input_ids": torch.zeros((1, len(options), 1)),
        "attention_mask": torch.ones((1, len(options), 1)),
        "context": context,
        "options": options,
    }


def test_mqag_stubbed_predict(monkeypatch):
    module = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(
            from_pretrained=lambda name: _G1Tok() if name == "g1" else _G2Tok()
        ),
        AutoModelForSeq2SeqLM=types.SimpleNamespace(
            from_pretrained=lambda name: _GenModel()
        ),
        LongformerTokenizer=types.SimpleNamespace(from_pretrained=lambda name: _AnsTokenizer()),
        LongformerForMultipleChoice=types.SimpleNamespace(from_pretrained=lambda name: _AnswerModel()),
        AutoModelForSequenceClassification=types.SimpleNamespace(
            from_pretrained=lambda name: _AnsModel()
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", module)
    monkeypatch.setattr(sc_utils, "prepare_answering_input", _prepare_answering_input)
    monkeypatch.setattr(selfcheck_metrics, "prepare_answering_input", _prepare_answering_input)

    metric = SelfCheckMQAG(num_questions=1, g1_model="g1", g2_model="g2", qa_model="qa", answer_model="ans")
    sents = ["John loves pizza"]
    samples = ["John loves pizza", "John loves pasta"]
    scores, ans_stats = metric.predict(
        sents, samples, metric="counting", disagreement_threshold=0.5
    )
    assert math.isclose(scores[0], 0.5)
    assert math.isclose(ans_stats[0][0], 0.9, rel_tol=1e-5)
    assert metric.last_disagreement == scores
    assert metric.last_answerability == ans_stats

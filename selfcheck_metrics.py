"""Simplified SelfCheckGPT metrics implementations.

This module provides light-weight approximations of the five
SelfCheckGPT variants described in the paper.  The goal of this file is
not to perfectly reproduce the paper's results but to expose a clear API
that mirrors the original implementation.  Each class exposes a
``predict`` method which takes a list of sentences from a main passage
and a list of sample passages generated from the same prompt.  The method
returns a list of inconsistency scores where higher values indicate a
higher likelihood of hallucination.

Most heavy models used in the paper (e.g. DeBERTa for NLI) are optional.
When the required libraries or model weights are not available the code
falls back to light-weight heuristics so that the project remains
runnable in constrained environments.  The BERTScore variant now relies
on the real :mod:`bert_score` package and therefore requires access to
the corresponding model weights.
"""

from __future__ import annotations

from typing import Callable, Iterable, List
import collections
import math


# ---------------------------------------------------------------------------
# BERTScore -----------------------------------------------------------------
# ---------------------------------------------------------------------------

class SelfCheckBERTScore:
    """BERTScore-based SelfCheckGPT variant.

    This implementation initializes :class:`bert_score.BERTScorer` by default
    and computes the inconsistency score as ``1 - F1``.  Model selection and
    baseline rescaling can be configured via the constructor.

    Parameters
    ----------
    model: str, optional
        HuggingFace model name, e.g. ``"roberta-large"``.
    baseline: bool, optional
        If ``True`` scores are rescaled with the BERTScore baseline.
    """

    def __init__(self, model: str = "roberta-large", baseline: bool = True) -> None:
        try:
            from bert_score import BERTScorer  # type: ignore

            self.scorer = BERTScorer(
                lang="en", rescale_with_baseline=baseline, model_type=model
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("BERTScore is unavailable") from exc

    def predict(self, sentences: Iterable[str], samples: Iterable[str]) -> List[float]:
        joined_samples = " ".join(samples)
        scores: List[float] = []
        for sent in sentences:
            _, _, F = self.scorer.score([sent], [joined_samples])
            score = 1 - F.mean().item()
            scores.append(float(score))
        return scores


# ---------------------------------------------------------------------------
# MQAG (Question Answering) --------------------------------------------------
# ---------------------------------------------------------------------------

class SelfCheckMQAG:
    """Question generation and answering based scorer.

    The real MQAG variant in the SelfCheckGPT paper generates questions
    from every sentence and runs a QA model over sampled passages.  The
    reference answer derived from the original sentence is compared with
    the answers from each sample.  The inconsistency score is the ratio
    of disagreements (``1 - matches/num_samples``).  ``SelfCheckMQAG``
    now always loads a T5-based question generation model and a QA model
    (DistilBERT by default).  Custom callables ``qg_fn`` and ``qa_fn``
    can be provided for testing purposes.  For each sentence the method
    also records the fraction of unanswerable samples in
    ``last_unanswerable``.
    """

    def __init__(
        self,
        qg_fn: Callable[[str], str] | None = None,
        qa_fn: Callable[[str, str], str] | None = None,
        batch_size: int = 8,
    ) -> None:
        self.qg_fn = qg_fn
        self.qa_fn = qa_fn
        self.batch_size = batch_size
        self.last_unanswerable: List[float] = []

        if self.qg_fn is None or self.qa_fn is None:
            try:  # pragma: no cover - heavy branch
                from transformers import pipeline  # type: ignore

                self.qg_pipe = pipeline(
                    "text2text-generation", model="valhalla/t5-small-qg-hl"
                )
                self.qa_pipe = pipeline(
                    "question-answering",
                    model="distilbert-base-uncased-distilled-squad",
                )
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("transformers models unavailable") from exc
        else:
            self.qg_pipe = None
            self.qa_pipe = None

    def predict(self, sentences: Iterable[str], samples: Iterable[str]) -> List[float]:
        sentences = list(sentences)
        samples = list(samples)
        total = len(samples) or 1

        if getattr(self, "qg_pipe", None) is not None:
            # Batch question generation
            qg_outputs = self.qg_pipe(sentences, batch_size=self.batch_size)
            questions = [o["generated_text"].strip() for o in qg_outputs]

            # Reference answers in batch
            ref_inputs = [
                {"question": q, "context": s} for q, s in zip(questions, sentences)
            ]
            ref_outputs = self.qa_pipe(ref_inputs, batch_size=self.batch_size)
            ref_answers = [o.get("answer", "").strip().lower() for o in ref_outputs]

            # Sample answers for all question/sample pairs in a single batch
            qa_inputs = []
            for sample in samples:
                qa_inputs.extend(
                    {"question": q, "context": sample} for q in questions
                )
            qa_outputs = self.qa_pipe(qa_inputs, batch_size=self.batch_size)

            answers_per_question: List[List[str]] = [[] for _ in questions]
            for j, _sample in enumerate(samples):
                for i in range(len(questions)):
                    ans = qa_outputs[j * len(questions) + i].get("answer", "").strip().lower()
                    answers_per_question[i].append(ans)

            scores: List[float] = []
            unans_ratios: List[float] = []
            for ref_answer, ans_list in zip(ref_answers, answers_per_question):
                matches = sum(1 for ans in ans_list if ans == ref_answer and ans)
                unans = sum(1 for ans in ans_list if not ans)
                scores.append(1 - matches / total)
                unans_ratios.append(unans / total)

            self.last_unanswerable = unans_ratios
            return scores

        # Custom functions (typically used in tests)
        questions = [self.qg_fn(s).strip() for s in sentences]
        ref_answers = [
            self.qa_fn(q, s).strip().lower() for q, s in zip(questions, sentences)
        ]
        scores: List[float] = []
        unans_ratios: List[float] = []
        for q, ref_answer in zip(questions, ref_answers):
            matches = 0
            unans = 0
            for sample in samples:
                ans = self.qa_fn(q, sample).strip().lower()
                if not ans:
                    unans += 1
                elif ans == ref_answer and ref_answer:
                    matches += 1
            scores.append(1 - matches / total)
            unans_ratios.append(unans / total)

        self.last_unanswerable = unans_ratios
        return scores


# ---------------------------------------------------------------------------
# n-gram --------------------------------------------------------------------
# ---------------------------------------------------------------------------

class SelfCheckNgram:
    """Unigram-based approximation used for hallucination detection."""

    def predict(self, sentences: Iterable[str], samples: Iterable[str]) -> List[float]:
        counter = collections.Counter()
        for text in samples:
            counter.update(text.lower().split())
        vocab_size = len(counter) or 1
        total = sum(counter.values()) + vocab_size
        scores: List[float] = []
        for sent in sentences:
            min_prob = 1.0
            for tok in sent.lower().split():
                prob = (counter.get(tok, 0) + 1) / total
                if prob < min_prob:
                    min_prob = prob
            scores.append(-math.log(min_prob))
        return scores


# ---------------------------------------------------------------------------
# NLI -----------------------------------------------------------------------
# ---------------------------------------------------------------------------

class SelfCheckNLI:
    """Toy NLI scorer based on substring matching.

    The real implementation would use a trained NLI model.  Here we
    simply check whether each sentence appears in the sample passages.
    """

    def predict(self, sentences: Iterable[str], samples: Iterable[str]) -> List[float]:
        sample_text = " ".join(s.lower() for s in samples)
        scores: List[float] = []
        for sent in sentences:
            score = 0.0 if sent.lower() in sample_text else 1.0
            scores.append(score)
        return scores


# ---------------------------------------------------------------------------
# LLM Prompt ----------------------------------------------------------------
# ---------------------------------------------------------------------------

class SelfCheckPrompt:
    """Prompt an external LLM with a Yes/No question.

    The constructor accepts an ``ask_fn`` callable used to query the LLM.
    This makes the class easy to test as the heavy API call can be
    replaced with a stub.
    """

    def __init__(self, ask_fn: Callable[[str, str], str] | None = None) -> None:
        self.ask_fn = ask_fn or self._openai_ask

    # -- Actual API call -----------------------------------------------------
    def _openai_ask(self, context: str, sentence: str) -> str:  # pragma: no cover - requires network
        import openai

        prompt = (
            f"Context: {context}\nSentence: {sentence}\n"
            "Is the sentence supported by the context above?\nAnswer Yes or No:"
        )
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return res["choices"][0]["message"]["content"].strip()

    def predict(self, sentences: Iterable[str], samples: Iterable[str]) -> List[float]:
        samples = list(samples)
        scores: List[float] = []
        for sent in sentences:
            total = 0.0
            for sample in samples:
                ans = self.ask_fn(sample, sent).strip().lower()
                if ans.startswith("y"):
                    val = 0.0
                elif ans.startswith("n"):
                    val = 1.0
                else:
                    val = 0.5
                total += val
            scores.append(total / max(1, len(samples)))
        return scores


__all__ = [
    "SelfCheckBERTScore",
    "SelfCheckMQAG",
    "SelfCheckNgram",
    "SelfCheckNLI",
    "SelfCheckPrompt",
]


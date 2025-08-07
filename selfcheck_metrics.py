"""Simplified SelfCheckGPT metrics implementations.

This module provides light-weight approximations of the five
SelfCheckGPT variants described in the paper.  The goal of this file is
not to perfectly reproduce the paper's results but to expose a clear API
that mirrors the original implementation.  Each class exposes a
``predict`` method which takes a list of sentences from a main passage
and a list of sample passages generated from the same prompt.  The method
returns a list of inconsistency scores where higher values indicate a
higher likelihood of hallucination.

The heavy models used in the paper (e.g. RoBERTa-large for BERTScore or
DeBERTa for NLI) are optional.  When the required libraries or model
weights are not available the code falls back to light-weight heuristics
so that the project remains runnable in constrained environments.
"""

from __future__ import annotations

from typing import Callable, Iterable, List
import collections
import math


# ---------------------------------------------------------------------------
# BERTScore -----------------------------------------------------------------
# ---------------------------------------------------------------------------

class SelfCheckBERTScore:
    """Approximation of the SelfCheckGPT-BERTScore variant.

    Parameters
    ----------
    use_bert_score: bool, optional
        If ``True`` and the :mod:`bert_score` package is available the real
        BERTScore implementation is used.  Otherwise a simple Jaccard
        similarity between tokens is employed.  The inconsistency score is
        ``1 - similarity`` so that higher means more likely hallucinated.
    """

    def __init__(self, use_bert_score: bool = False) -> None:
        self.scorer = None
        if use_bert_score:
            try:
                from bert_score import BERTScorer  # type: ignore

                self.scorer = BERTScorer(lang="en", rescale_with_baseline=True)
            except Exception:  # pragma: no cover - optional dependency
                self.scorer = None

    def _jaccard(self, a: str, b: str) -> float:
        ta = set(a.lower().split())
        tb = set(b.lower().split())
        if not ta and not tb:
            return 1.0
        return len(ta & tb) / len(ta | tb)

    def predict(self, sentences: Iterable[str], samples: Iterable[str]) -> List[float]:
        joined_samples = " ".join(samples)
        scores: List[float] = []
        for sent in sentences:
            if self.scorer is not None:  # pragma: no cover - heavy branch
                _, _, F = self.scorer.score([sent], [joined_samples])
                score = 1 - F.mean().item()
            else:
                score = 1 - self._jaccard(sent, joined_samples)
            scores.append(float(score))
        return scores


# ---------------------------------------------------------------------------
# MQAG (Question Answering) --------------------------------------------------
# ---------------------------------------------------------------------------

class SelfCheckMQAG:
    """Very small proxy for the MQAG approach.

    The implementation is intentionally simple: the "answer" to a
    sentence is assumed to be its final token.  The score counts in how
    many of the sample passages this token does *not* appear.
    """

    def predict(self, sentences: Iterable[str], samples: Iterable[str]) -> List[float]:
        samples = list(samples)
        scores: List[float] = []
        for sent in sentences:
            if not sent.split():
                scores.append(0.0)
                continue
            answer = sent.split()[-1].strip(". ,")
            missing = sum(1 for s in samples if answer not in s)
            scores.append(missing / max(1, len(samples)))
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


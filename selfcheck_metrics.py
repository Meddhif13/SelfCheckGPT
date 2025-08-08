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

    This implementation initializes :class:`bert_score.BERTScorer` and compares
    each sentence with every sampled passage separately.  The inconsistency score
    for a sentence is the average ``1 - F1`` across all samples.  Model selection
    and baseline rescaling can be configured via the constructor.  When a GPU is
    available the underlying BERTScore model runs on ``cuda``.

    Parameters
    ----------
    model: str, optional
        HuggingFace model name, e.g. ``"roberta-large"``.
    baseline: bool, optional
        If ``True`` scores are rescaled with the BERTScore baseline.
    """

    def __init__(self, model: str = "roberta-large", baseline: bool = True) -> None:
        try:
            import torch
            from bert_score import BERTScorer  # type: ignore

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.scorer = BERTScorer(
                lang="en",
                rescale_with_baseline=baseline,
                model_type=model,
                device=device,
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("BERTScore is unavailable") from exc

    def predict(self, sentences: Iterable[str], samples: Iterable[str]) -> List[float]:
        samples = list(samples)
        scores: List[float] = []
        for sent in sentences:
            gaps: List[float] = []
            for sample in samples:
                P, R, F = self.scorer.score([sent], [sample])
                gaps.append(1 - F.mean().item())
            if gaps:
                scores.append(float(sum(gaps) / len(gaps)))
            else:
                scores.append(0.0)
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
    can load arbitrary HuggingFace models for both question generation
    and answering and allows explicitly setting the device for each
    pipeline.  By default it uses a T5-based QG model and a DistilBERT QA
    model.  Multiple questions are generated for each sentence which are
    individually answered on the samples.  Scores and unanswerable ratios
    are averaged over these questions and exposed via ``last_disagreement``
    and ``last_unanswerable``.  The attributes ``avg_disagreement`` and
    ``avg_unanswerable`` summarize these statistics over all sentences.
    """

    def __init__(
        self,
        qg_fn: Callable[[str], Iterable[str]] | None = None,
        qa_fn: Callable[[str, str], str] | None = None,
        batch_size: int = 8,
        num_questions: int = 3,
        qg_model: str = "valhalla/t5-small-qg-hl",
        qa_model: str = "distilbert-base-uncased-distilled-squad",
        qg_device: int | str | None = None,
        qa_device: int | str | None = None,
    ) -> None:
        self.qg_fn = qg_fn
        self.qa_fn = qa_fn
        self.batch_size = batch_size
        self.num_questions = max(1, num_questions)
        self.last_unanswerable: List[float] = []
        self.last_disagreement: List[float] = []
        self.avg_unanswerable: float = 0.0
        self.avg_disagreement: float = 0.0

        if self.qg_fn is None or self.qa_fn is None:
            try:  # pragma: no cover - heavy branch
                from transformers import pipeline  # type: ignore
                import torch  # type: ignore

                def _resolve(dev):
                    if dev is not None:
                        return dev
                    return 0 if torch.cuda.is_available() else -1

                self.qg_pipe = pipeline(
                    "text2text-generation",
                    model=qg_model,
                    device=_resolve(qg_device),
                )
                self.qa_pipe = pipeline(
                    "question-answering",
                    model=qa_model,
                    device=_resolve(qa_device),
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
            # Generate multiple questions per sentence
            all_questions: List[List[str]] = []
            for sent in sentences:
                outputs = self.qg_pipe(
                    sent,
                    num_return_sequences=self.num_questions,
                    num_beams=self.num_questions,
                )
                all_questions.append([o["generated_text"].strip() for o in outputs])

            # Reference answers for every question
            all_ref_answers: List[List[str]] = []
            for qs, sent in zip(all_questions, sentences):
                ref_list = []
                for q in qs:
                    ans = (
                        self.qa_pipe({"question": q, "context": sent})
                        .get("answer", "")
                        .strip()
                        .lower()
                    )
                    ref_list.append(ans)
                all_ref_answers.append(ref_list)

            scores: List[float] = []
            unans_avgs: List[float] = []
            for qs, refs in zip(all_questions, all_ref_answers):
                q_scores: List[float] = []
                q_unans: List[float] = []
                for q, ref in zip(qs, refs):
                    matches = 0
                    unans = 0
                    for sample in samples:
                        ans = (
                            self.qa_pipe({"question": q, "context": sample})
                            .get("answer", "")
                            .strip()
                            .lower()
                        )
                        if not ans:
                            unans += 1
                        elif ans == ref and ref:
                            matches += 1
                    q_scores.append(1 - matches / total)
                    q_unans.append(unans / total)
                scores.append(sum(q_scores) / len(q_scores))
                unans_avgs.append(sum(q_unans) / len(q_unans))

            self.last_unanswerable = unans_avgs
            self.last_disagreement = scores
            self.avg_unanswerable = sum(unans_avgs) / len(unans_avgs) if unans_avgs else 0.0
            self.avg_disagreement = sum(scores) / len(scores) if scores else 0.0
            return scores

        # Custom functions (typically used in tests)
        all_questions: List[List[str]] = []
        for s in sentences:
            q_res = self.qg_fn(s)
            if isinstance(q_res, str):
                q_list = [q_res]
            else:
                q_list = list(q_res)
            all_questions.append([q.strip() for q in q_list])

        all_ref_answers: List[List[str]] = []
        for qs, s in zip(all_questions, sentences):
            all_ref_answers.append([self.qa_fn(q, s).strip().lower() for q in qs])

        scores: List[float] = []
        unans_avgs: List[float] = []
        for qs, refs in zip(all_questions, all_ref_answers):
            q_scores: List[float] = []
            q_unans: List[float] = []
            for q, ref in zip(qs, refs):
                matches = 0
                unans = 0
                for sample in samples:
                    ans = self.qa_fn(q, sample).strip().lower()
                    if not ans:
                        unans += 1
                    elif ans == ref and ref:
                        matches += 1
                q_scores.append(1 - matches / total)
                q_unans.append(unans / total)
            scores.append(sum(q_scores) / len(q_scores))
            unans_avgs.append(sum(q_unans) / len(q_unans))

        self.last_unanswerable = unans_avgs
        self.last_disagreement = scores
        self.avg_unanswerable = sum(unans_avgs) / len(unans_avgs) if unans_avgs else 0.0
        self.avg_disagreement = sum(scores) / len(scores) if scores else 0.0
        return scores


# ---------------------------------------------------------------------------
# n-gram --------------------------------------------------------------------
# ---------------------------------------------------------------------------

class SelfCheckNgram:
    """n-gram based approximation used for hallucination detection.

    Parameters
    ----------
    n: int, optional
        Order of the language model (default 1).
    smoothing: str, optional
        Smoothing method to use.  Supported values are ``"backoff"`` and
        ``"kneser_ney"`` (default ``"backoff"``).
    """

    def __init__(self, n: int = 1, smoothing: str = "backoff", discount: float = 0.75) -> None:
        self.n = max(1, n)
        self.smoothing = smoothing
        self.discount = discount

    # -- Utility -------------------------------------------------------------
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()

    def _build_model(self, samples: Iterable[str]):
        counts = [collections.Counter() for _ in range(self.n)]
        followers = [collections.defaultdict(set) for _ in range(max(0, self.n - 1))]
        histories = collections.defaultdict(set)

        for text in samples:
            tokens = self._tokenize(text)
            for i in range(len(tokens)):
                for k in range(1, self.n + 1):
                    if i + k <= len(tokens):
                        gram = tuple(tokens[i : i + k])
                        counts[k - 1][gram] += 1
                        if self.smoothing == "kneser_ney" and k < self.n and i + k < len(tokens):
                            followers[k - 1][gram].add(tokens[i + k])
                if self.smoothing == "kneser_ney" and i > 0:
                    histories[(tokens[i],)].add(tokens[i - 1])

        total_tokens = sum(counts[0].values())
        vocab_size = len(counts[0]) or 1
        total_continuations = sum(len(v) for v in histories.values()) or 1
        return counts, followers, histories, total_tokens, vocab_size, total_continuations

    # -- Probability estimation ---------------------------------------------
    def _prob_backoff(
        self,
        gram: tuple[str, ...],
        counts: list[collections.Counter],
        total_tokens: int,
        vocab_size: int,
    ) -> float:
        k = len(gram)
        if k == 0:
            return 1.0 / vocab_size
        if k == 1:
            return (counts[0].get(gram, 0) + 1) / (total_tokens + vocab_size)
        count = counts[k - 1].get(gram, 0)
        if count > 0:
            prefix = gram[:-1]
            prefix_count = counts[k - 2].get(prefix, 0)
            return (count + 1) / (prefix_count + vocab_size)
        return self._prob_backoff(gram[1:], counts, total_tokens, vocab_size)

    def _prob_kneser_ney(
        self,
        gram: tuple[str, ...],
        counts: list[collections.Counter],
        followers: list[collections.defaultdict],
        histories: collections.defaultdict,
        total_continuations: int,
        vocab_size: int,
    ) -> float:
        k = len(gram)
        if k == 1:
            cont = len(histories.get(gram, []))
            if total_continuations == 0:
                return 1.0 / vocab_size
            return cont / total_continuations
        count = counts[k - 1].get(gram, 0)
        prefix = gram[:-1]
        prefix_count = counts[k - 2].get(prefix, 0)
        if prefix_count == 0:
            return self._prob_kneser_ney(
                gram[1:], counts, followers, histories, total_continuations, vocab_size
            )
        uniq = len(followers[k - 2].get(prefix, []))
        discount = self.discount
        lower = self._prob_kneser_ney(
            gram[1:], counts, followers, histories, total_continuations, vocab_size
        )
        return max(count - discount, 0) / prefix_count + (discount * uniq / prefix_count) * lower

    # -- Public API ----------------------------------------------------------
    def predict(self, sentences: Iterable[str], samples: Iterable[str]) -> List[float]:
        counts, followers, histories, total_tokens, vocab_size, total_continuations = self._build_model(samples)
        scores: List[float] = []

        for sent in sentences:
            tokens = self._tokenize(sent)
            if len(tokens) < self.n:
                grams = [tuple(tokens)]
            else:
                grams = [tuple(tokens[i : i + self.n]) for i in range(len(tokens) - self.n + 1)]

            min_prob = 1.0
            for gram in grams:
                if self.smoothing == "kneser_ney":
                    prob = self._prob_kneser_ney(
                        gram, counts, followers, histories, total_continuations, vocab_size
                    )
                else:
                    prob = self._prob_backoff(gram, counts, total_tokens, vocab_size)
                if prob < min_prob:
                    min_prob = prob
            scores.append(-math.log(min_prob))

        return scores


# ---------------------------------------------------------------------------
# NLI -----------------------------------------------------------------------
# ---------------------------------------------------------------------------

class SelfCheckNLI:
    """NLI based scorer using a pretrained model.

    By default :class:`SelfCheckNLI` loads the
    ``microsoft/deberta-base-mnli`` model to obtain Natural Language
    Inference probabilities.  A different model name can be supplied, for
    example ``microsoft/deberta-large-mnli``.  For each pair of
    ``(sample, sentence)`` it computes the probability of contradiction and
    entailment and derives an inconsistency score
    ``0.5 * (p_contra + (1 - p_entail))``.  Higher scores therefore indicate
    that the sentence is not supported by the sampled passages.

    A custom ``nli_fn`` can be supplied for testing which should accept a
    premise and hypothesis and return a tuple ``(p_contra, p_entail)``.
    """

    def __init__(
        self,
        model: str = "microsoft/deberta-base-mnli",
        nli_fn: Callable[[str, str], tuple[float, float]] | None = None,
        device: str | None = None,
        temperature: float = 1.0,
    ) -> None:
        self.temperature = temperature
        if nli_fn is None:
            try:  # pragma: no cover - heavy branch
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )  # type: ignore
                import torch  # type: ignore

                self.device = torch.device(
                    device or ("cuda" if torch.cuda.is_available() else "cpu")
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model)
                self.model = AutoModelForSequenceClassification.from_pretrained(model)
                self.model.to(self.device)
                self.model.eval()

                def _hf_nli(premise: str, hypothesis: str) -> tuple[float, float]:
                    inputs = self.tokenizer(
                        premise,
                        hypothesis,
                        return_tensors="pt",
                        truncation=True,
                    ).to(self.device)
                    with torch.no_grad():
                        logits = self.model(**inputs).logits
                        if self.temperature != 1.0:
                            logits = logits / self.temperature
                    probs = logits.softmax(dim=-1)[0].tolist()
                    if len(probs) == 3:
                        p_contra, _p_neutral, p_entail = probs
                    else:  # pragma: no cover - edge case
                        p_contra, p_entail = probs[0], probs[-1]
                    return float(p_contra), float(p_entail)

                self.nli_fn = _hf_nli
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("transformers NLI model unavailable") from exc
        else:
            self.nli_fn = nli_fn
            self.device = device

    def predict(self, sentences: Iterable[str], samples: Iterable[str]) -> List[float]:
        sentences = list(sentences)
        samples = list(samples)
        total = len(samples) or 1
        scores: List[float] = []
        for sent in sentences:
            agg = 0.0
            for sample in samples:
                p_contra, p_entail = self.nli_fn(sample, sent)
                agg += 0.5 * (p_contra + (1 - p_entail))
            scores.append(agg / total)
        return scores


# ---------------------------------------------------------------------------
# LLM Prompt ----------------------------------------------------------------
# ---------------------------------------------------------------------------

class SelfCheckPrompt:
    """Prompt an external LLM with a Yes/No question.

    The constructor accepts an ``ask_fn`` callable used to query the LLM.
    This makes the class easy to test as the heavy API call can be
    replaced with a stub.  Results are cached so repeated queries with
    the same context/sentence pair do not trigger additional API calls.
    """

    def __init__(
        self,
        ask_fn: Callable[[str, str], str] | None = None,
        model: str = "gpt-3.5-turbo",
        max_retries: int = 3,
        retry_wait: float = 1.0,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self._client = None

        self._raw_ask = ask_fn or self._openai_ask
        self._cache: dict[tuple[str, str], str] = {}

        def cached_ask(context: str, sentence: str) -> str:
            key = (context, sentence)
            if key not in self._cache:
                self._cache[key] = self._raw_ask(context, sentence)
            return self._cache[key]

        self.ask_fn = cached_ask

    # -- Actual API call -----------------------------------------------------
    def _openai_ask(
        self, context: str, sentence: str
    ) -> str:  # pragma: no cover - requires network
        import os
        import time
        from openai import OpenAI, RateLimitError

        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            self._client = OpenAI(api_key=api_key)

        prompt = (
            f"Context: {context}\nSentence: {sentence}\n"
            "Is the sentence supported by the context above?\nAnswer Yes or No:"
        )
        for attempt in range(self.max_retries):
            try:
                res = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                return res.choices[0].message.content.strip()
            except RateLimitError:
                time.sleep(self.retry_wait * (2**attempt))
        raise RuntimeError("OpenAI API request failed after retries")

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


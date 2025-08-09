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
        # Per-sentence disagreement scores and answerability statistics
        self.last_disagreement: List[float] = []
        # ``last_answerability`` stores per-question answerability ratios for
        # every sentence.  ``avg_answerability`` summarizes them globally.  For
        # backward compatibility we also expose ``last_unanswerable`` and
        # ``avg_unanswerable`` which are derived from the answerability values.
        self.last_answerability: List[List[float]] = []
        self.last_unanswerable: List[float] = []
        self.avg_answerability: float = 0.0
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

    def predict(
        self,
        sentences: Iterable[str],
        samples: Iterable[str],
        *,
        scoring_method: str = "counting",
        beta1: float = 0.1,
        beta2: float = 0.5,
        answerability_threshold: float = 0.5,
    ) -> tuple[List[float], List[List[float]]]:
        """Score ``sentences`` against ``samples``.

        Parameters
        ----------
        sentences, samples:
            Main passage sentences and sampled passages.
        scoring_method: {"counting", "bayes", "bayes_with_alpha"}
            Strategy used to convert matches into inconsistency scores.
        beta1, beta2:
            Beta hyper-parameters for Bayesian scoring.
        answerability_threshold: float
            Minimum answerability score for a sample to be considered in the
            counting and vanilla Bayes methods.
        """

        sentences = list(sentences)
        samples = list(samples)
        total = len(samples) or 1
        assert scoring_method in {"counting", "bayes", "bayes_with_alpha"}

        # Prepare question generation and answering helpers
        if getattr(self, "qg_pipe", None) is not None:
            def _gen_questions(text: str) -> List[str]:
                outputs = self.qg_pipe(
                    text,
                    num_return_sequences=self.num_questions,
                    num_beams=self.num_questions,
                )
                return [o["generated_text"].strip() for o in outputs]

            def _answer(q: str, ctx: str) -> str:
                return (
                    self.qa_pipe({"question": q, "context": ctx})
                    .get("answer", "")
                    .strip()
                    .lower()
                )
        else:
            def _gen_questions(text: str) -> List[str]:
                q_res = self.qg_fn(text)
                if isinstance(q_res, str):
                    return [q_res.strip()]
                return [q.strip() for q in q_res]

            def _answer(q: str, ctx: str) -> str:
                return self.qa_fn(q, ctx).strip().lower()

        all_questions: List[List[str]] = [_gen_questions(s) for s in sentences]
        all_ref_answers: List[List[str]] = [
            [_answer(q, s) for q in qs] for qs, s in zip(all_questions, sentences)
        ]

        sent_scores: List[float] = []
        answerability_stats: List[List[float]] = []
        for qs, refs in zip(all_questions, all_ref_answers):
            q_scores: List[float] = []
            q_ans_stats: List[float] = []
            for q, ref in zip(qs, refs):
                ref_ans_score = 1.0 if ref else 0.0
                matches = 0
                mismatches = 0
                soft_match = 0.0
                soft_mismatch = 0.0
                answerable = 0
                for sample in samples:
                    ans = _answer(q, sample)
                    ans_score = 1.0 if ans else 0.0
                    if ans_score >= answerability_threshold:
                        answerable += 1
                        if ans == ref and ref:
                            matches += 1
                        else:
                            mismatches += 1
                    if ans == ref and ref:
                        soft_match += ans_score
                    else:
                        soft_mismatch += ans_score
                q_ans_stats.append(answerable / total)

                if scoring_method == "counting":
                    if ref_ans_score < answerability_threshold:
                        score = 0.5
                    elif answerable == 0:
                        score = 0.5
                    else:
                        score = (answerable - matches) / answerable
                elif scoring_method == "bayes":
                    if ref_ans_score < answerability_threshold:
                        score = 0.5
                    else:
                        gamma1 = beta2 / (1.0 - beta1)
                        gamma2 = beta1 / (1.0 - beta2)
                        score = (gamma2 ** mismatches) / (
                            (gamma1 ** matches) + (gamma2 ** mismatches)
                        )
                else:  # bayes_with_alpha
                    gamma1 = beta2 / (1.0 - beta1)
                    gamma2 = beta1 / (1.0 - beta2)
                    score = (gamma2 ** soft_mismatch) / (
                        (gamma1 ** soft_match) + (gamma2 ** soft_mismatch)
                    )
                q_scores.append(score)

            sent_scores.append(sum(q_scores) / len(q_scores))
            answerability_stats.append(q_ans_stats)

        self.last_disagreement = sent_scores
        self.last_answerability = answerability_stats

        # Per-sentence and global statistics derived from the per-question data
        self.last_unanswerable = [
            1 - (sum(qs) / len(qs) if qs else 0.0) for qs in answerability_stats
        ]
        if answerability_stats:
            total_q = sum(len(qs) for qs in answerability_stats)
            self.avg_answerability = sum(sum(qs) for qs in answerability_stats) / total_q
        else:
            self.avg_answerability = 0.0
        self.avg_unanswerable = (
            sum(self.last_unanswerable) / len(self.last_unanswerable)
            if self.last_unanswerable
            else 0.0
        )
        self.avg_disagreement = (
            sum(sent_scores) / len(sent_scores) if sent_scores else 0.0
        )

        return sent_scores, answerability_stats


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
    def predict(
        self,
        sentences: Iterable[str],
        samples: Iterable[str],
        *,
        n: int | None = None,
    ) -> dict[str, list[float] | float]:
        """Estimate n-gram probabilities for ``sentences``.

        Parameters
        ----------
        sentences:
            Sentences from the main passage to score.
        samples:
            Sampled passages used to build the n-gram model.
        n: int, optional
            Override the n-gram order specified at construction time.

        Returns
        -------
        dict
            A dictionary containing per-sentence average and maximum negative
            log probabilities as well as document level aggregates:

            ``{
                'sentence_scores': [...],
                'sentence_max_scores': [...],
                'avg_neg_logprob': float,
                'avg_max_neg_logprob': float
            }``
        """

        order = max(1, n or self.n)
        self_n_backup = self.n
        self.n = order
        counts, followers, histories, total_tokens, vocab_size, total_continuations = self._build_model(samples)
        self.n = self_n_backup

        sent_avgs: List[float] = []
        sent_maxes: List[float] = []

        for sent in sentences:
            tokens = self._tokenize(sent)
            if len(tokens) < order:
                grams = [tuple(tokens)] if tokens else []
            else:
                grams = [tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1)]

            neg_logs: List[float] = []
            for gram in grams:
                if self.smoothing == "kneser_ney":
                    prob = self._prob_kneser_ney(
                        gram, counts, followers, histories, total_continuations, vocab_size
                    )
                else:
                    prob = self._prob_backoff(gram, counts, total_tokens, vocab_size)
                neg_logs.append(-math.log(max(prob, 1e-12)))

            if neg_logs:
                sent_avgs.append(sum(neg_logs) / len(neg_logs))
                sent_maxes.append(max(neg_logs))
            else:
                sent_avgs.append(0.0)
                sent_maxes.append(0.0)

        avg_neg = sum(sent_avgs) / len(sent_avgs) if sent_avgs else 0.0
        avg_max_neg = sum(sent_maxes) / len(sent_maxes) if sent_maxes else 0.0

        return {
            "sentence_scores": sent_avgs,
            "sentence_max_scores": sent_maxes,
            "avg_neg_logprob": avg_neg,
            "avg_max_neg_logprob": avg_max_neg,
        }


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

    ``SelfCheckPrompt`` also supports running a local HuggingFace
    ``transformers`` model instead of the OpenAI API and allows
    customizing both the question template and the mapping from raw model
    outputs to numerical scores.
    """

    _DEFAULT_TEMPLATE = (
        "Context: {context}\nSentence: {sentence}\n"
        "Is the sentence supported by the context above?\nAnswer Yes or No:"
    )

    def __init__(
        self,
        ask_fn: Callable[[str, str], str] | None = None,
        model: str = "gpt-3.5-turbo",
        max_retries: int = 3,
        retry_wait: float = 1.0,
        *,
        prompt_template: str | None = None,
        map_fn: Callable[[str], float] | None = None,
        hf_model: str | None = None,
        hf_device: int | str | None = None,
        hf_max_new_tokens: int = 16,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self.prompt_template = prompt_template or self._DEFAULT_TEMPLATE
        self.map_fn = map_fn or self._default_map
        self._client = None
        self._hf_pipe = None
        self._hf_max_new_tokens = hf_max_new_tokens

        if ask_fn is not None:
            self._raw_ask = ask_fn
        elif hf_model is not None:
            try:  # pragma: no cover - optional dependency
                from transformers import pipeline  # type: ignore
                import torch  # type: ignore

                def _resolve(dev):
                    if dev is not None:
                        return dev
                    return 0 if torch.cuda.is_available() else -1

                self._hf_pipe = pipeline(
                    "text-generation",
                    model=hf_model,
                    device=_resolve(hf_device),
                )
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("transformers models unavailable") from exc
            self._raw_ask = self._hf_ask
        else:
            self._raw_ask = self._openai_ask

        self._cache: dict[tuple[str, str], str] = {}

        def cached_ask(context: str, sentence: str) -> str:
            key = (context, sentence)
            if key not in self._cache:
                self._cache[key] = self._raw_ask(context, sentence)
            return self._cache[key]

        self.ask_fn = cached_ask

    # -- configuration -------------------------------------------------------
    def set_prompt_template(self, template: str) -> None:
        """Set a new prompt ``template`` for Yes/No questions."""

        self.prompt_template = template

    # -- backends ------------------------------------------------------------
    def _openai_ask(
        self, context: str, sentence: str
    ) -> str:  # pragma: no cover - requires network
        import os
        import time
        from openai import OpenAI, RateLimitError

        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            self._client = OpenAI(api_key=api_key)

        prompt = self.prompt_template.format(context=context, sentence=sentence)
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

    def _hf_ask(self, context: str, sentence: str) -> str:
        prompt = self.prompt_template.format(context=context, sentence=sentence)
        assert self._hf_pipe is not None  # for mypy
        res = self._hf_pipe(
            prompt,
            max_new_tokens=self._hf_max_new_tokens,
            return_full_text=False,
        )
        return res[0]["generated_text"].strip()

    # -- default mapping -----------------------------------------------------
    @staticmethod
    def _default_map(ans: str) -> float:
        ans = ans.strip().lower()
        if ans.startswith("y"):
            return 0.0
        if ans.startswith("n"):
            return 1.0
        return 0.5

    # -- prediction ----------------------------------------------------------
    def predict(self, sentences: Iterable[str], samples: Iterable[str]) -> List[float]:
        samples = list(samples)
        scores: List[float] = []
        for sent in sentences:
            total = 0.0
            for sample in samples:
                ans = self.ask_fn(sample, sent)
                total += self.map_fn(ans)
            scores.append(total / max(1, len(samples)))
        return scores


__all__ = [
    "SelfCheckBERTScore",
    "SelfCheckMQAG",
    "SelfCheckNgram",
    "SelfCheckNLI",
    "SelfCheckPrompt",
]


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

from typing import Callable, Iterable, List, Sequence
import collections
import math
import os
import re
import string
from pathlib import Path
from selfcheckgpt.utils import (
    MQAGConfig,
    prepare_answering_input,
    prepare_distractor_input,
    prepare_qa_input,
)


def get_prob_distances(
    p_ref: Sequence[float], p_other: Sequence[float]
) -> dict[str, float]:
    """Return distance measures between two probability vectors.

    Parameters
    ----------
    p_ref, p_other:
        Reference and comparison probability distributions.  They are
        automatically normalised and a small epsilon is added to avoid
        numerical issues with zeros.

    Returns
    -------
    dict
        Mapping with the following keys: ``"kl"`` for Kullback-Leibler
        divergence, ``"counting"`` which measures ``1 - P_ref"s option in
        the other distribution, ``"hellinger"`` and ``"total_variation"``.
    """

    import numpy as np

    ref = np.array(p_ref, dtype=float)
    other = np.array(p_other, dtype=float)
    if ref.sum() == 0:
        ref = np.ones_like(ref) / len(ref)
    if other.sum() == 0:
        other = np.ones_like(other) / len(other)
    ref = ref / ref.sum()
    other = other / other.sum()
    eps = 1e-8

    kl = float(np.sum(ref * np.log((ref + eps) / (other + eps))))
    counting = 1.0 - float(other[np.argmax(ref)])
    hellinger = float(
        np.sqrt(0.5 * np.sum((np.sqrt(ref) - np.sqrt(other)) ** 2))
    )
    total_variation = float(0.5 * np.sum(np.abs(ref - other)))
    return {
        "kl": kl,
        "counting": counting,
        "hellinger": hellinger,
        "total_variation": total_variation,
    }


def find_optimal_temperature(
    logits: Iterable[Sequence[float]], labels: Iterable[int]
) -> float:
    """Compute the temperature that minimizes NLL on validation logits.

    Parameters
    ----------
    logits:
        Iterable of raw logits for each validation example.
    labels:
        Iterable of gold label indices corresponding to the logits.

    Returns
    -------
    float
        Optimal temperature scalar.
    """

    try:  # pragma: no cover - optional dependency
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for temperature calibration") from exc

    logits_t = torch.tensor(list(logits), dtype=torch.float)
    labels_t = torch.tensor(list(labels), dtype=torch.long)

    temperature = torch.ones(1, requires_grad=True, dtype=torch.float)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    criterion = torch.nn.CrossEntropyLoss()

    def _eval():
        optimizer.zero_grad()
        loss = criterion(logits_t / temperature, labels_t)
        loss.backward()
        return loss

    optimizer.step(_eval)
    return float(temperature.item())


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
    individually answered on the samples.  Scores and answerability
    ratios are averaged over these questions and exposed via
    ``last_disagreement`` and ``last_answerability``.  The attributes
    ``avg_disagreement`` and ``avg_answerability`` summarize these
    statistics over all sentences.

    Parameters
    ----------
    qg_fn, qa_fn: callable, optional
        Custom question generation / answering functions.  When omitted
        the class loads real ``transformers`` models specified via the
        ``g1_model``, ``g2_model``, ``qa_model`` and ``answer_model``
        parameters.
    batch_size: int, optional
        Number of samples processed together by the generation models.
    num_questions: int, optional
        How many questions to sample per sentence (default 3).
    g1_model, g2_model, qa_model, answer_model: str, optional
        HuggingFace model identifiers for the first and second question
        generator, the multiple‑choice answerer and the answerability
        classifier.
    device: int | str | None, optional
        Device string or CUDA device id used for all models.
    """

    def __init__(
        self,
        qg_fn: Callable[[str], Iterable[dict]] | None = None,
        qa_fn: Callable[[str, Sequence[str], str], Sequence[float]] | None = None,
        batch_size: int = 8,
        num_questions: int = 3,
        g1_model: str | None = None,
        g2_model: str | None = None,
        qa_model: str | None = None,
        answer_model: str | None = None,
        device: int | str | None = None,
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
                from transformers import (
                    AutoModelForSeq2SeqLM,
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                    LongformerForMultipleChoice,
                    LongformerTokenizer,
                )
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("transformers models unavailable") from exc

            try:  # pragma: no cover - optional dependency
                import torch  # type: ignore
            except Exception:  # pragma: no cover - optional dependency
                torch = None  # type: ignore

            def _resolve(dev):
                if dev is not None:
                    return dev
                if torch is not None and getattr(torch.cuda, "is_available", lambda: False)():
                    return "cuda"
                return "cpu"

            self.device = _resolve(device)
            self._torch = torch

            g1_model = g1_model or MQAGConfig.generation1_squad
            g2_model = g2_model or MQAGConfig.generation2
            qa_model = qa_model or MQAGConfig.answering
            answer_model = answer_model or MQAGConfig.answerable

            self.g1_tokenizer = AutoTokenizer.from_pretrained(g1_model)
            self.g1_model = AutoModelForSeq2SeqLM.from_pretrained(g1_model)
            self.g2_tokenizer = AutoTokenizer.from_pretrained(g2_model)
            self.g2_model = AutoModelForSeq2SeqLM.from_pretrained(g2_model)
            self.a_tokenizer = LongformerTokenizer.from_pretrained(qa_model)
            self.a_model = LongformerForMultipleChoice.from_pretrained(qa_model)
            self.ans_tokenizer = LongformerTokenizer.from_pretrained(answer_model)
            self.ans_model = AutoModelForSequenceClassification.from_pretrained(
                answer_model
            )

            # Move to device and set eval mode
            for model in (self.g1_model, self.g2_model, self.a_model, self.ans_model):
                model.to(self.device)
                model.eval()
        else:
            self.g1_model = self.g2_model = self.a_model = self.ans_model = None
            self.g1_tokenizer = self.g2_tokenizer = self.a_tokenizer = self.ans_tokenizer = None
            self.device = device

    @staticmethod
    def _normalize(text: str) -> list[str]:
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        return text.split()

    @staticmethod
    def _f1(pred: str, ref: str) -> float:
        pred_tokens = SelfCheckMQAG._normalize(pred)
        ref_tokens = SelfCheckMQAG._normalize(ref)
        if not pred_tokens or not ref_tokens:
            return 0.0
        common = collections.Counter(pred_tokens) & collections.Counter(ref_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    def predict(
        self,
        sentences: Iterable[str],
        samples: Iterable[str],
        *,
        metric: str = "counting",
        disagreement_threshold: float = 0.5,
        answerability_threshold: float = 0.5,
    ) -> tuple[List[float], List[List[float]]]:
        """Score ``sentences`` against ``samples`` using probability distances.

        ``metric`` selects which distance from :func:`get_prob_distances` is
        used for aggregating disagreement.  The method also records
        per-question answerability ratios which mirror the original API.
        """

        sentences = list(sentences)
        samples = list(samples)
        total = len(samples) or 1

        assert metric in {"kl", "counting", "hellinger", "total_variation"}

        if getattr(self, "g1_model", None) is not None:
            torch = self._torch

            def _gen_questions(text: str) -> List[dict]:
                qa_input_ids = prepare_qa_input(
                    self.g1_tokenizer, context=text, device=self.device
                )
                questions: List[dict] = []
                for _ in range(self.num_questions):
                    outputs = self.g1_model.generate(
                        qa_input_ids, max_new_tokens=128, do_sample=True
                    )
                    qa_text = self.g1_tokenizer.decode(
                        outputs[0], skip_special_tokens=False
                    )
                    qa_text = qa_text.replace(
                        self.g1_tokenizer.pad_token or "", ""
                    ).replace(self.g1_tokenizer.eos_token or "", "")
                    qa_split = [x.strip() for x in qa_text.split(self.g1_tokenizer.sep_token)]
                    if len(qa_split) != 2:
                        continue
                    question, answer = qa_split
                    distractor_ids = prepare_distractor_input(
                        self.g2_tokenizer,
                        context=text,
                        question=question,
                        answer=answer,
                        device=self.device,
                        separator=self.g2_tokenizer.sep_token,
                    )
                    outputs = self.g2_model.generate(
                        distractor_ids, max_new_tokens=128, do_sample=True
                    )
                    distractors = self.g2_tokenizer.decode(
                        outputs[0], skip_special_tokens=False
                    )
                    distractors = distractors.replace(
                        self.g2_tokenizer.pad_token or "", ""
                    ).replace(self.g2_tokenizer.eos_token or "", "")
                    distractors = re.sub(
                        "<extra\\S+>", self.g2_tokenizer.sep_token, distractors
                    )
                    opts = [y.strip() for y in distractors.split(self.g2_tokenizer.sep_token)]
                    options = [answer] + opts
                    while len(options) < 4:
                        options.append(options[-1] if options else "")
                    questions.append({"question": question, "options": options})
                return questions

            def _answer_probs(q: dict, ctx: str) -> Sequence[float]:
                if torch is None:  # pragma: no cover - optional dependency
                    raise RuntimeError("PyTorch is required for MQAG")
                encoded = prepare_answering_input(
                    self.a_tokenizer,
                    q["question"],
                    q["options"],
                    ctx,
                    device=self.device,
                )
                logits = self.a_model(**encoded).logits[0]
                probs = torch.softmax(logits, dim=-1)
                return probs.tolist()

            def _answerable_prob(q: dict, ctx: str, _probs: Sequence[float] | None = None) -> float:
                if torch is None:  # pragma: no cover - optional dependency
                    raise RuntimeError("PyTorch is required for MQAG")
                tokenized = self.ans_tokenizer(
                    q["question"],
                    ctx,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                )
                if self.device is not None:
                    tokenized = tokenized.to(self.device)
                logits = self.ans_model(**tokenized).logits[0]
                probs = torch.softmax(logits, dim=-1)
                return float(probs[1]) if probs.numel() >= 2 else 0.0

        else:
            def _gen_questions(text: str) -> List[dict]:
                q_res = self.qg_fn(text)
                return list(q_res)

            def _answer_probs(q: dict, ctx: str) -> Sequence[float]:
                return self.qa_fn(q["question"], q["options"], ctx)

            def _answerable_prob(
                q: dict, ctx: str, probs: Sequence[float] | None = None
            ) -> float:
                if probs is None:
                    probs = _answer_probs(q, ctx)
                return max(probs) if probs else 0.0

        all_questions = [_gen_questions(s) for s in sentences]

        # reference distributions from the original sentences
        ref_dists = [
            [_answer_probs(q, sent) for q in qs] for qs, sent in zip(all_questions, sentences)
        ]

        sent_scores: List[float] = []
        answerability_stats: List[List[float]] = []
        for qs, refs in zip(all_questions, ref_dists):
            q_scores: List[float] = []
            q_ans_stats: List[float] = []
            for q, ref_prob in zip(qs, refs):
                disagreements = 0.0
                considered = 0
                ans_scores: List[float] = []
                for sample in samples:
                    probs = _answer_probs(q, sample)
                    ans_score = _answerable_prob(q, sample, probs)
                    ans_scores.append(ans_score)
                    if ans_score >= answerability_threshold:
                        considered += 1
                        distances = get_prob_distances(ref_prob, probs)
                        if distances[metric] > disagreement_threshold:
                            disagreements += 1
                q_ans_stats.append(sum(ans_scores) / total)
                if considered == 0:
                    q_scores.append(0.5)
                else:
                    q_scores.append(disagreements / considered)
            sent_scores.append(sum(q_scores) / len(q_scores) if q_scores else 0.0)
            answerability_stats.append(q_ans_stats)

        self.last_disagreement = sent_scores
        self.last_answerability = answerability_stats

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
    corpus: str | os.PathLike | Iterable[str], optional
        Reference corpus used to pre-build n-gram counts.  If provided the
        counts are combined with sample based statistics on every call to
        :meth:`predict`.
    """

    def __init__(
        self,
        n: int = 1,
        smoothing: str = "backoff",
        discount: float = 0.75,
        corpus: str | os.PathLike | Iterable[str] | None = None,
    ) -> None:
        self.n = max(1, n)
        self.smoothing = smoothing
        self.discount = discount
        self._corpus_texts: list[str] | None = None
        self._corpus_models: dict[int, tuple] = {}
        if corpus is not None:
            if isinstance(corpus, (str, os.PathLike, Path)):
                text = Path(corpus).read_text(encoding="utf8")
                self._corpus_texts = [text]
            else:
                self._corpus_texts = list(corpus)
            # Pre-build model for default n
            self._corpus_models[self.n] = self._build_model(self._corpus_texts)

    # -- Utility -------------------------------------------------------------
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()

    def _ensure_corpus_model(self, order: int):
        if self._corpus_texts is None:
            return None
        if order not in self._corpus_models:
            self_n_backup = self.n
            self.n = order
            self._corpus_models[order] = self._build_model(self._corpus_texts)
            self.n = self_n_backup
        return self._corpus_models[order]

    @staticmethod
    def _combine_models(model_a, model_b, order: int):
        counts_a, followers_a, histories_a, total_a, _, _ = model_a
        counts_b, followers_b, histories_b, total_b, _, _ = model_b
        counts = [counts_a[i] + counts_b[i] for i in range(order)]
        followers: list[collections.defaultdict] = []
        for i in range(max(0, order - 1)):
            merged = collections.defaultdict(set)
            for d in (followers_a[i], followers_b[i]):
                for key, vals in d.items():
                    merged[key].update(vals)
            followers.append(merged)
        histories = collections.defaultdict(set)
        for d in (histories_a, histories_b):
            for key, vals in d.items():
                histories[key].update(vals)
        total_tokens = total_a + total_b
        vocab_size = len(counts[0]) or 1
        total_continuations = sum(len(v) for v in histories.values()) or 1
        return counts, followers, histories, total_tokens, vocab_size, total_continuations

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
        sample_model = self._build_model(samples)
        self.n = self_n_backup

        corpus_model = self._ensure_corpus_model(order)
        if corpus_model is not None:
            (
                counts,
                followers,
                histories,
                total_tokens,
                vocab_size,
                total_continuations,
            ) = self._combine_models(corpus_model, sample_model, order)
        else:
            (
                counts,
                followers,
                histories,
                total_tokens,
                vocab_size,
                total_continuations,
            ) = sample_model

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
    ``microsoft/deberta-v3-large-mnli`` model to obtain Natural Language
    Inference probabilities.  A different model name can be supplied.  For
    each pair of ``(sample, sentence)`` it computes the probability of
    *contradiction* and uses this value directly as the inconsistency score.
    Higher scores therefore indicate that the sentence is not supported by
    the sampled passages.

    A custom ``nli_fn`` can be supplied for testing which should accept a
    premise and hypothesis and return raw logits for the NLI labels.  The
    logits are expected to follow the usual MNLI convention where index ``0``
    corresponds to ``contradiction`` and the last index denotes
    ``entailment``.
    """

    def __init__(
        self,
        model: str = "microsoft/deberta-v3-large-mnli",
        nli_fn: Callable[[str, str], Sequence[float]] | None = None,
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

                def _hf_logits(premise: str, hypothesis: str) -> List[float]:
                    inputs = self.tokenizer(
                        premise,
                        hypothesis,
                        return_tensors="pt",
                        truncation=True,
                    ).to(self.device)
                    with torch.no_grad():
                        logits = self.model(**inputs).logits[0]
                    return logits.tolist()

                self.nli_fn = _hf_logits
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("transformers NLI model unavailable") from exc
        else:
            self.nli_fn = nli_fn
            self.device = device

    def predict(
        self,
        sentences: Iterable[str],
        samples: Iterable[str],
        *,
        return_logits: bool = False,
    ) -> List[float] | tuple[List[float], List[List[List[float]]]]:
        sentences = list(sentences)
        samples = list(samples)
        total = len(samples) or 1
        scores: List[float] = []
        all_logits: List[List[List[float]]] = []
        for sent in sentences:
            agg = 0.0
            sent_logits: List[List[float]] = []
            for sample in samples:
                raw_logits = self.nli_fn(sample, sent)
                import torch  # type: ignore

                logits_t = torch.tensor(raw_logits, dtype=torch.float)
                if self.temperature != 1.0:
                    logits_t = logits_t / self.temperature
                probs = torch.softmax(logits_t, dim=-1).tolist()
                if len(probs) >= 1:
                    p_contra = probs[0]
                else:  # pragma: no cover - edge case
                    p_contra = 0.0
                agg += p_contra
                if return_logits:
                    sent_logits.append(list(raw_logits))
            scores.append(agg / total)
            if return_logits:
                all_logits.append(sent_logits)
        if return_logits:
            return scores, all_logits
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
        normalize: bool = True,
        temperature: float = 1.0,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self.prompt_template = prompt_template or self._DEFAULT_TEMPLATE
        self.map_fn = map_fn or self._default_map
        self.normalize = normalize
        self.temperature = temperature
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

    # -- normalisation ------------------------------------------------------
    @staticmethod
    def _normalise(ans: str) -> str:
        ans = ans.strip().lower()
        ans = ans.translate(str.maketrans("", "", string.punctuation))
        mapping = {
            "y": "yes",
            "yes": "yes",
            "yeah": "yes",
            "yep": "yes",
            "affirmative": "yes",
            "sure": "yes",
            "true": "yes",
            "correct": "yes",
            "n": "no",
            "no": "no",
            "nope": "no",
            "nah": "no",
            "negative": "no",
            "false": "no",
            "incorrect": "no",
        }
        return mapping.get(ans, ans)

    # -- prediction ----------------------------------------------------------
    def predict(
        self,
        sentences: Iterable[str],
        samples: Iterable[str],
        *,
        return_probs: bool = False,
    ) -> List[float] | tuple[List[float], List[List[float]]]:
        samples = list(samples)
        scores: List[float] = []
        all_probs: List[List[float]] = []
        for sent in sentences:
            total = 0.0
            sent_probs: List[float] = []
            for sample in samples:
                ans = self.ask_fn(sample, sent)
                if self.normalize:
                    ans = self._normalise(ans)
                prob = self.map_fn(ans)
                if self.temperature != 1.0:
                    p = min(max(prob, 1e-8), 1 - 1e-8)
                    logit = math.log(p / (1 - p))
                    p = 1 / (1 + math.exp(-logit / self.temperature))
                    prob = p
                total += prob
                if return_probs:
                    sent_probs.append(prob)
            avg = total / max(1, len(samples))
            scores.append(avg)
            if return_probs:
                all_probs.append(sent_probs)
        if return_probs:
            return scores, all_probs
        return scores


__all__ = [
    "SelfCheckBERTScore",
    "SelfCheckMQAG",
    "SelfCheckNgram",
    "SelfCheckNLI",
    "SelfCheckPrompt",
]


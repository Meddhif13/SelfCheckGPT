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

import re
import numpy as np
import torch
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import LongformerTokenizer, LongformerForMultipleChoice

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
    _HF,
    _HUB,
)


# Small helper to remain compatible with stubbed transformers in tests
def _from_pretrained_compat(cls, model_name: str, **kwargs):
    """Call cls.from_pretrained trying local_files_only when supported.

    Test stubs often don't accept the ``local_files_only`` kwarg. We first try
    passing it, and if a TypeError arises we retry without the kwarg.
    """
    try:
        return cls.from_pretrained(model_name, local_files_only=True, **kwargs)
    except TypeError:
        # Stubs don't accept this kwarg
        return cls.from_pretrained(model_name, **kwargs)


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

            self._fallback = False
            self._fb_tokenizer = None
            self._fb_model = None
            self._device = None

            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                self.scorer = BERTScorer(
                    lang="en",
                    rescale_with_baseline=baseline,
                    model_type=model,
                    device=device,
                )
            except Exception:
                # If baseline rescaling files are missing offline, retry without baseline.
                import logging as _logging
                _logging.warning(
                    "BERTScore baseline unavailable offline; proceeding without baseline rescaling"
                )
                self.scorer = BERTScorer(
                    lang="en",
                    rescale_with_baseline=False,
                    model_type=model,
                    device=device,
                )
        except Exception:
            # transformers-based cosine-similarity fallback using local model
            try:
                import torch  # type: ignore
                from transformers import AutoTokenizer, AutoModel  # type: ignore

                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._fb_tokenizer = _from_pretrained_compat(AutoTokenizer, model)
                self._fb_model = _from_pretrained_compat(AutoModel, model).to(self._device)
                self._fb_model.eval()
                self._fallback = True
                self.scorer = None  # type: ignore
            except Exception as exc2:  # pragma: no cover - optional dependency
                raise RuntimeError("BERTScore is unavailable") from exc2

    def predict(self, sentences: Iterable[str], samples: Iterable[str]) -> List[float]:
        samples = list(samples)
        scores: List[float] = []
        if not getattr(self, "_fallback", False):
            for sent in sentences:
                gaps: List[float] = []
                for sample in samples:
                    P, R, F = self.scorer.score([sent], [sample])  # type: ignore[attr-defined]
                    gaps.append(1 - F.mean().item())
                scores.append(float(sum(gaps) / len(gaps)) if gaps else 0.0)
            return scores

        # Fallback path: cosine distance between mean pooled embeddings
        import torch  # type: ignore
        assert self._fb_model is not None and self._fb_tokenizer is not None and self._device is not None

        with torch.no_grad():
            sample_embs: List[torch.Tensor] = []
            for s in samples:
                toks = self._fb_tokenizer(
                    s,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                ).to(self._device)
                out = self._fb_model(**toks).last_hidden_state  # (1, seq, hidden)
                emb = out.mean(dim=1).squeeze(0)  # (hidden)
                sample_embs.append(emb)

            for sent in sentences:
                toks = self._fb_tokenizer(
                    sent,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                ).to(self._device)
                out = self._fb_model(**toks).last_hidden_state
                sent_emb = out.mean(dim=1).squeeze(0)
                gaps: List[float] = []
                for emb in sample_embs:
                    # cosine similarity
                    num = torch.dot(sent_emb, emb)
                    denom = (sent_emb.norm(p=2) * emb.norm(p=2) + 1e-8)
                    cos = (num / denom).clamp(-1.0, 1.0).item()
                    gaps.append(1.0 - float(max(0.0, cos)))
                scores.append(float(sum(gaps) / len(gaps)) if gaps else 0.0)
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
        generator, the multipleâ€‘choice answerer and the answerability
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
                import torch
                from transformers import (
                    AutoModelForSeq2SeqLM,
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                    LongformerForMultipleChoice,
                    LongformerTokenizer,
                )
                
                # Set to local paths and print for debugging
                if g1_model is None:
                    g1_model = str(_HF / "lmqg__flan-t5-base-squad-qg")
                if g2_model is None:
                    g2_model = str(_HF / "potsawee__t5-large-generation-race-Distractor")
                if qa_model is None:
                    qa_model = str(_HF / "potsawee__longformer-large-4096-answering-race")
                if answer_model is None:
                    # Prefer HF repo-style default for tests; callers can override with local path
                    answer_model = "potsawee/longformer-large-4096-answerable-squad2"
                
                print("\nDEBUG - Model paths being used:")
                print(f"g1_model: {g1_model}")
                print(f"g2_model: {g2_model}")
                print(f"qa_model: {qa_model}")
                print(f"answer_model: {answer_model}")
                print(f"HF_HOME: {os.environ.get('HF_HOME')}")
                print(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE')}")
                
                self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
                
                # Check if model files exist before loading
                for model_path in [g1_model, g2_model, qa_model, answer_model]:
                    print(f"\nChecking files in {model_path}:")
                    if os.path.exists(model_path):
                        files = os.listdir(model_path)
                        print(f"Found files: {files}")
                    else:
                        print(f"Directory does not exist!")
                
                # Load all models
                print("\nAttempting to load g1_model...")
                self.g1_tokenizer = _from_pretrained_compat(AutoTokenizer, g1_model)
                self.g1_model = _from_pretrained_compat(AutoModelForSeq2SeqLM, g1_model)
                self.g1_model.to(self.device)
                self.g1_model.eval()
                
                self.g2_tokenizer = _from_pretrained_compat(AutoTokenizer, g2_model)
                self.g2_model = _from_pretrained_compat(AutoModelForSeq2SeqLM, g2_model)
                self.g2_model.to(self.device)
                self.g2_model.eval()
                
                self.qa_tokenizer = _from_pretrained_compat(LongformerTokenizer, qa_model)
                self.qa_model = _from_pretrained_compat(LongformerForMultipleChoice, qa_model)
                self.qa_model.to(self.device)
                self.qa_model.eval()
                
                self.answer_tokenizer = _from_pretrained_compat(LongformerTokenizer, answer_model)
                self.answer_model = _from_pretrained_compat(AutoModelForSequenceClassification, answer_model)
                self.answer_model.to(self.device)
                self.answer_model.eval()
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
                """Generate questions and answers with distractors following MQAG paper."""
                from selfcheckgpt.mqag_utils import prepare_qa_input, prepare_distractor_input
                
                # Initialize containers
                questions: List[dict] = []
                num_valid_questions = 0
                max_tries = int(self.num_questions * 1.5)  # Allow some retries for invalid outputs

                # Stage G.1: Question + Answer Generation following original implementation
                qa_input_ids = prepare_qa_input(
                    self.g1_tokenizer,
                    context=text,
                    device=self.device
                )
                
                for _ in range(max_tries):
                    gen_kwargs = {}
                    for _k in ("pad_token_id", "eos_token_id"):
                        _v = getattr(self.g1_tokenizer, _k, None)
                        if isinstance(_v, int):
                            gen_kwargs[_k] = _v
                    outputs = self.g1_model.generate(
                        qa_input_ids,
                        max_new_tokens=128,   # Following original implementation
                        do_sample=True,       # Enable sampling as per original
                        **gen_kwargs,
                    )
                    qa_text = self.g1_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                    print(f"\nGenerated text: {qa_text}")
                    
                                    # Decode and clean QA text
                    qa_text = self.g1_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    qa_text = qa_text.strip()
                    print(f"\nGenerated text: {qa_text}")
                    
                    # If the model generated just a question, we can use it
                    if qa_text.strip().endswith("?"):
                        question = qa_text.strip()
                        # Try to extract answer from the input context
                        # Remove question words and punctuation to get key terms
                        search_terms = question.lower()
                        for word in ["what", "who", "where", "when", "how", "why", "did", "was", "were", "is", "are", "?", "the"]:
                            search_terms = search_terms.replace(word, "").strip()
                            
                        # Find the most relevant part of the context
                        words = text.split()
                        best_match = None
                        max_matches = 0
                        
                        # Slide through context with a window to find best matching segment
                        window_size = 8  # Adjust as needed
                        for i in range(len(words) - window_size + 1):
                            window = " ".join(words[i:i + window_size])
                            matches = sum(1 for term in search_terms.split() if term in window.lower())
                            if matches > max_matches:
                                max_matches = matches
                                best_match = window
                        
                        # If we found a reasonable match, use it as the answer
                        if best_match and max_matches >= 1:
                            answer = best_match.strip()
                            print(f"Extracted answer from context: {answer}")
                        else:
                            print("Could not extract answer from context, skipping...")
                            continue
                    else:
                        model: str = "gpt-5-preview",
                        question = None
                        answer = None
                        
                        # Strategy 1: Look for "question:" and "answer:" markers
                        if "question:" in qa_text.lower() and "answer:" in qa_text.lower():
                            try:
                                parts = qa_text.lower().split("answer:")
                                if len(parts) >= 2:
                                    q_parts = parts[0].split("question:")
                                    if len(q_parts) >= 2:
                                        question = q_parts[1].strip()
                                        answer = parts[1].strip()
                            except:
                                pass
                                
                        # Strategy 2: Split on separator token if it exists
                        if (not question or not answer) and hasattr(self.g1_tokenizer, 'sep_token'):
                            try:
                                parts = qa_text.split(self.g1_tokenizer.sep_token)
                                if len(parts) == 2:
                                    question = parts[0].strip()
                                    answer = parts[1].strip()
                            except:
                                pass
                                
                        if not question or not answer:
                            print("Could not parse explicit QA pair, skipping...")
                            continue
                        
                    print(f"Extracted Question: {question}")
                    print(f"Extracted Answer: {answer}")
                    
                    # Remove any remaining special tokens
                    for token in [
                        getattr(self.g1_tokenizer, "pad_token", None),
                        getattr(self.g1_tokenizer, "eos_token", None),
                        getattr(self.g1_tokenizer, "bos_token", None),
                        "<sep>",
                    ]:
                        if token:
                            question = question.replace(token, "").strip()
                            answer = answer.replace(token, "").strip()
                    
                    # Basic validation
                    if not question or not answer or len(question) < 5 or len(answer) < 2:
                        print("Question or answer too short, skipping...")
                        continue
                        
                    print(f"Parsed Question: {question}")
                    print(f"Parsed Answer: {answer}")
                    
                    # Stage G.2: Distractor Generation
                    # Format distractor input following T5 format
                    distractor_input = f"context: {text} question: {question} answer: {answer}"
                    distractor_inputs = self.g2_tokenizer(
                        distractor_input,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                        padding=True
                    ).to(self.device)
                    
                    gen2_kwargs = {}
                    for _k in ("pad_token_id", "eos_token_id"):
                        _v = getattr(self.g2_tokenizer, _k, None)
                        if isinstance(_v, int):
                            gen2_kwargs[_k] = _v
                    distractor_outputs = self.g2_model.generate(
                        distractor_inputs.input_ids,
                        max_new_tokens=128,   # Longer outputs for multiple distractors
                        num_beams=3,          # Use beam search for better quality
                        do_sample=True,       # Enable sampling for diversity
                        temperature=0.8,      # Higher temperature for creative distractors
                        top_p=0.95,          # High top_p for quality
                        no_repeat_ngram_size=2,  # Avoid repetition
                        length_penalty=1.0,   # Don't penalize length
                        num_return_sequences=3,  # Generate multiple distractors at once
                        **gen2_kwargs,
                    )
                    
                    # Process all generated distractors
                    distractors = []
                    for output in distractor_outputs:
                        text = self.g2_tokenizer.decode(output, skip_special_tokens=False)
                        text = text.replace(self.g2_tokenizer.pad_token or "", "").replace(self.g2_tokenizer.eos_token or "", "")
                        
                        # Handle different separator formats
                        if "<sep>" in text:
                            parts = text.split("<sep>")
                        else:
                            parts = text.split(self.g2_tokenizer.sep_token)
                            
                        # Add valid parts as distractors
                        distractors.extend(p.strip() for p in parts if p.strip() and p.strip() != answer)
                    
                    # Ensure we have exactly 3 distractors
                    while len(distractors) < 3:
                        distractors.append(distractors[-1] if distractors else answer)
                    distractors = distractors[:3]  # Limit to 3 distractors
                    
                    options = [answer] + distractors  # Correct answer is always first
                    answer = None
                    
                    # Try to identify question and answer parts
                    if "?" in qa_text:
                        # Split on first question mark
                        parts = qa_text.split("?", 1)
                        question = parts[0].strip() + "?"
                        
                        # Look for answer indicators in the remaining text
                        remaining = parts[1].strip()
                        if remaining:
                            # Remove any follow-up questions from the answer
                            answer_part = remaining.split("?")[0].strip()
                            # Clean up the answer - remove common prefixes and extra punctuation
                            answer_part = re.sub(r'^[,\s]*', '', answer_part)
                            answer_part = re.sub(r'^(answer|a):\s*', '', answer_part, flags=re.IGNORECASE)
                            if answer_part:
                                answer = answer_part
                            else:
                                # Extract answer from context based on question
                                answer = f"According to the text: {text}"
                    
                    question_item = {
                        'question': question,
                        'options': options
                    }
                    questions.append(question_item)
                    
                    # Print for debugging
                    print(f"Question: {question}")
                    print("Options:")
                    for i, opt in enumerate(options):
                        print(f"  {i+1}. {opt}")
                    print("--------------------")
                    
                    num_valid_questions += 1
                    if num_valid_questions >= self.num_questions:
                        break
                        
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
                questions = list(q_res)
                # If no questions were generated, return a default question to avoid division by zero
                if not questions:
                    return [{"question": "What is this text about?", "options": ["Unknown", "Not specified", "Cannot determine", "No information"]}]
                return questions

            def _answer_probs(q: dict, ctx: str) -> Sequence[float]:
                return self.qa_fn(q["question"], q["options"], ctx)

            def _answerable_prob(
                q: dict, ctx: str, probs: Sequence[float] | None = None
            ) -> float:
                if probs is None:
                    probs = _answer_probs(q, ctx)
                return max(probs) if probs else 0.0

        # Debug output for question generation
        all_questions = []
        for i, s in enumerate(sentences):
            questions = _gen_questions(s)
            print(f"Generated {len(questions)} questions for sentence {i + 1}")
            all_questions.append(questions)

        # reference distributions from the original sentences 
        ref_dists = []
        for i, (qs, sent) in enumerate(zip(all_questions, sentences)):
            dists = []
            for q in qs:
                dist = _answer_probs(q, sent)
                dists.append(dist)
            print(f"Got {len(dists)} answer distributions for sentence {i + 1}")
            ref_dists.append(dists)

        sent_scores: List[float] = []
        answerability_stats: List[List[float]] = []
        for qs, refs in zip(all_questions, ref_dists):
            q_scores: List[float] = []
            q_ans_stats: List[float] = []
            for q, ref_prob in zip(qs, refs):
                disagreements = 0.0
                considered = 0
                ans_scores: List[float] = []
                print(f"\nProcessing question: {q['question']}")
                for i, sample in enumerate(samples):
                    probs = _answer_probs(q, sample)
                    ans_score = _answerable_prob(q, sample, probs)
                    print(f"Sample {i + 1} answerability score: {ans_score}")
                    ans_scores.append(ans_score)
                    if ans_score >= answerability_threshold:
                        considered += 1
                        distances = get_prob_distances(ref_prob, probs)
                        if distances[metric] > disagreement_threshold:
                            disagreements += 1
                print(f"Total considered samples: {considered}")
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
    ``microsoft/deberta-large-mnli`` model to obtain Natural Language
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
        model: str = None,  # Will set default below
        nli_fn: Callable[[str, str], Sequence[float]] | None = None,
        device: str | None = None,
        temperature: float = 1.0,
        *,
        batch_size: int = 16,
        max_length: int = 256,
    ) -> None:
        self.temperature = temperature
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(32, int(max_length))
        if nli_fn is None:
            try:  # pragma: no cover - heavy branch
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )  # type: ignore
                
                # Set default model path
                if model is None:
                    model = str(_HF / "microsoft__deberta-large-mnli")
                import torch  # type: ignore

                self.device = torch.device(
                    device or ("cuda" if torch.cuda.is_available() else "cpu")
                )
                self.tokenizer = _from_pretrained_compat(AutoTokenizer, model)
                self.model = _from_pretrained_compat(AutoModelForSequenceClassification, model)
                self.model.to(self.device)
                self.model.eval()
                self._use_hf = True
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("transformers NLI model unavailable") from exc
        else:
            self.nli_fn = nli_fn
            self.device = device
            self._use_hf = False

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
        import torch  # type: ignore

        for sent in sentences:
            agg = 0.0
            sent_logits: List[List[float]] = []

            if getattr(self, "_use_hf", False):
                # Batch over samples for this sentence
                for i in range(0, len(samples), self.batch_size):
                    batch_samples = samples[i : i + self.batch_size]
                    try:
                        inputs = self.tokenizer(
                            batch_samples,
                            [sent] * len(batch_samples),
                            return_tensors="pt",
                            truncation=True,
                            max_length=self.max_length,
                            padding=True,
                        )
                        if hasattr(inputs, "to"):
                            inputs = inputs.to(self.device)
                        with torch.no_grad():
                            logits = self.model(**inputs).logits
                        if self.temperature != 1.0:
                            logits = logits / self.temperature
                        probs = torch.softmax(logits, dim=-1)
                        agg += float(probs[:, 0].sum().item())
                        if return_logits:
                            for row in logits.tolist():
                                sent_logits.append(list(row))
                    except TypeError:
                        # Stubs path: call model with explicit premise/hypothesis
                        for prem in batch_samples:
                            logits_t = self.model(premise=prem, hypothesis=sent).logits
                            if self.temperature != 1.0:
                                logits_t = logits_t / self.temperature
                            probs = torch.softmax(logits_t, dim=-1)
                            agg += float(probs[0, 0].item())
                            if return_logits:
                                sent_logits.append(list(logits_t.squeeze(0).tolist()))
            else:
                # Pure function path for tests
                for sample in samples:
                    raw_logits = self.nli_fn(sample, sent)
                    logits_t = torch.tensor(raw_logits, dtype=torch.float)
                    if self.temperature != 1.0:
                        logits_t = logits_t / self.temperature
                    probs = torch.softmax(logits_t, dim=-1)
                    agg += float(probs[0].item())
                    if return_logits:
                        sent_logits.append(list(logits_t.squeeze(0).tolist()))

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
    model: str = "gpt-5-preview",
        max_retries: int = 3,
        retry_wait: float = 1.0,
        *,
        prompt_template: str | None = None,
        map_fn: Callable[[str], float] | None = None,
    hf_model: str | None = None,
    hf_task: str | None = None,
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

                def _resolve_device(dev: int | str | None):
                    if isinstance(dev, int):
                        return dev
                    if isinstance(dev, str):
                        # allow 'cpu', 'cuda', or CUDA index as string
                        if dev.lower() == "cpu":
                            return -1
                        if dev.lower().startswith("cuda"):
                            parts = dev.split(":")
                            if len(parts) == 2 and parts[1].isdigit():
                                return int(parts[1])
                            return 0 if torch.cuda.is_available() else -1
                    return 0 if torch.cuda.is_available() else -1

                # choose task automatically if not provided
                task = hf_task
                if task is None:
                    # Simple heuristic: FLAN/T5 -> text2text-generation, otherwise text-generation
                    low = hf_model.lower()
                    if any(x in low for x in ("t5", "flan")):
                        task = "text2text-generation"
                    else:
                        task = "text-generation"

                self._hf_pipe = pipeline(
                    task,
                    model=hf_model,
                    device=_resolve_device(hf_device),
                )
                self._hf_task = task
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
        from pathlib import Path
        # Import defensively to support test stubs that don't expose APIError
        import importlib
        openai = importlib.import_module("openai")
        OpenAI = getattr(openai, "OpenAI")
        RateLimitError = getattr(openai, "RateLimitError", Exception)
        APIError = getattr(openai, "APIError", Exception)

        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                key_file = os.getenv("OPENAI_API_KEY_FILE")
                if key_file and Path(key_file).exists():
                    api_key = Path(key_file).read_text(encoding="utf-8").strip()
                else:
                    repo_root = Path(__file__).resolve().parents[1]
                    candidate_files = [
                        repo_root / ".secrets" / "openai.key",
                        repo_root / ".env",
                    ]
                    for fp in candidate_files:
                        if fp.exists():
                            text = fp.read_text(encoding="utf-8")
                            if "OPENAI_API_KEY=" in text:
                                for line in text.splitlines():
                                    if line.strip().startswith("OPENAI_API_KEY="):
                                        api_key = (
                                            line.split("=", 1)[1].strip().strip('"').strip("'")
                                        )
                                        break
                            else:
                                api_key = text.strip()
                            if api_key:
                                break
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY not found (set env var, OPENAI_API_KEY_FILE, .secrets/openai.key or .env)"
                )
            try:
                self._client = OpenAI(api_key=api_key, timeout=60.0)
            except TypeError:
                # Some test stubs don't accept timeout
                self._client = OpenAI(api_key=api_key)

        prompt = self.prompt_template.format(context=context, sentence=sentence)
        # primary attempts with backoff and jitter
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                res = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                return res.choices[0].message.content.strip()
            except (RateLimitError, APIError, Exception) as e:
                last_exc = e
                delay = self.retry_wait * (2**attempt) + 0.05 * (attempt + 1)
                time.sleep(min(delay, 10.0))
        # fallbacks if preview model is flaky
        for fb_model in ("gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"):
            try:
                res = self._client.chat.completions.create(
                    model=fb_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                import logging as _logging
                _logging.warning(
                    "OpenAI model '%s' failed; fell back to '%s'", self.model, fb_model
                )
                return res.choices[0].message.content.strip()
            except Exception as e:
                last_exc = e
                continue
        raise RuntimeError("OpenAI API request failed after retries") from last_exc

    def _hf_ask(self, context: str, sentence: str) -> str:
        prompt = self.prompt_template.format(context=context, sentence=sentence)
        assert self._hf_pipe is not None  # for mypy
        kwargs = {"max_new_tokens": self._hf_max_new_tokens}
        # Avoid return_full_text for text2text-generation
        task = getattr(self, "_hf_task", None)
        if task != "text2text-generation":
            kwargs["return_full_text"] = False
        res = self._hf_pipe(prompt, **kwargs)
        out = res[0]
        text = (
            out.get("generated_text")
            if isinstance(out, dict)
            else getattr(out, "generated_text", "")
        )
        if not text and isinstance(out, dict):
            # Some pipelines use different keys
            text = out.get("summary_text") or out.get("text") or ""
        return str(text).strip()

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



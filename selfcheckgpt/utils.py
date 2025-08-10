"""Minimal stubs for the original :mod:`selfcheckgpt` utilities.

This project only relies on a tiny subset of the real library.  The
functions defined here provide the interfaces expected by
``selfcheck_metrics`` so that the tests can run without the heavy
dependencies of the original project.  Only a handful of convenience
wrappers around HuggingFace tokenizers are included; they implement the
minimal behaviour required by the tests.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MQAGConfig:
    """Model names used by the MQAG metric.

    These defaults mirror the identifiers cited in the original
    SelfCheckGPT paper and point to publicly available HuggingFace
    checkpoints.  They are used when no explicit model names are provided
    to :class:`selfcheck_metrics.SelfCheckMQAG`.
    """

    # First and second question generation models
    generation1_squad: str = "potsawee/t5-base-squad-qg"
    generation2: str = "potsawee/t5-base-distractor-generation"
    # Multiple-choice answerer
    answering: str = "potsawee/longformer-large-4096-mc-squad2"
    # Answerability classifier
    answerable: str = "potsawee/longformer-large-4096-answerable-squad2"


# ---------------------------------------------------------------------------
# Tokenisation helpers mirroring the original project
# ---------------------------------------------------------------------------

def prepare_qa_input(t5_tokenizer, *, context: str, device: str | int | None = None):
    """Tokenise ``context`` for question generation.

    Parameters
    ----------
    t5_tokenizer:
        HuggingFace tokenizer used by the QG model.
    context:
        Source passage from which questions are generated.
    device:
        Device identifier understood by :meth:`torch.Tensor.to`.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, seq_len)`` containing token ids for the model.
    """

    encoding = t5_tokenizer([context], return_tensors="pt")
    input_ids = encoding.input_ids
    if device is not None:
        input_ids = input_ids.to(device)
    return input_ids


def prepare_distractor_input(
    t5_tokenizer,
    *,
    context: str,
    question: str,
    answer: str,
    device: str | int | None = None,
    separator: str = "<sep>",
):
    """Tokenise the question/answer pair for distractor generation.

    Returns a tensor of shape ``(1, seq_len)`` suitable for the second
    question generation model which predicts distractors.
    """

    input_text = f"{question} {separator} {answer} {separator} {context}"
    encoding = t5_tokenizer([input_text], return_tensors="pt")
    input_ids = encoding.input_ids
    if device is not None:
        input_ids = input_ids.to(device)
    return input_ids


def prepare_answering_input(
    tokenizer,
    question: str,
    options: list[str],
    context: str,
    *,
    device: str | int | None = None,
    max_seq_length: int = 4096,
):
    """Tokenise multiple-choice QA inputs for ``LongformerForMultipleChoice``.

    Returns a mapping with ``input_ids`` and ``attention_mask`` tensors of
    shape ``(1, num_options, seq_len)``.
    """

    c_plus_q = context + " " + tokenizer.bos_token + " " + question
    repeated_context = [c_plus_q] * len(options)
    tokenized = tokenizer(
        repeated_context,
        options,
        max_length=max_seq_length,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    if device is not None:
        tokenized = tokenized.to(device)

    input_ids = tokenized["input_ids"].unsqueeze(0)
    attention_mask = tokenized["attention_mask"].unsqueeze(0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


"""Utility functions for SelfCheckGPT metrics."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import torch
import numpy as np

# Resolve repo root and hf-cache folder
_ROOT = Path(__file__).resolve().parents[1]
_HF   = _ROOT / "hf-cache"
_HUB  = _HF / "hub"

# Set HuggingFace cache directory and offline mode
os.environ["TRANSFORMERS_CACHE"] = str(_HF)
os.environ["HF_HOME"] = str(_HF)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def _abs(p: Path) -> str:
    """Convert path to absolute with forward slashes"""
    return str(p.resolve()).replace("\\", "/")

    # First and second question generation models
    generation1_squad: str = _QG_DEFAULT
    generation2: str       = _DIS_DEFAULT

    # Multiple-choice answerer
    answering:  str = _abs(_HF / "potsawee__longformer-large-4096-answering-race")
    # Answerability classifier
    answerable: str = _abs(_HF / "potsawee__longformer-large-4096-answerable-squad2")

    def __post_init__(self):
        # If caller passed None/empty, repair to defaults
        if not self.generation1_squad:
            self.generation1_squad = _QG_DEFAULT
        if not self.generation2:
            self.generation2 = _DIS_DEFAULT

# ---------------------------------------------------------------------------
# Tokenisation helpers mirroring the original project
# ---------------------------------------------------------------------------

@dataclass
class MQAGConfig:
    """Model names used by the MQAG metric."""
    # Question generation models
    generation1_squad: str = _abs(_HF / "lmqg__flan-t5-base-squad-qg")
    generation1_race: str = _abs(_HF / "lmqg__flan-t5-base-race")
    generation2: str = _abs(_HF / "potsawee__t5-large-generation-race-Distractor")
    # QA models
    answering: str = _abs(_HF / "potsawee__longformer-large-4096-answering-race")
    answerability: str = _abs(_HF / "potsawee__longformer-large-4096-answerable-squad2")

@dataclass
class NLIConfig:
    """Model name for NLI."""
    nli_model: str = _abs(_HF / "microsoft__deberta-large-mnli")

def prepare_qa_input(t5_tokenizer, context: str, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    """Prepare input for question generation."""
    encoding = t5_tokenizer([context], return_tensors="pt")
    input_ids = encoding.input_ids
    if device is not None:
        input_ids = input_ids.to(device)
    return input_ids

# Import MQAG-specific utilities
from .mqag_utils import prepare_distractor_input, prepare_answering_input

# Utility functions
def expand_list1(mylist: List[Any], num: int) -> List[Any]:
    """Expand a list by repeating each element num times."""
    expanded = []
    for x in mylist:
        for _ in range(num):
            expanded.append(x)
    return expanded

def expand_list2(mylist: List[Any], num: int) -> List[Any]:
    """Expand a list by repeating the whole list num times."""
    expanded = []
    for _ in range(num):
        expanded.extend(mylist)
    return expanded


# MQAG utility functions have been moved to mqag_utils.py
from .mqag_utils import prepare_distractor_input, prepare_answering_input


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







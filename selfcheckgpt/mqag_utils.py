"""MQAG utility functions based on the original implementation."""

import torch
import numpy as np
from typing import List, Dict, Any

def prepare_qa_input(t5_tokenizer, context, device):
    """
    Prepare input for T5 question-answer generation.
    Uses format: "generate question and answer: <context>"
    Expected output: "question: <question> answer: <answer>"
    """
    # Normalize and clean context
    context = context.strip()
    
    # Add instruction prefix to match T5 fine-tuning format
    input_text = f"generate question and answer: {context}"
    
    # Tokenize with proper truncation and padding
    encoding = t5_tokenizer(
        [input_text],
        return_tensors="pt",
        max_length=512,  # T5 typical input length
        truncation=True,
        padding=True,
    )
    
    # Move to target device
    input_ids = encoding.input_ids.to(device)
    return input_ids

def prepare_distractor_input(t5_tokenizer, context, question, answer, device, separator='<sep>'):
    """
    input: question <sep> answer <sep> article
    output: distractor1 <sep> distractor2 <sep> distractor3
    """
    input_text = question + ' ' + separator + ' ' + answer + ' ' + separator + ' ' + context
    encoding = t5_tokenizer(
        [input_text],
        return_tensors="pt",
    )
    input_ids = encoding.input_ids.to(device)
    return input_ids

def prepare_answering_input(tokenizer, question, options, context, device, max_seq_length=4096):
    """Prepare input for multiple-choice QA."""
    c_plus_q = context + ' ' + tokenizer.bos_token + ' ' + question
    c_plus_q_4 = [c_plus_q] * len(options)

    tokenized_examples = tokenizer(
        c_plus_q_4, options,
        max_length=max_seq_length,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    tokenized_examples = tokenized_examples.to(device)
    input_ids = tokenized_examples['input_ids'].unsqueeze(0)
    attention_mask = tokenized_examples['attention_mask'].unsqueeze(0)

    example_encoded = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    return example_encoded

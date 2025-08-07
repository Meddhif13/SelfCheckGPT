"""Minimal evaluation script for the simplified SelfCheckGPT metrics.

This script is intentionally lightweight.  It loads a small portion of
`potsawee/wiki_bio_gpt3_hallucination` and evaluates the simplified
``SelfCheckNgram`` metric.  The goal is to demonstrate how the metrics
may be invoked rather than to reproduce the exact numbers from the paper.
"""

import logging
from typing import List

from datasets import load_dataset
from sklearn.metrics import average_precision_score

from selfcheck_metrics import SelfCheckNgram

logging.basicConfig(level=logging.INFO)


def load_annotations(example) -> List[int]:
    """Convert annotation strings to binary non-factual labels."""
    labels = []
    for ann in example["annotation"]:
        labels.append(0 if ann == "accurate" else 1)
    return labels


def main() -> None:
    logging.info("Loading dataset ...")
    ds = load_dataset("potsawee/wiki_bio_gpt3_hallucination", split="test[:5]")

    logging.info("Evaluating SelfCheckNgram on a handful of samples ...")
    metric = SelfCheckNgram()
    all_scores: List[float] = []
    all_labels: List[int] = []
    for example in ds:
        sentences = example["gpt3_sentences"]
        samples = example["gpt3_text_samples"]
        scores = metric.predict(sentences, samples)
        labels = load_annotations(example)
        all_scores.extend(scores)
        all_labels.extend(labels)

    ap = average_precision_score(all_labels, all_scores)
    logging.info("Average precision: %.3f", ap)


if __name__ == "__main__":
    main()

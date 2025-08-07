"""Run simple experiments with the toy SelfCheckGPT metrics.

The original :mod:`selfcheckgpt` project evaluates several different
hallucination detectors.  The earlier version of this repository only
demonstrated the n‑gram variant.  This script offers a slightly more
flexible evaluation loop that can score multiple metrics on a slice of
the ``potsawee/wiki_bio_gpt3_hallucination`` dataset.  The
implementations are still intentionally lightweight but the structure of
the experiment is closer to the setup described in the paper.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Callable, Dict, Iterable, List

from datasets import load_dataset
from sklearn.metrics import average_precision_score

from selfcheck_metrics import (
    SelfCheckBERTScore,
    SelfCheckMQAG,
    SelfCheckNLI,
    SelfCheckNgram,
    SelfCheckPrompt,
)

logging.basicConfig(level=logging.INFO)


def load_annotations(example) -> List[int]:
    """Convert annotation strings to binary non‑factual labels."""

    labels: List[int] = []
    for ann in example["annotation"]:
        labels.append(0 if ann == "accurate" else 1)
    return labels


# ---------------------------------------------------------------------------
# Metric factory ------------------------------------------------------------
# ---------------------------------------------------------------------------

_PROMPT_CLIENT = SelfCheckPrompt(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    rate_limit=float(os.getenv("OPENAI_RATE_LIMIT", "1.0")),
)


def _prompt_heuristic(context: str, sentence: str) -> str:
    """Proxy that delegates to the real OpenAI prompt call."""

    return _PROMPT_CLIENT._openai_ask(context, sentence)


MetricFactory = Dict[str, Callable[[], object]]

METRICS: MetricFactory = {
    "bertscore": lambda: SelfCheckBERTScore(use_bert_score=False),
    "mqag": SelfCheckMQAG,
    "ngram": SelfCheckNgram,
    "nli": SelfCheckNLI,
    "prompt": lambda: SelfCheckPrompt(ask_fn=_prompt_heuristic),
}


def evaluate(metric, dataset: Iterable[dict]) -> float:
    """Return average precision of ``metric`` on ``dataset``."""

    all_scores: List[float] = []
    all_labels: List[int] = []
    for example in dataset:
        sentences = example["gpt3_sentences"]
        samples = example["gpt3_text_samples"]
        scores = metric.predict(sentences, samples)
        labels = load_annotations(example)
        all_scores.extend(scores)
        all_labels.extend(labels)
    return average_precision_score(all_labels, all_scores)


def main() -> None:  # pragma: no cover - exercised via CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["ngram"],
        help="Which metrics to evaluate (or 'all').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of test examples to load from the dataset.",
    )
    args = parser.parse_args()

    metric_names = list(METRICS) if "all" in args.metrics else args.metrics

    logging.info("Loading dataset slice of %d examples ...", args.limit)
    ds = load_dataset(
        "potsawee/wiki_bio_gpt3_hallucination", split=f"test[:{args.limit}]"
    )

    for name in metric_names:
        if name not in METRICS:
            logging.warning("Unknown metric '%s' -- skipping", name)
            continue
        metric = METRICS[name]()
        ap = evaluate(metric, ds)
        logging.info("%s average precision: %.3f", name, ap)


if __name__ == "__main__":
    main()

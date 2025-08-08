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

def _prompt_heuristic(context: str, sentence: str) -> str:
    """Very small stand‑in for the real LLM prompt call.

    The function simply checks whether the final token of ``sentence``
    appears in ``context`` and returns "Yes" or "No" accordingly.  This
    keeps the example runnable without external API access.
    """

    token = sentence.split()[-1].strip(". ,") if sentence.split() else ""
    return "Yes" if token and token in context else "No"


MetricFactory = Dict[str, Callable[[], object]]

METRICS: MetricFactory = {
    "bertscore": SelfCheckBERTScore,
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
    parser.add_argument(
        "--ngram-n",
        type=int,
        default=1,
        help="Order of the n-gram model used by the 'ngram' metric.",
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
        if name == "ngram":
            metric = SelfCheckNgram(n=args.ngram_n)
        else:
            metric = METRICS[name]()
        ap = evaluate(metric, ds)
        logging.info("%s average precision: %.3f", name, ap)


if __name__ == "__main__":
    main()

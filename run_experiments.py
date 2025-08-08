"""Run SelfCheckGPT metrics over the WikiBio hallucination dataset.

This script loads one or more splits of the WikiBio hallucination dataset,
optionally regenerates model samples with the lightweight sampling pipeline and
then scores a selection of SelfCheckGPT metrics.  For every metric we compute
the usual classification statistics (precision, recall, F1, average precision
and Brier score) and also derive calibration curves.  Results are written to a
CSV file and precision/recall and calibration plots are stored alongside it.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, List
import math

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from data.utils import load_wikibio_hallucination
from sampling.generator import generate_samples, OpenAIChatLLM
from selfcheck_metrics import (
    SelfCheckBERTScore,
    SelfCheckMQAG,
    SelfCheckNLI,
    SelfCheckNgram,
    SelfCheckPrompt,
)

logging.basicConfig(level=logging.INFO)


def load_annotations(example: dict) -> List[int]:
    """Convert annotation strings to binary non-factual labels."""

    labels: List[int] = []
    for ann in example["annotation"]:
        labels.append(0 if ann == "accurate" else 1)
    return labels


# ---------------------------------------------------------------------------
# Metric factory ------------------------------------------------------------
# ---------------------------------------------------------------------------


MetricFactory = Dict[str, Callable[[], object]]

METRICS: MetricFactory = {
    "bertscore": SelfCheckBERTScore,
    "mqag": SelfCheckMQAG,
    "ngram": SelfCheckNgram,
    "nli": SelfCheckNLI,
    "prompt": SelfCheckPrompt,
}


# ---------------------------------------------------------------------------
# Evaluation ----------------------------------------------------------------
# ---------------------------------------------------------------------------


def evaluate(metric, dataset: Iterable[dict], *, bins: int = 10) -> dict:
    """Return a dictionary with scoring results for ``metric`` on ``dataset``."""

    all_scores: List[float] = []
    all_labels: List[int] = []
    for example in dataset:
        sentences = example["gpt3_sentences"]
        samples = example["gpt3_text_samples"]
        scores = metric.predict(sentences, samples)
        labels = load_annotations(example)
        # Ensure scores fall into [0, 1]. Some metrics (e.g. ngram) return
        # unbounded values which we squash using ``1 - exp(-score)``.
        for s in scores:
            if s < 0 or s > 1:
                all_scores.append(1 - math.exp(-max(0.0, s)))
            else:
                all_scores.append(s)
        all_labels.extend(labels)

    precision_curve, recall_curve, thresholds = precision_recall_curve(
        all_labels, all_scores
    )
    ap = average_precision_score(all_labels, all_scores)

    preds = [1 if s >= 0.5 else 0 for s in all_scores]
    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)
    brier = brier_score_loss(all_labels, all_scores)

    bins = max(1, min(bins, len(all_labels)))
    prob_true, prob_pred = calibration_curve(
        all_labels, all_scores, n_bins=bins, strategy="quantile"
    )

    return {
        "average_precision": ap,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "brier": brier,
        "pr_curve": (recall_curve, precision_curve),
        "pr_thresholds": thresholds,
        "calibration": (prob_pred, prob_true),
    }


def _save_plots(name: str, stats: dict, out_dir: Path) -> None:
    """Persist precision/recall and calibration plots for ``name``."""

    recall_curve, precision_curve = stats["pr_curve"]
    plt.figure()
    plt.plot(recall_curve, precision_curve)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{name} Precision-Recall")
    plt.savefig(out_dir / f"{name}_pr.png")
    plt.close()

    prob_pred, prob_true = stats["calibration"]
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Mean predicted value")
    plt.ylabel("Fraction of positives")
    plt.title(f"{name} Calibration")
    plt.savefig(out_dir / f"{name}_calibration.png")
    plt.close()


# ---------------------------------------------------------------------------
# CLI -----------------------------------------------------------------------
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover - exercised via CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["ngram"],
        help="Which metrics to evaluate (or 'all').",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        default=["test"],
        help=(
            "Dataset split(s) or slices to load, e.g. 'test[:100]'."
            " Use 'all' to evaluate train, validation and test."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of examples after loading the split.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory where CSV files and plots will be written.",
    )
    parser.add_argument(
        "--ngram-n",
        type=int,
        default=1,
        help="Order of the n-gram model used by the 'ngram' metric.",
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        help="Regenerate samples using the simple sampling pipeline.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=1,
        help="Number of samples per prompt to use or generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the LLM.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-3.5-turbo",
        help="OpenAI model for sampling and prompt-based metric.",
    )
    parser.add_argument(
        "--calib-bins",
        type=int,
        default=10,
        help="Number of bins for the calibration curve.",
    )
    args = parser.parse_args()

    metric_names = list(METRICS) if "all" in args.metrics else args.metrics

    llm: OpenAIChatLLM | None = None
    if args.resample or "prompt" in metric_names:
        llm = OpenAIChatLLM(model=args.llm_model)

    split_names = args.split
    if "all" in split_names:
        split_names = ["train", "validation", "test"]

    base_out = Path(args.output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    for split in split_names:
        logging.info("Loading dataset split '%s' ...", split)
        ds = load_wikibio_hallucination(split=split)

        examples = list(ds)
        if args.limit is not None:
            examples = examples[: args.limit]

        if args.resample or not all("gpt3_text_samples" in ex for ex in examples):
            logging.info("Generating samples with LLM ...")
            prompts = [" ".join(ex["gpt3_sentences"]) for ex in examples]
            out_dir = base_out / split if len(split_names) > 1 else base_out
            out_dir.mkdir(parents=True, exist_ok=True)
            sample_file = out_dir / "samples.jsonl"
            generate_samples(
                llm,
                prompts,
                sample_file,
                num_samples=args.sample_count,
                temperature=args.temperature,
            )
            prompt_to_samples: dict[str, list[str]] = collections.defaultdict(list)
            with sample_file.open("r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    prompt_to_samples[obj["prompt"]].append(obj["sample"])
            for ex, prompt in zip(examples, prompts):
                ex["gpt3_text_samples"] = prompt_to_samples[prompt]
        else:
            for ex in examples:
                ex["gpt3_text_samples"] = ex.get("gpt3_text_samples", [])[: args.sample_count]

        out_dir = base_out / split if len(split_names) > 1 else base_out
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_rows: List[dict[str, float | str]] = []
        for name in metric_names:
            if name not in METRICS:
                logging.warning("Unknown metric '%s' -- skipping", name)
                continue
            try:
                if name == "ngram":
                    metric = SelfCheckNgram(n=args.ngram_n)
                elif name == "prompt":
                    metric = SelfCheckPrompt(ask_fn=llm.ask_yes_no if llm else None)
                else:
                    metric = METRICS[name]()
                stats = evaluate(metric, examples, bins=args.calib_bins)
            except Exception as exc:  # pragma: no cover - optional dependencies
                logging.warning("Metric %s failed: %s", name, exc)
                continue

            _save_plots(name, stats, out_dir)
            summary_rows.append(
                {
                    "metric": name,
                    "average_precision": stats["average_precision"],
                    "precision": stats["precision"],
                    "recall": stats["recall"],
                    "f1": stats["f1"],
                    "brier": stats["brier"],
                }
            )

        if summary_rows:
            summary_path = out_dir / "summary.csv"
            with summary_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
                writer.writeheader()
                writer.writerows(summary_rows)
            logging.info("Wrote results to %s", summary_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()


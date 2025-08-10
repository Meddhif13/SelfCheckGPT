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
from sklearn.model_selection import StratifiedKFold

from data.utils import load_wikibio_hallucination
from sampling.generator import generate_samples, OpenAIChatLLM
from selfcheck_metrics import (
    SelfCheckBERTScore,
    SelfCheckMQAG,
    SelfCheckNLI,
    SelfCheckNgram,
    SelfCheckPrompt,
    find_optimal_temperature,
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


def _compute_stats(scores: List[float], labels: List[int], *, bins: int = 10) -> dict:
    """Compute evaluation statistics for ``scores`` against ``labels``."""

    precision_curve, recall_curve, thresholds = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)

    preds = [1 if s >= 0.5 else 0 for s in scores]
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    brier = brier_score_loss(labels, scores)

    bins = max(1, min(bins, len(labels)))
    prob_true, prob_pred = calibration_curve(
        labels, scores, n_bins=bins, strategy="quantile"
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


def evaluate(
    metric, dataset: Iterable[dict], *, bins: int = 10, return_scores: bool = False
) -> dict | tuple[dict, List[float], List[int]]:
    """Return scoring results and optionally raw scores and labels."""

    all_scores: List[float] = []
    all_labels: List[int] = []
    for example in dataset:
        sentences = example["gpt3_sentences"]
        samples = example["gpt3_text_samples"]
        scores = metric.predict(sentences, samples)
        if isinstance(scores, tuple):  # MQAG returns (scores, answerability)
            scores, _ = scores
        elif isinstance(scores, dict):  # ngram returns detailed stats
            scores = scores.get("sentence_scores", [])
        labels = load_annotations(example)
        # Ensure scores fall into [0, 1]. Some metrics (e.g. ngram) return
        # unbounded values which we squash using ``1 - exp(-score)``.
        for s in scores:
            if s < 0 or s > 1:
                all_scores.append(1 - math.exp(-max(0.0, s)))
            else:
                all_scores.append(s)
        all_labels.extend(labels)

    stats = _compute_stats(all_scores, all_labels, bins=bins)
    if return_scores:
        return stats, all_scores, all_labels
    return stats


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
        "--train-split",
        type=str,
        default="train",
        help="Dataset split or slice for training the combiner.",
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default="validation",
        help="Optional dataset split or slice for validation/hyperparameters.",
    )
    parser.add_argument(
        "--test-split",
        type=str,
        default="test",
        help="Dataset split or slice for final evaluation.",
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
        default=20,
        help="Number of samples per prompt to use or generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the LLM (used if --temperatures is not set).",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=None,
        help="Optional list of temperatures to sweep over.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling cutoff.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p nucleus sampling cutoff.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic sampling (temperature=0).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching LLM generations.",
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
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified folds for combiner cross-validation (set to 1 to disable).",
    )
    parser.add_argument(
        "--paper-config",
        action="store_true",
        help=(
            "Use configuration matching the SelfCheckGPT paper (20 samples, top-k=50, top-p=0.95, resampling, 5-fold CV)."
        ),
    )
    args = parser.parse_args()

    if args.paper_config:
        args.sample_count = 20
        if args.top_k is None:
            args.top_k = 50
        if args.top_p is None:
            args.top_p = 0.95
        args.resample = True
        args.cv_folds = max(args.cv_folds, 5)

    if (args.top_k is not None or args.top_p is not None) and not args.resample:
        logging.info(
            "top-k/top-p provided without --resample; enabling resampling to match settings"
        )
        args.resample = True

    metric_names = list(METRICS) if "all" in args.metrics else args.metrics

    llm: OpenAIChatLLM | None = None
    if args.resample or "prompt" in metric_names:
        llm = OpenAIChatLLM(model=args.llm_model)

    base_out = Path(args.output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    temps = args.temperatures if args.temperatures is not None else [args.temperature]

    for temp in temps:
        logging.info(
            "Running configuration: temperature=%s top_k=%s top_p=%s deterministic=%s",
            temp,
            args.top_k,
            args.top_p,
            args.deterministic,
        )
        temp_dir = base_out / f"temp_{str(temp).replace('.', '_')}" if len(temps) > 1 else base_out
        temp_dir.mkdir(parents=True, exist_ok=True)

        def prepare_split(examples: List[dict], name: str) -> List[dict]:
            if args.limit is not None:
                examples = examples[: args.limit]
            if args.resample or not all("gpt3_text_samples" in ex for ex in examples):
                logging.info("Generating samples with LLM for %s split...", name)
                prompts = [" ".join(ex["gpt3_sentences"]) for ex in examples]
                split_dir = temp_dir / name
                split_dir.mkdir(parents=True, exist_ok=True)
                sample_file = split_dir / "samples.jsonl"
                generate_samples(
                    llm,
                    prompts,
                    sample_file,
                    num_samples=args.sample_count,
                    temperature=temp,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    deterministic=args.deterministic,
                    cache_dir=args.cache_dir,
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
            return examples

        # Load and prepare datasets
        logging.info("Loading dataset split '%s' for training ...", args.train_split)
        train_ds = load_wikibio_hallucination(split=args.train_split)
        train_examples = prepare_split(list(train_ds), "train")

        val_examples: List[dict] = []
        if args.val_split:
            logging.info("Loading dataset split '%s' for validation ...", args.val_split)
            val_ds = load_wikibio_hallucination(split=args.val_split)
            val_examples = prepare_split(list(val_ds), "val")

        logging.info("Loading dataset split '%s' for testing ...", args.test_split)
        test_ds = load_wikibio_hallucination(split=args.test_split)
        test_examples = prepare_split(list(test_ds), "test")

        nli_temperature = 1.0
        prompt_temperature = 1.0
        if "nli" in metric_names and val_examples:
            try:
                metric_cal = SelfCheckNLI()
                calib_logits: list[list[float]] = []
                calib_labels: list[int] = []
                for ex in val_examples:
                    _, per_sent_logits = metric_cal.predict(
                        ex["gpt3_sentences"],
                        ex["gpt3_text_samples"],
                        return_logits=True,
                    )
                    labels = load_annotations(ex)
                    for lbl, sent_logits in zip(labels, per_sent_logits):
                        for logit in sent_logits:
                            calib_logits.append(logit)
                            calib_labels.append(lbl)
                nli_temperature = find_optimal_temperature(calib_logits, calib_labels)
            except Exception as exc:  # pragma: no cover - optional dependency
                logging.warning("NLI temperature calibration failed: %s", exc)

        if "prompt" in metric_names and val_examples:
            try:
                metric_cal = SelfCheckPrompt(ask_fn=llm.ask_yes_no if llm else None)
                calib_probs: list[float] = []
                calib_labels: list[int] = []
                for ex in val_examples:
                    _, per_sent_probs = metric_cal.predict(
                        ex["gpt3_sentences"],
                        ex["gpt3_text_samples"],
                        return_probs=True,
                    )
                    labels = load_annotations(ex)
                    for lbl, sent_probs in zip(labels, per_sent_probs):
                        for prob in sent_probs:
                            p = min(max(prob, 1e-8), 1 - 1e-8)
                            logit = math.log(p / (1 - p))
                            calib_probs.append(logit)
                            calib_labels.append(1 - lbl)
                calib_logits = [[p, 0.0] for p in calib_probs]
                prompt_temperature = find_optimal_temperature(
                    calib_logits, calib_labels
                )
            except Exception as exc:  # pragma: no cover - optional dependency
                logging.warning("Prompt temperature calibration failed: %s", exc)

        def get_metric(name: str):
            if name == "ngram":
                return SelfCheckNgram(n=args.ngram_n)
            if name == "prompt":
                return SelfCheckPrompt(
                    ask_fn=llm.ask_yes_no if llm else None,
                    temperature=prompt_temperature,
                )
            if name == "nli":
                return SelfCheckNLI(temperature=nli_temperature)
            return METRICS[name]()

        # Evaluate metrics on training data for combiner fitting
        train_score_matrix: dict[str, List[float]] = {}
        train_labels: List[int] | None = None
        for name in metric_names:
            if name not in METRICS:
                logging.warning("Unknown metric '%s' -- skipping", name)
                continue
            try:
                metric = get_metric(name)
                _, scores, labels = evaluate(
                    metric, train_examples, bins=args.calib_bins, return_scores=True
                )
            except Exception as exc:  # pragma: no cover - optional dependencies
                logging.warning("Metric %s failed: %s", name, exc)
                continue
            if train_labels is None:
                train_labels = labels
            train_score_matrix[name] = scores

        # Evaluate metrics on test data
        summary_rows: List[dict[str, float | str]] = []
        test_score_matrix: dict[str, List[float]] = {}
        test_labels: List[int] | None = None
        for name in metric_names:
            if name not in METRICS:
                continue
            if name not in train_score_matrix:
                # skip metrics that failed on training data
                logging.warning("Skipping metric '%s' due to missing training scores", name)
                continue
            try:
                metric = get_metric(name)
                stats, scores, labels = evaluate(
                    metric, test_examples, bins=args.calib_bins, return_scores=True
                )
            except Exception as exc:  # pragma: no cover - optional dependencies
                logging.warning("Metric %s failed on test data: %s", name, exc)
                continue
            if test_labels is None:
                test_labels = labels
            test_score_matrix[name] = scores
            _save_plots(name, stats, temp_dir)
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

        # Train combiner on train split and evaluate on test split
        if (
            train_score_matrix
            and test_score_matrix
            and train_labels is not None
            and test_labels is not None
        ):
            try:
                import numpy as np
                import torch
                from selfcheck_combiner import SelfCheckCombiner

                feature_names = [n for n in metric_names if n in train_score_matrix and n in test_score_matrix]
                if feature_names:
                    X_train = np.column_stack([train_score_matrix[n] for n in feature_names])
                    y_train = np.array(train_labels)

                    cv_stats = None
                    if args.cv_folds > 1:
                        try:
                            class_counts = np.bincount(y_train)
                            max_folds = int(class_counts.min())
                            n_splits = min(args.cv_folds, max_folds)
                        except ValueError:
                            n_splits = 0
                        if n_splits >= 2:
                            skf = StratifiedKFold(
                                n_splits=n_splits, shuffle=True, random_state=0
                            )
                            fold_stats = []
                            for tr_idx, val_idx in skf.split(X_train, y_train):
                                comb_fold = SelfCheckCombiner()
                                comb_fold.fit(X_train[tr_idx], y_train[tr_idx])
                                val_scores = comb_fold.predict(X_train[val_idx])
                                fold_stats.append(
                                    _compute_stats(
                                        val_scores, y_train[val_idx], bins=args.calib_bins
                                    )
                                )
                            cv_stats = {
                                k: sum(fs[k] for fs in fold_stats) / len(fold_stats)
                                for k in (
                                    "average_precision",
                                    "precision",
                                    "recall",
                                    "f1",
                                    "brier",
                                )
                            }

                    comb = SelfCheckCombiner()
                    comb.fit(X_train, y_train)
                    torch.save(comb._model.state_dict(), temp_dir / "combiner.pt")

                    X_test = np.column_stack([test_score_matrix[n] for n in feature_names])
                    comb_scores = comb.predict(X_test)
                    comb_stats = _compute_stats(comb_scores, test_labels, bins=args.calib_bins)
                    _save_plots("combined", comb_stats, temp_dir)

                    summary_row = {
                        "metric": "combined",
                        "average_precision": comb_stats["average_precision"],
                        "precision": comb_stats["precision"],
                        "recall": comb_stats["recall"],
                        "f1": comb_stats["f1"],
                        "brier": comb_stats["brier"],
                    }
                    if cv_stats is not None:
                        summary_row.update({f"cv_{k}": v for k, v in cv_stats.items()})
                    summary_rows.append(summary_row)
                else:
                    logging.warning("Combiner training skipped: no common metrics")
            except Exception as exc:  # pragma: no cover - optional dependency
                logging.warning("Combiner failed: %s", exc)

        if summary_rows:
            summary_path = temp_dir / "summary.csv"
            fieldnames = sorted({k for row in summary_rows for k in row.keys()})
            with summary_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_rows)
            logging.info("Wrote results to %s", summary_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()


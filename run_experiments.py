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
import functools
import time

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


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


def _compute_stats(
    scores: List[float], labels: List[int], *, bins: int = 10, threshold: float | None = None
) -> dict:
    """Compute evaluation statistics for ``scores`` against ``labels``."""

    precision_curve, recall_curve, thresholds = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)

    thr = 0.5 if threshold is None else float(threshold)
    preds = [1 if s >= thr else 0 for s in scores]
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
    "threshold": thr,
    }


def evaluate(
    metric,
    dataset: Iterable[dict],
    *,
    bins: int = 10,
    return_scores: bool = False,
    temperature: float = 1.0,
) -> dict | tuple[dict, List[float], List[int]]:
    """Return scoring results and optionally raw scores and labels.

    ``temperature`` allows optional temperature scaling of the resulting scores
    using ``find_optimal_temperature`` calibrated on a validation split.
    """

    all_scores: List[float] = []
    all_labels: List[int] = []
    try:
        _n = len(dataset)  # type: ignore[arg-type]
    except Exception:
        _n = -1
    if _n >= 0:
        logging.debug("evaluate(): scoring %d examples", _n)
    for example in dataset:
        sentences = example["gpt3_sentences"]
        samples = example["gpt3_text_samples"]
        logging.debug("evaluate(): %d sentences vs %d samples", len(sentences), len(samples))
        t_eval = time.perf_counter()
        scores = metric.predict(sentences, samples)
        logging.debug("evaluate(): metric.predict took %.3fs", time.perf_counter() - t_eval)
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

    if temperature != 1.0:
        logging.debug("evaluate(): applying temperature scaling=%.3f", temperature)
        scaled: List[float] = []
        for s in all_scores:
            p = min(max(s, 1e-8), 1 - 1e-8)
            logit = math.log(p / (1 - p))
            p = 1 / (1 + math.exp(-logit / temperature))
            scaled.append(p)
        all_scores = scaled

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
        default="evaluation",
        help="Dataset split or slice for training the combiner.",
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default=None,
        help="Optional dataset split or slice for validation/hyperparameters.",
    )
    parser.add_argument(
        "--test-split",
        type=str,
        default="evaluation",
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
    default="gpt-5-preview",
        help="OpenAI model for sampling and prompt-based metric.",
    )
    parser.add_argument(
        "--prompt-backend",
        type=str,
        default="openai",
        choices=["openai", "hf"],
        help="Backend for the prompt metric: OpenAI API or local HuggingFace.",
    )
    parser.add_argument(
        "--prompt-hf-model",
        type=str,
        default=None,
        help="HF model path/name for local prompt backend (e.g., hf-cache/lmqg__flan-t5-base-squad-qg).",
    )
    parser.add_argument(
        "--prompt-hf-task",
        type=str,
        default=None,
        help="HF pipeline task for local prompt backend (text-generation or text2text-generation).",
    )
    parser.add_argument(
        "--prompt-hf-device",
        type=str,
        default=None,
        help="Device for HF prompt backend (e.g., cpu, cuda, or CUDA index).",
    )
    parser.add_argument(
        "--prompt-hf-max-new-tokens",
        type=int,
        default=16,
        help="Max new tokens for HF prompt backend generations.",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=None,
        help="Override the Yes/No prompt template.",
    )
    parser.add_argument(
        "--calib-bins",
        type=int,
        default=10,
        help="Number of bins for the calibration curve.",
    )
    parser.add_argument(
        "--nli-batch-size",
        type=int,
        default=16,
        help="Batch size for NLI model inference (larger is faster on GPU).",
    )
    parser.add_argument(
        "--nli-max-length",
        type=int,
        default=256,
        help="Max sequence length for NLI tokenization (smaller is faster).",
    )
    parser.add_argument(
        "--bertscore-no-baseline",
        action="store_true",
        help="Disable BERTScore baseline rescaling (recommended in offline mode).",
    )
    parser.add_argument(
        "--bertscore-model",
        type=str,
        default=str(Path("hf-cache") / "roberta-large"),
        help="HF model path/name for BERTScore (use local cache path for offline).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified folds for combiner cross-validation (set to 1 to disable).",
    )
    parser.add_argument(
        "--combiner-l2",
        type=float,
        default=0.0,
        help="L2 regularisation strength for the combiner (weight decay).",
    )
    parser.add_argument(
        "--combiner-patience",
        type=int,
        default=0,
        help="Early stopping patience for the combiner (0 disables early stopping).",
    )
    parser.add_argument(
        "--mqag-metric",
        type=str,
        default="counting",
        choices=["kl", "counting", "hellinger", "total_variation"],
        help="Disagreement metric used by MQAG.",
    )
    parser.add_argument(
        "--mqag-answerability-threshold",
        type=float,
        default=0.5,
        help="Answerability threshold for MQAG.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase logging verbosity (DEBUG level).",
    )
    parser.add_argument(
        "--tune-thresholds",
        action="store_true",
        help="Sweep decision thresholds on a split to maximize F1 and apply them for single-point metrics.",
    )
    parser.add_argument(
        "--tune-split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Dataset split to use for threshold tuning (defaults to train; falls back to train if val missing).",
    )
    parser.add_argument(
        "--paper-config",
        action="store_true",
        help=(
            "Use configuration matching the SelfCheckGPT paper (20 samples, top-k=50, top-p=0.95, resampling, 5-fold CV)."
        ),
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Verbose logging enabled")

    if args.paper_config:
        args.sample_count = 20
        if args.top_k is None:
            args.top_k = 50
        if args.top_p is None:
            args.top_p = 0.95
        args.resample = True
        args.cv_folds = max(args.cv_folds, 5)
        args.mqag_metric = "kl"
        args.mqag_answerability_threshold = 0.9
        if args.combiner_l2 == 0.0:
            args.combiner_l2 = 1e-4
        args.combiner_patience = max(args.combiner_patience, 5)

    if (args.top_k is not None or args.top_p is not None) and not args.resample:
        logging.info(
            "top-k/top-p provided without --resample; enabling resampling to match settings"
        )
        args.resample = True

    metric_names = list(METRICS) if "all" in args.metrics else args.metrics

    llm: OpenAIChatLLM | None = None
    if args.resample or ("prompt" in metric_names and args.prompt_backend == "openai"):
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
            logging.info(
                "Preparing %s split: %d examples (resample=%s, sample_count=%d)",
                name,
                len(examples),
                bool(args.resample),
                args.sample_count,
            )
            if args.resample or not all("gpt3_text_samples" in ex for ex in examples):
                logging.info("Generating samples with LLM for %s split...", name)
                prompts = [" ".join(ex["gpt3_sentences"]) for ex in examples]
                split_dir = temp_dir / name
                split_dir.mkdir(parents=True, exist_ok=True)
                sample_file = split_dir / "samples.jsonl"
                _t0 = time.perf_counter()
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
                logging.info(
                    "Generated samples for %s split in %.2fs -> %s",
                    name,
                    time.perf_counter() - _t0,
                    sample_file,
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
            avg_samples = (
                sum(len(ex.get("gpt3_text_samples", [])) for ex in examples) / max(1, len(examples))
            )
            logging.info("Prepared %s split: %d examples (avg samples per prompt ~%.2f)", name, len(examples), avg_samples)
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

        def get_metric(name: str, *, temperature: float = 1.0):
            if name == "ngram":
                return SelfCheckNgram(n=args.ngram_n)
            if name == "prompt":
                if args.prompt_backend == "hf" or args.prompt_hf_model:
                    logging.info(
                        "Init prompt metric (HF): model=%s task=%s device=%s max_new_tokens=%d",
                        args.prompt_hf_model or "sshleifer/tiny-gpt2",
                        args.prompt_hf_task,
                        args.prompt_hf_device,
                        args.prompt_hf_max_new_tokens,
                    )
                    return SelfCheckPrompt(
                        ask_fn=None,
                        temperature=temperature,
                        hf_model=args.prompt_hf_model or "sshleifer/tiny-gpt2",
                        hf_device=args.prompt_hf_device,
                        hf_max_new_tokens=args.prompt_hf_max_new_tokens,
                        prompt_template=args.prompt_template,
                        hf_task=args.prompt_hf_task,
                    )
                logging.info("Init prompt metric (OpenAI): model=%s", args.llm_model)
                return SelfCheckPrompt(
                    ask_fn=llm.ask_yes_no if llm else None,
                    temperature=temperature,
                    prompt_template=args.prompt_template,
                )
            if name == "nli":
                logging.info("Init NLI metric: batch_size=%d max_length=%d", args.nli_batch_size, args.nli_max_length)
                return SelfCheckNLI(
                    temperature=temperature,
                    batch_size=args.nli_batch_size,
                    max_length=args.nli_max_length,
                )
            if name == "mqag":
                metric = SelfCheckMQAG()
                metric.predict = functools.partial(
                    metric.predict,
                    metric=args.mqag_metric,
                    answerability_threshold=args.mqag_answerability_threshold,
                )
                return metric
            if name == "bertscore":
                logging.info("Init BERTScore: model=%s baseline=%s", args.bertscore_model, not args.bertscore_no_baseline)
                return SelfCheckBERTScore(
                    model=args.bertscore_model,
                    baseline=not args.bertscore_no_baseline,
                )
            return METRICS[name]()

        def _best_threshold(scores: List[float], labels: List[int]) -> float:
            """Return threshold maximizing F1 (ties broken by higher recall)."""
            if not scores:
                return 0.5
            import numpy as _np
            from sklearn.metrics import f1_score as _f1

            # Consider unique score cutoffs and a couple of sentinels
            cand = sorted(set(scores))
            # Use mid-points between consecutive unique scores to reduce ties
            mids: List[float] = []
            for a, b in zip(cand[:-1], cand[1:]):
                mids.append((a + b) / 2)
            cands = [0.0] + mids + [1.0]
            best_t = 0.5
            best_f1 = -1.0
            best_rec = -1.0
            y = _np.array(labels)
            s = _np.array(scores)
            for t in cands:
                preds = (s >= t).astype(int)
                f1 = _f1(y, preds, zero_division=0)
                # recall for tie-breaker
                tp = int(((preds == 1) & (y == 1)).sum())
                fn = int(((preds == 0) & (y == 1)).sum())
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                if f1 > best_f1 or (abs(f1 - best_f1) <= 1e-12 and rec > best_rec):
                    best_f1 = f1
                    best_rec = rec
                    best_t = float(t)
            return best_t

        metric_temps: dict[str, float] = {n: 1.0 for n in metric_names}
        if val_examples:
            for name in metric_names:
                if name not in METRICS:
                    continue
                try:
                    metric = get_metric(name)
                    _, scores, labels = evaluate(
                        metric, val_examples, bins=args.calib_bins, return_scores=True
                    )
                    logits = []
                    for s in scores:
                        p = min(max(s, 1e-8), 1 - 1e-8)
                        logits.append([math.log(p / (1 - p)), 0.0])
                    metric_temps[name] = find_optimal_temperature(logits, labels)
                except Exception as exc:  # pragma: no cover - optional dependency
                    logging.warning("Calibration for %s failed: %s", name, exc)

        # Evaluate metrics on training data for combiner fitting
        train_score_matrix: dict[str, List[float]] = {}
        train_labels: List[int] | None = None
        for name in metric_names:
            if name not in METRICS:
                logging.warning("Unknown metric '%s' -- skipping", name)
                continue
            try:
                metric = get_metric(name, temperature=metric_temps.get(name, 1.0))
                eval_temp = 1.0 if name in {"prompt", "nli"} else metric_temps.get(name, 1.0)
                logging.info("[TRAIN] Evaluating '%s' (temp=%s) ...", name, eval_temp)
                _t0 = time.perf_counter()
                _, scores, labels = evaluate(
                    metric,
                    train_examples,
                    bins=args.calib_bins,
                    return_scores=True,
                    temperature=eval_temp,
                )
                logging.info("[TRAIN] Done '%s' in %.2fs | scores=%d", name, time.perf_counter() - _t0, len(scores))
            except Exception as exc:  # pragma: no cover - optional dependencies
                logging.warning("Metric %s failed: %s", name, exc)
                continue
            if train_labels is None:
                train_labels = labels
            train_score_matrix[name] = scores

        # Evaluate metrics on validation data for combiner early stopping
        val_score_matrix: dict[str, List[float]] = {}
        val_labels: List[int] | None = None
        if val_examples:
            for name in metric_names:
                if name not in METRICS or name not in train_score_matrix:
                    continue
                try:
                    metric = get_metric(name, temperature=metric_temps.get(name, 1.0))
                    eval_temp = 1.0 if name in {"prompt", "nli"} else metric_temps.get(name, 1.0)
                    logging.info("[VAL] Evaluating '%s' (temp=%s) ...", name, eval_temp)
                    _t0 = time.perf_counter()
                    _, scores, labels = evaluate(
                        metric,
                        val_examples,
                        bins=args.calib_bins,
                        return_scores=True,
                        temperature=eval_temp,
                    )
                    logging.info("[VAL] Done '%s' in %.2fs | scores=%d", name, time.perf_counter() - _t0, len(scores))
                except Exception as exc:  # pragma: no cover - optional dependencies
                    logging.warning("Metric %s failed on validation data: %s", name, exc)
                    continue
                if val_labels is None:
                    val_labels = labels
                val_score_matrix[name] = scores

        # Evaluate metrics on test data
        summary_rows: List[dict[str, float | str]] = []
        test_score_matrix: dict[str, List[float]] = {}
        test_labels: List[int] | None = None
        tuned_thresholds: dict[str, float] = {}

        # If requested, tune thresholds using selected split
        if args.tune_thresholds:
            tune_on_val = args.tune_split == "val" and bool(val_score_matrix)
            src_scores = val_score_matrix if tune_on_val else train_score_matrix
            src_labels = val_labels if tune_on_val else train_labels
            logging.info("Tuning thresholds on %s split (%d metrics)", "val" if tune_on_val else "train", len(src_scores))
            if src_labels is not None:
                for name in metric_names:
                    if name in src_scores:
                        try:
                            _t0 = time.perf_counter()
                            thr = _best_threshold(src_scores[name], src_labels)
                            tuned_thresholds[name] = thr
                            # quick stats at tuned threshold
                            import numpy as _np
                            from sklearn.metrics import f1_score as _f1
                            y = _np.array(src_labels)
                            s = _np.array(src_scores[name])
                            preds = (s >= thr).astype(int)
                            f1v = _f1(y, preds, zero_division=0)
                            tp = int(((preds == 1) & (y == 1)).sum())
                            fn = int(((preds == 0) & (y == 1)).sum())
                            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                            logging.info(
                                "Tuned threshold '%s': %.4f (F1=%.4f Recall=%.4f) in %.2fs",
                                name,
                                thr,
                                f1v,
                                rec,
                                time.perf_counter() - _t0,
                            )
                        except Exception as _exc:  # pragma: no cover - robustness
                            logging.warning("Threshold tuning failed for %s: %s", name, _exc)
            else:
                logging.warning("Threshold tuning skipped (no labels available for chosen split)")
        for name in metric_names:
            if name not in METRICS:
                continue
            if name not in train_score_matrix:
                # skip metrics that failed on training data
                logging.warning("Skipping metric '%s' due to missing training scores", name)
                continue
            try:
                metric = get_metric(name, temperature=metric_temps.get(name, 1.0))
                eval_temp = 1.0 if name in {"prompt", "nli"} else metric_temps.get(name, 1.0)
                logging.info("[TEST] Evaluating '%s' (temp=%s) ...", name, eval_temp)
                _t0 = time.perf_counter()
                stats, scores, labels = evaluate(
                    metric,
                    test_examples,
                    bins=args.calib_bins,
                    return_scores=True,
                    temperature=eval_temp,
                )
                logging.info(
                    "[TEST] Done '%s' in %.2fs | AP=%.4f Brier=%.4f",
                    name,
                    time.perf_counter() - _t0,
                    stats.get("average_precision", float('nan')),
                    stats.get("brier", float('nan')),
                )
            except Exception as exc:  # pragma: no cover - optional dependencies
                logging.warning("Metric %s failed on test data: %s", name, exc)
                continue
            if test_labels is None:
                test_labels = labels
            test_score_matrix[name] = scores
            # Recompute single-point stats using tuned threshold if available
            if args.tune_thresholds and name in tuned_thresholds:
                stats = _compute_stats(scores, labels, bins=args.calib_bins, threshold=tuned_thresholds[name])
            _save_plots(name, stats, temp_dir)
            logging.info("Saved plots for '%s' in %s", name, temp_dir)
            summary_rows.append(
                {
                    "metric": name,
                    "average_precision": stats["average_precision"],
                    "precision": stats["precision"],
                    "recall": stats["recall"],
                    "f1": stats["f1"],
                    "brier": stats["brier"],
                    "threshold": stats.get("threshold", 0.5),
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

                feature_names = [
                    n
                    for n in metric_names
                    if n in train_score_matrix
                    and n in test_score_matrix
                    and (not val_score_matrix or n in val_score_matrix)
                ]
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
                                comb_fold = SelfCheckCombiner(
                                    l2=args.combiner_l2, patience=args.combiner_patience
                                )
                                comb_fold.fit(
                                    X_train[tr_idx],
                                    y_train[tr_idx],
                                    X_val=X_train[val_idx],
                                    y_val=y_train[val_idx],
                                )
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

                    comb = SelfCheckCombiner(
                        l2=args.combiner_l2, patience=args.combiner_patience
                    )
                    X_val = None
                    y_val_arr = None
                    if val_score_matrix and val_labels is not None:
                        X_val = np.column_stack([val_score_matrix[n] for n in feature_names])
                        y_val_arr = np.array(val_labels)
                    logging.info("Fitting combiner on %s examples with %d features", X_train.shape[0], X_train.shape[1])
                    _t0 = time.perf_counter()
                    comb.fit(X_train, y_train, X_val=X_val, y_val=y_val_arr)
                    logging.info("Combiner trained in %.2fs", time.perf_counter() - _t0)
                    torch.save(comb._model.state_dict(), temp_dir / "combiner.pt")
                    logging.info("Saved combiner weights -> %s", temp_dir / "combiner.pt")

                    X_test = np.column_stack([test_score_matrix[n] for n in feature_names])
                    comb_scores = comb.predict(X_test)
                    # Optionally tune threshold for combined score using chosen split
                    comb_thr = None
                    if args.tune_thresholds:
                        # Select source split for combiner tuning
                        if args.tune_split == "val" and val_score_matrix and val_labels is not None:
                            X_src = np.column_stack([val_score_matrix[n] for n in feature_names])
                            y_src = np.array(val_labels)
                        else:
                            X_src = X_train
                            y_src = y_train
                        try:
                            src_scores = comb.predict(X_src)
                            comb_thr = _best_threshold(list(map(float, src_scores)), list(map(int, y_src)))
                        except Exception as _exc:  # pragma: no cover - robustness
                            logging.warning("Combiner threshold tuning failed: %s", _exc)
                    comb_stats = _compute_stats(
                        comb_scores, test_labels, bins=args.calib_bins, threshold=comb_thr
                    )
                    _save_plots("combined", comb_stats, temp_dir)
                    logging.info("Saved plots for 'combined' in %s", temp_dir)

                    summary_row = {
                        "metric": "combined",
                        "average_precision": comb_stats["average_precision"],
                        "precision": comb_stats["precision"],
                        "recall": comb_stats["recall"],
                        "f1": comb_stats["f1"],
                        "brier": comb_stats["brier"],
                        "threshold": comb_stats.get("threshold", 0.5),
                    }
                    if cv_stats is not None:
                        summary_row.update({f"cv_{k}": v for k, v in cv_stats.items()})
                    summary_rows.append(summary_row)
                else:
                    logging.warning("Combiner training skipped: no common metrics")
            except Exception as exc:  # pragma: no cover - optional dependency
                logging.warning("Combiner failed: %s", exc)

        # Persist tuned thresholds if any
        if tuned_thresholds:
            try:
                with (temp_dir / "thresholds.json").open("w", encoding="utf-8") as f:
                    json.dump(tuned_thresholds, f, indent=2)
                logging.info("Wrote tuned thresholds -> %s", temp_dir / "thresholds.json")
            except Exception as _exc:  # pragma: no cover - robustness
                logging.warning("Failed to write thresholds.json: %s", _exc)

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


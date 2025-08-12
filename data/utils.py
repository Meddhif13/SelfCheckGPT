"""Helpers for loading and validating datasets used by the project."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from datasets import Dataset, load_dataset

DATASET_NAME = "potsawee/wiki_bio_gpt3_hallucination"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "selfcheckgpt"


def load_wikibio_hallucination(
    split: str = "evaluation",
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
) -> Dataset:
    """Download and return a slice of the WikiBio hallucination dataset.

    The dataset is cached locally so subsequent calls are fast.

    Args:
        split: Dataset split to load. Supports slice notation such as
            ``"evaluation[:1]"`` for a tiny sample used in tests.
        cache_dir: Directory where the dataset will be cached.

    Returns:
        A :class:`datasets.Dataset` containing the requested split.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET_NAME, split=split, cache_dir=str(cache_path))
    _validate_dataset(dataset)
    return dataset


def _validate_dataset(dataset: Dataset) -> None:
    """Basic validation to ensure the dataset is usable."""
    if len(dataset) == 0:
        raise ValueError("Loaded dataset split is empty")
    if not dataset.column_names:
        raise ValueError("Dataset has no columns")

    required_columns = {
        "gpt3_sentences",
        "gpt3_text_samples",
        "annotation",
    }
    missing = required_columns.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

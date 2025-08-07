import pathlib
import sys

import spacy
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from data.utils import load_wikibio_hallucination


def test_spacy_model_loads():
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("This is a test.")
    assert doc.text == "This is a test."


def test_wikibio_dataset_loads(tmp_path):
    try:
        ds = load_wikibio_hallucination(
            split="train[:1]", cache_dir=tmp_path
        )
    except Exception as exc:  # pragma: no cover - network failure
        pytest.skip(f"dataset download failed: {exc}")
    assert len(ds) == 1

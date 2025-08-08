import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from datasets import Dataset
import run_experiments


def test_run_experiments_tiny(tmp_path, monkeypatch):
    ds = Dataset.from_dict(
        {
            "gpt3_sentences": [["Paris is in France.", "The sky is green."]],
            "gpt3_text_samples": [["Paris is in France.", "The sky is blue."]],
            "annotation": [["accurate", "inaccurate"]],
        }
    )

    def fake_loader(split="test"):
        return ds

    monkeypatch.setattr(run_experiments, "load_wikibio_hallucination", fake_loader)

    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiments.py",
            "--metrics",
            "ngram",
            "--limit",
            "1",
            "--output-dir",
            str(out_dir),
        ],
    )
    run_experiments.main()

    summary = out_dir / "summary.csv"
    assert summary.exists()
    assert (out_dir / "ngram_pr.png").exists()
    assert (out_dir / "ngram_calibration.png").exists()
    content = summary.read_text()
    assert "ngram" in content

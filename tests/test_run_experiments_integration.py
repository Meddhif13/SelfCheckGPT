import sys
import json
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

    seen_splits = []

    def fake_loader(split="test"):
        seen_splits.append(split)
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
            "--train-split",
            "evaluation[:1]",
            "--val-split",
            "evaluation[:1]",
            "--test-split",
            "evaluation[:1]",
        ],
    )
    run_experiments.main()

    summary = out_dir / "summary.csv"
    assert summary.exists()
    assert (out_dir / "ngram_pr.png").exists()
    assert (out_dir / "ngram_calibration.png").exists()
    assert (out_dir / "combiner.pt").exists()
    content = summary.read_text()
    assert "ngram" in content
    assert seen_splits == ["evaluation[:1]", "evaluation[:1]", "evaluation[:1]"]


def test_run_experiments_temperature_sweep(tmp_path, monkeypatch):
    ds = Dataset.from_dict(
        {
            "gpt3_sentences": [["Paris is in France."]],
            "gpt3_text_samples": [["Paris is in France."]],
            "annotation": [["accurate"]],
        }
    )

    monkeypatch.setattr(run_experiments, "load_wikibio_hallucination", lambda split="test": ds)

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
            "--temperatures",
            "0.1",
            "0.2",
        ],
    )
    run_experiments.main()

    assert (out_dir / "temp_0_1" / "summary.csv").exists()
    assert (out_dir / "temp_0_2" / "summary.csv").exists()


def test_run_experiments_passes_sampling_params(tmp_path, monkeypatch):
    ds = Dataset.from_dict(
        {
            "gpt3_sentences": [["Paris is in France."]],
            "annotation": [["accurate"]],
        }
    )

    monkeypatch.setattr(run_experiments, "load_wikibio_hallucination", lambda split="test": ds)

    captured = {}

    def fake_generate(
        llm,
        prompts,
        output_path,
        *,
        num_samples,
        temperature,
        top_k,
        top_p,
        deterministic,
        cache_dir=None,
    ):
        captured.update(
            {
                "num_samples": num_samples,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "deterministic": deterministic,
            }
        )
        with Path(output_path).open("w", encoding="utf-8") as f:
            for p in prompts:
                json.dump({"prompt": p, "sample": "s"}, f)
                f.write("\n")

    monkeypatch.setattr(run_experiments, "generate_samples", fake_generate)

    class DummyLLM:
        def ask_yes_no(self, context, sentence):
            return "Yes"

    monkeypatch.setattr(run_experiments, "OpenAIChatLLM", lambda model: DummyLLM())

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
            "--resample",
            "--sample-count",
            "16",
            "--temperature",
            "0.8",
            "--top-k",
            "5",
            "--top-p",
            "0.9",
            "--deterministic",
        ],
    )

    run_experiments.main()

    assert captured == {
        "num_samples": 16,
        "temperature": 0.8,
        "top_k": 5,
        "top_p": 0.9,
        "deterministic": True,
    }


def test_run_experiments_paper_config(tmp_path, monkeypatch):
    ds = Dataset.from_dict(
        {
            "gpt3_sentences": [["Paris is in France."]],
            "annotation": [["accurate"]],
        }
    )

    monkeypatch.setattr(run_experiments, "load_wikibio_hallucination", lambda split="test": ds)

    captured = {}

    def fake_generate(
        llm,
        prompts,
        output_path,
        *,
        num_samples,
        temperature,
        top_k,
        top_p,
        deterministic,
        cache_dir=None,
    ):
        captured.update(
            {
                "num_samples": num_samples,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "deterministic": deterministic,
            }
        )
        with Path(output_path).open("w", encoding="utf-8") as f:
            for p in prompts:
                json.dump({"prompt": p, "sample": "s"}, f)
                f.write("\n")

    monkeypatch.setattr(run_experiments, "generate_samples", fake_generate)

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
            "--paper-config",
        ],
    )

    run_experiments.main()

    assert captured == {
        "num_samples": 20,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
        "deterministic": False,
    }


def test_run_experiments_combiner(tmp_path, monkeypatch):
    # create dataset with 10 examples and alternating labels
    sentences = [[f"S{i}"] for i in range(10)]
    samples = [[f"S{i}"] for i in range(10)]
    annotations = [["accurate"] if i % 2 == 0 else ["inaccurate"] for i in range(10)]
    ds = Dataset.from_dict(
        {
            "gpt3_sentences": sentences,
            "gpt3_text_samples": samples,
            "annotation": annotations,
        }
    )

    monkeypatch.setattr(run_experiments, "load_wikibio_hallucination", lambda split="test": ds)

    class DummyMetric:
        def __init__(self, value):
            self.value = value

        def predict(self, sentences, samples):
            return [self.value for _ in sentences]

    run_experiments.METRICS["m1"] = lambda: DummyMetric(0.1)
    run_experiments.METRICS["m2"] = lambda: DummyMetric(0.9)

    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiments.py",
            "--metrics",
            "m1",
            "m2",
            "--limit",
            "10",
            "--output-dir",
            str(out_dir),
            "--train-split",
            "train",
            "--test-split",
            "test",
        ],
    )

    run_experiments.main()

    summary = out_dir / "summary.csv"
    assert summary.exists()
    content = summary.read_text()
    assert "combined" in content

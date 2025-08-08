import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sampling.generator import generate_samples


class StubLLM:
    def __init__(self):
        self.calls: list[tuple[str, float, int | None, float | None, bool]] = []

    def __call__(
        self,
        prompt: str,
        *,
        temperature: float,
        top_k: int | None = None,
        top_p: float | None = None,
        deterministic: bool = False,
    ) -> str:
        self.calls.append((prompt, temperature, top_k, top_p, deterministic))
        return f"sample-{len(self.calls)}"


def test_sampling_parameters_and_logging(tmp_path: Path):
    prompts = ["Hello"]
    out_file = tmp_path / "samples.jsonl"
    llm = StubLLM()

    generate_samples(
        llm,
        prompts,
        out_file,
        num_samples=16,
        temperature=0.3,
        top_k=5,
        top_p=0.9,
        deterministic=True,
    )

    assert len(llm.calls) == 16
    assert all(call == ("Hello", 0.3, 5, 0.9, True) for call in llm.calls)

    lines = out_file.read_text().splitlines()
    assert len(lines) == 16
    records = [json.loads(line) for line in lines]
    assert all(r["metadata"]["top_k"] == 5 for r in records)
    assert records[0]["metadata"]["deterministic"] is True


def test_sampling_cache(tmp_path: Path):
    prompts = ["Hello"]
    cache_dir = tmp_path / "cache"

    out1 = tmp_path / "out1.jsonl"
    llm1 = StubLLM()
    generate_samples(llm1, prompts, out1, cache_dir=cache_dir)
    assert len(llm1.calls) == 1

    out2 = tmp_path / "out2.jsonl"
    llm2 = StubLLM()
    generate_samples(llm2, prompts, out2, cache_dir=cache_dir)
    assert len(llm2.calls) == 0
    rec1 = json.loads(out1.read_text().splitlines()[0])
    rec2 = json.loads(out2.read_text().splitlines()[0])
    assert rec1["sample"] == rec2["sample"]


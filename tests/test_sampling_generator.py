import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sampling.generator import generate_samples


class StubLLM:
    def __init__(self):
        self.calls: list[tuple[str, float]] = []

    def __call__(self, prompt: str, *, temperature: float) -> str:
        self.calls.append((prompt, temperature))
        return f"sample-{len(self.calls)}"


def test_sampling_deterministic(tmp_path: Path):
    prompts = ["Hello", "World"]
    out_file = tmp_path / "samples.jsonl"
    llm = StubLLM()

    generate_samples(llm, prompts, out_file, num_samples=2, temperature=0.3)

    assert len(llm.calls) == 4
    assert all(t == 0.3 for _, t in llm.calls)

    lines = out_file.read_text().splitlines()
    assert len(lines) == 4
    records = [json.loads(line) for line in lines]
    assert [r["prompt"] for r in records] == ["Hello", "Hello", "World", "World"]
    assert [r["sample"] for r in records] == [
        "sample-1",
        "sample-2",
        "sample-3",
        "sample-4",
    ]


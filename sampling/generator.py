"""Utilities for generating and storing samples from an LLM."""
from __future__ import annotations

from pathlib import Path
import json
from typing import Iterable, Protocol


class LLM(Protocol):
    """Protocol for language model clients."""

    def __call__(self, prompt: str, *, temperature: float) -> str:  # pragma: no cover - interface
        """Generate a completion for ``prompt``."""


def generate_samples(
    llm: LLM,
    prompts: Iterable[str],
    output_path: str | Path,
    *,
    num_samples: int = 1,
    temperature: float = 0.7,
) -> None:
    """Query ``llm`` for each prompt and persist the results.

    Each line in ``output_path`` will contain a JSON object with keys
    ``prompt`` and ``sample``.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for prompt in prompts:
            for _ in range(num_samples):
                sample = llm(prompt, temperature=temperature)
                json.dump({"prompt": prompt, "sample": sample}, f, ensure_ascii=False)
                f.write("\n")


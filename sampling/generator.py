"""Utilities for generating and storing samples from an LLM."""
from __future__ import annotations

from pathlib import Path
import json
from typing import Iterable, Protocol
import os
import time

from openai import OpenAI, RateLimitError


class LLM(Protocol):
    """Protocol for language model clients."""

    def __call__(self, prompt: str, *, temperature: float) -> str:  # pragma: no cover - interface
        """Generate a completion for ``prompt``."""


class OpenAIChatLLM:
    """Simple OpenAI Chat Completions client with caching and retries.

    The class reads the API key from the ``OPENAI_API_KEY`` environment
    variable and uses a small in-memory cache to avoid repeat requests for
    identical prompts.  It exposes ``ask_yes_no`` for convenience when a
    Yes/No judgement is required.
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        max_retries: int = 3,
        retry_wait: float = 1.0,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self._client: OpenAI | None = None
        # cache keyed by (prompt, temperature)
        self._cache: dict[tuple[str, float], str] = {}

    # -- internal ---------------------------------------------------------
    def _ensure_client(self) -> OpenAI:
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY environment variable not set")
            self._client = OpenAI(api_key=api_key)
        return self._client

    # -- public -----------------------------------------------------------
    def __call__(self, prompt: str, *, temperature: float) -> str:  # pragma: no cover - network
        key = (prompt, temperature)
        if key in self._cache:
            return self._cache[key]

        client = self._ensure_client()
        for attempt in range(self.max_retries):
            try:
                res = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                text = res.choices[0].message.content or ""
                self._cache[key] = text
                return text
            except RateLimitError:
                time.sleep(self.retry_wait * (2**attempt))
        raise RuntimeError("OpenAI API request failed after retries")

    def ask_yes_no(self, context: str, sentence: str) -> str:  # pragma: no cover - network
        prompt = (
            f"Context: {context}\nSentence: {sentence}\n"
            "Is the sentence supported by the context above?\nAnswer Yes or No:"
        )
        return self(prompt, temperature=0.0)


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


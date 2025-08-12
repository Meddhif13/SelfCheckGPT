"""Utilities for generating and storing samples from an LLM."""
from __future__ import annotations

from pathlib import Path
import json
from typing import Iterable, Protocol
import os
import time
import hashlib
import logging

from openai import OpenAI, RateLimitError, APIError


class LLM(Protocol):
    """Protocol for language model clients."""

    def __call__(
        self,
        prompt: str,
        *,
        temperature: float,
        top_k: int | None = None,
        top_p: float | None = None,
        deterministic: bool = False,
    ) -> str:  # pragma: no cover - interface
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
        # cache keyed by (prompt, temperature, top_k, top_p, deterministic)
        self._cache: dict[tuple[str, float, int | None, float | None, bool], str] = {}

    # -- internal ---------------------------------------------------------
    def _ensure_client(self) -> OpenAI:
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY environment variable not set")
            self._client = OpenAI(api_key=api_key)
        return self._client

    # -- public -----------------------------------------------------------
    def __call__(
        self,
        prompt: str,
        *,
        temperature: float,
        top_k: int | None = None,
        top_p: float | None = None,
        deterministic: bool = False,
    ) -> str:  # pragma: no cover - network
        if deterministic:
            temperature = 0.0
        key = (prompt, temperature, top_k, top_p, deterministic)
        if key in self._cache:
            return self._cache[key]

        client = self._ensure_client()
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                res = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    top_p=top_p,
                )
                text = res.choices[0].message.content or ""
                self._cache[key] = text
                return text
            except (RateLimitError, APIError) as e:
                last_exc = e
                time.sleep(self.retry_wait * (2**attempt))
        raise RuntimeError("OpenAI API request failed after retries") from last_exc

    def ask_yes_no(self, context: str, sentence: str) -> str:  # pragma: no cover - network
        prompt = (
            f"Context: {context}\nSentence: {sentence}\n"
            "Is the sentence supported by the context above?\nAnswer Yes or No:"
        )
        return self(
            prompt,
            temperature=0.0,
            top_k=None,
            top_p=None,
            deterministic=True,
        )


def generate_samples(
    llm: LLM,
    prompts: Iterable[str],
    output_path: str | Path,
    *,
    num_samples: int = 1,
    temperature: float = 0.7,
    top_k: int | None = None,
    top_p: float | None = None,
    deterministic: bool = False,
    cache_dir: str | Path | None = None,
) -> None:
    """Query ``llm`` for each prompt and persist the results.

    Each line in ``output_path`` will contain a JSON object with keys
    ``prompt`` and ``sample`` along with a ``metadata`` field logging the
    sampling parameters and timestamp.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
    else:
        cache_path = None

    logging.info(
        "Generating %d samples per prompt with temperature=%s top_k=%s top_p=%s deterministic=%s",
        num_samples,
        temperature,
        top_k,
        top_p,
        deterministic,
    )

    with path.open("w", encoding="utf-8") as f:
        for prompt in prompts:
            for _ in range(num_samples):
                if cache_path is not None:
                    key_obj = {
                        "prompt": prompt,
                        "temperature": temperature,
                        "top_k": top_k,
                        "top_p": top_p,
                        "deterministic": deterministic,
                    }
                    key = hashlib.sha256(
                        json.dumps(key_obj, sort_keys=True).encode("utf-8")
                    ).hexdigest()
                    cache_file = cache_path / f"{key}.txt"
                    if cache_file.exists():
                        sample = cache_file.read_text(encoding="utf-8")
                    else:
                        sample = llm(
                            prompt,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            deterministic=deterministic,
                        )
                        cache_file.write_text(sample, encoding="utf-8")
                else:
                    sample = llm(
                        prompt,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        deterministic=deterministic,
                    )

                record = {
                    "prompt": prompt,
                    "sample": sample,
                    "metadata": {
                        "temperature": temperature,
                        "top_k": top_k,
                        "top_p": top_p,
                        "deterministic": deterministic,
                        "timestamp": time.time(),
                    },
                }
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
                f.flush()


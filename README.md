# SlefCheckGPT

This repository provides a **light‑weight educational reimplementation** of
several ideas from the paper
[*SelfCheckGPT: Zero‑Resource Black‑Box Hallucination Detection for Generative LLMs*](https://arxiv.org/abs/2305.11617).

The goal is to expose simple, easy to read code rather than to perfectly
reproduce the original results.  The project offers minimal Python
classes for five scoring strategies:

* **BERTScore** – semantic similarity to sampled passages.
* **MQAG** – a tiny proxy for question answering consistency.
* **n‑gram** – unigram language model scoring.
* **NLI** – entailment check via substring matching.
* **LLM Prompt** – ask an external model whether a sentence is supported.

The `run_experiments.py` script can evaluate any of the simplified
metrics on a slice of the WikiBio hallucination dataset.  It mirrors the
evaluation loop of the original project but uses light‑weight stand‑ins so
that the code runs in restricted environments.

## Installation

```bash
pip install -r requirements.txt
```

## Running experiments

By default the script evaluates the n‑gram metric on fifty examples:

```bash
python run_experiments.py
```

To score several metrics at once specify them via ``--metrics`` and
optionally change the number of evaluated examples with ``--limit``:

```bash
python run_experiments.py --metrics ngram mqag nli --limit 25
```

## Running tests

```bash
pytest -q
```

The code is intentionally compact and designed for instructional
purposes.  It should serve as a starting point for more complete
replications of the original SelfCheckGPT system.

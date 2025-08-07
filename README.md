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

A small demonstration script `run_experiments.py` evaluates the n‑gram
metric on a subset of the WikiBio hallucination dataset.

## Installation

```bash
pip install -r requirements.txt
```

## Running the example experiment

```bash
python run_experiments.py
```

## Running tests

```bash
pytest -q
```

The code is intentionally compact and designed for instructional
purposes.  It should serve as a starting point for more complete
replications of the original SelfCheckGPT system.

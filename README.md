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
* **NLI** – entailment check using a pretrained NLI model.
* **LLM Prompt** – ask an external model whether a sentence is supported.

The `run_experiments.py` script can evaluate any of the simplified
metrics on a slice of the WikiBio hallucination dataset.  It mirrors the
evaluation loop of the original project but uses light‑weight stand‑ins so
that the code runs in restricted environments.

## Installation

The project relies on the WikiBio hallucination dataset and spaCy's English
model. After installing the Python dependencies, download these resources:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "from data.utils import load_wikibio_hallucination; load_wikibio_hallucination(split='train[:1]')"
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

## LLM configuration

Some features such as sample generation (`--resample`) and the `prompt` metric
require access to an external language model.  The code uses OpenAI's Chat
Completions API and expects the API key to be available via the
`OPENAI_API_KEY` environment variable.  The model can be selected with
`--llm-model`:

```bash
export OPENAI_API_KEY=sk-YOUR_KEY
python run_experiments.py --metrics prompt --llm-model gpt-3.5-turbo
```

The same model is reused for both sample generation and Yes/No judgements.  The
`--temperature` flag controls sampling temperature.

## Running tests

```bash
pytest -q
```

The code is intentionally compact and designed for instructional
purposes.  It should serve as a starting point for more complete
replications of the original SelfCheckGPT system.

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
metrics on the WikiBio hallucination dataset.  It mirrors the evaluation
loop of the original project but uses light‑weight stand‑ins so that the
code runs in restricted environments.  The script now supports running on
multiple dataset splits and allows configuration of how many sampled
passages per prompt are used for scoring.  Sampling parameters such as
temperature, top‑k/top‑p cut‑offs and deterministic mode can be
configured and swept over.

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

Use ``--train-split``, ``--val-split`` and ``--test-split`` to select dataset
partitions for training the combiner, optional validation, and final
evaluation.  Splits accept the usual Hugging Face slicing syntax:

```bash
python run_experiments.py --train-split train[:1000] --val-split validation[:200] --test-split test --metrics all --sample-count 20
```

The ``--sample-count`` flag controls how many sampled passages per
prompt are used.  When ``--resample`` is given this number of samples is
regenerated with the lightweight sampling pipeline.  Otherwise the
pre‑computed samples in the dataset are truncated to the requested count.
By default the script now follows the paper and uses 20 samples per
prompt.

The logistic‑regression combiner is trained on the specified training
split and evaluated on the test split.  To mirror the exact settings
from the paper, pass ``--paper-config`` which enables resampling, sets
``--sample-count`` to 20 and applies the original top‑k/top‑p cut‑offs.

Every run writes a ``summary.csv`` file and generates precision/recall
and calibration plots for each metric, reproducing the statistics
reported in the paper (precision, recall, F1, average precision, Brier
score and calibration curves).

## LLM configuration

Some features such as sample generation (`--resample`) and the `prompt` metric
require access to an external language model.  The code uses OpenAI's Chat
Completions API and expects the API key to be available via the
`OPENAI_API_KEY` environment variable.  The model can be selected with
`--llm-model`:

```bash
export OPENAI_API_KEY=sk-YOUR_KEY
python run_experiments.py --metrics prompt --llm-model gpt-3.5-turbo --top-p 0.9 --top-k 50
```

The same model is reused for both sample generation and Yes/No judgements.
Generation can be tailored with ``--temperature``, ``--top-k`` and
``--top-p``.  Passing ``--deterministic`` forces greedy decoding.  The
``--temperatures`` flag accepts multiple values to run a sweep which
stores results for each configuration in a separate directory.  When
``--cache-dir`` is supplied GPU or API based generations are cached on
disk for reproducibility.

## Running tests

```bash
pytest -q
```

The code is intentionally compact and designed for instructional
purposes.  It should serve as a starting point for more complete
replications of the original SelfCheckGPT system.

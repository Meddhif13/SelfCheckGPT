# SlefCheckGPT

This repository provides a **light‑weight educational reimplementation** of
several ideas from the paper
[*SelfCheckGPT: Zero‑Resource Black‑Box Hallucination Detection for Generative LLMs*](https://arxiv.org/abs/2305.11617).

The goal is to expose simple, easy to read code rather than to perfectly
reproduce the original results.  The project offers minimal Python
classes for five scoring strategies:

 For a detailed project status, evolution timeline, and change log, see: [docs/status.md](docs/status.md)

* **BERTScore** – semantic similarity to sampled passages.
* **MQAG** – a tiny proxy for question answering consistency.
* **n‑gram** – unigram language model scoring.
* **NLI** – entailment check using a pretrained NLI model.
* **LLM Prompt** – ask an external model whether a sentence is supported.

## Quick MQAG example

Run the full MQAG pipeline with real HuggingFace models (matching the cached
models under `hf-cache/` in this repo):

```python
from selfcheck_metrics import SelfCheckMQAG

sentences = ["Paris is the capital of Germany."]
samples = [
    "Berlin is the capital of Germany.",
    "Paris is the capital of France.",
]

mqag = SelfCheckMQAG(
    # Local cache paths (preferred for offline):
    g1_model="hf-cache/lmqg__flan-t5-base-squad-qg",
    g2_model="hf-cache/potsawee__t5-large-generation-race-Distractor",
    qa_model="hf-cache/potsawee__longformer-large-4096-answering-race",
    answer_model="hf-cache/potsawee__longformer-large-4096-answerable-squad2",
    # Alternatively, the equivalent HF IDs:
    # g1_model="lmqg/flan-t5-base-squad-qg",
    # g2_model="potsawee/t5-large-generation-race-Distractor",
    # qa_model="potsawee/longformer-large-4096-answering-race",
    # answer_model="potsawee/longformer-large-4096-answerable-squad2",
)

scores, answerability = mqag.predict(sentences, samples)
print(scores[0], answerability[0])
```

The first call downloads the model weights and may take a moment.

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
python -c "from data.utils import load_wikibio_hallucination; load_wikibio_hallucination(split='evaluation[:1]')"
```

## GPU setup (RTX 4060)

Running the heavier metrics on GPU requires a recent NVIDIA driver and a
CUDA‑enabled PyTorch build.  An RTX 4060 supports CUDA 12, which in turn needs
driver version 535 or newer.  The repository ships a helper script that
installs the CUDA 12.1 toolkit, PyTorch with matching CUDA wheels and downloads
all Hugging Face weights used by BERTScore, MQAG and NLI:

```bash
bash scripts/setup_gpu_env.sh
```

The script fetches the following models in advance so that the first run does
not need to contact Hugging Face:

- `roberta-large` for BERTScore
- `lmqg/flan-t5-base-squad-qg`, `potsawee/t5-large-generation-race-Distractor`,
  `potsawee/longformer-large-4096-answering-race` and
  `potsawee/longformer-large-4096-answerable-squad2` for MQAG
- `microsoft/deberta-large-mnli` for NLI

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
evaluation.  The WikiBio hallucination dataset only provides an
``evaluation`` split, so different slices of it can be used for the
individual stages:

```bash
python run_experiments.py --train-split evaluation[:1000] --val-split evaluation[1000:1200] --test-split evaluation[1200:2000] --metrics all --sample-count 20
```

The ``--sample-count`` flag controls how many sampled passages per
prompt are used.  When ``--resample`` is given this number of samples is
regenerated with the lightweight sampling pipeline.  Otherwise the
pre‑computed samples in the dataset are truncated to the requested count.
By default the script now follows the paper and uses 20 samples per
prompt.

The MQAG scorer exposes its disagreement metric and answerability
threshold via ``--mqag-metric`` and ``--mqag-answerability-threshold``.
Passing ``--paper-config`` sets them to the values used in the SelfCheckGPT
paper (``kl`` and ``0.9`` respectively).

The logistic‑regression combiner is trained on the specified training
split and evaluated on the test split.  To mirror the exact settings
from the paper, pass ``--paper-config`` which enables resampling, sets
``--sample-count`` to 20 and applies the original top‑k/top‑p cut‑offs.

Every run writes a ``summary.csv`` file and generates precision/recall
and calibration plots for each metric, reproducing the statistics
reported in the paper (precision, recall, F1, average precision, Brier
score and calibration curves).

To mirror the paper exactly with GPU‑accelerated metrics run:

```bash
python run_experiments.py --metrics ngram bertscore mqag nli --paper-config
```

Results are placed in the directory given by ``--output-dir`` (``results`` by
default).  It will contain ``summary.csv``, per‑metric ``*_pr.png`` and
``*_calibration.png`` plots and, when multiple metrics are combined, the trained
logistic‑regression weights in ``combiner.pt``.

### Threshold tuning (optional)

You can automatically sweep decision thresholds on a chosen split to maximize
F1, then apply those thresholds to the single operating point reported in
``summary.csv`` and saved into ``thresholds.json``:

```bash
python run_experiments.py --metrics ngram nli bertscore prompt \
    --limit 200 --output-dir results/gpu_demo --tune-thresholds --tune-split train
```

Average precision (AP) and PR curves are unaffected by thresholds; only the
reported precision/recall/F1 at the single operating point use the tuned value.

## Offline and GPU usage

All heavy Hugging Face models can run fully offline once cached. To force
offline mode, set:

```powershell
$env:HF_HOME=(Resolve-Path .\hf-cache).Path
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
```

For BERTScore, run once online with ``--bertscore-model roberta-large`` to let
the library fetch baseline rescaling stats. After that, you can switch back to
offline using the cached files (or pass ``--bertscore-no-baseline`` to disable
rescaling entirely).

To use GPU, ensure PyTorch detects CUDA and export:

```powershell
$env:CUDA_VISIBLE_DEVICES='0'
```

Recommended GPU run (offline) using local cache:

```powershell
.\.venv\Scripts\Activate
$env:HF_HOME=(Resolve-Path .\hf-cache).Path
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:CUDA_VISIBLE_DEVICES='0'

python run_experiments.py --metrics ngram nli bertscore prompt --limit 200 `
    --output-dir results\gpu_demo --deterministic `
    --nli-batch-size 16 --nli-max-length 160 `
    --bertscore-model hf-cache\roberta-large `
    --prompt-backend hf `
    --prompt-hf-model hf-cache\lmqg__flan-t5-base-squad-qg `
    --prompt-hf-task text2text-generation --prompt-hf-device cuda --prompt-hf-max-new-tokens 24 `
    --tune-thresholds --tune-split train
```

## Results (sample)

### Final checked-in demo (results/)

These artifacts are included in the repo for a minimal, reproducible demo:

- Files: `results/summary.csv`, `ngram_pr.png`, `ngram_calibration.png`,
    `combined_pr.png`, `combined_calibration.png`, and `combiner.pt`.
- Metrics from `results/summary.csv` (rounded):
    - ngram — AP 0.8295, Brier 0.2671, F1 0.8436, Precision 0.7296, Recall 1.00
    - combined — AP 0.8294, Brier 0.1973, F1 0.8436, Precision 0.7296, Recall 1.00

Use these as a quick reference and to validate plotting/calibration. For larger
or GPU/offline runs, see the examples below.

### GPU Offline Run (results/gpu_demo)

This run used all metrics (ngram, NLI, BERTScore, prompt) on 200 examples, fully offline and GPU-accelerated. All model weights were loaded from the local Hugging Face cache. Plots and summary statistics are in `results/gpu_demo`:

- `summary.csv`: Per-metric and combined scores (average precision, Brier, F1, precision, recall)
- `*_pr.png`, `*_calibration.png`: Precision-recall and calibration curves for each metric and the combiner
- `combiner.pt`: Trained logistic regression weights for the combined metric

#### Example metrics (limit=200, offline, GPU)

| Metric     | AP     | Brier  | F1    | Precision | Recall |
|------------|--------|--------|-------|-----------|--------|
| ngram      | varies | varies | varies| varies    | varies |
| nli        | varies | varies | varies| varies    | varies |
| bertscore  | varies | varies | varies| varies    | varies |
| prompt     | varies | varies | varies| varies    | varies |
| combined   | varies | varies | varies| varies    | varies |

Plots:
- `results/gpu_demo/combined_pr.png`, `results/gpu_demo/combined_calibration.png`
- Per-metric: `ngram_pr.png`, `nli_pr.png`, `bertscore_pr.png`, `prompt_pr.png` (and calibration variants)

#### Workflow evolution
- Added CLI flags for offline mode, local model paths, and GPU device selection
- Implemented threshold tuning (`--tune-thresholds`) for optimal F1 at a single operating point
- Documented all steps and results in README for reproducibility

See the Changelog for a full list of recent improvements.

### Smoke‑tuned Run (results/gpu_demo_smoke_tuned)

This smaller run demonstrates threshold tuning and verbose logging end‑to‑end.

- Command (PowerShell):
    ```powershell
    .\.venv\Scripts\Activate
    $env:HF_HOME=(Resolve-Path .\hf-cache).Path
    $env:HF_HUB_OFFLINE='1'
    $env:TRANSFORMERS_OFFLINE='1'
    $env:CUDA_VISIBLE_DEVICES='0'

    python run_experiments.py --metrics ngram nli bertscore prompt --limit 20 `
        --output-dir results\gpu_demo_smoke_tuned --deterministic `
        --nli-batch-size 16 --nli-max-length 160 `
        --bertscore-model hf-cache\roberta-large `
        --prompt-backend hf `
        --prompt-hf-model hf-cache\lmqg__flan-t5-base-squad-qg `
        --prompt-hf-task text2text-generation --prompt-hf-device cuda --prompt-hf-max-new-tokens 24 `
        --tune-thresholds --tune-split train `
        --verbose
    ```

- Artifacts: `summary.csv` (includes per‑metric threshold column), `thresholds.json`, per‑metric and combined PR/calibration plots, `combiner.pt`.

- Tuned thresholds will be saved to `thresholds.json` and reflected in
    `summary.csv` under the `threshold` column. Exact values vary with the
    sample and tuning split.

Reproducing: rerun the command above; for larger runs, increase `--limit` and reuse `--tune-thresholds` and `--verbose`.

A small offline GPU run (limit=5) produced per‑metric PR/calibration plots and
``summary.csv`` in ``results/gpu_demo``. After a one‑time online BERTScore run
to cache baselines, BERTScore reported non‑zero P/R at the single operating
point, and the combined model achieved strong AP and calibration on the sample.
For robust conclusions, increase ``--limit`` (e.g., 200–1000) and inspect
``combined_pr.png`` and ``combined_calibration.png``.

## Changelog

- Added Hugging Face prompt backend (local pipelines) and device control.
- Added offline support and local model path flags (BERTScore, MQAG, NLI).
- Added GPU optimizations and batching flags for heavy models.
- Added BERTScore baseline rescaling handling and guidance for offline usage.
- Added threshold tuning (``--tune-thresholds``/``--tune-split``) with persisted
    ``thresholds.json`` and reporting of the applied threshold in ``summary.csv``.

## LLM configuration

Some features such as sample generation (`--resample`) and the `prompt` metric
can use an external OpenAI model or a local HuggingFace model.

- OpenAI backend: Set `OPENAI_API_KEY` and pass `--llm-model`. We validated
    online runs with `gpt-4o-mini`; the code default is a placeholder, so pass
    your model explicitly.
- Local HF backend: Use `--prompt-backend hf` and provide a local or hub model
    (e.g., FLAN‑T5). This requires no API key and works offline once cached.

```bash
export OPENAI_API_KEY=sk-YOUR_KEY
python run_experiments.py --metrics prompt --llm-model gpt-4o-mini --top-p 0.9 --top-k 50
```

The same model is reused for both sample generation and Yes/No judgements (for
the OpenAI backend). Generation can be tailored with ``--temperature``, ``--top-k`` and
``--top-p``.  Passing ``--deterministic`` forces greedy decoding.  The
``--temperatures`` flag accepts multiple values to run a sweep which
stores results for each configuration in a separate directory.  When
``--cache-dir`` is supplied GPU or API based generations are cached on
disk for reproducibility.

Note on prompt HF backend defaults: if you select `--prompt-backend hf` without
providing `--prompt-hf-model`, the script uses `sshleifer/tiny-gpt2` for a quick
smoke test. For realistic prompt scoring, pass a stronger model such as
`lmqg/flan-t5-base-squad-qg` (or the equivalent local cache path under
`hf-cache/`).

## Running tests

```bash
pytest -q
```

The code is intentionally compact and designed for instructional
purposes.  It should serve as a starting point for more complete
replications of the original SelfCheckGPT system.

## Testing status

- Environment: Python 3.11/3.13 (Windows), PyTorch with CUDA when available
- Current result: full test suite passes (35 passed, 4 skipped, 0 failed)
- Scope covered by tests:
    - MQAG pipeline (stubbed models) with disagreement and answerability stats
    - NLI scorer (HF model path and pure-stub path), including temperature calibration
    - BERTScore scorer integration (skipped if model unavailable)
    - n‑gram scorer for multiple n and smoothing setups
    - Prompt scorer (OpenAI/huggingface backends, caching, normalization)

For reproducible local runs on Windows PowerShell:

```powershell
& .\.venv\Scripts\Activate
pytest -q
```

## Docs

- Project status, evolution, and change log: [docs/status.md](docs/status.md)
- Quick MQAG example and usage: see the top of this README
- Experiments and GPU/offline guidance: "Running experiments", "Offline and GPU usage"

## Overnight runs (online + offline)

Use the orchestrator to run online (OpenAI) and offline (HF) experiments and save a transcript plus a manifest to a timestamped directory.

- Offline only:

```powershell
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File .\scripts\overnight.ps1 -SkipOpenAI
```

- Online + offline:

```powershell
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File .\scripts\overnight.ps1 -OpenAIModel gpt-4o-mini
```

Outputs are saved under `results/overnight/<timestamp>/` with:
- `transcript_<timestamp>.log`: full PowerShell transcript
- `manifest.json`, `run_config.json`: step outcomes and configuration
- `env/`: captured Python and GPU info
- `online/` and `offline/` subfolders containing per-run artifacts

Latest example (2025-08-14 18:42:13) under `results/overnight/20250814_184213/`:

- Offline GPU smoke-tuned (limit=20): `offline/gpu_demo_smoke_tuned/summary.csv`
    - ngram: AP 0.8137, Brier 0.2650, F1 0.8551, P 0.7563, R 0.9837, Thr 0.9915
    - nli: AP 0.8680, Brier 0.4667, F1 0.8521, P 0.7516, R 0.9837, Thr 0.00326
    - bertscore: AP 0.7334, Brier 0.7240, F1 0.8502, P 0.7439, R 0.9919, Thr 0.00313
    - prompt: AP 0.7542, Brier 0.2626, F1 0.8511, P 0.7547, R 0.9756, Thr 0.9750
    - combined: AP 0.8710, Brier 0.1893, F1 0.8592, P 0.7578, R 0.9919, Thr 0.7084

- Online (OpenAI gpt-4o-mini):
    - prompt_smoke_gpt-4o-mini (limit=20): `online/prompt_smoke_gpt-4o-mini/summary.csv`
        - prompt: AP 0.9340, Brier 0.1203, F1 0.9120, P 0.8976, R 0.9268
        - combined: AP 0.9340, Brier 0.1682, F1 0.8454, P 0.7321, R 1.0000
    - resample_smoke_gpt-4o-mini (limit=20): `online/resample_smoke_gpt-4o-mini/summary.csv`
        - ngram: AP 0.7498, Brier 0.2640, F1 0.8454, P 0.7321, R 1.0000
        - combined: AP 0.7498, Brier 0.1961, F1 0.8454, P 0.7321, R 1.0000
    - combined_100_gpt-4o-mini (limit=100): `online/combined_100_gpt-4o-mini/summary.csv`
        - ngram: AP 0.6717, Brier 0.2466, F1 0.8577, P 0.7509, R 1.0000
        - prompt: AP 0.7578, Brier 0.6375, F1 0.3294, P 0.7840, R 0.2085
        - combined: AP 0.6940, Brier 0.1873, F1 0.8577, P 0.7509, R 1.0000

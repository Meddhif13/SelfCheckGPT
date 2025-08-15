# Project status and change log

This document summarizes the repository’s current state, evolution, issues encountered, and how they were resolved. It also highlights differences vs. the original SelfCheckGPT repository.

## Summary
- Goal: lightweight, educational reimplementation of SelfCheckGPT metrics with clean APIs and offline/GPU-friendly execution.
- Current status: full test suite green (25/25). Heavy models support offline mode using the local `hf-cache` directory. GPU execution recommended for NLI and MQAG.

## Evolution timeline (high level)
1) Context and setup
   - Read the repo/paper and original SelfCheckGPT to define scope and interfaces.
   - Added secure key discovery for OpenAI (env var, file, .env) and guidance in README.

2) Early runs and API reliability issues
   - Prompt metric via OpenAI intermittently failed with HTTP 520 errors (network/edge instability).
   - Verified keys via curl and Python; issues persisted even after network changes.

3) Offline-first pivot and HF backends
   - Added Hugging Face prompt backend (local pipelines) and device control.
   - Ensured all heavy models use local `hf-cache` and work offline (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`).
   - Enabled batching and GPU usage for NLI and MQAG; added flags for batch size and max sequence length.

4) Metric hardening and CLI improvements
   - Added threshold tuning (`--tune-thresholds`, `--tune-split`) and verbose logging (`--verbose`).
   - Improved plotting and artifact generation (`summary.csv`, PR/calibration PNGs, `combiner.pt`).
   - Documented a GPU offline run and a smoke-tuned run with thresholds.

5) Test reliability and stub compatibility
   - Fixed several stub-related issues in MQAG and NLI:
     - MQAG: missing tokenizer token attributes (e.g., `bos_token`) and flexible output parsing.
     - NLI: tokenizer/model signature mismatches and temperature calibration path.
   - Result: all tests pass (including stubbed MQAG/NLI and calibration tests).

## Issues encountered and resolutions

1) OpenAI prompt metric failures (HTTP 520)
   - Symptom: frequent 520 errors during chat completions; persisted across key rotation and network change.
   - Resolution:
     - Implemented backoff and fallback to stable models in the OpenAI client.
     - Added an HF local pipeline option for the prompt metric (`--prompt-backend hf`), enabling fully offline runs.

2) NLI temperature calibration and stub signature errors
   - Symptom: TypeError in tests expecting specific tokenizer call signature; logits handling inconsistent.
   - Resolution:
     - Rewrote `SelfCheckNLI.predict` to cleanly separate three paths:
       (a) HF model + tokenizer batching, (b) stubbed model path using `premise/hypothesis`, (c) pure function `nli_fn`.
     - Ensured temperature scaling and `return_logits=True` work consistently.

3) MQAG tokenizer attributes and parsing
   - Symptom: AttributeError on missing `bos_token`; fragile QA/distractor parsing in stub context.
   - Resolution:
     - Defensive token cleanup using `getattr(..., None)` and safe replacements.
     - More robust question/answer parsing with fallbacks; ensured minimum viable options.

4) BERTScore offline baseline
   - Symptom: BERTScore required baseline rescaling stats not present offline.
   - Resolution:
     - Attempt baseline rescaling; on failure, automatically retry without baseline and log a warning.
     - Documented how to fetch baselines once online, then run offline thereafter.

5) Performance issues (NLI on CPU)
   - Symptom: NLI was slow on CPU for moderate batch sizes.
   - Resolution:
     - Added GPU support, batch size, and max length flags; documented recommended GPU run.

## Current functionality (high level)
- `selfcheck_metrics.py`
  - SelfCheckPrompt: OpenAI or HF pipeline backend, normalization/mapping hooks, caching, temperature scaling.
  - SelfCheckNLI: DeBERTa-large-MNLI by default, batching, GPU support, temperature scaling, robust stub path.
  - SelfCheckMQAG: QG + distractor + MCQA + answerability; supports local HF paths and test stubs.
  - SelfCheckBERTScore: `bert_score` first, transformer fallback if needed; baseline handling for offline.
  - SelfCheckNgram: backoff and Kneser–Ney; optional corpus merge.
- `run_experiments.py`
  - CLI for choosing metrics, splits, sample counts; offline/GPU flags; threshold tuning; plots and CSV output.
- `selfcheckgpt/utils.py`
  - Centralized HF cache paths and offline env; helper tokenization and model path configs.

## Differences vs. original SelfCheckGPT
- Scope: compact, instructional subset prioritizing readability and offline execution.
- MQAG: simplified generation/answering pipeline; aims to mirror behavior but not exact outputs.
- Prompt metric: supports an HF local backend in addition to OpenAI.
- Combiner: simple logistic regression; results may differ from paper.

## Testing status
- Passed: 25/25 tests covering MQAG (stubbed), NLI (HF and stub paths), BERTScore integration (skips if unavailable), n-gram scoring, and prompt metric behaviors.
- Notes: tests validate calibration routines and threshold application; full paper-scale replication is out of scope.

## Reproduction and environment
- Offline mode: use local `hf-cache` and set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.
- GPU: recommended for NLI and MQAG; set `CUDA_VISIBLE_DEVICES` appropriately.
- Windows PowerShell quick start:

```powershell
& .\.venv\Scripts\Activate
$env:HF_HOME=(Resolve-Path .\hf-cache).Path
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
pytest -q
```

## Known limitations
- Exact numerical parity with the paper is not guaranteed due to simplified pipelines.
- BERTScore baseline files must be available for rescaling; otherwise we fall back without baseline.
- OpenAI backend reliability depends on external service; HF backend is recommended for offline stability.

## Next steps (optional)
- Add larger, end-to-end benchmarks with artifact snapshots.
- Expand MQAG parsing robustness and add more tests for edge cases.
- Provide a small CLI for per-document multi-metric scoring.

## Overnight runs (2025-08-14)

We added `scripts/overnight.ps1` to run online (OpenAI) and offline (HF) sweeps with a transcript, manifest, and environment snapshot. Example timestamped output directory: `results/overnight/20250814_184213/`.

Offline GPU smoke (limit=20) in `offline/gpu_demo_smoke_tuned/summary.csv`:
- ngram: AP 0.8137, Brier 0.2650, F1 0.8551
- nli: AP 0.8680, Brier 0.4667, F1 0.8521
- bertscore: AP 0.7334, Brier 0.7240, F1 0.8502 (baseline disabled offline)
- prompt: AP 0.7542, Brier 0.2626, F1 0.8511
- combined: AP 0.8710, Brier 0.1893, F1 0.8592

Online runs (gpt-4o-mini):
- `online/prompt_smoke_gpt-4o-mini` (limit=20): prompt AP 0.9340; combined AP 0.9340
- `online/resample_smoke_gpt-4o-mini` (limit=20): ngram AP 0.7498; combined AP 0.7498
- `online/combined_100_gpt-4o-mini` (limit=100): combined AP 0.6940; ngram AP 0.6717; prompt AP 0.7578

Notes:
- At the end of the run, README generation attempted to compute relative paths for all manifest entries. The `openai_ping` entry doesn’t have an `outDir`, which caused a non-fatal Resolve-Path null argument error during README creation. We hardened the script to skip entries without `outDir` and the next runs won’t show this error. All experiments completed successfully and artifacts were saved.

---
This document will be updated as the repository evolves to track deltas and maintainers’ decisions.

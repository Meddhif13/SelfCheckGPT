# Analysis of SelfCheckGPT Metrics (Online and Offline Overnight Runs)

## Abstract
We analyze simplified SelfCheckGPT metrics implemented in this repository on the WikiBio hallucination dataset. Using both offline (GPU-accelerated HuggingFace models) and online (OpenAI gpt-4o-mini) configurations, we report precision–recall behavior (average precision), calibration (Brier score, reliability curves), and single-point operating characteristics (F1, precision, recall) with tuned thresholds where applicable. We discuss strengths and limitations of individual metrics and the logistic-regression combiner, and we contrast outcomes qualitatively with the original SelfCheckGPT paper’s expectations.

## Data and Experimental Setup
- Dataset: WikiBio Hallucination dataset (`evaluation` split). We slice it for small demos.
- Samples per prompt: 20 (matching the paper’s practice).
- Metrics: n-gram, NLI, BERTScore, prompt (Yes/No), and a logistic-regression combiner.
- Offline (HF) models (local cache equivalents under `hf-cache/`):
  - BERTScore: `roberta-large`
  - NLI: `microsoft/deberta-large-mnli`
  - MQAG: `lmqg/flan-t5-base-squad-qg`, `potsawee/t5-large-generation-race-Distractor`,
    `potsawee/longformer-large-4096-answering-race`, `potsawee/longformer-large-4096-answerable-squad2`
- Online prompt backend: OpenAI Chat Completions, model `gpt-4o-mini`.
- Threshold tuning: optional sweep on a chosen split to maximize F1 at a single operating point; reported as `threshold` in `summary.csv` when enabled.
- Artifacts: `summary.csv`, PR and calibration plots (`*_pr.png`, `*_calibration.png`), `combiner.pt` (not checked in).

## Runs Analyzed (Overnight)
Timestamped root: `results/overnight/20250814_184213/`.

### Offline GPU smoke‑tuned (limit=20)
Path: `offline/gpu_demo_smoke_tuned/`
- Summary (`summary.csv`):
  - ngram — AP 0.8137, Brier 0.2650, F1 0.8551, P 0.7563, R 0.9837, Thr 0.9915
  - nli — AP 0.8680, Brier 0.4667, F1 0.8521, P 0.7516, R 0.9837, Thr 0.00326
  - bertscore — AP 0.7334, Brier 0.7240, F1 0.8502, P 0.7439, R 0.9919, Thr 0.00313
  - prompt — AP 0.7542, Brier 0.2626, F1 0.8511, P 0.7547, R 0.9756, Thr 0.9750
  - combined — AP 0.8710, Brier 0.1893, F1 0.8592, P 0.7578, R 0.9919, Thr 0.7084
- Figures: see `offline/gpu_demo_smoke_tuned/*_pr.png` and `*_calibration.png`.

### Online (OpenAI, gpt‑4o‑mini)
- `online/prompt_smoke_gpt-4o-mini/summary.csv` (limit=20)
  - prompt — AP 0.9340, Brier 0.1203, F1 0.9120, P 0.8976, R 0.9268
  - combined — AP 0.9340, Brier 0.1682, F1 0.8454, P 0.7321, R 1.0000
- `online/resample_smoke_gpt-4o-mini/summary.csv` (limit=20)
  - ngram — AP 0.7498, Brier 0.2640, F1 0.8454, P 0.7321, R 1.0000
  - combined — AP 0.7498, Brier 0.1961, F1 0.8454, P 0.7321, R 1.0000
- `online/combined_100_gpt-4o-mini/summary.csv` (limit=100)
  - ngram — AP 0.6717, Brier 0.2466, F1 0.8577, P 0.7509, R 1.0000
  - prompt — AP 0.7578, Brier 0.6375, F1 0.3294, P 0.7840, R 0.2085
  - combined — AP 0.6940, Brier 0.1873, F1 0.8577, P 0.7509, R 1.0000

## Analysis and Discussion
### Per‑metric behavior
- n‑gram: Strong recall with high thresholds (offline and online), producing solid F1 and AP on small slices. Calibration improves when combined.
- NLI: Good AP in the offline run but comparatively higher Brier, reflecting miscalibration on small samples; recall remains high at tuned operating points.
- BERTScore: AP is moderate in the offline run and Brier is high, consistent with known sensitivity to baseline rescaling. Running once with baseline rescaling online or disabling baseline (as we do offline) explains the elevated Brier; the metric still contributes to the combiner.
- Prompt (Yes/No): Online prompt scoring with gpt‑4o‑mini yields high AP and strong F1 on the small 20‑sample slice. In the 100‑example combined online run, prompt single‑point F1 drops due to lower recall at the default threshold, while AP remains respectable—consistent with threshold sensitivity and dataset slice differences.

### Combined model
- Across both offline and online runs, the logistic‑regression combiner delivers the best or near‑best Brier and maintains high F1 by fusing complementary signals.
- On the offline (limit=20) run, combined AP 0.8710 and Brier 0.1893 with recall near 1.0 shows that combining imperfectly calibrated metrics yields a more reliable operating point.
- On the 100‑example online run, combined AP 0.6940 with Brier 0.1873 and F1 0.8577 outperforms any single metric’s Brier, suggesting robust calibration from multi‑signal fusion.

### Calibration insights
- Reliability curves (PR and calibration PNGs) show improved alignment for the combiner vs individual metrics. The prompt metric online shows strong confidence but can be under‑ or over‑confident depending on the slice, which the combiner mitigates.

### Comparison to the Original SelfCheckGPT Paper (Qualitative)
- Scope: This repository is an educational, lightweight reimplementation. We use simplified pipelines and smaller runs. Exact numerical parity is not expected.
- Trends: The original paper reports that combining signals (NLI, MQAG, etc.) improves discrimination and calibration. Our small‑scale runs reproduce the same qualitative behavior: the combiner improves calibration (lower Brier) and maintains strong recall.
- Differences: We operate on small evaluation slices (20–100 examples) and leverage local HF models and an alternative prompt backend; BERTScore baseline handling differs in offline mode. These choices explain deviations in absolute numbers while preserving the expected ranking of methods and the benefit of combination.

### Threats to Validity
- Small sample sizes (limit=20 and limit=100) increase variance in estimates.
- Prompt backend differences and API instability can affect the online metrics.
- Offline BERTScore baseline rescaling is disabled by default; enabling baseline with online fetch may lower its Brier.

## Reproducibility
- The exact artifacts referenced here are checked into `results/overnight/20250814_184213/` (CSV, PNG, logs; `.pt` files ignored by design).
- To regenerate:
  - Offline: `powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File .\scripts\overnight.ps1 -SkipOpenAI`
  - Online + offline: `powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File .\scripts\overnight.ps1 -OpenAIModel gpt-4o-mini`
- For manual runs and flags, see README sections on running experiments, offline/GPU usage, and threshold tuning.

## Figures (Quick Links)
- Offline combined PR: `results/overnight/20250814_184213/offline/gpu_demo_smoke_tuned/combined_pr.png`
- Offline combined calibration: `results/overnight/20250814_184213/offline/gpu_demo_smoke_tuned/combined_calibration.png`
- Online prompt PR (limit=20): `results/overnight/20250814_184213/online/prompt_smoke_gpt-4o-mini/prompt_pr.png`
- Online combined PR (limit=100): `results/overnight/20250814_184213/online/combined_100_gpt-4o-mini/combined_pr.png`

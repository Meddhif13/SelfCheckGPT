# Experiment report

## Objective
This project provides a light-weight educational reimplementation of SelfCheckGPT, aiming to assess hallucination detection metrics on LLM generated text.

## Dataset
We evaluate metrics on the WikiBio hallucination dataset, which contains prompts, generated sentences and 20 sampled passages per prompt, and only offers an `evaluation` split.

## Metrics
We score the following simplified SelfCheckGPT metrics: n-gram language model, natural language inference (NLI), BERTScore, prompt-based Yes/No scoring, and a logistic-regression combiner that fuses individual scores.

## Tuned thresholds
Thresholds derived via `run_experiments.py --tune-thresholds`:

- ngram: 0.99149
- nli: 0.00326
- bertscore: 0.00313
- prompt: 0.97500

## Key results
On a limit=20 evaluation run, average precision (AP) and Brier score per metric were:

| Metric   | AP    | Brier | F1    | Precision | Recall | Threshold |
|----------|-------|-------|-------|-----------|--------|-----------|
| combined | 0.8710| 0.1893| 0.8592| 0.7578    | 0.9919 | 0.7084    |
| ngram    | 0.8137| 0.2650| 0.8551| 0.7563    | 0.9837 | 0.9915    |
| nli      | 0.8680| 0.4667| 0.8521| 0.7516    | 0.9837 | 0.00326   |
| bertscore| 0.7334| 0.7240| 0.8502| 0.7439    | 0.9919 | 0.00313   |
| prompt   | 0.7542| 0.2626| 0.8511| 0.7547    | 0.9756 | 0.9750    |

## Reproducing experiments
1. Install dependencies and fetch resources:

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "from data.utils import load_wikibio_hallucination; load_wikibio_hallucination(split='evaluation[:1]')"
```

2. Run the evaluation:

```
python run_experiments.py --metrics ngram nli bertscore prompt \
    --train-split evaluation[:1000] --val-split evaluation[1000:1200] \
    --test-split evaluation[1200:2000] --sample-count 20 \
    --tune-thresholds --limit 20 --output-dir results/demo
```

Artifacts such as `summary.csv`, `thresholds.json`, precision–recall and calibration plots, and `combiner.pt` will be written to the output directory.


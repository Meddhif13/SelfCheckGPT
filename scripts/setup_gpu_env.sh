#!/usr/bin/env bash
set -euo pipefail

# Reinstall PyTorch with CUDA 12.1 wheels
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121


# Pre-download Hugging Face models (as in the original script)
python3 - <<'PY'
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    LongformerForMultipleChoice,
)

models = {
    "roberta-large": AutoModel,
    "potsawee/t5-base-squad-qg": AutoModelForSeq2SeqLM,
    "potsawee/t5-base-distractor-generation": AutoModelForSeq2SeqLM,
    "potsawee/longformer-large-4096-mc-squad2": LongformerForMultipleChoice,
    "potsawee/longformer-large-4096-answerable-squad2": AutoModelForSequenceClassification,
    "microsoft/deberta-v3-large-mnli": AutoModelForSequenceClassification,
}

for name, cls in models.items():
    print(f"Downloading {name}...")
    cls.from_pretrained(name)
    AutoTokenizer.from_pretrained(name)
PY

echo "GPU environment setup complete."


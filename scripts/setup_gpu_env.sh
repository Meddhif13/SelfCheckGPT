#!/usr/bin/env bash
# Sample environment setup for running SelfCheckGPT metrics on an RTX 4060.
# Installs CUDA Toolkit 12.1, matching PyTorch wheels and downloads
# required Hugging Face model weights.

set -euo pipefail

# Install NVIDIA driver and CUDA Toolkit (Ubuntu 22.04 example)
sudo apt-get update
sudo apt-get install -y nvidia-driver-535
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-1

# CUDA environment variables
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Pre-download required Hugging Face model weights
python - <<'PY'
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

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$PROJECT_ROOT"

uv venv .venv_trl --python 3.10
source .venv_trl/bin/activate

uv pip install -U pip setuptools wheel packaging ninja

uv pip install \
  torch==2.6.0 \
  torchvision==0.21.0 \
  torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

uv pip install \
  "transformers>=4.57,<5" \
  "accelerate>=1.6,<2" \
  "datasets>=3.5,<4" \
  "peft>=0.15,<1" \
  "trl @ git+https://github.com/huggingface/trl.git" \
  "vllm" \
  "kernels" \
  pandas pyarrow swanlab tqdm

python - <<'PY'
import torch
import transformers
import peft
import trl
from trl.experimental.distillation import DistillationTrainer
import vllm
import swanlab

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), torch.cuda.device_count())
print("transformers:", transformers.__version__)
print("trl:", trl.__version__)
print("trl distillation: ok", DistillationTrainer.__name__)
print("vllm:", vllm.__version__)
print("swanlab: ok")

assert torch.cuda.is_available(), "CUDA is not available"
assert torch.__version__.endswith("+cu124"), f"Unexpected torch build: {torch.__version__}"
PY

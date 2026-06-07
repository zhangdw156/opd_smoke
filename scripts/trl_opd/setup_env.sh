#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$PROJECT_ROOT"

export UV_PROJECT_ENVIRONMENT="$PROJECT_ROOT/.venv_trl"
uv sync --python 3.10
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

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
PY

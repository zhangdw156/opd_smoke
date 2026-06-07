#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$PROJECT_ROOT/configs/env.sh"

cd "$PROJECT_ROOT"
source .venv_trl/bin/activate

export CUDA_VISIBLE_DEVICES=${TEACHER_CUDA_VISIBLE_DEVICES:-1}
HOST=${TEACHER_SERVER_HOST:-127.0.0.1}
PORT=${TEACHER_SERVER_PORT:-8000}
TP_SIZE=${TEACHER_TENSOR_PARALLEL_SIZE:-1}
GPU_UTIL=${TEACHER_GPU_MEMORY_UTILIZATION:-0.85}
MAX_MODEL_LEN=${TEACHER_MAX_MODEL_LEN:-1024}

if [[ ! -f "$TEACHER_MODEL/config.json" ]]; then
  echo "ERROR: teacher model not found: $TEACHER_MODEL" >&2
  exit 1
fi

printf 'Teacher model:   %s\n' "$TEACHER_MODEL"
printf 'CUDA devices:    %s\n' "$CUDA_VISIBLE_DEVICES"
printf 'Server URL:      http://%s:%s\n' "$HOST" "$PORT"
printf 'Tensor parallel: %s\n' "$TP_SIZE"
printf 'Max model len:   %s\n' "$MAX_MODEL_LEN"

trl vllm-serve \
  --model "$TEACHER_MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --trust-remote-code

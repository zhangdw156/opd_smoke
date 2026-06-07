#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$PROJECT_ROOT/configs/env.sh"

cd "$PROJECT_ROOT"
source .venv_trl/bin/activate

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export SWANLAB_MODE=${SWANLAB_MODE:-cloud}

RUN_NAME=${RUN_NAME:-qwen25_3b_from_7b_gsm8k_trl_opd_distill}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="$CKPT_ROOT/${RUN_NAME}_${TIMESTAMP}"
LOG_FILE="$LOG_ROOT/${RUN_NAME}_${TIMESTAMP}.log"

NUM_PROCESSES=${NUM_PROCESSES:-1}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-16}
NUM_GENERATIONS=${NUM_GENERATIONS:-1}
GLOBAL_BATCH_SIZE=$((NUM_PROCESSES * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))

mkdir -p "$OUT_DIR" "$LOG_ROOT" "$SWANLAB_LOG_DIR"

if [[ ! -f "$STUDENT_MODEL/config.json" ]]; then
  echo "ERROR: student model not found: $STUDENT_MODEL" >&2
  exit 1
fi

if [[ ! -f "$TEACHER_MODEL/config.json" ]]; then
  echo "ERROR: teacher model not found: $TEACHER_MODEL" >&2
  exit 1
fi

if [[ ! -f "$DATA_ROOT/train.parquet" ]]; then
  echo "ERROR: train parquet not found: $DATA_ROOT/train.parquet" >&2
  exit 1
fi

USE_TEACHER_SERVER=${USE_TEACHER_SERVER:-1}
TEACHER_MODEL_SERVER_URL=${TEACHER_MODEL_SERVER_URL:-http://127.0.0.1:8000}
TEACHER_SERVER_ARGS=()
if [[ "$USE_TEACHER_SERVER" == "1" ]]; then
  TEACHER_SERVER_ARGS=(--use_teacher_server --teacher_model_server_url "$TEACHER_MODEL_SERVER_URL")
fi

SWANLAB_ARGS=()
if [[ "${ENABLE_SWANLAB:-1}" == "1" ]]; then
  SWANLAB_ARGS=(--swanlab)
fi

printf 'Project root:      %s\n' "$OPD_PROJECT_ROOT"
printf 'Student model:     %s\n' "$STUDENT_MODEL"
printf 'Teacher model:     %s\n' "$TEACHER_MODEL"
printf 'Train file:        %s\n' "$DATA_ROOT/train.parquet"
printf 'Output dir:        %s\n' "$OUT_DIR"
printf 'Log file:          %s\n' "$LOG_FILE"
printf 'CUDA devices:      %s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
printf 'Num processes:     %s\n' "$NUM_PROCESSES"
printf 'Per-device batch:  %s\n' "$PER_DEVICE_TRAIN_BATCH_SIZE"
printf 'Grad accum steps:  %s\n' "$GRADIENT_ACCUMULATION_STEPS"
printf 'Global batch size: %s\n' "$GLOBAL_BATCH_SIZE"
printf 'Generation batch:  %s\n' "$GENERATION_BATCH_SIZE"
printf 'Teacher server:    %s %s\n' "$USE_TEACHER_SERVER" "$TEACHER_MODEL_SERVER_URL"

accelerate launch --num_processes "$NUM_PROCESSES" scripts/trl_opd/train.py \
  --student_model "$STUDENT_MODEL" \
  --teacher_model "$TEACHER_MODEL" \
  --train_file "$DATA_ROOT/train.parquet" \
  --output_dir "$OUT_DIR" \
  --project_name "${PROJECT_NAME:-opd_smoke}" \
  --run_name "$RUN_NAME" \
  --max_steps "${MAX_STEPS:-640}" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --generation_batch_size "$GENERATION_BATCH_SIZE" \
  --num_generations "$NUM_GENERATIONS" \
  --max_prompt_length "${MAX_PROMPT_LENGTH:-512}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH:-256}" \
  --lr "${LR:-1e-6}" \
  --temperature "${TEMPERATURE:-1.0}" \
  --top_p "${TOP_P:-0.95}" \
  --top_k "${TOP_K:-0}" \
  --loss_top_k "${LOSS_TOP_K:-1}" \
  --lmbda "${LMBDA:-1.0}" \
  --beta "${BETA:-1.0}" \
  --lora_rank "${LORA_RANK:-16}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --dtype "${DTYPE:-bfloat16}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  --logging_steps "${LOGGING_STEPS:-1}" \
  --save_steps "${SAVE_STEPS:-100}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --report_to "${REPORT_TO:-none}" \
  "${TEACHER_SERVER_ARGS[@]}" \
  "${SWANLAB_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"

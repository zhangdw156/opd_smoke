#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$PROJECT_ROOT/configs/env.sh"

cd "$PROJECT_ROOT"
source .venv_trl/bin/activate

RUN_NAME=${RUN_NAME:-qwen25_3b_from_7b_gsm8k_trl_opd_mvp}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

OUT_DIR="$CKPT_ROOT/${RUN_NAME}_${TIMESTAMP}"
LOG_FILE="$LOG_ROOT/${RUN_NAME}_${TIMESTAMP}.log"

mkdir -p "$OUT_DIR" "$LOG_ROOT" "$SWANLAB_LOG_DIR"

echo "Project root:   $OPD_PROJECT_ROOT"
echo "Student model:  $STUDENT_MODEL"
echo "Teacher model:  $TEACHER_MODEL"
echo "Train file:     $DATA_ROOT/train.parquet"
echo "Output dir:     $OUT_DIR"
echo "Log file:       $LOG_FILE"
echo "SwanLab mode:   ${SWANLAB_MODE:-unset}"
echo "CUDA devices:   ${CUDA_VISIBLE_DEVICES:-unset}"

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

python scripts/train_trl_opd_mvp.py \
  --student_model "$STUDENT_MODEL" \
  --teacher_model "$TEACHER_MODEL" \
  --train_file "$DATA_ROOT/train.parquet" \
  --output_dir "$OUT_DIR" \
  --project_name "${PROJECT_NAME:-opd_smoke}" \
  --run_name "$RUN_NAME" \
  --max_steps "${MAX_STEPS:-20}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH:-512}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-256}" \
  --lr "${LR:-1e-5}" \
  --temperature "${TEMPERATURE:-1.0}" \
  --top_p "${TOP_P:-0.95}" \
  --lora_rank "${LORA_RANK:-16}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --save_every "${SAVE_EVERY:-10}" \
  --student_device "${STUDENT_DEVICE:-cuda:0}" \
  --teacher_device "${TEACHER_DEVICE:-cuda:1}" \
  --seed "${SEED:-42}" \
  2>&1 | tee "$LOG_FILE"

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

python scripts/train_trl_opd_mvp.py \
  --student_model "$STUDENT_MODEL" \
  --teacher_model "$TEACHER_MODEL" \
  --train_file "$DATA_ROOT/train.parquet" \
  --output_dir "$OUT_DIR" \
  --project_name "opd_smoke" \
  --run_name "$RUN_NAME" \
  --max_steps 20 \
  --batch_size 1 \
  --max_prompt_length 512 \
  --max_new_tokens 256 \
  --lr 1e-5 \
  2>&1 | tee "$LOG_FILE"

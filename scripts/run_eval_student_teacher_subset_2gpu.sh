#!/usr/bin/env bash
set -euo pipefail

cd /data/zhangdw12/work/opd_smoke

source configs/env.sh
source .venv_trl/bin/activate

EVAL_LIMIT=${EVAL_LIMIT:-200}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-512}
EVAL_SEED=${EVAL_SEED:-42}
EVAL_RUN_NAME=${EVAL_RUN_NAME:-base_student_teacher_subset}

EVAL_DIR="$RUN_ROOT/eval_${EVAL_RUN_NAME}_$(date +"%Y%m%d_%H%M%S")"
mkdir -p "$EVAL_DIR"

echo "EVAL_DIR=$EVAL_DIR"
echo "EVAL_LIMIT=$EVAL_LIMIT"
echo "EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE"
echo "EVAL_MAX_NEW_TOKENS=$EVAL_MAX_NEW_TOKENS"

echo "[eval] base student on physical GPU 2"
CUDA_VISIBLE_DEVICES=2 python scripts/eval_gsm8k.py \
  --model "$STUDENT_MODEL" \
  --test_file "$DATA_ROOT/test.parquet" \
  --output_file "$EVAL_DIR/student_base.jsonl" \
  --limit "$EVAL_LIMIT" \
  --seed "$EVAL_SEED" \
  --batch_size "$EVAL_BATCH_SIZE" \
  --max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
  --device cuda:0 \
  > "$EVAL_DIR/student_base.log" 2>&1 &

student_pid=$!

echo "[eval] base teacher on physical GPU 3"
CUDA_VISIBLE_DEVICES=3 python scripts/eval_gsm8k.py \
  --model "$TEACHER_MODEL" \
  --test_file "$DATA_ROOT/test.parquet" \
  --output_file "$EVAL_DIR/teacher_base.jsonl" \
  --limit "$EVAL_LIMIT" \
  --seed "$EVAL_SEED" \
  --batch_size "$EVAL_BATCH_SIZE" \
  --max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
  --device cuda:0 \
  > "$EVAL_DIR/teacher_base.log" 2>&1 &

teacher_pid=$!

echo "student_pid=$student_pid"
echo "teacher_pid=$teacher_pid"

wait "$student_pid"
wait "$teacher_pid"

echo "[done] summaries:"
cat "$EVAL_DIR/student_base.summary.json"
echo
cat "$EVAL_DIR/teacher_base.summary.json"
echo
echo "[done] logs and predictions are under $EVAL_DIR"

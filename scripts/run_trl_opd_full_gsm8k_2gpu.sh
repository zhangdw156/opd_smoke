#!/usr/bin/env bash
set -euo pipefail

cd /data/zhangdw12/work/opd_smoke

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

source configs/env.sh
source .venv_trl/bin/activate

export SWANLAB_MODE=${SWANLAB_MODE:-cloud}
export RUN_NAME=${RUN_NAME:-qwen25_3b_from_7b_gsm8k_trl_opd_topk_full_2gpu}
export STUDENT_DEVICE=${STUDENT_DEVICE:-cuda:0}
export TEACHER_DEVICE=${TEACHER_DEVICE:-cuda:1}

export MAX_STEPS=${MAX_STEPS:-$(python - <<'PY'
import pandas as pd
df = pd.read_parquet("/data/zhangdw12/datasets/gsm8k/train.parquet")
print(len(df))
PY
)}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
export SAVE_EVERY=${SAVE_EVERY:-500}
export LORA_RANK=${LORA_RANK:-16}
export LORA_ALPHA=${LORA_ALPHA:-32}
export LR=${LR:-1e-5}
export TEMPERATURE=${TEMPERATURE:-1.0}
export TOP_P=${TOP_P:-0.95}
export TOP_K=${TOP_K:-64}

echo "RUN_NAME=$RUN_NAME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "STUDENT_DEVICE=$STUDENT_DEVICE"
echo "TEACHER_DEVICE=$TEACHER_DEVICE"
echo "MAX_STEPS=$MAX_STEPS"
echo "MAX_PROMPT_LENGTH=$MAX_PROMPT_LENGTH"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "SAVE_EVERY=$SAVE_EVERY"
echo "LORA_RANK=$LORA_RANK"
echo "LORA_ALPHA=$LORA_ALPHA"
echo "LR=$LR"
echo "TEMPERATURE=$TEMPERATURE"
echo "TOP_P=$TOP_P"
echo "TOP_K=$TOP_K"
echo "SWANLAB_MODE=$SWANLAB_MODE"

bash scripts/run_trl_opd_mvp.sh

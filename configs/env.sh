#!/usr/bin/env bash

# ===== Project paths =====
export OPD_PROJECT_ROOT=${OPD_PROJECT_ROOT:-/data/zhangdw12/work/opd_smoke}

# ===== External data/model paths =====
export MODEL_ROOT=${MODEL_ROOT:-/data/zhangdw12/models}
export DATA_ROOT=${DATA_ROOT:-/data/zhangdw12/datasets/gsm8k}

export STUDENT_MODEL=${STUDENT_MODEL:-$MODEL_ROOT/Qwen2.5-3B-Instruct}
export TEACHER_MODEL=${TEACHER_MODEL:-$MODEL_ROOT/Qwen2.5-7B-Instruct}

# ===== Project artifact paths =====
export LOG_ROOT=${LOG_ROOT:-$OPD_PROJECT_ROOT/logs}
export CKPT_ROOT=${CKPT_ROOT:-$OPD_PROJECT_ROOT/checkpoints}
export RUN_ROOT=${RUN_ROOT:-$OPD_PROJECT_ROOT/runs}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-$OPD_PROJECT_ROOT/swanlab}

# ===== Cache paths =====
export HF_HOME=${HF_HOME:-$OPD_PROJECT_ROOT/cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$OPD_PROJECT_ROOT/cache/huggingface/transformers}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$OPD_PROJECT_ROOT/cache/triton}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$OPD_PROJECT_ROOT/cache/xdg}

# ===== Runtime settings =====
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ===== SwanLab =====
export SWANLAB_MODE=${SWANLAB_MODE:-offline}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-$OPD_PROJECT_ROOT/swanlab}

#!/usr/bin/env bash

# ===== Project paths =====
export OPD_PROJECT_ROOT=/data/zhangdw12/work/opd_smoke
export VERL_ROOT=/data/zhangdw12/work/verl

# ===== External data/model paths =====
export MODEL_ROOT=/data/zhangdw12/models
export DATA_ROOT=/data/zhangdw12/datasets/gsm8k

# 修改成你的真实本地模型目录名
export STUDENT_MODEL=${STUDENT_MODEL:-$MODEL_ROOT/Qwen2.5-3B-Instruct}
export TEACHER_MODEL=${TEACHER_MODEL:-$MODEL_ROOT/Qwen2.5-7B-Instruct}

# ===== Project artifact paths =====
export LOG_ROOT=$OPD_PROJECT_ROOT/logs
export CKPT_ROOT=$OPD_PROJECT_ROOT/checkpoints
export RUN_ROOT=$OPD_PROJECT_ROOT/runs
export SWANLAB_LOG_DIR=$OPD_PROJECT_ROOT/swanlab
export RAY_TMPDIR=$OPD_PROJECT_ROOT/ray_tmp

# ===== Cache paths =====
export HF_HOME=$OPD_PROJECT_ROOT/cache/huggingface
export TRANSFORMERS_CACHE=$OPD_PROJECT_ROOT/cache/huggingface/transformers
export TRITON_CACHE_DIR=$OPD_PROJECT_ROOT/cache/triton
export XDG_CACHE_HOME=$OPD_PROJECT_ROOT/cache/xdg

# ===== Runtime settings =====
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

# ===== SwanLab =====
# 在线模式：
# export SWANLAB_API_KEY=你的_swanlab_api_key
# export SWANLAB_MODE=cloud
#
# 离线模式：
export SWANLAB_MODE=${SWANLAB_MODE:-offline}
export SWANLAB_LOG_DIR=$OPD_PROJECT_ROOT/swanlab

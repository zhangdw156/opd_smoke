#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

source "$PROJECT_ROOT/configs/env.sh"

cd "$VERL_ROOT"
source .venv/bin/activate

RUN_NAME=${RUN_NAME:-qwen25_3b_gsm8k_grpo_lora_smoke_v071}
PROJECT_NAME=${PROJECT_NAME:-opd_smoke_v071}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

RUN_DIR="$RUN_ROOT/${RUN_NAME}_${TIMESTAMP}"
LOG_FILE="$LOG_ROOT/${RUN_NAME}_${TIMESTAMP}.log"
CKPT_DIR="$CKPT_ROOT/${RUN_NAME}_${TIMESTAMP}"

mkdir -p "$RUN_DIR" "$CKPT_DIR" "$LOG_ROOT" "$SWANLAB_LOG_DIR" "$RAY_TMPDIR"

echo "Project root: $OPD_PROJECT_ROOT"
echo "VERL root:    $VERL_ROOT"
echo "Run dir:      $RUN_DIR"
echo "Log file:     $LOG_FILE"
echo "Checkpoint:   $CKPT_DIR"
echo "Student:      $STUDENT_MODEL"
echo "Data root:    $DATA_ROOT"
echo "SwanLab mode: $SWANLAB_MODE"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$DATA_ROOT/train.parquet" \
  data.val_files="$DATA_ROOT/test.parquet" \
  data.train_batch_size=16 \
  data.max_prompt_length=512 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=False \
  actor_rollout_ref.model.path="$STUDENT_MODEL" \
  actor_rollout_ref.model.lora_rank=64 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=3e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.critic_warmup=0 \
  trainer.logger='["console","swanlab"]' \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$RUN_NAME" \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.val_before_train=False \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.save_freq=20 \
  trainer.test_freq=5 \
  trainer.total_epochs=1 \
  "$@" \
  2>&1 | tee "$LOG_FILE"

echo "Finished run: $RUN_NAME"
echo "Run dir:      $RUN_DIR"
echo "Log file:     $LOG_FILE"
echo "Checkpoint:   $CKPT_DIR"

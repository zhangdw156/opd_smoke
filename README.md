# opd_smoke

This repository contains a formal TRL OPD training path for distilling a local Qwen2.5-7B teacher into a local Qwen2.5-3B student on GSM8K.

The formal OPD scripts are isolated under `scripts/trl_opd/`. They reuse the same model, dataset, log, and checkpoint paths defined in `configs/env.sh`, which is also used by the MVP scripts. The MVP scripts remain separate and should not be used as the training implementation for this workflow.

## Formal TRL OPD

- Trainer: TRL `experimental.distillation.DistillationTrainer`
- Objective: on-policy reverse KL with `BETA=1.0` and `LMBDA=1.0`
- Teacher scoring: TRL vLLM server via `trl vllm-serve`
- Student updates: LoRA adapters saved under `checkpoints/`
- Data and model paths: inherited from `configs/env.sh`

## Layout

- `configs/env.sh`: shared local paths for project, models, data, logs, checkpoints, cache, and SwanLab
- `scripts/trl_opd/setup_env.sh`: server-only environment setup for formal TRL OPD
- `scripts/trl_opd/run_teacher_server.sh`: starts the teacher vLLM server
- `scripts/trl_opd/run_train.sh`: starts TRL OPD training
- `scripts/trl_opd/train.py`: repo-local Python wrapper around `DistillationTrainer`
- `scripts/*mvp*`: old MVP path, kept separate

## Server Setup

Run this on the H20 server only:

```bash
cd /data/zhangdw12/work/opd_smoke
UV_PROJECT_ENVIRONMENT=.venv_trl uv sync --python 3.10
```

Or use the wrapper, which runs the same sync and then performs an import/CUDA self-check:

```bash
bash scripts/trl_opd/setup_env.sh
```

Dependencies are declared in `pyproject.toml`; the environment path remains `.venv_trl`, matching the existing scripts.

## Run TRL OPD

Use two terminals. Put the teacher vLLM server on a GPU that is not used by the training process.

Terminal 1, teacher server:

```bash
cd /data/zhangdw12/work/opd_smoke
source configs/env.sh

TEACHER_CUDA_VISIBLE_DEVICES=1 TEACHER_SERVER_PORT=8000 TEACHER_MAX_MODEL_LEN=1024 bash scripts/trl_opd/run_teacher_server.sh
```

Terminal 2, student training:

```bash
cd /data/zhangdw12/work/opd_smoke
source configs/env.sh

CUDA_VISIBLE_DEVICES=0 NUM_PROCESSES=1 MAX_STEPS=640 PER_DEVICE_TRAIN_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=8 GENERATION_BATCH_SIZE=16 MAX_COMPLETION_LENGTH=256 LOSS_TOP_K=1 BETA=1.0 LMBDA=1.0 TEACHER_MODEL_SERVER_URL=http://127.0.0.1:8000 bash scripts/trl_opd/run_train.sh
```

With the defaults above, effective global batch size is:

```text
NUM_PROCESSES * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
= 1 * 2 * 8 = 16
```

For multiple training GPUs, set `CUDA_VISIBLE_DEVICES` and `NUM_PROCESSES` consistently. For example, `CUDA_VISIBLE_DEVICES=0,1 NUM_PROCESSES=2` with the same per-device batch and accumulation gives global batch size `32`. Keep the teacher server on another GPU, for example `TEACHER_CUDA_VISIBLE_DEVICES=2`.

## Tuning Knobs

- Increase throughput first with `GENERATION_BATCH_SIZE`, then `PER_DEVICE_TRAIN_BATCH_SIZE`, then `GRADIENT_ACCUMULATION_STEPS`.
- If student training OOMs, lower `PER_DEVICE_TRAIN_BATCH_SIZE` or set `MAX_COMPLETION_LENGTH=128`.
- If teacher server OOMs, lower `TEACHER_GPU_MEMORY_UTILIZATION` or `TEACHER_MAX_MODEL_LEN`.
- `BETA=1.0` is reverse KL, the OPD objective.
- `LOSS_TOP_K=1` is required by TRL's teacher-server path when `BETA>0`.
- `LMBDA=1.0` means fully on-policy. Lower values require dataset assistant completions and are not the default for this GSM8K prompt-only path.

## Expected Outputs

- Logs: `logs/`
- LoRA adapters: `checkpoints/`
- SwanLab records: `swanlab/`

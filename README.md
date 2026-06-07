# opd_smoke

Smoke-test project for running verl v0.7.1 experiments on GSM8K with local Qwen2.5 models.

## Layout

- `configs/`: environment configuration
- `scripts/`: experiment launch scripts
- `logs/`: local logs, ignored by git
- `checkpoints/`: verl checkpoints, ignored by git
- `swanlab/`: SwanLab local records, ignored by git
- `ray_tmp/`: Ray temp files, ignored by git
- `cache/`: local caches, ignored by git
- `runs/`: per-run artifacts, ignored by git

## Current setup

- verl root: `/data/zhangdw12/work/verl`
- data root: `/data/zhangdw12/datasets/gsm8k`
- model root: `/data/zhangdw12/models`
- project root: `/data/zhangdw12/work/opd_smoke`

## Run

```bash
cd /data/zhangdw12/work/opd_smoke
source configs/env.sh
bash scripts/run_qwen25_3b_gsm8k_grpo_lora_smoke_v071.sh

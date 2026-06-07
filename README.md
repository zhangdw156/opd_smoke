# opd_smoke

Minimal on-policy distillation MVP for distilling a local Qwen2.5-7B teacher into a local Qwen2.5-3B student on GSM8K.

## Current implementation

- Framework: PyTorch + Transformers + PEFT + TRL ecosystem
- Student: local Qwen2.5-3B-Instruct
- Teacher: local Qwen2.5-7B-Instruct
- Data: local GSM8K parquet
- Logging: SwanLab
- Training: student on-policy generation + token-level teacher KL on generated response tokens
- Output: LoRA adapters saved under `checkpoints/`

## Layout

- `configs/`: environment variables and local paths
- `scripts/`: setup and training scripts
- `logs/`: ignored runtime logs
- `checkpoints/`: ignored model outputs
- `swanlab/`: ignored SwanLab local records
- `cache/`: ignored local caches
- `runs/`: ignored run artifacts

## Setup

```bash
cd /data/zhangdw12/work/opd_smoke
bash scripts/setup_trl_env.sh
Run a tiny OPD MVP
cd /data/zhangdw12/work/opd_smoke
source configs/env.sh

MAX_STEPS=5 \
MAX_NEW_TOKENS=128 \
SAVE_EVERY=5 \
bash scripts/run_trl_opd_mvp.sh
Run the default MVP
cd /data/zhangdw12/work/opd_smoke
source configs/env.sh
bash scripts/run_trl_opd_mvp.sh
Expected outputs
Logs: logs/
LoRA adapters: checkpoints/
SwanLab records: swanlab/

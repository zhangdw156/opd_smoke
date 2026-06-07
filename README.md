opd_smoke

Minimal OPD MVP for Qwen2.5-3B student distilled from Qwen2.5-7B teacher on GSM8K.

Current implementation:
- framework: PyTorch + Transformers + PEFT + TRL ecosystem
- logging: SwanLab
- data: local GSM8K parquet
- models: local Qwen2.5 model directories

Run:
1. bash scripts/setup_trl_env.sh
2. source configs/env.sh
3. bash scripts/run_trl_opd_mvp.sh

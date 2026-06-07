#!/usr/bin/env python3
"""Repo-local wrapper around TRL's experimental DistillationTrainer."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import TrainerCallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRL on-policy distillation training")
    parser.add_argument("--student_model", required=True)
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--project_name", default="opd_smoke")
    parser.add_argument("--run_name", default="qwen25_3b_from_7b_gsm8k_trl_distillation")
    parser.add_argument("--max_steps", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--generation_batch_size", type=int, default=16)
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0, help="Sampling top-k; 0 disables top-k sampling")
    parser.add_argument("--loss_top_k", type=int, default=1)
    parser.add_argument("--lmbda", type=float, default=1.0, help="1.0 means fully on-policy")
    parser.add_argument("--beta", type=float, default=1.0, help="0.0=forward KL, 0.5=JSD, 1.0=reverse KL")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--dtype", default="bfloat16", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_teacher_server", action="store_true")
    parser.add_argument("--teacher_model_server_url", default="http://127.0.0.1:8000")
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--swanlab", action="store_true")
    return parser.parse_args()


def maybe_to_python(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def row_to_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    if "prompt" in row:
        prompt = maybe_to_python(row["prompt"])
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            return [{"role": str(msg["role"]), "content": str(msg["content"])} for msg in prompt]
        if isinstance(prompt, list):
            return [{"role": "user", "content": "\n".join(str(x) for x in prompt)}]
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]

    for key in ("question", "problem", "input"):
        value = row.get(key)
        if isinstance(value, str):
            return [{"role": "user", "content": value}]

    raise ValueError(f"Cannot find prompt-like field in row keys: {list(row.keys())}")


def load_messages_dataset(train_file: str, seed: int) -> Dataset:
    df = pd.read_parquet(train_file)
    if len(df) == 0:
        raise ValueError(f"Empty train file: {train_file}")
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    records = [{"messages": row_to_messages(row)} for row in df.to_dict(orient="records")]
    return Dataset.from_list(records)


class SwanLabCallback(TrainerCallback):
    def __init__(self, project_name: str, run_name: str, config: dict[str, Any]) -> None:
        self.swanlab = None
        try:
            import swanlab

            self.swanlab = swanlab
            swanlab.init(project=project_name, experiment_name=run_name, config=config)
        except Exception as exc:
            print(f"[warn] SwanLab init failed; continuing without SwanLab: {exc}")

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if self.swanlab is not None and logs:
            self.swanlab.log(dict(logs), step=int(state.global_step))

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ANN001
        if self.swanlab is not None:
            try:
                self.swanlab.finish()
            except Exception:
                pass


def main() -> None:
    args = parse_args()

    try:
        from trl.experimental.distillation import DistillationConfig, DistillationTrainer
    except Exception as exc:
        raise RuntimeError(
            "This script requires a TRL build with trl.experimental.distillation. "
            "Run scripts/trl_opd/setup_env.sh on the training server, or install TRL from the Hugging Face main branch."
        ) from exc

    if args.temperature <= 0:
        raise ValueError("--temperature must be positive.")
    if args.num_generations <= 0:
        raise ValueError("--num_generations must be positive.")
    if args.generation_batch_size % args.num_generations != 0:
        raise ValueError("--generation_batch_size must be divisible by --num_generations.")
    if args.use_teacher_server and args.beta > 0 and args.loss_top_k != 1:
        raise ValueError("TRL server-backed distillation with beta > 0 currently requires --loss_top_k 1.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = load_messages_dataset(args.train_file, args.seed)
    print(f"[info] loaded train rows={len(train_dataset)} from {args.train_file}")

    model_kwargs = {
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
        "dtype": args.dtype,
        "use_cache": not args.gradient_checkpointing,
    }
    teacher_model_kwargs = {
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
        "dtype": args.dtype,
        "use_cache": True,
    }
    report_to = [] if args.report_to == "none" else [x.strip() for x in args.report_to.split(",") if x.strip()]

    training_args = DistillationConfig(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        max_steps=args.max_steps,
        seed=args.seed,
        data_seed=args.seed,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        generation_batch_size=args.generation_batch_size,
        num_generations=args.num_generations,
        learning_rate=args.lr,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_length=args.max_prompt_length + args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        loss_top_k=args.loss_top_k,
        lmbda=args.lmbda,
        beta=args.beta,
        use_teacher_server=args.use_teacher_server,
        teacher_model_server_url=args.teacher_model_server_url,
        teacher_model_name_or_path=args.teacher_model,
        teacher_model_init_kwargs=teacher_model_kwargs,
        model_init_kwargs=model_kwargs,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.dtype == "bfloat16",
        fp16=args.dtype == "float16",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="steps",
        eval_strategy="no",
        report_to=report_to,
        remove_unused_columns=False,
        run_name=args.run_name,
    )

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    trainer = DistillationTrainer(
        model=args.student_model,
        teacher_model=None if args.use_teacher_server else args.teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    if args.swanlab:
        trainer.add_callback(SwanLabCallback(args.project_name, args.run_name, vars(args)))

    trainer.train()
    trainer.save_model(str(output_dir / "final"))


if __name__ == "__main__":
    main()

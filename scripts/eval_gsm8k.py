#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Subset GSM8K evaluator")
    p.add_argument("--model", required=True)
    p.add_argument("--adapter_dir", default=None)
    p.add_argument("--test_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def maybe_to_python(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def row_to_prompt(row: dict[str, Any], tokenizer) -> str:
    if "prompt" in row:
        prompt = maybe_to_python(row["prompt"])
        if isinstance(prompt, str):
            if "<|im_start|>" in prompt or "<|im_end|>" in prompt:
                return prompt
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        if isinstance(prompt, list):
            if prompt and isinstance(prompt[0], dict):
                return tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": "\n".join(str(x) for x in prompt)}],
                tokenize=False,
                add_generation_prompt=True,
            )

    for key in ("question", "problem", "input"):
        value = row.get(key)
        if isinstance(value, str):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": value}],
                tokenize=False,
                add_generation_prompt=True,
            )

    raise ValueError(f"Cannot find prompt-like field in row keys: {list(row.keys())}")


def normalize_answer(text: str | None) -> str | None:
    if text is None:
        return None
    s = str(text).strip()
    if "####" in s:
        s = s.split("####")[-1].strip()
    s = s.replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not matches:
        return None
    ans = matches[-1]
    if "." in ans:
        try:
            val = float(ans)
            if val.is_integer():
                return str(int(val))
            return str(val)
        except ValueError:
            return ans
    return str(int(ans))


def extract_ground_truth(row: dict[str, Any]) -> str:
    candidates = []

    reward_model = maybe_to_python(row.get("reward_model"))
    if isinstance(reward_model, dict):
        for key in ("ground_truth", "answer", "target", "solution"):
            if key in reward_model:
                candidates.append(reward_model[key])
    elif reward_model is not None:
        candidates.append(reward_model)

    extra_info = maybe_to_python(row.get("extra_info"))
    if isinstance(extra_info, dict):
        for key in ("answer", "ground_truth", "target", "solution"):
            if key in extra_info:
                candidates.append(extra_info[key])

    for key in ("answer", "ground_truth", "target"):
        if key in row:
            candidates.append(row[key])

    for candidate in candidates:
        norm = normalize_answer(str(candidate))
        if norm is not None:
            return norm

    raise ValueError(f"Cannot extract ground truth from row keys: {list(row.keys())}")


def load_model(args: argparse.Namespace, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    if args.adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.to(args.device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args, dtype)

    df = pd.read_parquet(args.test_file)
    limit = min(args.limit, len(df))
    df = df.sample(n=limit, random_state=args.seed).reset_index(drop=True)

    print(f"[info] model={args.model}")
    print(f"[info] adapter_dir={args.adapter_dir}")
    print(f"[info] test_file={args.test_file}")
    print(f"[info] limit={limit}")
    print(f"[info] batch_size={args.batch_size}")
    print(f"[info] max_new_tokens={args.max_new_tokens}")
    print(f"[info] dtype={dtype}")
    print(f"[info] device={args.device}")
    print(f"[info] output_file={output_file}")

    correct = 0
    total = 0

    with output_file.open("w", encoding="utf-8") as fout:
        for start in tqdm(range(0, limit, args.batch_size), desc="GSM8K eval"):
            rows = [df.iloc[i].to_dict() for i in range(start, min(start + args.batch_size, limit))]
            prompts = [row_to_prompt(row, tokenizer) for row in rows]
            golds = [extract_ground_truth(row) for row in rows]

            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_prompt_length,
                add_special_tokens=False,
            ).to(args.device)

            prompt_lens = enc["attention_mask"].sum(dim=1).tolist()

            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if args.temperature > 0:
                gen_kwargs["temperature"] = args.temperature

            with torch.no_grad():
                generated = model.generate(**enc, **gen_kwargs)

            input_width = enc["input_ids"].shape[1]
            for j, row in enumerate(rows):
                completion_ids = generated[j, input_width:]
                completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

                pred = normalize_answer(completion)
                gold = golds[j]
                is_correct = pred == gold

                correct += int(is_correct)
                total += 1

                fout.write(json.dumps({
                    "idx": int(start + j),
                    "correct": bool(is_correct),
                    "pred": pred,
                    "gold": gold,
                    "prompt_len": int(prompt_lens[j]),
                    "completion": completion,
                }, ensure_ascii=False) + "\n")
                fout.flush()

    summary = {
        "model": args.model,
        "adapter_dir": args.adapter_dir,
        "test_file": args.test_file,
        "limit": limit,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "correct": correct,
        "total": total,
        "accuracy": correct / max(total, 1),
    }

    summary_file = output_file.with_suffix(".summary.json")
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[save] details={output_file}")
    print(f"[save] summary={summary_file}")


if __name__ == "__main__":
    main()

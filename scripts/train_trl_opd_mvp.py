import argparse
import os
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--student_model", required=True)
    p.add_argument("--teacher_model", required=True)
    p.add_argument("--train_file", required=True)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--project_name", default="opd_smoke")
    p.add_argument("--run_name", default="qwen25_3b_from_7b_gsm8k_trl_opd_mvp")

    p.add_argument("--max_steps", type=int, default=20)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=1.0)

    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--student_device", default="cuda:0")
    p.add_argument("--teacher_device", default="cuda:1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=10)

    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def row_to_prompt(row, tokenizer):
    if "prompt" in row:
        prompt = row["prompt"]

        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()

        if isinstance(prompt, str):
            return prompt

        if isinstance(prompt, list):
            if len(prompt) > 0 and isinstance(prompt[0], dict):
                return tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            return "\n".join(str(x) for x in prompt)

    for key in ["question", "problem", "input"]:
        if key in row and isinstance(row[key], str):
            messages = [{"role": "user", "content": row[key]}]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    raise ValueError(f"Cannot find prompt-like field in row keys: {list(row.keys())}")


def tokenize_prompt(tokenizer, prompt, max_prompt_length, device):
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length,
    )
    return {k: v.to(device) for k, v in enc.items()}


def build_lora_student(model, args):
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def init_swanlab(args):
    try:
        import swanlab

        run = swanlab.init(
            project=args.project_name,
            experiment_name=args.run_name,
            config=vars(args),
        )
        return swanlab, run
    except Exception as exc:
        print(f"[warn] SwanLab init failed, continue without SwanLab: {exc}")
        return None, None


def main():
    args = parse_args()
    set_seed(args.seed)

    assert torch.cuda.is_available(), "CUDA is required for this MVP"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    dtype = get_dtype()

    tokenizer = AutoTokenizer.from_pretrained(
        args.student_model,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    student = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(args.student_device)

    student = build_lora_student(student, args)
    student.train()

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(args.teacher_device)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)

    df = pd.read_parquet(args.train_file)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    swanlab, _ = init_swanlab(args)

    progress = tqdm(range(args.max_steps), desc="OPD MVP")

    for step in progress:
        row = df.iloc[step % len(df)].to_dict()
        prompt = row_to_prompt(row, tokenizer)

        prompt_inputs = tokenize_prompt(
            tokenizer,
            prompt,
            args.max_prompt_length,
            args.student_device,
        )

        prompt_len = int(prompt_inputs["input_ids"].shape[1])

        student.eval()
        with torch.no_grad():
            generated = student.generate(
                **prompt_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        student.train()

        seq = generated[:, :].detach()
        total_len = int(seq.shape[1])
        response_len = total_len - prompt_len

        if response_len <= 1:
            print(f"[warn] empty response at step {step}, skip")
            continue

        student_seq = seq.to(args.student_device)
        teacher_seq = seq.to(args.teacher_device)

        with torch.no_grad():
            teacher_logits = teacher(teacher_seq).logits[:, :-1, :]
            teacher_logits = teacher_logits.to(args.student_device)

        student_logits = student(student_seq).logits[:, :-1, :]

        target_positions = torch.arange(
            1,
            total_len,
            device=args.student_device,
        )

        response_mask = target_positions >= prompt_len

        s_logits = student_logits[0, response_mask, :]
        t_logits = teacher_logits[0, response_mask, :]

        temp = args.temperature
        s_log_probs = F.log_softmax(s_logits.float() / temp, dim=-1)
        t_probs = F.softmax(t_logits.float() / temp, dim=-1)

        loss = F.kl_div(
            s_log_probs,
            t_probs,
            reduction="batchmean",
        ) * (temp ** 2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            completion_ids = seq[0, prompt_len:]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

        metrics = {
            "train/loss": float(loss.item()),
            "train/grad_norm": float(grad_norm),
            "train/response_len": int(response_len),
            "train/prompt_len": int(prompt_len),
            "train/total_len": int(total_len),
            "train/lr": args.lr,
        }

        progress.set_postfix(loss=metrics["train/loss"], response_len=response_len)

        if swanlab is not None:
            swanlab.log(metrics, step=step)
            if step == 0 or (step + 1) % args.save_every == 0:
                swanlab.log(
                    {
                        "sample/prompt": prompt[:2000],
                        "sample/completion": completion[:2000],
                    },
                    step=step,
                )

        if (step + 1) % args.save_every == 0:
            save_dir = Path(args.output_dir) / f"step_{step + 1}"
            student.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"[save] {save_dir}")

        del teacher_logits, student_logits, s_logits, t_logits, s_log_probs, t_probs
        torch.cuda.empty_cache()

    final_dir = Path(args.output_dir) / "final"
    student.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[save] {final_dir}")

    if swanlab is not None:
        try:
            swanlab.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()

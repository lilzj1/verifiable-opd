"""Protocol SFT for a larger frozen OPD teacher using LoRA.

The adapter is trained only on the existing train split's exact expert
decisions.  The optional merged output is an ordinary Hugging Face checkpoint
that verl can load as a frozen teacher; no adapter logic enters OPD itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import expert_messages, read_records
from .environment import Task
from .tokenization import expert_decisions


def labeled_example(decision: object) -> dict[str, list[int]]:
    """Build standard causal-LM labels: prompt ignored, action supervised."""
    input_ids = list(decision.prefix) + list(decision.response)
    labels = [-100] * len(decision.prefix) + list(decision.response)
    if len(input_ids) != len(labels) or not decision.response:
        raise ValueError("invalid expert decision")
    return {"input_ids": input_ids, "labels": labels}


def collate(examples: list[dict[str, list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("empty batch")
    width = max(len(row["input_ids"]) for row in examples)
    ids, labels, masks = [], [], []
    for row in examples:
        padding = width - len(row["input_ids"])
        ids.append(row["input_ids"] + [pad_id] * padding)
        labels.append(row["labels"] + [-100] * padding)
        masks.append([1] * len(row["input_ids"]) + [0] * padding)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.gradient_accumulation, args.lora_rank, args.lora_alpha) < 1:
        parser.error("steps, batch and LoRA sizes must be positive")
    if args.lr <= 0 or args.max_length < 2:
        parser.error("invalid learning rate or max length")

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    rows = read_records(args.data, {"train"})
    examples = []
    for row in rows:
        messages = expert_messages(Task.from_dict(row["task"]))
        examples.extend(labeled_example(d) for d in expert_decisions(tokenizer, messages))
    if not examples or max(map(lambda row: len(row["input_ids"]), examples)) > args.max_length:
        raise ValueError("empty data or expert decision exceeds max length")

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        examples,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(args.device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or any(parameter.requires_grad for name, parameter in model.named_parameters() if "lora_" not in name):
        raise RuntimeError("LoRA freeze boundary is invalid")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    config = {
        **vars(args),
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "examples": len(examples),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(loader)
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        step_loss = 0.0
        supervised_tokens = 0
        for _ in range(args.gradient_accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {key: value.to(args.device) for key, value in batch.items()}
            output_value = model(**batch, use_cache=False)
            loss = output_value.loss / args.gradient_accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
            step_loss += float(loss.detach())
            supervised_tokens += int((batch["labels"] != -100).sum())
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        metrics = {
            "step": step,
            "loss": step_loss,
            "grad_norm": float(grad_norm),
            "supervised_tokens": supervised_tokens,
            "seconds": time.monotonic() - started,
            "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0,
        }
        with (output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        print(json.dumps(metrics, sort_keys=True), flush=True)

    adapter_dir = output / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    if args.merge:
        merged = model.merge_and_unload()
        merged.config.use_cache = True
        merged_dir = output / "merged_model"
        merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="4GB")
        tokenizer.save_pretrained(merged_dir)
    (output / "COMPLETE").write_text("ok\n")


if __name__ == "__main__":
    main()

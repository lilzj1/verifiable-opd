"""Compare a frozen OPD teacher and student on held-out expert decisions.

Greedy episode success is useful, but it is stricter than the contract needed
by online policy distillation: the teacher scores actions produced at visited
student states.  This gate therefore also measures teacher-forced likelihood
on the exact same held-out expert action tokens for both models.  Only the dev
split is accepted; test data must remain untouched until the experiment is
frozen.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import expert_messages, read_records
from .environment import Task
from .hf_backend import load_model
from .tokenization import causal_target_slice, check_tokenizers, expert_decisions


def score_model(model, tokenizer, rows):
    token_count = 0
    nll_sum = 0.0
    correct = 0
    decisions = 0
    per_task = []
    with torch.inference_mode():
        for row in rows:
            task_tokens = 0
            task_nll = 0.0
            task_correct = 0
            for decision in expert_decisions(
                tokenizer, expert_messages(Task.from_dict(row["task"]))
            ):
                device = next(model.parameters()).device
                ids = torch.tensor(
                    [decision.prefix + decision.response], dtype=torch.long, device=device
                )
                output = model(
                    input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
                )
                span = causal_target_slice(len(decision.prefix), len(decision.response))
                logits = output.logits[0, span].float()
                targets = ids[0, len(decision.prefix) :]
                losses = F.cross_entropy(logits, targets, reduction="none")
                count = int(targets.numel())
                task_tokens += count
                task_nll += float(losses.sum())
                task_correct += int((logits.argmax(dim=-1) == targets).sum())
                decisions += 1
            token_count += task_tokens
            nll_sum += task_nll
            correct += task_correct
            per_task.append(
                {
                    "task_id": row["task_id"],
                    "length": row["length"],
                    "tokens": task_tokens,
                    "nll": task_nll / task_tokens,
                    "token_accuracy": task_correct / task_tokens,
                }
            )
    if token_count == 0:
        raise ValueError("teacher gate received no expert action tokens")
    mean_nll = nll_sum / token_count
    return {
        "tasks": len(rows),
        "decisions": decisions,
        "tokens": token_count,
        "mean_nll": mean_nll,
        "perplexity": math.exp(min(mean_nll, 50.0)),
        "token_accuracy": correct / token_count,
        "per_task": per_task,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = read_records(args.data, {"dev"})
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        rows = rows[: args.limit]
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)

    # Tokenizer identity is checked without keeping both large models resident.
    from transformers import AutoTokenizer

    student_tokenizer = AutoTokenizer.from_pretrained(args.student, local_files_only=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True)
    check_tokenizers(student_tokenizer, teacher_tokenizer)

    student, student_tokenizer = load_model(args.student, args.device)
    student_metrics = score_model(student, student_tokenizer, rows)
    del student
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    teacher, teacher_tokenizer = load_model(args.teacher, args.device)
    teacher_metrics = score_model(teacher, teacher_tokenizer, rows)
    del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "student": student_metrics,
        "teacher": teacher_metrics,
        "teacher_student_nll_ratio": teacher_metrics["mean_nll"]
        / student_metrics["mean_nll"],
        "teacher_nll_improvement": student_metrics["mean_nll"]
        - teacher_metrics["mean_nll"],
        "teacher_token_accuracy_improvement": teacher_metrics["token_accuracy"]
        - student_metrics["token_accuracy"],
        "gate_pass": bool(
            teacher_metrics["mean_nll"] < student_metrics["mean_nll"]
            and teacher_metrics["token_accuracy"] > student_metrics["token_accuracy"]
        ),
        "config": vars(args),
        "scope": "held-out dev expert actions; teacher forcing; no test data",
    }
    output.mkdir(parents=True)
    (output / "gate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"student", "teacher"}}, indent=2))
    print(
        json.dumps(
            {
                "student": {k: v for k, v in student_metrics.items() if k != "per_task"},
                "teacher": {k: v for k, v in teacher_metrics.items() if k != "per_task"},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

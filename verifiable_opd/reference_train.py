"""Correctness-first SFT/OPD/replay runner (Transformers, single student).

The teacher can be placed on a second GPU. This runner is not a distributed
verl benchmark. Use only small budgets until teacher/student probes are passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch

from .data import canonical, read_records, expert_messages
from .environment import Task
from .hf_backend import load_model, rollout
from .losses import backward_update
from .replay import select_replay
from .tokenization import check_tokenizers, expert_decisions


def save_checkpoint(directory, student, tokenizer, optimizer, step, config, rng):
    final = directory / f"step_{step:06d}"
    temporary = directory / f"step_{step:06d}.incomplete"
    temporary.mkdir()
    student.save_pretrained(temporary / "model", safe_serialization=True)
    tokenizer.save_pretrained(temporary / "model")
    state = {"step": step, "optimizer": optimizer.state_dict(), "config": config,
        "python_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    torch.save(state, temporary / "training_state.pt")
    (temporary / "metadata.json").write_text(json.dumps(config, indent=2) + "\n")
    temporary.rename(final)
    (directory / "latest.json").write_text(json.dumps({"step": step, "checkpoint": str(final.resolve())}) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["sft", "opd", "uniform", "failure"], required=True)
    p.add_argument("--student", required=True, help="local HF model directory")
    p.add_argument("--teacher", help="local HF teacher directory")
    p.add_argument("--data", required=True, help="train.jsonl only")
    p.add_argument("--output", required=True)
    p.add_argument("--student-device", default="cpu")
    p.add_argument("--teacher-device", default="cpu")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--replay-tokens", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--max-context", type=int, default=4096)
    p.add_argument("--save-every", type=int, default=0, help="0 saves final checkpoint only")
    p.add_argument("--resume", help="our own step directory, not arbitrary pickle")
    a = p.parse_args()
    if a.steps < 1 or a.batch_size < 1 or a.lr <= 0 or a.alpha < 0 or a.replay_tokens < 0:
        p.error("invalid training budget")
    if a.mode != "sft" and not a.teacher:
        p.error("OPD requires --teacher")
    if a.mode in ("uniform", "failure") and (a.alpha <= 0 or a.replay_tokens < 1):
        p.error("replay modes require positive alpha and token budget")
    if a.max_new_tokens < 1 or a.max_context < 2 or a.save_every < 0:
        p.error("invalid context or checkpoint budget")
    rows = read_records(a.data, {"train"})
    if a.batch_size > len(rows):
        p.error("batch size exceeds training pool")
    config = vars(a).copy()
    config["backend"] = "transformers_reference_not_verl"
    config["data_sha256"] = hashlib.sha256(Path(a.data).read_bytes()).hexdigest()
    config["lr_schedule"] = "constant"
    config["torch_version"] = torch.__version__
    import transformers
    config["transformers_version"] = transformers.__version__
    directory = Path(a.output)
    state = None
    if a.resume:
        # Only load artifacts produced by this project. This contains Python RNG.
        state = torch.load(Path(a.resume) / "training_state.pt", map_location="cpu", weights_only=False)
        for field in ("mode", "student", "teacher", "data_sha256", "seed", "lr", "batch_size",
                      "alpha", "replay_tokens", "max_context", "max_new_tokens", "student_device",
                      "teacher_device", "torch_version", "transformers_version"):
            if state["config"][field] != config[field]:
                raise ValueError(f"resume configuration changed: {field}")
        if a.steps <= state["step"]:
            raise ValueError("--steps is the final global step, must exceed saved step")
    elif directory.exists():
        raise FileExistsError("output exists; use a fresh directory or explicit --resume")
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    student_path = str(Path(a.resume) / "model") if a.resume else a.student
    student, tokenizer = load_model(student_path, a.student_device, trainable=True)
    if hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()
    teacher = None
    if a.mode != "sft":
        teacher, teacher_tokenizer = load_model(a.teacher, a.teacher_device)
        check_tokenizers(tokenizer, teacher_tokenizer)
    optimizer = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=0.0)
    experts = {}
    for row in rows:
        decisions = expert_decisions(tokenizer, expert_messages(Task.from_dict(row["task"])))
        if any(len(d.prefix) + len(d.response) > a.max_context for d in decisions):
            raise ValueError("expert trajectory exceeds max context; change the frozen budget explicitly")
        experts[row["task_id"]] = decisions
    costs = {uid: sum(len(d.response) for d in ds) for uid, ds in experts.items()}
    if a.mode in ("uniform", "failure") and a.replay_tokens < min(costs.values()):
        raise ValueError("replay token budget cannot fit any complete expert trajectory")
    start = 0
    if state:
        optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start = state["step"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"config_from_step_{start}.json").write_text(json.dumps(config, indent=2) + "\n")
    for step in range(start + 1, a.steps + 1):
        started = time.monotonic()
        batch = rng.sample(rows, a.batch_size)
        online, traces, replay_ds = [], [], []
        selection = None
        if a.mode == "sft":
            replay_ds = [d for row in batch for d in experts[row["task_id"]]]
        else:
            for row in batch:
                ds, trace = rollout(student, tokenizer, row, a.max_new_tokens, a.max_context, sample=True)
                if not ds:
                    raise ValueError("empty rollout: initial context does not fit")
                online.extend(ds)
                traces.append(trace)
            if a.mode in ("uniform", "failure"):
                failures = {t["task_id"] for t in traces if not t["success"]}
                selection = select_replay(costs, failures, a.replay_tokens, rng.randrange(2**32), a.mode)
                replay_ds = [d for uid in selection["task_ids"] for d in experts[uid]]
        alpha = a.alpha if a.mode in ("uniform", "failure") else 0.0
        metrics = backward_update(student, teacher, optimizer, online, replay_ds, alpha)
        metrics.update(step=step, mode=a.mode, seconds=time.monotonic() - started,
                       backend=config["backend"], replay_selection=selection,
                       success_rate=sum(t["success"] for t in traces) / len(traces) if traces else None,
                       teacher_scored_context_tokens=sum(t["scored_context_tokens"] for t in traces),
                       train_task_ids=[r["task_id"] for r in batch])
        with (directory / "metrics.jsonl").open("a") as f:
            f.write(canonical(metrics) + "\n")
        with (directory / "rollouts.jsonl").open("a") as f:
            for trace in traces:
                f.write(canonical({"update_step": step, **trace}) + "\n")
        print(canonical(metrics), flush=True)
        if step == a.steps or (a.save_every and step % a.save_every == 0):
            save_checkpoint(directory, student, tokenizer, optimizer, step, config, rng)


if __name__ == "__main__":
    main()

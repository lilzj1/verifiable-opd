"""Generate disjoint task splits and expert trajectories, without an LLM."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from .environment import ArithmeticEnv, Task, oracle_action
from .prompts import SYSTEM_PROMPT, TOOL_SCHEMA, user_prompt


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def task_id(task):
    return hashlib.sha256(canonical(task.to_dict()).encode()).hexdigest()


def initial_messages(task):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(task)}]


def expert_messages(task, mode="sparse"):
    env = ArithmeticEnv(task, mode)
    messages = initial_messages(task)
    while not env.done:
        action = json.loads(oracle_action(env))
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "submit_value", "arguments": action}}]})
        obs, _, _ = env.step(canonical(action))
        messages.append({"role": "tool", "content": canonical(obs)})
    # Training/evaluation stop immediately when the environment terminates.
    # Do not add an unsampled final assistant utterance to expert supervision.
    assert env.success
    return messages


def make_record(task, split):
    return {"task_id": task_id(task), "split": split, "task": task.to_dict(),
            "length": len(task.operations)}


def build_splits(train=1024, dev=128, test=128, ood=128, seed=17):
    counts = {"train": train, "dev": dev, "test_id": test, "test_ood": ood}
    if any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError("every split must contain at least one task")
    rng = random.Random(seed)
    used = set()
    result = {}
    for split, count in counts.items():
        lengths = [12, 16] if split == "test_ood" else [3, 5, 8]
        rows = []
        while len(rows) < count:
            length = lengths[len(rows) % len(lengths)]
            task = Task(rng.randrange(97), tuple(
                (rng.choice(["add", "mul"]), rng.randint(2, 29)) for _ in range(length)))
            uid = task_id(task)
            if uid in used:
                continue
            used.add(uid)
            rows.append(make_record(task, split))
        result[split] = rows
    return result


def read_records(path, allowed_splits=None):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty task file")
    seen = set()
    for row in rows:
        task = Task.from_dict(row["task"])
        if row["task_id"] != task_id(task) or row["task_id"] in seen:
            raise ValueError("task hash mismatch or duplicate task")
        if row["length"] != len(task.operations):
            raise ValueError("length mismatch")
        if allowed_splits is not None and row["split"] not in allowed_splits:
            raise ValueError(f"split {row['split']} is not permitted here")
        seen.add(row["task_id"])
    return rows


def export(output, splits, seed, parquet=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite dataset directory: {output}")
    if parquet:
        import pyarrow as pa
        import pyarrow.parquet as pq
    final_output = output
    output = output.with_name(output.name + ".incomplete")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "seed": seed, "modulus": 97,
                "scope": "synthetic modular arithmetic, not a public benchmark",
                "files": {}, "splits": {}}
    for split, rows in splits.items():
        path = output / f"{split}.jsonl"
        path.write_text("".join(canonical(row) + "\n" for row in rows))
        manifest["splits"][split] = {"count": len(rows), "lengths": dict(Counter(r["length"] for r in rows))}
        if split in ("train", "dev"):
            experts = [{"task_id": row["task_id"], "split": split,
                        "messages": expert_messages(Task.from_dict(row["task"])),
                        "tools": [TOOL_SCHEMA]} for row in rows]
            (output / f"sft_{split}.jsonl").write_text("".join(canonical(r) + "\n" for r in experts))
            if parquet:
                pq.write_table(pa.Table.from_pylist(experts), output / f"sft_{split}.parquet")
        if parquet:
            rl_rows = []
            for row in rows:
                task = Task.from_dict(row["task"])
                rl_rows.append({"data_source": "verifiable_arithmetic", "ability": "arithmetic",
                    "agent_name": "arithmetic_tool_agent", "prompt": initial_messages(task),
                    "reward_model": {"style": "rule", "ground_truth": "environment_success"},
                    "extra_info": {"task_id": row["task_id"], "split": split,
                        "need_tools_kwargs": True,
                        "tools_kwargs": {"submit_value": {"create_kwargs": {
                            "task_json": canonical(row["task"]), "mode": "sparse", "max_calls": 2 * row["length"]}}}}})
            pq.write_table(pa.Table.from_pylist(rl_rows), output / f"{split}.parquet")
    for path in sorted(output.iterdir()):
        manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    output.rename(final_output)
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    for name, default in [("train", 1024), ("dev", 128), ("test", 128), ("ood", 128), ("seed", 17)]:
        p.add_argument(f"--{name}", type=int, default=default)
    p.add_argument("--parquet", action="store_true")
    a = p.parse_args()
    manifest = export(a.output, build_splits(a.train, a.dev, a.test, a.ood, a.seed), a.seed, a.parquet)
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()

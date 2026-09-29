"""Frozen greedy evaluation, separate from train-only replay selection."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from .data import canonical, read_records
from .hf_backend import load_model, rollout


def summarize(traces):
    if not traces:
        raise ValueError("no evaluation episodes")
    by_length = {}
    for length in sorted({t["length"] for t in traces}):
        subset = [t for t in traces if t["length"] == length]
        by_length[str(length)] = {"count": len(subset), "success_rate": sum(t["success"] for t in subset) / len(subset)}
    return {"count": len(traces), "success_rate": sum(t["success"] for t in traces) / len(traces),
        "by_length": by_length, "calls": sum(t["calls"] for t in traces),
        "action_tokens": sum(t["action_tokens"] for t in traces),
        "context_truncated": sum(t["context_truncated"] for t in traces),
        "invalid_actions": sum(a["parse_error"] is not None or a["observation"].get("feedback") == "invalid_json_or_value"
                               for t in traces for a in t["actions"])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--max-context", type=int, default=4096)
    p.add_argument("--limit", type=int, help="dev probe only; test must use all records")
    p.add_argument("--seed", type=int, default=123)
    a = p.parse_args()
    rows = read_records(a.data, {"dev", "test_id", "test_ood"})
    if a.max_new_tokens < 1 or a.max_context < 2:
        p.error("invalid token budget")
    if a.limit is not None:
        if a.limit < 1 or any(r["split"] != "dev" for r in rows):
            p.error("--limit must be positive and is allowed only for dev probes")
        rows = rows[:a.limit]
    output = Path(a.output)
    if output.exists():
        raise FileExistsError(output)
    torch.manual_seed(a.seed)
    model, tokenizer = load_model(a.model, a.device)
    output.mkdir(parents=True)
    start = time.monotonic()
    traces = []
    for row in rows:
        _, trace = rollout(model, tokenizer, row, a.max_new_tokens, a.max_context, sample=False)
        traces.append(trace)
        with (output / "episodes.jsonl").open("a") as f:
            f.write(canonical(trace) + "\n")
        print(canonical({k: trace[k] for k in ("task_id", "success", "calls")}), flush=True)
    summary = {**summarize(traces), "seconds": time.monotonic() - start,
        "config": vars(a), "backend": "transformers_reference_not_verl", "decoding": "greedy",
        "data_sha256": hashlib.sha256(Path(a.data).read_bytes()).hexdigest()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

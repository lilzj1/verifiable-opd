"""Offline planning checks only: no torch, model, network, GPU or training launch.

This validates a proposed interchange schema, NOT verl's runtime DataProto.
Do not use it as a GPU correctness or teacher-quality gate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_draft(plan):
    require(plan["status"] == "DRAFT_CPU_ONLY", "not a CPU planning draft")
    require(plan["allow_training"] is False, "draft must not authorize training")
    domains = set(plan["domains"])
    require(domains == {"math", "if"}, "unexpected domains")
    require(set(plan["teacher_routes"]) == domains, "incomplete teacher routing")
    require(len(set(plan["teacher_routes"].values())) == 2, "need two teacher roles")
    require(plan["groups"]["S2"]["replay_alpha"] == 0, "S2 must have no CE")
    require(plan["groups"]["S3"]["replay_alpha"] > 0, "S3 needs positive CE alpha")
    c = plan["candidate"]
    require(c["max_prompt_tokens"] + c["max_response_tokens"] <= c["max_context_tokens"],
            "context shorter than prompt plus response")
    require(c["online_prompts_per_update"] == len(domains) * c["online_prompts_per_domain"],
            "domain batch counts do not sum")
    require(c["teacher_gpu_count"] == len(domains) * c["teacher_tp"] * c["teacher_replicas_per_domain"],
            "teacher resource pool mismatch")
    require(c["student_gpu_count"] + c["teacher_gpu_count"] == len(c["physical_gpus"]),
            "total resource pool mismatch")
    require(len(set(c["physical_gpus"])) == len(c["physical_gpus"]), "duplicate GPUs")
    require(c["keep_complete_recovery_checkpoints"] >= 2, "need rollback checkpoint")
    require(c["stage_check_update"] <= c["formal_updates_max"], "invalid stage budget")


def public_messages(row):
    """Whitelist model input; never serialize the entire training/eval record."""
    messages = row["messages"]
    require(isinstance(messages, list) and bool(messages), "missing messages")
    for message in messages:
        require(set(message) == {"role", "content"}, "unexpected message field")
        require(message["role"] in {"system", "user"}, "single-turn input cannot contain answer")
        require(isinstance(message["content"], str) and message["content"].strip(), "empty message")
    require(messages[-1]["role"] == "user", "prompt must end with user")
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def check_records(rows):
    """Require offline-produced dedup clusters; does not discover semantic duplicates."""
    seen_ids, cluster_split = set(), {}
    for row in rows:
        require(row["domain"] in {"math", "if"}, "unknown domain: do not use default teacher")
        require(row["split"] in {"train", "dev", "test"}, "unknown split")
        require(row["kind"] in {"online", "expert", "eval"}, "unknown row kind")
        require(row["kind"] not in {"online", "expert"} or row["split"] == "train",
                "only train may supply updates or replay")
        require(row["kind"] != "eval" or row["split"] in {"dev", "test"}, "eval split required")
        for field in ("example_id", "dedup_cluster_id", "source_revision"):
            require(isinstance(row[field], str) and bool(row[field].strip()), f"missing {field}")
        require(row["example_id"] not in seen_ids, "duplicate example_id")
        seen_ids.add(row["example_id"])
        previous = cluster_split.setdefault(row["dedup_cluster_id"], row["split"])
        require(previous == row["split"], "dedup cluster leaks across splits")
        public_messages(row)
        if row["kind"] == "expert":
            require(isinstance(row.get("expert_response"), str) and bool(row["expert_response"].strip()),
                    "expert row lacks complete response")


def evaluation_response_budget(plan):
    e = plan["evaluation_candidate"]
    per_model = (e["aime24_problems"] + e["aime25_problems"]) * e["math_samples_per_problem"]
    per_model += (e["ifeval_prompts"] + e["ifbench_prompts"]) * e["if_samples_per_prompt"]
    return {"per_checkpoint": per_model, "all_student_checkpoints": per_model * e["student_checkpoints_to_test"]}


def source_fingerprints(root):
    paths = [
        "verifiable_opd/verl_replay_batch.py", "verifiable_opd/verl_mixed_loss.py",
        "checkpoint_rotator.py", "run_verl_profile.sh",
        "upstream/verl-v0.8.0/verl/trainer/ppo/ray_trainer.py",
        "upstream/verl-v0.8.0/verl/workers/config/distillation.py",
        "upstream/verl-v0.8.0/verl/experimental/teacher_loop/teacher_manager.py",
        "upstream/verl-v0.8.0/verl/experimental/agent_loop/single_turn_agent_loop.py",
    ]
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    plan = json.loads((directory / "protocol.draft.json").read_text())
    check_draft(plan)
    print(json.dumps({
        "status": "CPU_DRAFT_CHECK_ONLY_NOT_TRAINING_READY",
        "test_response_budget": evaluation_response_budget(plan),
        "unfrozen_fields": [k for k, v in plan["must_freeze_before_training"].items() if v is None],
        "source_sha256": source_fingerprints(directory.parent),
    }, ensure_ascii=False, indent=2))

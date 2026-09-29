"""Driver-side expert replay selection and DataProto assembly for verl."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import numpy as np
import torch

from .environment import Task
from .expert_agent_loop import expert_action_token_cost
from .replay import select_replay


def _task_payload(row: dict) -> tuple[str, Task, str]:
    extra = row.get("extra_info") or {}
    task_id = extra.get("task_id")
    split = extra.get("split")
    try:
        task_json = extra["tools_kwargs"]["submit_value"]["create_kwargs"]["task_json"]
    except (KeyError, TypeError) as exc:
        raise ValueError("replay pool row lacks arithmetic task_json") from exc
    if not task_id or split != "train":
        raise ValueError("expert replay pool must contain only identified train rows")
    return str(task_id), Task.from_dict(json.loads(task_json)), str(split)


@dataclass(frozen=True)
class ReplayPool:
    task_to_index: dict[str, int]
    costs: dict[str, int]

    @classmethod
    def from_dataset(cls, dataset, tokenizer):
        task_to_index: dict[str, int] = {}
        costs: dict[str, int] = {}
        for index in range(len(dataset)):
            row = dataset.dataframe[index]
            task_id, task, _ = _task_payload(row)
            if task_id in task_to_index:
                raise ValueError(f"duplicate replay task_id: {task_id}")
            task_to_index[task_id] = index
            costs[task_id] = expert_action_token_cost(tokenizer, task)
        if not costs:
            raise ValueError("empty expert replay pool")
        return cls(task_to_index=task_to_index, costs=costs)

    def select(self, failures, *, budget, max_trajectories, seed, strategy):
        result = select_replay(
            self.costs,
            set(failures),
            budget,
            seed,
            strategy,
            max_trajectories=max_trajectories,
        )
        if not result["task_ids"]:
            raise ValueError(
                "expert replay selected no trajectory; token budget is smaller than every complete trajectory"
            )
        if len(result["task_ids"]) != max_trajectories:
            raise ValueError(
                f"expert replay selected {len(result['task_ids'])} trajectories, expected "
                f"exactly {max_trajectories}; increase the token budget"
            )
        result["indices"] = [self.task_to_index[task_id] for task_id in result["task_ids"]]
        return result


def failed_task_ids(online_output) -> set[str]:
    """Extract failed online train tasks; missing success is conservatively a failure."""
    extras = online_output.non_tensor_batch.get("extra_info")
    success = online_output.non_tensor_batch.get("environment_success")
    if extras is None:
        raise KeyError("online rollout lacks extra_info needed for failure-aware replay")
    if success is None:
        success = np.full(len(online_output), None, dtype=object)
    failures = set()
    for extra, succeeded in zip(extras, success, strict=True):
        if not bool(succeeded):
            task_id = (extra or {}).get("task_id")
            if task_id:
                failures.add(str(task_id))
    return failures


def restore_rollout_identity(rollout_output, generation_input):
    """Restore dataset identity omitted by verl's async reward-loop output.

    Agent-loop outputs preserve request order, but in the async reward path they
    need not echo the input ``extra_info`` column.  Replay selection happens
    before the trainer unions output with its original batch, so copy only this
    identity column from the exact generation input when it is absent.
    """
    if "extra_info" in rollout_output.non_tensor_batch:
        return rollout_output
    source = generation_input.non_tensor_batch.get("extra_info")
    if source is None:
        raise KeyError("generation input lacks extra_info needed for expert replay")
    if len(source) != len(rollout_output):
        raise ValueError("generation input/output length mismatch while restoring replay identity")
    rollout_output.non_tensor_batch["extra_info"] = np.asarray(
        copy.deepcopy(list(source)), dtype=object
    )
    return rollout_output


def make_expert_input(dataset, collate_fn, selection, *, global_steps, temperature):
    """Build a normal verl dataset batch routed to the oracle agent loop."""
    from verl.protocol import DataProto

    samples = []
    for task_id, index in zip(selection["task_ids"], selection["indices"], strict=True):
        sample = dataset[index]
        sample["agent_name"] = "arithmetic_expert_replay_agent"
        sample["index"] = int(index)
        # Keep task identity explicit in both the normal extra_info and tracing fields.
        sample["extra_info"] = dict(sample.get("extra_info") or {})
        if sample["extra_info"].get("task_id") != task_id:
            raise ValueError("replay selection index no longer matches task_id")
        samples.append(sample)
    batch = DataProto.from_single_dict(collate_fn(samples))
    batch.non_tensor_batch["uid"] = np.array(
        [f"expert-replay-{global_steps}-{task_id}" for task_id in selection["task_ids"]], dtype=object
    )
    batch.meta_info["global_steps"] = global_steps
    # ``DataProto.to_tensordict`` promotes meta_info into the worker input.
    # The v0.8 FSDP engine requires a temperature entry even when it only
    # recomputes log-probabilities.  Expert replay is built on a separate
    # driver path, so it must explicitly mirror the online rollout setting.
    batch.meta_info["temperature"] = float(temperature)
    return batch


def mark_replay_rows(batch, expert: bool):
    batch.batch["expert_replay"] = torch.full(
        (len(batch),), bool(expert), dtype=torch.bool, device=batch.batch.device
    )


def _zero_rows_like(value: torch.Tensor, rows: int) -> torch.Tensor:
    return torch.zeros((rows, *value.shape[1:]), dtype=value.dtype, device=value.device)


def concat_for_joint_update(online, expert):
    """Concat while retaining online-only advantage fields as zero-valued expert placeholders."""
    from verl.protocol import DataProto

    mark_replay_rows(online, False)
    mark_replay_rows(expert, True)

    online_tensor_keys = set(online.batch.keys())
    expert_tensor_keys = set(expert.batch.keys())
    for key in online_tensor_keys - expert_tensor_keys:
        expert.batch[key] = _zero_rows_like(online.batch[key], len(expert))
    for key in expert_tensor_keys - online_tensor_keys:
        online.batch[key] = _zero_rows_like(expert.batch[key], len(online))

    online_non_tensor_keys = set(online.non_tensor_batch)
    expert_non_tensor_keys = set(expert.non_tensor_batch)
    for key in online_non_tensor_keys - expert_non_tensor_keys:
        expert.non_tensor_batch[key] = np.full(len(expert), None, dtype=object)
    for key in expert_non_tensor_keys - online_non_tensor_keys:
        online.non_tensor_batch[key] = np.full(len(online), None, dtype=object)

    # DataProto.concat requires overlapping metadata to be identical.  Actor update
    # reads its runtime values from tensors/non-tensors, so mirroring metadata is safe.
    expert.meta_info = copy.deepcopy(online.meta_info)
    return DataProto.concat([online, expert])

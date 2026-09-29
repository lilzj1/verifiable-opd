"""Select whole expert trajectories using a fixed assistant-token budget."""
from __future__ import annotations

import random


def select_replay(costs, failures, budget, seed, strategy="uniform", max_trajectories=None):
    if strategy not in ("uniform", "failure") or type(budget) is not int or budget < 0:
        raise ValueError("invalid strategy or token budget")
    if max_trajectories is not None and (type(max_trajectories) is not int or max_trajectories < 1):
        raise ValueError("max_trajectories must be a positive integer")
    if any(type(v) is not int or v < 1 for v in costs.values()):
        raise ValueError("expert token costs must be positive integers")
    if not set(failures).issubset(costs):
        raise ValueError("failures must refer only to the training pool")
    rng = random.Random(seed)
    keys = sorted(costs)
    rng.shuffle(keys)
    if strategy == "failure":
        keys.sort(key=lambda key: key not in failures)
    chosen, used = [], 0
    for key in keys:
        if max_trajectories is not None and len(chosen) >= max_trajectories:
            break
        if used + costs[key] <= budget:
            chosen.append(key)
            used += costs[key]
    return {"task_ids": chosen, "tokens": used, "budget": budget,
            "failure_selected": sum(key in failures for key in chosen),
            "uniform_fallback": sum(key not in failures for key in chosen) if strategy == "failure" else 0}

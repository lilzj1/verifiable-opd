"""Explicit no-task-reward adapter for sampled OPD, NOT an evaluation metric."""


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source not in {"math", "if"}:
        raise ValueError(f"Unmapped teacher domain: {data_source}")
    return {"score": 0.0}

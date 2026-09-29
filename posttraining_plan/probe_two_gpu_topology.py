"""Exercise the real local verl resource validator without allocating GPUs.

Does not load model weights or measure device memory. Fake model paths are
intentional: validation happens before loading and cannot authorize a launch.
"""
import argparse
import inspect
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("This resource-validator test is CPU-only")
    from verl.workers.config.distillation import DistillationConfig, DistillationTeacherModelConfig
    from verl.workers.config.rollout import RolloutConfig

    root = Path(__file__).resolve().parent.parent
    actual = Path(inspect.getfile(DistillationConfig)).resolve()
    expected = (root / "upstream/verl-v0.8.0/verl/workers/config/distillation.py").resolve()
    if actual != expected:
        raise RuntimeError(f"Wrong verl imported: {actual}")

    results = []
    for teacher_pool in (1, 2):
        teachers = {"teacher_model": DistillationTeacherModelConfig()}
        for domain in ("math", "if"):
            teachers[f"{domain}_teacher"] = DistillationTeacherModelConfig(
                key=domain, model_path=f"/NOT_A_REAL_MODEL/{domain}", num_replicas=1,
                inference=RolloutConfig(name="vllm", tensor_model_parallel_size=1,
                    data_parallel_size=1, pipeline_model_parallel_size=1,
                    prompt_length=2048, response_length=4096, max_model_len=8192),
            )
        item = {"teacher_pool_gpus": teacher_pool, "teachers": 2, "teacher_tp_each": 1}
        try:
            config = DistillationConfig(enabled=True, n_gpus_per_node=teacher_pool,
                                        nnodes=1, teacher_models=teachers)
            item.update(result="CONFIG_VALID", resolved_keys=sorted(config.teacher_models))
        except ValueError as exc:
            item.update(result="CONFIG_REJECTED", reason=str(exc))
        results.append(item)
    report = {"kind": "REAL_VERL_CONFIG_CHECK_NOT_GPU_MEMORY_TEST", "source": str(actual),
              "cases": results,
              "conclusion": "Current separate-pool topology needs >=2 teacher GPUs plus >=1 student GPU. "
                            "Two GPUs require scheduling/colocation changes, not merely smaller microbatches."}
    if results[0]["result"] != "CONFIG_REJECTED" or results[1]["result"] != "CONFIG_VALID":
        raise RuntimeError(f"Unexpected validator behavior: {report}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

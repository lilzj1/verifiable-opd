"""Summarize the frozen formal training logs and greedy evaluations.

This script deliberately reads artifacts instead of rerunning training or
evaluation.  It makes the comparison auditable and keeps the one-seed / small
test-set limitations visible in the generated report.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path


ANSI = re.compile(r"\x1b\[[0-9;]*m")
NUMBER = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
EXPERIMENTS = {
    "opd": "formal_opd_s50_seed17_v2",
    "uniform": "formal_uniform_s50_seed17_v1",
    "failure": "formal_failure_s50_seed17_v1",
}
SPLITS = ("dev", "test_id", "test_ood")


def _metric(line: str, name: str) -> float | None:
    match = re.search(rf"(?:^| - ){re.escape(name)}:(?:np\.\w+\()?{NUMBER}", line)
    return float(match.group(1)) if match else None


def parse_training_log(path: Path) -> dict[str, float | int | bool]:
    text = ANSI.sub("", path.read_text(errors="replace"))
    lines = [line for line in text.splitlines() if re.search(r"(?:^|\s)step:\d+ - ", line)]
    if not lines:
        raise ValueError(f"no step metrics found in {path}")

    def values(name: str) -> list[float]:
        return [value for line in lines if (value := _metric(line, name)) is not None]

    step_ids = [int(re.search(r"(?:^|\s)step:(\d+) - ", line).group(1)) for line in lines]
    distill = values("actor/distillation/loss")
    actor_loss = values("actor/loss")
    grad_norm = values("actor/grad_norm")
    step_seconds = values("timing_s/step")
    save_seconds = values("timing_s/save_checkpoint")
    selected = values("expert_replay/selected_action_tokens")
    trajectories = values("expert_replay/trajectories")
    failures = values("expert_replay/failure_selected")
    fallbacks = values("expert_replay/uniform_fallback")
    expert_ce = values("actor/expert_replay/ce")
    weighted_ce = values("actor/expert_replay/weighted_ce")
    checked = distill + actor_loss + grad_norm + expert_ce + weighted_ce

    result: dict[str, float | int | bool] = {
        "steps": len(lines),
        "first_step": min(step_ids),
        "last_step": max(step_ids),
        "all_finite": bool(checked) and all(math.isfinite(value) for value in checked),
        "distill_loss_first": distill[0],
        "distill_loss_last": distill[-1],
        "distill_loss_mean": statistics.fmean(distill),
        "actor_loss_mean": statistics.fmean(actor_loss),
        "grad_norm_mean": statistics.fmean(grad_norm),
        "grad_norm_max": max(grad_norm),
        "step_seconds_median": statistics.median(step_seconds),
        "step_seconds_sum": sum(step_seconds),
        "checkpoint_seconds": max(save_seconds, default=0.0),
    }
    if selected:
        result.update(
            replay_action_tokens_sum=int(sum(selected)),
            replay_action_tokens_mean=statistics.fmean(selected),
            replay_trajectories_sum=int(sum(trajectories)),
            failure_selected_sum=int(sum(failures)),
            failure_selected_fraction=(sum(failures) / sum(trajectories)) if sum(trajectories) else 0.0,
            uniform_fallback_sum=int(sum(fallbacks)),
            expert_ce_mean=statistics.fmean(expert_ce),
            weighted_ce_mean=statistics.fmean(weighted_ce),
        )
    return result


def _local_action_metrics(summary: dict, episodes_path: Path) -> dict[str, float | int]:
    data_path = Path(summary["config"]["data"])
    records = {
        row["task_id"]: row
        for row in (json.loads(line) for line in data_path.read_text().splitlines() if line.strip())
    }
    correct = 0
    attempts = 0
    for line in episodes_path.read_text().splitlines():
        trace = json.loads(line)
        task = records[trace["task_id"]]["task"]
        value = task["initial"]
        operation_index = 0
        for action in trace["actions"]:
            attempts += 1
            observation = action["observation"]
            # A valid sparse-mode action advances exactly one operation.  Its
            # submitted value is the next observation's public state.
            if observation.get("step") != operation_index + 1:
                continue
            kind, operand = task["operations"][operation_index]
            raw = value + operand if kind == "add" else value * operand
            expected = raw % task["modulus"]
            submitted = observation["value"]
            correct += int(submitted == expected)
            value = submitted
            operation_index += 1
    return {
        "correct_local_actions": correct,
        "action_attempts": attempts,
        "local_action_accuracy": correct / attempts if attempts else 0.0,
    }


def load_evaluations(root: Path) -> dict[str, dict[str, dict]]:
    evaluations: dict[str, dict[str, dict]] = {}
    for model in ("sft", *EXPERIMENTS):
        evaluations[model] = {}
        for split in SPLITS:
            path = root / model / split / "summary.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            summary = json.loads(path.read_text())
            summary.update(_local_action_metrics(summary, path.with_name("episodes.jsonl")))
            evaluations[model][split] = summary
    return evaluations


def markdown_report(payload: dict) -> str:
    lines = [
        "# Verifiable-OPD 正式实验结果",
        "",
        "固定协议：Qwen2.5-1.5B SFT 起点、Qwen2.5-7B LoRA 冻结教师、seed=17、50 个训练 step。",
        "回放系数 0.1、动作 token 上限 512；评测为 seed=123 的 greedy 解码，每个 split 24 题。",
        "",
        "## 冻结评测",
        "",
        "| 模型 | dev | ID test | OOD test | 局部动作准确率（dev/ID/OOD） | 非法动作（dev/ID/OOD） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, split_data in payload["evaluation"].items():
        success = [f'{split_data[s]["success_rate"]:.3f}' for s in SPLITS]
        action_accuracy = "/".join(f'{split_data[s]["local_action_accuracy"]:.3f}' for s in SPLITS)
        invalid = "/".join(str(split_data[s]["invalid_actions"]) for s in SPLITS)
        lines.append(f"| {model} | {' | '.join(success)} | {action_accuracy} | {invalid} |")

    lines.extend(
        [
            "",
            "## 训练与回放审计",
            "",
            "| 方案 | steps | 有限数值 | 蒸馏 loss 均值 | grad norm 均值/最大 | step中位数/累计秒 | replay tokens | 失败优先占比 | fallback |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model, metrics in payload["training"].items():
        replay = metrics.get("replay_action_tokens_sum", 0)
        failure_fraction = metrics.get("failure_selected_fraction", 0.0)
        fallback = metrics.get("uniform_fallback_sum", 0)
        lines.append(
            f'| {model} | {metrics["steps"]} | {metrics["all_finite"]} | '
            f'{metrics["distill_loss_mean"]:.4f} | {metrics["grad_norm_mean"]:.2f}/'
            f'{metrics["grad_norm_max"]:.2f} | {metrics["step_seconds_median"]:.2f}/'
            f'{metrics["step_seconds_sum"]:.1f} | {replay} | {failure_fraction:.3f} | {fallback} |'
        )

    lines.extend(
        [
            "",
            "## 本轮观察",
            "",
            "- SFT起点在ID test为0/24；纯OPD和均匀回放均为1/24，失败感知回放为0/24。样本太小，只能记为初步现象。",
            "- 纯OPD在dev/ID/OOD的局部动作准确率均最高，说明本轮回放没有提高算术决策准确率。",
            "- 均匀回放在三个split的非法动作数均最低，说明它更好地保持了工具调用格式，但该收益没有转化成更高的长链成功率。",
            "- 失败感知分支的100条回放轨迹全部来自本批失败任务、没有fallback，机制确实触发；它仍未优于均匀回放，因此不能把负结果解释成分支没运行。",
            "- 所有观测到的episode成功都来自长度3；长度5、8、12、16均为0，当前模型没有可靠的中长链执行能力。",
            "- 四个模型的OOD成功率均为0/24。50步训练不足以获得长度12/16的长程泛化。",
            "- 专家回放将训练step中位耗时从约8秒提高到约10秒，并实际使用约1.1万专家动作token；收益判断必须包含这部分成本。",
            "",
            "## 解释边界",
            "",
            "- 这是一个 seed、每个 split 24 题的小规模工程对照，不构成统计显著性结论。",
            "- 等训练 step 不等于等计算量；表中保留实际 replay token，原始日志保留逐步耗时。",
            "- dev 可用于诊断；ID/OOD test 只做冻结后的最终报告，不能据其回调超参数。",
            "- 若回放没有提升，结论应是该配置下未观察到收益，而不是删去负结果。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--eval-root", type=Path, default=Path("artifacts/formal_eval_v1"))
    parser.add_argument("--json", type=Path, default=Path("artifacts/formal_results_v1.json"))
    parser.add_argument("--markdown", type=Path, default=Path("FORMAL_RESULTS.md"))
    args = parser.parse_args()

    training = {
        name: parse_training_log(args.artifacts / f"{experiment}_launcher.log")
        for name, experiment in EXPERIMENTS.items()
    }
    payload = {"training": training, "evaluation": load_evaluations(args.eval_root)}
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.markdown.write_text(markdown_report(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

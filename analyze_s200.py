"""Summarize the frozen 200-step OPD continuation and its evaluations."""

from __future__ import annotations

import json
from pathlib import Path

from analyze_formal import SPLITS, _local_action_metrics, parse_training_log


ARTIFACTS = Path("artifacts")
EVAL_ROOT = ARTIFACTS / "formal_eval_s200_v1"
EXPERIMENTS = {
    "opd": "formal_opd_s200_seed17_v1",
    "uniform": "formal_uniform_s200_seed17_v1",
}


def load_evaluations() -> dict[str, dict[str, dict]]:
    result: dict[str, dict[str, dict]] = {}
    for mode in EXPERIMENTS:
        result[mode] = {}
        for split in SPLITS:
            path = EVAL_ROOT / mode / split / "summary.json"
            summary = json.loads(path.read_text())
            summary.update(_local_action_metrics(summary, path.with_name("episodes.jsonl")))
            result[mode][split] = summary
    return result


def markdown(payload: dict) -> str:
    lines = [
        "# Verifiable-OPD 200-step 延长实验",
        "",
        "两组实验分别从各自已保存的 step 50 FSDP checkpoint 恢复模型、optimizer、",
        "LR scheduler 与 RNG，续训至总计 200 step。训练与评测其余协议保持不变。",
        "评测使用 seed=123 的 greedy 解码，每个 split 完整 24 题，无 limit。",
        "",
        "## 冻结评测",
        "",
        "| 模型 | dev | ID test | OOD test | 局部动作准确率（dev/ID/OOD） | 非法动作（dev/ID/OOD） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, splits in payload["evaluation"].items():
        success = " | ".join(f'{splits[s]["success_rate"]:.3f}' for s in SPLITS)
        accuracy = "/".join(f'{splits[s]["local_action_accuracy"]:.3f}' for s in SPLITS)
        invalid = "/".join(str(splits[s]["invalid_actions"]) for s in SPLITS)
        lines.append(f"| {mode} | {success} | {accuracy} | {invalid} |")

    lines.extend(
        [
            "",
            "## 续训区间审计（step 51–200）",
            "",
            "| 方案 | 新增 steps | 有限数值 | 蒸馏 loss 均值 | grad norm 均值/最大 | step 中位数/累计秒 | 新增 replay tokens |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode, metrics in payload["training"].items():
        lines.append(
            f'| {mode} | {metrics["steps"]} | {metrics["all_finite"]} | '
            f'{metrics["distill_loss_mean"]:.4f} | {metrics["grad_norm_mean"]:.2f}/'
            f'{metrics["grad_norm_max"]:.2f} | {metrics["step_seconds_median"]:.2f}/'
            f'{metrics["step_seconds_sum"]:.1f} | {metrics.get("replay_action_tokens_sum", 0)} |'
        )

    lines.extend(
        [
            "",
            "## 与 50-step 结果的对照",
            "",
            "| 方案 | 阶段 | dev/ID/OOD 成功率 | 局部动作准确率（dev/ID/OOD） | 非法动作（dev/ID/OOD） |",
            "|---|---|---:|---:|---:|",
        ]
    )
    old = payload["step50_evaluation"]
    new = payload["evaluation"]
    for mode in EXPERIMENTS:
        for label, data in (("50", old[mode]), ("200", new[mode])):
            success = "/".join(f'{data[s]["success_rate"]:.3f}' for s in SPLITS)
            accuracy = "/".join(f'{data[s]["local_action_accuracy"]:.3f}' for s in SPLITS)
            invalid = "/".join(str(data[s]["invalid_actions"]) for s in SPLITS)
            lines.append(f"| {mode} | {label} | {success} | {accuracy} | {invalid} |")

    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "- 200-step 纯 OPD 的 dev/ID 均为 1/24，均匀回放的 dev 为 1/24、ID 为 0/24；两者 OOD 均为 0/24。",
            "- 成功仍只出现在长度 3；长度 5/8/12/16 没有成功样本。",
            "- 纯 OPD 在 dev/ID 的局部动作准确率更高；均匀回放在 OOD 的局部准确率更高且非法动作更少，但没有转化为整条轨迹成功。",
            "- 从 50 续训到 200 step 未提高完整 episode 成功率；在当前单 seed、24 题/split 条件下，不能宣称均匀回放带来性能收益。",
            "- 结果支持的是工程链路、稳定训练和负结果分析，不构成统计显著性或通用多步泛化结论。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    old_payload = json.loads((ARTIFACTS / "formal_results_v1.json").read_text())
    payload = {
        "training": {
            mode: parse_training_log(ARTIFACTS / f"{experiment}_launcher.log")
            for mode, experiment in EXPERIMENTS.items()
        },
        "evaluation": load_evaluations(),
        "step50_evaluation": {mode: old_payload["evaluation"][mode] for mode in EXPERIMENTS},
    }
    (ARTIFACTS / "formal_results_s200_v1.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    Path("FORMAL_RESULTS_S200.md").write_text(markdown(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""verl loss for one optimizer update containing online OPD and expert CE.

This extension intentionally supports only the project's frozen setting:
single-sample k1, policy-gradient OPD, no task-reward branch, token-mean.
The two terms use independent *global* action-token denominators.
"""
from __future__ import annotations

import torch

from verl.trainer.ppo.core_algos import get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_sum
from verl.workers.utils.padding import no_padding_2_padding


def split_replay_masks(response_mask: torch.Tensor, expert_replay: torch.Tensor):
    if response_mask.is_nested:
        response_mask = response_mask.to_padded_tensor(False)
    expert_replay = expert_replay.bool().reshape(-1)
    if response_mask.shape[0] != expert_replay.shape[0]:
        raise ValueError("expert replay row mask does not match batch size")
    expert_mask = response_mask.bool() & expert_replay[:, None]
    online_mask = response_mask.bool() & (~expert_replay[:, None])
    return online_mask, expert_mask


def mixed_opd_replay_loss(
    config,
    distillation_config,
    model_output=None,
    data=None,
    dp_group=None,
    student_logits=None,
    data_format="thd",
):
    del dp_group, data_format
    if student_logits is not None:
        raise ValueError("mixed replay is implemented for sampled k1, not top-k logits")
    loss_cfg = distillation_config.distillation_loss
    if (
        loss_cfg.loss_mode != "k1"
        or not loss_cfg.use_policy_gradient
        or loss_cfg.use_task_rewards
        or config.loss_agg_mode != "token-mean"
    ):
        raise ValueError("mixed replay requires k1 PG-OPD, no task rewards, and token-mean")
    if "expert_replay" not in data:
        raise KeyError("mixed replay batch lacks expert_replay row markers")

    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_prob = no_padding_2_padding(data["teacher_logprobs"], data)
    # Only remove an explicit top-k=1 dimension. A [batch, response=1]
    # tensor must retain its response axis.
    if teacher_log_prob.ndim == log_prob.ndim + 1 and teacher_log_prob.shape[-1] == 1:
        teacher_log_prob = teacher_log_prob.squeeze(-1)
    old_log_prob = data["old_log_probs"]
    if old_log_prob.is_nested:
        old_log_prob = old_log_prob.to_padded_tensor(0.0)
    online_mask, expert_mask = split_replay_masks(data["response_mask"], data["expert_replay"])
    if log_prob.shape != teacher_log_prob.shape or log_prob.shape != online_mask.shape:
        raise ValueError("student, teacher, and action masks are not aligned")

    online_total = int(tu.get_non_tensor_data(data, "opd_online_action_tokens", None))
    expert_total = int(tu.get_non_tensor_data(data, "opd_expert_action_tokens", None))
    dp_size = int(tu.get_non_tensor_data(data, "dp_size", None))
    if online_total <= 0 or expert_total <= 0:
        raise ValueError("global OPD/replay token denominators must be positive")

    # Mask BEFORE arithmetic: multiplication by zero does not remove NaN.
    # Teacher and old-policy scores are irrelevant on expert/padding rows.
    log_prob = torch.where(online_mask | expert_mask, log_prob, 0.0)
    online_log_prob = torch.where(online_mask, log_prob, 0.0)
    teacher_log_prob = torch.where(online_mask, teacher_log_prob, 0.0)
    old_log_prob = torch.where(online_mask, old_log_prob, 0.0)
    token_kl = kl_penalty(logprob=online_log_prob, ref_logprob=teacher_log_prob, kl_penalty="k1")
    # The official PG-OPD path treats -KL as a detached token advantage.
    loss_cfg.global_batch_info.update(
        {
            "dp_size": dp_size,
            "batch_num_tokens": online_total,
            "global_batch_size": None,
            "loss_scale_factor": config.loss_scale_factor,
        }
    )
    if online_mask.any():
        policy_loss_fn = get_policy_loss_fn(loss_cfg.policy_loss_mode)
        opd_loss, opd_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=online_log_prob,
            advantages=-token_kl.detach(),
            response_mask=online_mask,
            loss_agg_mode="token-mean",
            config=loss_cfg,
            rollout_is_weights=data.get("rollout_is_weights", None),
        )
        abs_kl = (masked_sum(token_kl.abs(), online_mask) / online_total * dp_size).detach()
    else:
        # A DP rank or microbatch may legitimately contain only expert rows.
        opd_loss = log_prob.sum() * 0.0
        opd_metrics = {}
        abs_kl = log_prob.new_zeros(())
    # Expert rows contain the exact oracle action tokens; observations are masked.
    sft_loss = -masked_sum(log_prob, expert_mask) / expert_total * dp_size
    total = opd_loss + loss_cfg.expert_replay_alpha * sft_loss

    metrics = {
        f"distillation/{key.removeprefix('actor/')}": value for key, value in opd_metrics.items()
    }
    metrics.update(
        {
            "distillation/loss": Metric(AggregationType.SUM, opd_loss.detach()),
            "expert_replay/ce": Metric(AggregationType.SUM, sft_loss.detach()),
            "expert_replay/weighted_ce": Metric(
                AggregationType.SUM, (loss_cfg.expert_replay_alpha * sft_loss).detach()
            ),
            "expert_replay/total_loss": Metric(AggregationType.SUM, total.detach()),
            "expert_replay/online_action_tokens": Metric(AggregationType.SUM, online_mask.sum().detach()),
            "expert_replay/action_tokens": Metric(AggregationType.SUM, expert_mask.sum().detach()),
            "distillation/abs_loss": Metric(AggregationType.SUM, abs_kl),
        }
    )
    return total, metrics

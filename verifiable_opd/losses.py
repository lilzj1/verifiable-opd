"""Reference token-level OPD surrogate and independently normalized SFT loss.

This is a single fresh-policy update, not a PPO multi-epoch implementation.
We stop gradients through the sampled log-ratio coefficient. It is a local
token-level reverse-KL surrogate on visited states, not the exact trajectory-KL
gradient through all future environment states.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .tokenization import causal_target_slice


def decision_logprobs(model, decision):
    device = next(model.parameters()).device
    ids = torch.tensor([decision.prefix + decision.response], dtype=torch.long, device=device)
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    span = causal_target_slice(len(decision.prefix), len(decision.response))
    logits = out.logits[0, span].float()
    targets = ids[0, len(decision.prefix):]
    return -F.cross_entropy(logits, targets, reduction="none")


def opd_surrogate(current_logp, rollout_logp, teacher_logp):
    if current_logp.shape != rollout_logp.shape or current_logp.shape != teacher_logp.shape:
        raise ValueError("logprob alignment mismatch")
    if current_logp.numel() == 0:
        raise ValueError("empty action loss")
    coefficient = (rollout_logp - teacher_logp).detach().to(current_logp.device)
    return (coefficient * current_logp).sum()


def backward_update(student, teacher, optimizer, online, experts, alpha, max_grad_norm=1.0):
    if alpha < 0 or (not online and not experts):
        raise ValueError("invalid update inputs")
    if online and teacher is None:
        raise ValueError("online distillation needs a teacher")
    if alpha > 0 and not experts:
        raise ValueError("positive replay coefficient but no experts fit the budget")
    online_tokens = sum(len(d.response) for d in online)
    expert_tokens = sum(len(d.response) for d in experts)
    optimizer.zero_grad(set_to_none=True)
    # Disable dropout during both fresh-policy sampling and differentiable scoring.
    # eval() does not disable gradients.
    student.eval()
    if teacher is not None:
        teacher.eval()
        if any(p.requires_grad for p in teacher.parameters()):
            raise ValueError("teacher must be frozen")
    surrogate = 0.0
    sampled_kl = 0.0
    ce = 0.0
    for decision in online:
        with torch.no_grad():
            teacher_lp = decision_logprobs(teacher, decision)
        current = decision_logprobs(student, decision)
        # No optimizer step has occurred since rollout; this is the same policy.
        old = current.detach()
        loss = opd_surrogate(current, old, teacher_lp.to(current.device)) / online_tokens
        loss.backward()
        surrogate += float(loss.detach())
        sampled_kl += float((old - teacher_lp.to(old.device)).sum()) / online_tokens
    sft_weight = alpha if online else 1.0
    if sft_weight:
        for decision in experts:
            loss = -decision_logprobs(student, decision).sum() / expert_tokens
            (sft_weight * loss).backward()
            ce += float(loss.detach())
    norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return {"opd_surrogate": surrogate, "sampled_reverse_kl": sampled_kl,
            "sft_ce": ce, "alpha": sft_weight, "grad_norm": float(norm),
            "online_action_tokens": online_tokens, "expert_action_tokens": expert_tokens}

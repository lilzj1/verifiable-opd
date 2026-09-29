"""Exact-oracle rollout used only by the expert replay branch.

The loop deliberately does not ask the student server to generate.  It renders the
same Qwen chat template as the online tool loop and marks only assistant tool-call
tokens as trainable.  Tool observations and generation prompts remain context with
response_mask=0.
"""
from __future__ import annotations

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

from .data import expert_messages
from .environment import Task
from .tokenization import expert_decisions, prompt_ids


def task_from_tools_kwargs(tools_kwargs: dict) -> Task:
    try:
        task_json = tools_kwargs["submit_value"]["create_kwargs"]["task_json"]
    except (KeyError, TypeError) as exc:
        raise ValueError("expert replay requires submit_value.create_kwargs.task_json") from exc
    import json

    return Task.from_dict(json.loads(task_json))


def render_expert_trajectory(tokenizer, task: Task) -> tuple[list[int], list[int], list[int], int]:
    """Return prompt, response, action mask, and number of assistant turns."""
    messages = expert_messages(task)
    decisions = expert_decisions(tokenizer, messages)
    if not decisions:
        raise ValueError("expert trajectory has no assistant decisions")

    terminal_prefix = prompt_ids(tokenizer, messages)
    initial_prompt = decisions[0].prefix
    response: list[int] = []
    response_mask: list[int] = []
    cursor = list(initial_prompt)

    for index, decision in enumerate(decisions):
        if cursor != decision.prefix:
            raise ValueError("expert chat template lost prefix alignment")
        response.extend(decision.response)
        response_mask.extend([1] * len(decision.response))
        cursor.extend(decision.response)

        next_prefix = decisions[index + 1].prefix if index + 1 < len(decisions) else terminal_prefix
        if next_prefix[: len(cursor)] != cursor:
            raise ValueError("tool observation is not suffix-compatible with expert action")
        observation_suffix = next_prefix[len(cursor) :]
        response.extend(observation_suffix)
        response_mask.extend([0] * len(observation_suffix))
        cursor = list(next_prefix)

    if len(response) != len(response_mask) or sum(response_mask) == 0:
        raise ValueError("invalid expert response mask")
    return list(initial_prompt), response, response_mask, len(decisions)


def expert_action_token_cost(tokenizer, task: Task) -> int:
    return sum(render_expert_trajectory(tokenizer, task)[2])


@register("arithmetic_expert_replay_agent")
class ArithmeticExpertReplayAgentLoop(ToolAgentLoop):
    """Produce a complete correct trajectory without student generation."""

    async def run(self, sampling_params, priority: int = 0, **kwargs) -> AgentLoopOutput:
        del sampling_params, priority
        task = task_from_tools_kwargs(kwargs.get("tools_kwargs", {}))
        prompt, response, response_mask, assistant_turns = render_expert_trajectory(self.tokenizer, task)
        if len(prompt) > self.prompt_length:
            raise ValueError(
                f"expert prompt has {len(prompt)} tokens, exceeding prompt_length={self.prompt_length}"
            )
        if len(response) > self.response_length:
            raise ValueError(
                f"expert response has {len(response)} tokens, exceeding response_length={self.response_length}"
            )
        return AgentLoopOutput(
            prompt_ids=prompt,
            response_ids=response,
            response_mask=response_mask,
            reward_score=1.0,
            num_turns=2 * assistant_turns + 2,
            metrics=AgentLoopMetrics(),
            extra_fields={
                "environment_success": True,
                "environment_done": True,
                "episode_success": True,
                "expert_replay_generated": True,
                "turn_scores": [],
                "tool_rewards": [],
            },
        )

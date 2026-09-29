"""Small verl integration: expose exact environment success as trajectory reward."""

from __future__ import annotations

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop, AgentState


def finalize_environment_fields(extra_fields):
    """Keep outcome columns batch-aligned even when no tool call was parsed."""
    success = bool(extra_fields.get("environment_success", False))
    extra_fields.setdefault("environment_success", success)
    extra_fields.setdefault("environment_done", False)
    extra_fields["episode_success"] = success
    return float(success)


@register("arithmetic_tool_agent")
class ArithmeticToolAgentLoop(ToolAgentLoop):
    async def _handle_processing_tools_state(self, agent_data):
        state = await super()._handle_processing_tools_state(agent_data)
        env = getattr(agent_data, "_verifiable_opd_env", None)
        if env is not None:
            # Set the authoritative outcome from environment state. A later tool
            # error must never replace this with an unrelated last tool reward.
            agent_data.extra_fields["environment_success"] = env.success
            agent_data.extra_fields["environment_done"] = env.done
            if env.done:
                return AgentState.TERMINATED
        return state

    async def run(self, sampling_params, priority: int = 0, **kwargs):
        output = await super().run(sampling_params, priority=priority, **kwargs)
        output.reward_score = finalize_environment_fields(output.extra_fields)
        return output

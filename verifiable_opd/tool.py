"""Stateful verl tool for the arithmetic environment."""

from __future__ import annotations

import base64
import json
from typing import Any

from .environment import ArithmeticEnv, Task

try:
    from verl.tools.base_tool import BaseTool
    from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
except ImportError:  # Allows environment/data tests without installing verl.
    BaseTool = object
    OpenAIFunctionToolSchema = Any
    ToolResponse = None


class ArithmeticTool(BaseTool):
    """Keep per-trajectory state on verl's AgentData object.

    ToolAgentLoop calls ``create`` for every function call. Therefore the encoded
    instance id carries immutable task configuration, while the mutable env lives
    on AgentData, which is unique to one trajectory.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        if ToolResponse is None:
            raise ImportError("ArithmeticTool requires verl on PYTHONPATH")
        super().__init__(config, tool_schema)

    async def create(self, instance_id: str | None = None, **kwargs):
        create_kwargs = kwargs.get("create_kwargs", {})
        raw = json.dumps(create_kwargs, sort_keys=True, separators=(",", ":")).encode()
        encoded = base64.urlsafe_b64encode(raw).decode()
        return encoded, ToolResponse()

    @staticmethod
    def _decode_instance(instance_id: str) -> dict[str, Any]:
        raw = base64.urlsafe_b64decode(instance_id.encode()).decode()
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("invalid tool instance payload")
        return value

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        agent_data = kwargs.get("agent_data")
        if agent_data is None:
            raise ValueError("ArithmeticTool requires agent_data")
        env = getattr(agent_data, "_verifiable_opd_env", None)
        if env is None:
            payload = self._decode_instance(instance_id)
            task = Task.from_dict(json.loads(payload["task_json"]))
            env = ArithmeticEnv(task, mode=payload.get("mode", "sparse"), max_calls=payload.get("max_calls"))
            setattr(agent_data, "_verifiable_opd_env", env)

        observation, reward, done = env.step(json.dumps(parameters, separators=(",", ":")))
        metrics = {
            "episode_done": float(done),
            "episode_success": float(env.success),
            "env_calls": float(env.calls),
        }
        return ToolResponse(text=json.dumps(observation, separators=(",", ":"))), reward, metrics

    async def release(self, instance_id: str, **kwargs) -> None:
        # State belongs to AgentData and is discarded when that trajectory ends.
        return None

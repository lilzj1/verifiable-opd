"""Explicit integration check using the pinned, real verl classes; no GPU."""
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import HermesToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema
from verifiable_opd.agent_loop import ArithmeticToolAgentLoop, finalize_environment_fields
from verifiable_opd.environment import Task
from verifiable_opd.prompts import TOOL_SCHEMA
from verifiable_opd.tool import ArithmeticTool


def agent():
    return AgentData([], [], [], None, {}, {}, "test", {})


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_early_parse_failure_has_batch_stable_outcome_fields(self):
        fields = {}
        self.assertEqual(finalize_environment_fields(fields), 0.0)
        self.assertEqual(
            fields,
            {
                "environment_success": False,
                "environment_done": False,
                "episode_success": False,
            },
        )

    async def test_hermes_closing_tag_is_a_per_turn_stop_token(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                self.assert_args = (text, add_special_tokens)
                return [151658]

        parser = HermesToolParser(Tokenizer())
        self.assertEqual(parser.stop_token_ids, [151658])

    async def test_recreate_tool_preserves_trajectory_state_and_isolates_episodes(self):
        tool = ArithmeticTool({}, OpenAIFunctionToolSchema.model_validate(TOOL_SCHEMA))
        payload = {"task_json": json.dumps(Task(2, (("add", 3), ("mul", 2))).to_dict())}
        first, other = agent(), agent()
        iid, _ = await tool.create(create_kwargs=payload)
        _, reward, metrics = await tool.execute(iid, {"value": 5}, agent_data=first)
        self.assertEqual(reward, 0)
        self.assertEqual(metrics["env_calls"], 1)
        await tool.release(iid)
        iid, _ = await tool.create(create_kwargs=payload)
        _, reward, metrics = await tool.execute(iid, {"value": 10}, agent_data=first)
        self.assertEqual(reward, 1)
        self.assertEqual(metrics["env_calls"], 2)
        _, _, metrics = await tool.execute(iid, {"value": 5}, agent_data=other)
        self.assertEqual(metrics["env_calls"], 1)
        self.assertFalse(other._verifiable_opd_env.done)

        # The parent execution is mocked; the real environment and subclass
        # termination policy are exercised. This is not a vLLM rollout test.
        loop = object.__new__(ArithmeticToolAgentLoop)
        with patch.object(ToolAgentLoop, "_handle_processing_tools_state",
                          new=AsyncMock(return_value=AgentState.GENERATING)):
            state = await loop._handle_processing_tools_state(first)
        self.assertEqual(state, AgentState.TERMINATED)
        self.assertTrue(first.extra_fields["environment_success"])

    async def test_missing_agent_is_rejected(self):
        tool = ArithmeticTool({}, OpenAIFunctionToolSchema.model_validate(TOOL_SCHEMA))
        with self.assertRaisesRegex(ValueError, "agent_data"):
            await tool.execute("", {"value": 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Pure environment interaction shared by evaluation and replay diagnostics."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .data import canonical, initial_messages
from .environment import ArithmeticEnv, Task


@dataclass
class ActionResult:
    parameters: dict | None
    error: str | None


def parse_action(text):
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if len(matches) != 1:
        return ActionResult(None, "expected_exactly_one_tool_call")
    try:
        value = json.loads(matches[0])
    except (ValueError, TypeError):
        return ActionResult(None, "invalid_tool_json")
    if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
        return ActionResult(None, "invalid_tool_fields")
    if value["name"] != "submit_value" or not isinstance(value["arguments"], dict):
        return ActionResult(None, "invalid_tool_name_or_arguments")
    return ActionResult(value["arguments"], None)


def assistant_message(text):
    return {"role": "assistant", "content": text}


def execute_text(env, text):
    parsed = parse_action(text)
    action = canonical(parsed.parameters) if parsed.parameters is not None else "invalid"
    observation, reward, done = env.step(action)
    return observation, reward, done, parsed.error


def run_episode(record, policy, mode="sparse", max_calls=None):
    task = Task.from_dict(record["task"])
    env = ArithmeticEnv(task, mode, max_calls)
    messages = initial_messages(task)
    actions = []
    invalid = 0
    while not env.done:
        text = policy(messages)
        observation, reward, done, error = execute_text(env, text)
        invalid += int(error is not None or observation.get("feedback") == "invalid_json_or_value")
        actions.append({"text": text, "observation": observation, "reward": reward, "parse_error": error})
        messages.extend([assistant_message(text), {"role": "tool", "content": canonical(observation)}])
    return {"task_id": record["task_id"], "split": record["split"], "length": record["length"],
            "success": env.success, "calls": env.calls, "invalid_actions": invalid, "actions": actions}

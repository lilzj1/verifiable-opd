"""Shared prompts and tool schema; one source of truth for SFT and RL."""

from __future__ import annotations

import json

from .environment import Task


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_value",
        "description": "Submit the next modular-arithmetic state value.",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "integer",
                    "description": "The next state value after applying exactly one operation.",
                }
            },
            "required": ["value"],
        },
    },
}

SYSTEM_PROMPT = (
    "You are a tool-using arithmetic agent. Apply exactly one public operation at a time, "
    "reduce modulo the stated modulus, and call submit_value with only the next integer value. "
    "After the tool reports done=true, the episode ends."
)


def user_prompt(task: Task) -> str:
    operations = [f"{op} {arg}" for op, arg in task.operations]
    return (
        f"Initial value: {task.initial}. Modulus: {task.modulus}. "
        f"Operations in order: {json.dumps(operations)}. Execute the full chain."
    )

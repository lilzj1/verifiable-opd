"""Deterministic multi-step arithmetic environment used by SFT and OPD."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Task:
    initial: int
    operations: tuple[tuple[str, int], ...]
    modulus: int = 97

    def __post_init__(self) -> None:
        if type(self.modulus) is not int or self.modulus < 2:
            raise ValueError("modulus must be an integer >= 2")
        if type(self.initial) is not int or not 0 <= self.initial < self.modulus:
            raise ValueError("initial outside state space")
        if not isinstance(self.operations, tuple) or not self.operations:
            raise ValueError("nonempty tuple of operations required")
        for operation in self.operations:
            if not isinstance(operation, tuple) or len(operation) != 2:
                raise ValueError("each operation must be a pair")
            op, arg = operation
            if op not in ("add", "mul") or type(arg) is not int:
                raise ValueError("invalid operation")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["operations"] = [list(operation) for operation in self.operations]
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Task":
        if not isinstance(value, dict):
            raise ValueError("task must be an object")
        if set(value) != {"initial", "operations", "modulus"}:
            raise ValueError("task keys must be initial, operations and modulus")
        operations = value["operations"]
        if not isinstance(operations, (list, tuple)):
            raise ValueError("operations must be a list")
        return cls(
            initial=value["initial"],
            operations=tuple(tuple(operation) for operation in operations),
            modulus=value["modulus"],
        )


def transition(value: int, operation: tuple[str, int], modulus: int) -> int:
    op, arg = operation
    return (value + arg if op == "add" else value * arg) % modulus


class ArithmeticEnv:
    """Execute one submitted value per public arithmetic operation.

    ``sparse`` accepts every valid value and reveals no correctness until the
    episode ends. ``constraint`` rejects a wrong value and lets the agent retry,
    but never exposes the oracle value.
    """

    def __init__(self, task: Task, mode: str = "sparse", max_calls: int | None = None):
        if mode not in ("sparse", "constraint"):
            raise ValueError("invalid mode")
        self.task = task
        self.mode = mode
        self.max_calls = max_calls if max_calls is not None else 2 * len(task.operations)
        if type(self.max_calls) is not int or self.max_calls < 1:
            raise ValueError("invalid max_calls")
        self.reset()

    def reset(self) -> dict[str, Any]:
        self.index = 0
        self.calls = 0
        self.value = self.task.initial
        self.all_correct = True
        self.done = False
        self.success = False
        return self.observation()

    def observation(self) -> dict[str, Any]:
        next_operation = None
        if self.index < len(self.task.operations):
            next_operation = list(self.task.operations[self.index])
        return {
            "value": self.value,
            "step": self.index,
            "total_steps": len(self.task.operations),
            "next_operation": next_operation,
            "modulus": self.task.modulus,
            "calls_left": self.max_calls - self.calls,
            "done": self.done,
        }

    def step(self, action: str) -> tuple[dict[str, Any], float, bool]:
        if self.done:
            raise RuntimeError("episode already finished")
        self.calls += 1
        valid = False
        try:
            obj = json.loads(action)
            valid = (
                isinstance(obj, dict)
                and set(obj) == {"value"}
                and type(obj["value"]) is int
                and 0 <= obj["value"] < self.task.modulus
            )
        except (ValueError, TypeError):
            obj = None

        feedback = None
        if not valid:
            feedback = "invalid_json_or_value"
            self.all_correct = False
        else:
            expected = transition(self.value, self.task.operations[self.index], self.task.modulus)
            correct = obj["value"] == expected
            if self.mode == "constraint" and not correct:
                feedback = "rule_check_failed_retry"
            else:
                self.all_correct = self.all_correct and correct
                self.value = obj["value"]
                self.index += 1

        completed = self.index == len(self.task.operations)
        self.done = completed or self.calls >= self.max_calls
        self.success = bool(self.done and completed and self.all_correct)
        reward = float(self.success) if self.done else 0.0
        observation = self.observation()
        if feedback:
            observation["feedback"] = feedback
        if self.done:
            observation["episode_success"] = self.success
        return observation, reward, self.done


def oracle_action(env: ArithmeticEnv) -> str:
    """Return the expert action. Never call this in student evaluation."""
    if env.done:
        raise RuntimeError("episode already finished")
    result = transition(env.value, env.task.operations[env.index], env.task.modulus)
    return json.dumps({"value": result}, separators=(",", ":"))

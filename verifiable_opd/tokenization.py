"""Exact per-decision prefix/target alignment for the reference trainer."""
from __future__ import annotations

from dataclasses import dataclass

from .prompts import TOOL_SCHEMA


@dataclass
class Decision:
    prefix: list[int]
    response: list[int]

    def __post_init__(self):
        if not self.prefix or not self.response:
            raise ValueError("prefix and response must be nonempty")


def prompt_ids(tokenizer, messages):
    return list(tokenizer.apply_chat_template(messages, tools=[TOOL_SCHEMA],
                tokenize=True, add_generation_prompt=True))


def expert_decisions(tokenizer, messages):
    decisions = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = prompt_ids(tokenizer, messages[:index])
        full = list(tokenizer.apply_chat_template(messages[:index + 1], tools=[TOOL_SCHEMA],
                    tokenize=True, add_generation_prompt=False))
        if full[:len(prefix)] != prefix:
            raise ValueError("chat template is not prefix-compatible; refuse approximate label alignment")
        response = full[len(prefix):]
        # End at the model's EOS; Qwen's following newline is template-supplied.
        eos = tokenizer.eos_token_id
        if eos is not None and eos in response:
            response = response[:response.index(eos) + 1]
        decisions.append(Decision(prefix, response))
    return decisions


def check_tokenizers(student, teacher):
    for field in ("get_vocab", "get_added_vocab"):
        if getattr(student, field)() != getattr(teacher, field)():
            raise ValueError(f"teacher/student mismatch: {field}")
    for field in ("eos_token_id", "bos_token_id", "pad_token_id", "chat_template"):
        if getattr(student, field) != getattr(teacher, field):
            raise ValueError(f"teacher/student mismatch: {field}")


def causal_target_slice(prefix_length, response_length):
    if prefix_length < 1 or response_length < 1:
        raise ValueError("empty prefix/response")
    return slice(prefix_length - 1, prefix_length + response_length - 1)

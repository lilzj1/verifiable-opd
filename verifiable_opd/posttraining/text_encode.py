"""Single-source chat encoding and explicit next-token supervision boundaries."""
from dataclasses import dataclass


def validate_messages(messages):
    if not isinstance(messages, list) or not messages or messages[-1].get("role") != "user":
        raise ValueError("Prompt must be nonempty and end with a user message")
    for message in messages:
        if set(message) != {"role", "content"} or message["role"] not in {"system", "user"}:
            raise ValueError("Prompt must contain only whitelisted system/user role/content")
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise ValueError("Empty or non-text prompt")


def encode_prompt(tokenizer, messages, max_prompt=2048):
    validate_messages(messages)
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                       enable_thinking=True)
    if not ids or len(ids) > max_prompt:
        raise ValueError("Prompt length outside budget; silent truncation is forbidden")
    return list(ids)


@dataclass(frozen=True)
class EncodedExpert:
    prompt_ids: list[int]
    response_ids: list[int]

    @property
    def labels(self):
        # Input-aligned labels; the CE implementation shifts exactly once.
        return [-100] * len(self.prompt_ids) + self.response_ids


def encode_expert(tokenizer, messages, response, max_prompt=2048, max_response=4096):
    prompt = encode_prompt(tokenizer, messages, max_prompt)
    if not isinstance(response, str) or not response.strip():
        raise ValueError("Missing expert answer")
    if tokenizer.eos_token and tokenizer.eos_token in response:
        raise ValueError("Expert body must not embed conversation EOS")
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": response}],
        tokenize=True, add_generation_prompt=False, enable_thinking=True)
    if full[:len(prompt)] != prompt:
        raise ValueError("Chat-template prefix boundary mismatch")
    target = list(full[len(prompt):])
    if tokenizer.eos_token_id not in target:
        raise ValueError("Complete expert dialogue lacks EOS")
    target = target[:target.index(tokenizer.eos_token_id) + 1]
    if len(target) > max_response:
        raise ValueError("Expert answer exceeds budget; never truncate a demonstration")
    return EncodedExpert(prompt, target)

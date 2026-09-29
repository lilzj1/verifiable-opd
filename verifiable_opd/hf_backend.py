"""Simple Transformers backend for correctness checks before distributed verl.

Each decision stores the exact prefix and sampled action token IDs. Both models
score that very same tensor. History is re-rendered between decisions; this is
not verl's continuous-token rollout implementation and is reported separately.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import canonical, initial_messages
from .environment import ArithmeticEnv, Task
from .prompts import TOOL_SCHEMA
from .tokenization import Decision, prompt_ids
from .trajectory import execute_text, assistant_message


def load_model(path, device="cpu", trainable=False):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
                torch_dtype=dtype, attn_implementation="sdpa").to(device)
    model.requires_grad_(trainable)
    model.eval()
    # No hidden sampling defaults, penalties, forced tokens or top-k inherited
    # from a model repo: OPD samples the distribution scored by the teacher.
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    return model, tokenizer


def rollout(model, tokenizer, record, max_new_tokens=96, max_context=4096, sample=True):
    model.eval()
    env = ArithmeticEnv(Task.from_dict(record["task"]), "sparse")
    messages = initial_messages(env.task)
    decisions, actions = [], []
    truncated = False
    for _ in range(env.max_calls):
        prefix = prompt_ids(tokenizer, messages)
        if len(prefix) + max_new_tokens > max_context:
            truncated = True
            break
        device = next(model.parameters()).device
        ids = torch.tensor([prefix], device=device)
        kwargs = {"do_sample": sample, "max_new_tokens": max_new_tokens,
                  "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                  "eos_token_id": tokenizer.eos_token_id, "use_cache": True,
                  # Qwen2.5 may continue with a second tool call instead of
                  # emitting EOS.  One environment decision is exactly one
                  # tool call, so stop as soon as its closing tag is emitted.
                  "stop_strings": ["</tool_call>"], "tokenizer": tokenizer}
        if sample:
            kwargs.update(temperature=1.0, top_p=1.0, top_k=0)
        with torch.no_grad():
            output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **kwargs)
        response = output[0, len(prefix):].tolist()
        if not response:
            raise RuntimeError("empty generation")
        decisions.append(Decision(prefix, response))
        text = tokenizer.decode(response, skip_special_tokens=True)
        obs, reward, done, error = execute_text(env, text)
        actions.append({"text": text, "observation": obs, "reward": reward, "parse_error": error})
        messages.extend([assistant_message(text), {"role": "tool", "content": canonical(obs)}])
        if done:
            break
    return decisions, {"task_id": record["task_id"], "split": record["split"],
        "length": record["length"], "success": env.success, "calls": env.calls,
        "context_truncated": truncated, "actions": actions,
        "action_tokens": sum(len(d.response) for d in decisions),
        "scored_context_tokens": sum(len(d.prefix) + len(d.response) for d in decisions)}

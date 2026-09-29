"""Independent, complete-answer CE replay, selected deterministically by update."""
import hashlib
import json
from pathlib import Path

from .text_encode import encode_expert


class TextReplayPool:
    def __init__(self, path, tokenizer, *, max_prompt=2048, max_response=4096):
        self.path = Path(path).resolve(strict=True)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.examples = {}
        for line in self.path.read_text().split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["split"] != "train" or row["kind"] != "expert" or row["domain"] not in {"math", "if"}:
                raise ValueError("Replay accepts only identified Math/IF train demonstrations")
            key = row["example_id"]
            if key in self.examples:
                raise ValueError("Duplicate replay example")
            encoded = encode_expert(tokenizer, row["messages"], row["expert_response"],
                                    max_prompt, max_response)
            self.examples[key] = (row, encoded)
        if {v[0]["domain"] for v in self.examples.values()} != {"math", "if"}:
            raise ValueError("Both expert domains required")

    def select(self, *, step, seed, token_budget, rows_per_domain=1):
        if step < 0 or token_budget <= 0 or rows_per_domain <= 0:
            raise ValueError("Invalid replay selection budget")
        selected, costs = [], {}
        for domain in ("math", "if"):
            remaining = token_budget // 2
            candidates = sorted((k for k, v in self.examples.items() if v[0]["domain"] == domain),
                key=lambda k: hashlib.sha256(f"{seed}:{step}:{k}".encode()).hexdigest())
            chosen = []
            for key in candidates:
                cost = len(self.examples[key][1].response_ids)
                if cost <= remaining:
                    chosen.append(key)
                    remaining -= cost
                    if len(chosen) == rows_per_domain:
                        break
            if len(chosen) != rows_per_domain:
                raise ValueError(f"Insufficient complete {domain} examples within replay budget")
            selected.extend(chosen)
            costs[domain] = token_budget // 2 - remaining
        return {"task_ids": selected, "tokens": sum(costs.values()), "tokens_by_domain": costs,
                "budget": token_budget, "failure_selected": 0, "uniform_fallback": len(selected),
                "selector": "hash(seed,update,example_id), per-domain complete-answer budget",
                "pool_sha256": self.sha256, "step": step}

    def build_batch(self, selection, tokenizer, *, max_prompt, max_response, temperature):
        import numpy as np
        import torch
        from verl.protocol import DataProto
        from verl.utils.model import compute_position_id_with_mask
        encoded = [self.examples[key] for key in selection["task_ids"]]
        prompts, responses, prompt_masks, masks = [], [], [], []
        for row, data in encoded:
            plen, rlen = len(data.prompt_ids), len(data.response_ids)
            prompts.append([tokenizer.pad_token_id] * (max_prompt - plen) + data.prompt_ids)
            responses.append(data.response_ids + [tokenizer.pad_token_id] * (max_response - rlen))
            prompt_masks.append([0] * (max_prompt - plen) + [1] * plen)
            masks.append([1] * rlen + [0] * (max_response - rlen))
        prompt_tensor = torch.tensor(prompts, dtype=torch.long)
        response_tensor = torch.tensor(responses, dtype=torch.long)
        response_mask = torch.tensor(masks, dtype=torch.long)
        attention = torch.cat([torch.tensor(prompt_masks), response_mask], dim=1)
        batch = DataProto.from_dict(tensors={"prompts": prompt_tensor, "responses": response_tensor,
            "input_ids": torch.cat([prompt_tensor, response_tensor], dim=1),
            "attention_mask": attention, "position_ids": compute_position_id_with_mask(attention),
            "response_mask": response_mask}, non_tensors={
                "data_source": np.array([r[0]["domain"] for r in encoded], dtype=object),
                "uid": np.array([f"ce:{selection['step']}:{key}" for key in selection["task_ids"]], dtype=object),
                "extra_info": np.array([{"example_id": r[0]["example_id"], "split": "train",
                    "domain": r[0]["domain"], "pool_sha256": self.sha256} for r in encoded], dtype=object)},
                meta_info={"temperature": float(temperature), "global_steps": selection["step"]})
        return batch

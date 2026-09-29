"""Real-weight load/forward/cache probes, NOT a quality gate or verl training."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("This probe is authorized for physical GPU 1 only")
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from smollm3_compat import verify_effective
    torch.set_num_threads(2)
    torch.manual_seed(17)
    state = json.loads(args.assets.read_text())
    if state["phase"] != "ASSETS_READY_GPU_VALIDATION_PENDING":
        raise ValueError("Assets must be fully validated")
    result = {"kind": "REAL_WEIGHT_HF_GPU_INFRASTRUCTURE_PROBE_NOT_QUALITY_OR_VERL",
              "transformers": transformers.__version__, "torch": torch.__version__,
              "device": torch.cuda.get_device_name(0), "physical_gpu": 1, "models": {}}
    prompts = ["Compute 17 + 25. Give the final answer.",
               "Write exactly two sentences about rain. Do not use the word blue."]
    backend_hashes = []
    for role, info in state["models"].items():
        path = info["overlay"]
        # Preserve published SmolLM tokenizer; do not apply a Mistral regex rewrite.
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, fix_mistral_regex=False)
        backend = json.loads(tokenizer.backend_tokenizer.to_str())
        raw = json.loads((Path(path) / "tokenizer.json").read_text())
        if backend["pre_tokenizer"] != raw["pre_tokenizer"]:
            raise ValueError("Unexpected pre-tokenizer mutation")
        fingerprint = hashlib.sha256(json.dumps(backend, sort_keys=True).encode()).hexdigest()
        backend_hashes.append(fingerprint)
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0}).eval()
        model.requires_grad_(False)
        verify_effective(model.config)
        examples = []
        for prompt in prompts:
            inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True, enable_thinking=True,
                return_tensors="pt", return_dict=True).to("cuda")
            with torch.inference_mode():
                forward = model(**inputs, use_cache=False)
                assert torch.isfinite(forward.logits).all(), "non-finite real-weight logits"
                shifted = forward.logits[:, :-1].float().log_softmax(-1).gather(
                    -1, inputs["input_ids"][:, 1:, None]).squeeze(-1)
                assert torch.isfinite(shifted).all()
                started = time.monotonic()
                generated = model.generate(**inputs, max_new_tokens=96, do_sample=False,
                    eos_token_id=128012, pad_token_id=128012, use_cache=True)
                torch.cuda.synchronize()
            response = generated[0, inputs["input_ids"].shape[-1]:].tolist()
            examples.append({"prompt": prompt, "response": tokenizer.decode(response),
                "response_ids": response, "tokens": len(response),
                "seconds": time.monotonic() - started, "eos": response[-1] == 128012,
                "prefix_mean_logprob": shifted.mean().item()})
            del inputs, forward, shifted, generated
        result["models"][role] = {"phase": "REAL_WEIGHT_FORWARD_CACHE_PASS",
            "backend_sha256": fingerprint, "pretokenizer_unchanged": True,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "total_seconds": time.monotonic() - start, "examples": examples}
        (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        print(role, "FORWARD_CACHE_PASS", flush=True)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    result["all_tokenizer_backends_identical"] = len(set(backend_hashes)) == 1
    if not result["all_tokenizer_backends_identical"]:
        raise ValueError("Different complete tokenizer backends")
    result["phase"] = "PASS_INFRASTRUCTURE_ONLY"
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

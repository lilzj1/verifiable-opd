"""Single-GPU vLLM real model/teacher-prefix probe. No optimizer or quality claims."""
import gc
import json
import os
from pathlib import Path
import time


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("Probe restricted to physical GPU 1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    root = Path(__file__).resolve().parent.parent
    out = root / "artifacts/vllm_preflight_v1"
    out.mkdir(exist_ok=False)
    asset = json.loads((root / "artifacts/posttraining_assets_20260926_resume_v4/status.json").read_text())
    path = asset["models"]["shared_student_initialization"]["overlay"]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, fix_mistral_regex=False)
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, enable_thinking=True)
        for prompt in ["Compute 17 + 25. Give the final answer.", "Write two sentences about rain."]]
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": 0}, local_files_only=True).eval()
    hf_scores = []
    with torch.inference_mode():
        for ids in prompts:
            batch = torch.tensor([ids], device="cuda")
            scores = model(batch, use_cache=False).logits[0, :-1].float().log_softmax(-1)
            hf_scores.append(scores.gather(-1, batch[0, 1:, None]).squeeze(-1).cpu().tolist())
            del scores, batch
    del model
    gc.collect()
    torch.cuda.empty_cache()
    from vllm import LLM, SamplingParams
    started = time.monotonic()
    llm = LLM(model=path, tokenizer=path, dtype="bfloat16", tensor_parallel_size=1,
        max_model_len=8192, gpu_memory_utilization=0.35, enforce_eager=True,
        max_num_seqs=4, enable_prefix_caching=False, enable_chunked_prefill=False,
        trust_remote_code=False, seed=17)
    outputs = llm.generate([{"prompt_token_ids": ids} for ids in prompts],
        SamplingParams(temperature=0, max_tokens=128, prompt_logprobs=1,
                       stop_token_ids=[128012]), use_tqdm=False)
    records = []
    for ids, hf, output in zip(prompts, hf_scores, outputs, strict=True):
        actual = [entry[token].logprob for token, entry in zip(ids[1:], output.prompt_logprobs[1:], strict=True)]
        errors = torch.tensor(actual) - torch.tensor(hf)
        records.append({"max_abs_error": errors.abs().max().item(),
            "mean_abs_error": errors.abs().mean().item(), "prompt_token_ids": ids,
            "hf_logprobs": hf, "vllm_logprobs": actual,
            "response": output.outputs[0].text, "finish_reason": output.outputs[0].finish_reason})
    # Registered before seeing measurements; diagnostic tolerances for BF16 kernels.
    passed = all(r["max_abs_error"] <= 0.5 and r["mean_abs_error"] <= 0.05 for r in records)
    (out / "report.json").write_text(json.dumps({"phase": "PASS" if passed else "FAIL",
        "scope": "student architecture infrastructure; not full teacher 16-prefix gate",
        "max_abs_tolerance": 0.5, "mean_abs_tolerance": 0.05,
        "seconds": time.monotonic() - started, "records": records}, indent=2) + "\n")
    if not passed:
        raise ValueError("HF/vLLM prefix scores outside registered diagnostic tolerance")


if __name__ == "__main__":
    main()

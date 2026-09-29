"""Fixed local-dev autonomous generation probe; no training or final test access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, choices=("shared_student_initialization", "math_teacher", "instruction_following_teacher"))
    parser.add_argument("--gpu", required=True, choices=("1", "2", "3", "4"))
    parser.add_argument("--per-domain", type=int, default=16)
    parser.add_argument("--response", type=int, default=4096, choices=(4096, 8192, 30000))
    parser.add_argument("--domain", choices=("both", "math", "if"), default="both")
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=1.0, choices=(0.6,1.0))
    parser.add_argument("--top-k", type=int, default=-1, choices=(-1,20))
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16","float16"))
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu:
        raise ValueError("Physical device visibility mismatch")
    from transformers import AutoTokenizer
    from verifiable_opd.posttraining.text_encode import encode_prompt
    from verifiable_opd.posttraining.checkers import score
    from vllm import LLM, SamplingParams
    asset = json.loads((ROOT / "artifacts/posttraining_assets_20260926_resume_v4/status.json").read_text())
    source = ROOT / "data/posttraining_text_v1/dev.jsonl"
    rows = [json.loads(line) for line in source.read_text().split('\n') if line.strip()]
    domains = ("math", "if") if args.domain == "both" else (args.domain,)
    chosen = [r for domain in domains for r in [r for r in rows if r["domain"] == domain][:args.per_domain]]
    assert args.per_domain > 0 and len(chosen) == args.per_domain * len(domains) and all(r["split"] == "dev" for r in chosen)
    context = args.context or (2048 + args.response + 1)
    if context < 2048 + args.response + 1 or context > 32768:
        raise ValueError("Context must fit prompt + response and stay within source 32768 limit")
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {"role": args.role, "physical_gpu": args.gpu, "dev_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "ids": [r["example_id"] for r in chosen], "max_response_tokens": args.response,
        "domains": list(domains), "max_model_len": context,
        "temperature": args.temperature, "top_p": 0.95, "top_k":args.top_k,"seed": 17,"dtype":args.dtype,
        "scope": "development length/quality probe, not final performance comparison",
        "backend_prefix_gate": "See separate response-prefix audit; this is autonomous quality, not a numerical gate"}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    path = asset["models"][args.role]["overlay"]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, fix_mistral_regex=False)
    started = time.monotonic()
    llm = LLM(model=path, dtype=args.dtype, max_model_len=context,
        gpu_memory_utilization=0.50, enforce_eager=True, max_num_seqs=4,
        enable_prefix_caching=False, enable_chunked_prefill=False, seed=17, tensor_parallel_size=1)
    initialization = time.monotonic() - started
    results = []
    with (args.output / "predictions.jsonl").open("x") as stream:
        for start in range(0, len(chosen), 4):
            batch = chosen[start:start+4]
            inputs = [encode_prompt(tokenizer, r["messages"]) for r in batch]
            if any(len(ids) > 2048 for ids in inputs):
                raise ValueError("Diagnostic prompt exceeds frozen 2048-token allowance")
            before = time.monotonic()
            outputs = llm.generate([{"prompt_token_ids": ids} for ids in inputs],
                SamplingParams(temperature=args.temperature, top_p=0.95, top_k=args.top_k, max_tokens=args.response,
                               stop_token_ids=[128012], seed=17, logprobs=1), use_tqdm=False)
            elapsed = time.monotonic() - before
            for row, prompt_ids, output in zip(batch, inputs, outputs, strict=True):
                answer = output.outputs[0]
                item = {"example_id": row["example_id"], "domain": row["domain"], "role": args.role,
                    "prompt_token_ids": prompt_ids, "response": answer.text,
                    "response_token_ids": list(answer.token_ids), "finish_reason": answer.finish_reason,
                    "batch_wall_seconds": elapsed, "batch_index": start//4,
                    "sampled_logprobs": [entry[token].logprob for token, entry in zip(answer.token_ids, answer.logprobs, strict=True)]}
                try:
                    item["evaluation"] = score(row, answer.text)
                except Exception as exc:
                    item["checker_error"] = f"{type(exc).__name__}: {exc}"
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                stream.flush()
                results.append(item)
            print(args.role, f"{len(results)}/{len(chosen)}", "batch_seconds", round(elapsed, 2), flush=True)
    summary = {"phase": "PROBE_COMPLETE" if not any("checker_error" in r for r in results) else "CHECKER_FAILURE",
        "initialization_seconds": initialization, "wall_seconds": time.monotonic()-started, "domains": {}}
    for domain in domains:
        subset = [r for r in results if r["domain"] == domain]
        summary["domains"][domain] = {"n": len(subset),
            "correct": sum(r.get("evaluation", {}).get("score", 0) for r in subset),
            "truncated": sum(r["finish_reason"] == "length" for r in subset),
            "checker_errors": sum("checker_error" in r for r in subset),
            "generated_tokens": sum(len(r["response_token_ids"]) for r in subset)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    # Explicitly close the engine before verifier caches/interpreter teardown.
    # Otherwise multiprocessing + verifier imports can leave vLLM holding VRAM.
    llm.llm_engine.engine_core.shutdown()
    if summary["phase"] == "CHECKER_FAILURE":
        raise RuntimeError("Checker failures must be resolved before quality claims")


if __name__ == "__main__":
    main()

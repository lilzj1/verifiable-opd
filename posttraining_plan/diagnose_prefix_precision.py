"""Diagnose failed BF16 prefix comparison, preserving original failed result."""
import gc
import json
import os
from pathlib import Path


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("GPU 1 only")
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parent.parent
    path = root / "artifacts/posttraining_assets_20260926_resume_v4/models/shared_student_initialization"
    reference = json.loads((root / "artifacts/vllm_preflight_v1/report.json").read_text())
    report = {"original_gate": "FAIL", "variants": {}}
    for attention, dtype in [("flash_attention_2", torch.bfloat16), ("eager", torch.float32)]:
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
            torch_dtype=dtype, attn_implementation=attention, device_map={"": 0}).eval()
        rows = []
        with torch.inference_mode():
            for record in reference["records"]:
                ids = torch.tensor([record["prompt_token_ids"]], device="cuda")
                logits = model(ids, use_cache=False).logits[0, :-1].float()
                scores = logits.log_softmax(-1).gather(-1, ids[0, 1:, None]).flatten().cpu()
                errors = (scores - torch.tensor(record["vllm_logprobs"])).abs()
                rows.append({"max_abs_error": errors.max().item(), "mean_abs_error": errors.mean().item(),
                             "logprobs": scores.tolist()})
                del ids, logits
        report["variants"][str(dtype) + ":" + attention] = rows
        (root / "artifacts/prefix_precision_diagnosis_v1.json").write_text(json.dumps(report, indent=2))
        print(attention, dtype, [(r["max_abs_error"], r["mean_abs_error"]) for r in rows], flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

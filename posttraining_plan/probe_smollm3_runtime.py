"""CPU reference probe. Tiny random weights test runtime semantics, not model quality.

Run first in the declared reference runtime with --create-fixture, then in the
target runtime with the same fixture. No pretrained weights or GPU are used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


FIELDS = ("rope_theta", "rope_scaling", "rope_parameters", "tie_word_embeddings",
          "bos_token_id", "eos_token_id", "pad_token_id", "no_rope_layers",
          "no_rope_layer_interval", "layer_types", "max_position_embeddings")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--create-fixture", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU probe requires CUDA_VISIBLE_DEVICES=''")
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.create_fixture and args.fixture.exists():
        raise FileExistsError(args.fixture)
    if not args.create_fixture:
        fixtures = torch.load(args.fixture, map_location="cpu", weights_only=True)
    else:
        fixtures = {}
    report = {"transformers": transformers.__version__, "torch": torch.__version__,
              "kind": "CPU_TINY_RANDOM_MODEL_NOT_PRETRAINED_VALIDATION", "models": []}
    for path in sorted(args.configs.glob("*.config.json")):
        raw = json.loads(path.read_text())
        payload = dict(raw)
        kind = payload.pop("model_type")
        original = AutoConfig.for_model(kind, **payload)
        record = {"file": path.name, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "effective": {f: getattr(original, f, None) for f in FIELDS}}
        # Keep full-width rotary frequencies in a tiny four-layer model, including a NoPE layer.
        tiny = original.to_dict()
        tiny.update(vocab_size=64, hidden_size=256, intermediate_size=288,
                    num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
                    no_rope_layers=original.no_rope_layers[:4], layer_types=original.layer_types[:4],
                    bos_token_id=None, eos_token_id=1, pad_token_id=0, attention_dropout=0.0)
        tiny.pop("model_type", None)
        config = AutoConfig.for_model(kind, **tiny)
        config._attn_implementation = "eager"
        torch.manual_seed(17)
        model = AutoModelForCausalLM.from_config(config).float().eval()
        if not args.create_fixture:
            model.load_state_dict(fixtures[path.name]["weights"], strict=True)
        ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9], [3, 9, 8, 7, 6, 5, 4, 2]])
        positions = torch.stack((torch.arange(8), torch.arange(4096, 4104)))
        logits = model(ids, attention_mask=torch.ones_like(ids), position_ids=positions,
                       use_cache=False).logits
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 64), ids[:, 1:].reshape(-1))
        loss.backward()
        outputs = {"logits": logits.detach(), "loss": loss.detach(),
                   "embedding_grad": model.get_input_embeddings().weight.grad.detach().clone(),
                   "inv_freq": model.model.rotary_emb.inv_freq.detach().clone()}
        with torch.no_grad():
            prefix = model(ids[:, :-1], use_cache=True)
            outputs["cache_logits"] = model(ids[:, -1:], past_key_values=prefix.past_key_values,
                                             use_cache=True).logits.detach()
        if args.create_fixture:
            fixtures[path.name] = {"weights": {k: v.detach().clone() for k, v in model.state_dict().items()},
                                   "outputs": outputs}
            record["result"] = "REFERENCE_WRITTEN"
        else:
            expected = fixtures[path.name]["outputs"]
            record["max_abs_diff"] = {k: float((outputs[k] - expected[k]).abs().max()) for k in outputs}
            record["result"] = "PASS" if all(torch.allclose(outputs[k], expected[k], atol=2e-5, rtol=2e-5)
                                                   for k in outputs) else "FAIL"
        report["models"].append(record)
    if not report["models"]:
        raise RuntimeError("no config files")
    if args.create_fixture:
        args.fixture.parent.mkdir(parents=True, exist_ok=True)
        torch.save(fixtures, args.fixture)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return int(any(m["result"] == "FAIL" for m in report["models"]))


if __name__ == "__main__":
    raise SystemExit(main())

"""Explicit reference-runtime config normalization; never mutates source models.

Policy is the measured Transformers 4.57.6 interpretation, NOT a claim about
the authors' unpublished training jobs or the best theta for model quality.
Only audited config hashes are accepted. Full-weight validation is separate.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

POLICY = "open_mopd_published_runtime_tf4576_v1"
ROLES = {
    "shared_student_initialization": "487d07a23a948bc2ebfcc44444ed56d59be7ba4dd0a8a16d548f0ac1ab94c92f",
    "math_teacher": "79e45731ee406a59d20a8dbd5fd92e9e4504c85db5b72783d619c6f91eaee1ee",
    "instruction_following_teacher": "79e45731ee406a59d20a8dbd5fd92e9e4504c85db5b72783d619c6f91eaee1ee",
}


def normalize(body, role, reference):
    if role not in ROLES or hashlib.sha256(body).hexdigest() != ROLES[role]:
        raise ValueError("Unknown/changed source config; audit again, never silently normalize")
    if reference["transformers"] != "4.57.6":
        raise ValueError("Expected independently measured 4.57.6 reference")
    matches = [m for m in reference["models"] if m["file"] == f"{role}.config.json"]
    if len(matches) != 1 or matches[0]["source_sha256"] != ROLES[role]:
        raise ValueError("Reference does not match this source")
    measured = matches[0]["effective"]
    if matches[0]["result"] != "REFERENCE_WRITTEN":
        raise ValueError("Missing reference forward probe")
    if measured["rope_theta"] != 2_000_000 or measured["rope_scaling"] is not None:
        raise ValueError("Unexpected reference RoPE; stop for review")
    raw = json.loads(body)
    result = copy.deepcopy(raw)
    result.update(rope_theta=measured["rope_theta"], rope_scaling=None,
                  rope_parameters={"rope_theta": measured["rope_theta"], "rope_type": "default"},
                  tie_word_embeddings=measured["tie_word_embeddings"])
    changes = {k: {"source": raw.get(k), "normalized": result[k]} for k in result if raw.get(k) != result[k]}
    return result, {"policy": POLICY, "source_sha256": ROLES[role], "role": role, "changes": changes}


def verify_effective(config):
    """Guard target loader semantics; supports legacy and modern config field names."""
    theta = getattr(config, "rope_theta", None)
    nested = getattr(config, "rope_parameters", None)
    if theta is not None and theta != 2_000_000:
        raise ValueError("Legacy theta differs from frozen reference")
    if nested != {"rope_type": "default", "rope_theta": 2_000_000}:
        raise ValueError("Nested RoPE policy differs from frozen reference")
    if (config.no_rope_layers != [1, 1, 1, 0] * 9 or config.num_hidden_layers != 36
            or config.max_position_embeddings != 65536 or config.eos_token_id != 128012
            or config.pad_token_id != 128012 or not config.tie_word_embeddings):
        raise ValueError("Architecture/NoPE/token configuration drift")


def make_overlay(snapshot, output, role, reference):
    """Materialize a deployment view of EXISTING complete local weights; no download.

All weights/tokenizer assets stay read-only symlinks to the source. The only
replacement is config.json, with a separate manifest. Refuses partial snapshots.
"""
    snapshot = snapshot.resolve(strict=True)
    body = (snapshot / "config.json").read_bytes()
    config, manifest = normalize(body, role, reference)
    required = {"tokenizer.json", "tokenizer_config.json"}
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        if not weight_map:
            raise ValueError("Empty weights index")
        shards = set(weight_map.values())
        if any(not isinstance(s, str) or Path(s).name != s or not s.endswith(".safetensors") for s in shards):
            raise ValueError("Invalid weights shard paths")
        required |= shards | {"model.safetensors.index.json"}
    else:
        required.add("model.safetensors")
    for name in required:
        if not (snapshot / name).is_file() or (snapshot / name).stat().st_size == 0:
            raise ValueError(f"Missing/empty asset: {name}")
    output = output.absolute()
    if output == snapshot or snapshot in output.parents:
        raise ValueError("Overlay must not modify or nest inside source snapshot")
    output.mkdir(parents=True, exist_ok=False)
    optional = {"generation_config.json", "special_tokens_map.json", "chat_template.jinja",
                "vocab.json", "merges.txt", "added_tokens.json", "README.md"}
    for name in sorted(required | {n for n in optional if (snapshot / n).is_file()}):
        (output / name).symlink_to((snapshot / name).resolve())
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest.update(source_snapshot=str(snapshot), full_weight_validation="PENDING",
                    tokenizer_semantics_validation="PENDING")
    (output / "COMPAT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configs", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--role", choices=ROLES)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    if args.snapshot:
        if not args.role or args.configs:
            parser.error("--snapshot requires --role and excludes --configs")
        make_overlay(args.snapshot, args.output, args.role, reference)
        return
    if not args.configs or args.role:
        parser.error("config-only mode requires --configs and no --role")
    # Validate all configs before writing any output.
    converted = {r: normalize((args.configs / f"{r}.config.json").read_bytes(), r, reference) for r in ROLES}
    args.output.mkdir(parents=True, exist_ok=False)
    manifests = []
    for role, (config, manifest) in converted.items():
        (args.output / f"{role}.config.json").write_text(json.dumps(config, indent=2) + "\n")
        manifests.append(manifest)
    (args.output / "manifest.json").write_text(json.dumps({
        "status": "CONFIG_ONLY_NOT_LOADABLE_MODEL", "models": manifests,
    }, indent=2) + "\n")
    print(f"Wrote {len(converted)} config-only views. No weights changed or downloaded.")


if __name__ == "__main__":
    main()

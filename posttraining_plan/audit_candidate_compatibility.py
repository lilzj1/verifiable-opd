"""Bounded CPU-only audit of pinned candidate configs; never fetches weights.

Writes a fresh evidence directory and returns 2 if any check is blocked.
This is not a training, inference-quality, or GPU-memory benchmark.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import urllib.request


def fetch_config(asset):
    url = f"https://huggingface.co/{asset['repo_id']}/resolve/{asset['revision']}/config.json"
    with urllib.request.urlopen(url, timeout=25) as response:
        body = response.read(1_000_001)
    if len(body) > 1_000_000:
        raise ValueError("unexpected config size")
    return body, json.loads(body), url


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", type=Path,
                        help="Measured reference report; without it a field conflict is unresolved, not proof of wrong theta")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("run with CUDA_VISIBLE_DEVICES='' for this CPU audit")
    args.output.mkdir(parents=True, exist_ok=False)
    from transformers import AutoConfig

    root = Path(__file__).resolve().parent.parent
    assets = json.loads((root / "posttraining_candidate_assets.json").read_text())["models"]
    reference = json.loads(args.reference.read_text()) if args.reference else None
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "kind": "CPU_CONFIG_COMPATIBILITY_NOT_GPU_PROFILE",
        "versions": {p: version(p) for p in ("transformers", "torch", "vllm")},
        "models": [],
    }
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(fetch_config, asset) for asset in assets]
        for asset, future in zip(assets, futures):
            item = dict(asset)
            try:
                body, raw, url = future.result()
                (args.output / f"{asset['role']}.config.json").write_bytes(body)
                payload = dict(raw)
                model_type = payload.pop("model_type")
                effective = AutoConfig.for_model(model_type, **payload)
                declared = raw.get("rope_parameters", {}).get("rope_theta", raw.get("rope_theta"))
                used = getattr(effective, "rope_theta", None)
                if used is None:
                    used = (getattr(effective, "rope_parameters", None) or {}).get("rope_theta")
                expected = None
                if reference:
                    candidates = [m for m in reference["models"] if m["file"] == f"{asset['role']}.config.json"]
                    if len(candidates) != 1 or candidates[0]["source_sha256"] != hashlib.sha256(body).hexdigest():
                        raise ValueError("Reference source hash mismatch")
                    expected = candidates[0]["effective"]["rope_theta"]
                item.update(
                    source_url=url,
                    config_sha256=hashlib.sha256(body).hexdigest(),
                    saved_transformers_version=raw.get("transformers_version"),
                    declared_rope_theta=declared,
                    local_effective_rope_theta=used,
                    config_parsed=True,
                    expected_reference_theta=expected,
                    nested_and_effective_disagree=declared != used,
                    theta_matches_reference=expected is not None and expected == used,
                )
                item["remaining_checks"] = "Matching theta alone does not validate architecture/tokenizer/inference."
                if expected is None:
                    item["blocker"] = "No frozen reference semantics; nested theta is not automatically authoritative."
                elif expected != used:
                    item["blocker"] = "Runtime differs from chosen reference."
            except Exception as exc:
                item.update(theta_matches_reference=False, error=f"{type(exc).__name__}: {exc}")
            report["models"].append(item)
    report["status"] = ("BLOCKED_REFERENCE_UNRESOLVED_OR_MISMATCH" if any(
        not m["theta_matches_reference"] for m in report["models"]
    ) else "CONFIG_THETA_MATCH_ONLY_GPU_PENDING")
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 2 if report["status"].startswith("BLOCKED") else 0


if __name__ == "__main__":
    raise SystemExit(main())

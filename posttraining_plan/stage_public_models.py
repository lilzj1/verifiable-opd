"""Resume pinned public model downloads, validate files and tokenizer alignment.

CPU only; creates compatibility overlays but never starts training or GPU jobs.
Each run has its own lock and atomic status file. HF cache handles blob locks.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import traceback

from smollm3_compat import make_overlay


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def token_signature(path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    probes = [
        [{"role": "user", "content": "Compute 17 + 25. Give the final answer."}],
        [{"role": "user", "content": "Write exactly two sentences. Do not use the word blue."}],
    ]
    rendered = {}
    for thinking in (True, False):
        rendered[str(thinking)] = [tokenizer.apply_chat_template(
            p, tokenize=True, add_generation_prompt=True, enable_thinking=thinking
        ) for p in probes]
    vocab = json.dumps(tokenizer.get_vocab(), sort_keys=True, ensure_ascii=False).encode()
    return {"vocab_sha256": hashlib.sha256(vocab).hexdigest(),
            "vocab_count": len(tokenizer), "special_tokens_map": tokenizer.special_tokens_map,
            "eos_token_id": tokenizer.eos_token_id, "bos_token_id": tokenizer.bos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
            "probe_token_ids": rendered}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--weight-transport", choices=("hf", "aria2"), default="hf")
    parser.add_argument("--accelerated-cache", type=Path)
    args = parser.parse_args()
    if args.weight_transport == "aria2" and (not args.accelerated_cache or not shutil.which("aria2c")):
        parser.error("aria2 transport needs aria2c and --accelerated-cache")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Asset staging must have CUDA_VISIBLE_DEVICES=''")
    root = Path(__file__).resolve().parent.parent
    args.run_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.run_dir / "staging.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.run_dir / "status.json").exists():
        raise FileExistsError("Use a new run directory; existing HF cache downloads still resume")
    if shutil.disk_usage(root).free < 60 * 1024**3:
        raise RuntimeError("Need >=60GiB free before downloading; no cleanup is performed")
    assets = json.loads((root / "posttraining_candidate_assets.json").read_text())["models"]
    reference = json.loads((root / "artifacts/config_compat_v1/reference4576/report.json").read_text())
    state = {"phase": "DOWNLOADING", "started_at": datetime.now(timezone.utc).isoformat(),
             "gpu_jobs_started": False, "models": {}, "errors": []}
    write_lock = threading.Lock()
    metadata_lock = threading.Lock()

    def update(role=None, **fields):
        with write_lock:
            target = state if role is None else state["models"].setdefault(role, {})
            target.update(fields)
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            temporary = args.run_dir / "status.json.tmp"
            temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
            temporary.replace(args.run_dir / "status.json")
            print(state["updated_at"], role or "all", fields.get("phase", "updated"), flush=True)

    def stage(asset):
        from huggingface_hub import HfApi, snapshot_download
        from safetensors import safe_open
        role = asset["role"]
        update(role, phase="DOWNLOADING", repo_id=asset["repo_id"], revision=asset["revision"])
        patterns = ["*.json", "*.jinja", "*.txt", "README.md", "LICENSE*"]
        if args.weight_transport == "hf":
            patterns.append("*.safetensors")
        # HF's tqdm thread_map mutates a shared class lock; avoid concurrent
        # snapshot_download calls. Large aria2 weight transfers remain parallel.
        with metadata_lock:
            snapshot = Path(snapshot_download(
                asset["repo_id"], revision=asset["revision"], max_workers=2,
                allow_patterns=patterns,
            ))
        if args.weight_transport == "aria2":
            source = (args.accelerated_cache / role).resolve()
            source.mkdir(parents=True, exist_ok=True)
            for file in snapshot.iterdir():
                if not file.is_file() or file.name.endswith(".safetensors"):
                    continue
                target = source / file.name
                if target.exists() and target.resolve() != file.resolve():
                    raise ValueError(f"Unexpected cached metadata target: {target.name}")
                if not target.exists():
                    target.symlink_to(file.resolve())
            info = HfApi().model_info(asset["repo_id"], revision=asset["revision"], files_metadata=True)
            weights = [f for f in info.siblings if f.rfilename.endswith(".safetensors")]
            if not weights:
                raise ValueError("No public weight manifest")
            for file in weights:
                if Path(file.rfilename).name != file.rfilename or not file.lfs:
                    raise ValueError("Invalid weight path or missing official checksum")
                sha = file.lfs.sha256
                update(role, phase="DOWNLOADING_SEGMENTED", current_file=file.rfilename,
                       current_file_bytes=file.size, expected_sha256=sha, connections_per_file=8)
                url = f"https://huggingface.co/{asset['repo_id']}/resolve/{asset['revision']}/{file.rfilename}"
                command = ["aria2c", "--file-allocation=none", "--summary-interval=30",
                    "--console-log-level=warn", "--download-result=hide", "--enable-color=false",
                    "--show-console-readout=false", "-x", "8", "-s", "8", "-k", "1M",
                    "--continue=true", "--auto-file-renaming=false", "--allow-overwrite=false",
                    "--max-tries=10", "--retry-wait=5", "--connect-timeout=30", "--timeout=60",
                    "--auto-save-interval=15",
                    "--check-integrity=true", f"--checksum=sha-256={sha}",
                    f"--dir={source}", f"--out={file.rfilename}", url]
                with (args.run_dir / f"{role}.download.log").open("a") as log:
                    for attempt in range(1, 7):
                        update(role, phase="DOWNLOADING_SEGMENTED", download_attempt=attempt)
                        log.write(f"\nResume attempt {attempt}/6: {file.rfilename}\n")
                        log.flush()
                        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                        if result.returncode == 0:
                            break
                        # Retry transport failures only; checksum/filesystem errors fail closed.
                        if result.returncode not in {1, 2, 6, 7, 19, 22, 24, 29} or attempt == 6:
                            raise subprocess.CalledProcessError(result.returncode, command)
                        update(role, phase="RETRY_WAIT", last_download_exit=result.returncode)
                        log.write(f"Transport exited {result.returncode}; preserving partial file and retrying.\n")
                        log.flush()
                        time.sleep(min(5 * attempt, 30))
                if digest(source / file.rfilename) != sha:
                    raise ValueError("Official SHA256 mismatch after accelerated download")
            snapshot = source
        update(role, phase="VALIDATING", snapshot=str(snapshot))
        index = snapshot / "model.safetensors.index.json"
        weight_map = json.loads(index.read_text())["weight_map"] if index.is_file() else None
        names = sorted(set(weight_map.values())) if weight_map else ["model.safetensors"]
        tensors, files = {}, []
        for name in names:
            if Path(name).name != name or not name.endswith(".safetensors"):
                raise ValueError("Invalid weight filename")
            file = snapshot / name
            sha = digest(file)
            # HF LFS cache blob names are SHA256. Verify whenever available.
            blob_name = file.resolve().name
            if len(blob_name) == 64 and all(c in "0123456789abcdef" for c in blob_name) and sha != blob_name:
                raise ValueError(f"Weight checksum mismatch: {name}")
            with safe_open(file, framework="pt", device="cpu") as weights:
                for key in weights.keys():
                    if key in tensors:
                        raise ValueError(f"Duplicate tensor {key}")
                    if weight_map and weight_map.get(key) != name:
                        raise ValueError(f"Weight index mismatch {key}")
                    tensors[key] = weights.get_slice(key).get_shape()
            files.append({"name": name, "bytes": file.stat().st_size, "sha256": sha})
        if weight_map and set(tensors) != set(weight_map):
            raise ValueError("Missing indexed weights")
        overlay = args.run_dir / "models" / role
        make_overlay(snapshot, overlay, role, reference)
        signature = token_signature(overlay)
        update(role, phase="FILES_VALIDATED", overlay=str(overlay.resolve()), weight_files=files,
               tensor_count=len(tensors), tensor_shapes=tensors, tokenizer=signature,
               model_forward_validated=False)
        return role, signature

    update()
    signatures = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(stage, a): a["role"] for a in assets}
        for future in as_completed(futures):
            role = futures[future]
            try:
                role, signature = future.result()
                signatures[role] = signature
            except Exception as exc:
                traceback.print_exc()
                update(role, phase="FAILED", error=f"{type(exc).__name__}: {exc}")
                state["errors"].append(role)
    if state["errors"]:
        update(phase="FAILED_ASSET_STAGING")
        return 2
    student = signatures["shared_student_initialization"]
    matches = {role: signature == student for role, signature in signatures.items()}
    update(tokenizer_alignment=matches,
           phase="ASSETS_READY_GPU_VALIDATION_PENDING" if all(matches.values()) else "TOKENIZER_MISMATCH_REVIEW_REQUIRED")
    return 0 if all(matches.values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())

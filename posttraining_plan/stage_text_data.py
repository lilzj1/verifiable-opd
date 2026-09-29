"""Pinned raw dataset staging. No training, evaluation, or license inference."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
FILES = ["README.md", "rl_prompt_mix/manifest.json", "rl_prompt_mix/train.parquet",
         "sft/openr1_math_93k/train.parquet", "sft/instruction_nemotron_aligned/train-00000.parquet",
         "eval/math/aime24.parquet", "eval/math/aime25.parquet",
         "eval/if/ifeval_aligned.parquet", "eval/if/ifbench_test_aligned.parquet"]


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    from huggingface_hub import HfApi, hf_hub_download
    asset = json.loads((ROOT / "posttraining_candidate_assets.json").read_text())["dataset"]
    output = ROOT / "data/posttraining_raw_v1"
    output.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(asset["repo_id"], revision=asset["revision"], files_metadata=True)
    siblings = {file.rfilename: file for file in info.siblings}
    manifest = {"repository": asset, "files": {}, "phase": "STAGING_NOT_LICENSE_APPROVAL",
                "if_expert_subset": "first published parquet shard; fixed before model evaluation"}
    for name in FILES:
        file = siblings[name]
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not file.lfs:
            local = Path(hf_hub_download(asset["repo_id"], name, revision=asset["revision"], repo_type="dataset"))
            if not target.exists():
                target.symlink_to(local.resolve())
        else:
            sha = file.lfs.sha256
            if not (target.exists() and target.stat().st_size == file.size
                    and not Path(str(target) + ".aria2").exists() and digest(target) == sha):
                url = f"https://huggingface.co/datasets/{asset['repo_id']}/resolve/{asset['revision']}/{name}"
                command = ["aria2c", "--file-allocation=none", "--summary-interval=30",
                    "--console-log-level=warn", "--show-console-readout=false", "--enable-color=false",
                    "-x", "8", "-s", "8", "-k", "1M", "--continue=true", "--auto-save-interval=15",
                    "--auto-file-renaming=false", "--allow-overwrite=false", "--max-tries=10",
                    "--retry-wait=5", "--connect-timeout=30", "--timeout=60",
                    "--check-integrity=true", f"--checksum=sha-256={sha}",
                    f"--dir={target.parent}", f"--out={target.name}", url]
                for attempt in range(6):
                    result = subprocess.run(command)
                    if result.returncode == 0:
                        break
                    if result.returncode not in {1, 2, 6, 7, 19, 22, 24, 29} or attempt == 5:
                        raise subprocess.CalledProcessError(result.returncode, command)
                    time.sleep(5)
            if target.stat().st_size != file.size or digest(target) != sha:
                raise ValueError(f"Integrity failure: {name}")
        manifest["files"][name] = {"bytes": target.stat().st_size, "sha256": digest(target)}
        (output / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("VALIDATED", name, flush=True)
    manifest["phase"] = "RAW_FILES_READY_LICENSE_AND_SPLIT_AUDIT_PENDING"
    (output / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

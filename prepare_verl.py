"""Fetch an immutable official verl source snapshot; installs no dependencies."""
import argparse
import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", default="v0.8.0")
    p.add_argument("--output", default="upstream/verl-v0.8.0")
    a = p.parse_args()
    if not all(c.isalnum() or c in "._-" for c in a.ref):
        p.error("ref must be a simple tag or commit")
    output = Path(a.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    request = urllib.request.Request(f"https://api.github.com/repos/verl-project/verl/commits/{a.ref}",
                                     headers={"User-Agent": "verifiable-opd-setup"})
    with urllib.request.urlopen(request, timeout=60) as response:
        metadata = json.load(response)
    commit = metadata["sha"]
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("invalid resolved commit")
    url = f"https://codeload.github.com/verl-project/verl/tar.gz/{commit}"
    with urllib.request.urlopen(url, timeout=60) as response:
        archive = response.read()
    staging = output.with_name(output.name + ".incomplete")
    staging.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        tar.extractall(staging, filter="data")
    source = staging / f"verl-{commit}"
    lock = {"repository": "https://github.com/verl-project/verl", "requested_ref": a.ref,
            "commit": commit, "archive_sha256": hashlib.sha256(archive).hexdigest(),
            "url": url, "installed": False, "gpu_validated": False}
    (source / "OPD_SOURCE_LOCK.json").write_text(json.dumps(lock, indent=2) + "\n")
    source.rename(output)
    staging.rmdir()
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()

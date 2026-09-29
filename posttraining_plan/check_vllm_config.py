"""CPU-only test of the installed vLLM config loader, NOT its inference engine."""
import json
import os
from pathlib import Path
import tempfile

from smollm3_compat import verify_effective


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Run CPU config checks with CUDA_VISIBLE_DEVICES=''")
    from vllm.transformers_utils.config import get_config

    root = Path(__file__).resolve().parent.parent
    results = []
    for path in sorted((root / "artifacts/config_compat_v1/normalized").glob("*.config.json")):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "config.json").write_bytes(path.read_bytes())
            config = get_config(folder, trust_remote_code=False)
            verify_effective(config)
            results.append({"file": path.name, "status": "CONFIG_LOADER_PASS", "rope_theta": config.rope_theta})
    if len(results) != 3:
        raise RuntimeError("Expected all three configs")
    report = {"status": "CONFIG_ONLY_ENGINE_NOT_STARTED", "models": results}
    output = root / "artifacts/config_compat_v1/vllm_config_report.json"
    with output.open("x") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

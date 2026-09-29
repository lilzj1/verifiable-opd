"""Validate a local Hugging Face safetensors checkpoint without loading weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer


def validate_local_model(path: str | Path) -> dict[str, int | str]:
    root = Path(path)
    if not root.is_dir() or not (root / "config.json").is_file():
        raise FileNotFoundError(f"invalid local model directory: {root}")
    if list(root.glob("*.incomplete")):
        raise ValueError(f"incomplete files in model directory: {root}")

    index_path = root / "model.safetensors.index.json"
    expected_keys: set[str] | None = None
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("empty or invalid safetensors weight_map")
        expected_keys = set(weight_map)
        files = sorted({root / name for name in weight_map.values()})
    else:
        single = root / "model.safetensors"
        if not single.is_file():
            raise FileNotFoundError("no safetensors checkpoint or index")
        files = [single]

    actual_keys: set[str] = set()
    total_bytes = 0
    for file in files:
        if not file.is_file() or file.stat().st_size <= 8:
            raise ValueError(f"missing or empty model shard: {file}")
        total_bytes += file.stat().st_size
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
        if actual_keys & keys:
            raise ValueError(f"duplicate tensor keys in shard: {file}")
        actual_keys.update(keys)
    if expected_keys is not None and actual_keys != expected_keys:
        raise ValueError("safetensors index and shard tensor keys differ")

    config = AutoConfig.from_pretrained(root, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    return {
        "model_type": config.model_type,
        "shards": len(files),
        "tensors": len(actual_keys),
        "bytes": total_bytes,
        "tokenizer_size": len(tokenizer),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    args = parser.parse_args()
    print(json.dumps(validate_local_model(args.model), sort_keys=True))


if __name__ == "__main__":
    main()

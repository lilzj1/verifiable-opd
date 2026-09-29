#!/usr/bin/env python3
"""Safely retain the newest complete verl FSDP checkpoints.

The verl checkpoint manager only tracks checkpoints produced by the current
process.  A resumed run therefore needs this small companion to include older
checkpoints in retention.  A checkpoint is eligible for deletion only after
the trainer has atomically advanced ``latest_checkpointed_iteration.txt`` and
all model/optimizer/extra shards plus the dataloader state are present.
"""

from __future__ import annotations

import argparse
import re
import shutil
import time
from pathlib import Path


STEP_RE = re.compile(r"global_step_(\d+)$")


def checkpoint_step(path: Path) -> int | None:
    match = STEP_RE.fullmatch(path.name)
    return int(match.group(1)) if match else None


def latest_committed_step(root: Path) -> int | None:
    marker = root / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        return None
    try:
        return int(marker.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def complete_checkpoint(path: Path, committed_step: int | None) -> bool:
    step = checkpoint_step(path)
    if step is None or committed_step is None or step > committed_step:
        return False
    actor = path / "actor"
    if not (path / "data.pt").is_file() or not actor.is_dir():
        return False

    def ranks(prefix: str) -> set[str]:
        result: set[str] = set()
        for file in actor.glob(f"{prefix}_world_size_*_rank_*.pt"):
            if file.stat().st_size <= 0:
                continue
            result.add(file.name.split("_rank_", 1)[1].removesuffix(".pt"))
        return result

    model_ranks = ranks("model")
    return bool(model_ranks) and model_ranks == ranks("optim") == ranks("extra_state")


def rotate(root: Path, keep: int) -> list[Path]:
    if keep < 2:
        raise ValueError("keep must be at least 2 for recovery safety")
    committed = latest_committed_step(root)
    complete = sorted(
        (
            path
            for path in root.glob("global_step_*")
            if path.is_dir() and complete_checkpoint(path, committed)
        ),
        key=lambda path: checkpoint_step(path) or -1,
    )
    removed: list[Path] = []
    for path in complete[:-keep]:
        shutil.rmtree(path)
        removed.append(path)
        print(f"ROTATED {path}", flush=True)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--keep", type=int, default=2)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--watch-pid", type=int)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    while True:
        rotate(args.checkpoint_root, args.keep)
        if args.once:
            return
        if args.watch_pid is not None and not Path(f"/proc/{args.watch_pid}").exists():
            rotate(args.checkpoint_root, args.keep)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

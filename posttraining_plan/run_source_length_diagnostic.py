"""Bounded four-cell local-dev diagnostic; never launches training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "artifacts/source_length_diagnostic_v1"
CELLS = (
    ("student_8k", "shared_student_initialization", "1", 8192),
    ("math_8k", "math_teacher", "2", 8192),
    ("student_30k", "shared_student_initialization", "3", 30000),
    ("math_30k", "math_teacher", "4", 30000),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    source = ROOT / "data/posttraining_text_v1/dev.jsonl"
    rows = [json.loads(line) for line in source.read_text().split("\n") if line.strip()]
    selected = [row for row in rows if row["domain"] == "math"][:8]
    assert len(selected) == 8 and all(row["split"] == "dev" for row in selected)
    protocol = {
        "phase": "PLANNED", "allow_training": False,
        "scope": "Fixed first eight math dev rows, not selected by scores; diagnostic only",
        "dev_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "ids": [row["example_id"] for row in selected],
        "cells": CELLS, "temperature": .6, "top_p": .95, "top_k": -1,
        "dtype": "float16", "max_model_len": 32768, "seed": 17,
        "timeout_seconds_per_cell": 2700,
        "differences_from_source": ["DAPO dev, not AIME mean@64", "FP16", "eight questions, one sample each", "explicit response cap 30000, not a claim of exact source evaluation cap"],
        "decision": "Inspect paired correctness, truncation, raw reasoning and costs; no automatic gate pass, expansion or training",
        "probe_sha256": hashlib.sha256((ROOT / "posttraining_plan/run_teacher_probe.py").read_bytes()).hexdigest(),
    }
    if not args.execute:
        print(json.dumps(protocol, indent=2))
        return
    # Check the authorized devices only; never terminate existing GPU processes.
    gpu_info = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"
    ], text=True)
    memory = dict(tuple(int(v.strip()) for v in line.split(",")) for line in gpu_info.splitlines())
    if any(memory.get(int(cell[2]), 99999) > 100 for cell in CELLS):
        raise RuntimeError("An authorized device is no longer free; not launching")
    RUN.mkdir(exist_ok=False)
    (RUN / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    jobs = []
    started = time.time()
    try:
        for name, role, gpu, response in CELLS:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, TOKENIZERS_PARALLELISM="false",
                       OMP_NUM_THREADS="4", NCCL_P2P_DISABLE="1", NCCL_IB_DISABLE="1",
                       NLTK_DATA=str(ROOT / "data/nltk"))
            log = (RUN / f"{name}.log").open("x")
            command = ["timeout", "--signal=TERM", "--kill-after=30s", "45m",
                       str(ROOT / ".text_runtime/bin/python"), "-u",
                       "posttraining_plan/run_teacher_probe.py", "--role", role, "--gpu", gpu,
                       "--domain", "math", "--per-domain", "8", "--response", str(response),
                       "--context", "32768", "--temperature", "0.6", "--top-k", "-1",
                       "--dtype", "float16", "--output", str(RUN / name)]
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            jobs.append((name, process, log))
        while True:
            results = {}
            for name, process, _ in jobs:
                entry = {"pid": process.pid, "exit_code": process.poll()}
                summary = RUN / name / "summary.json"
                if summary.exists():
                    entry["summary"] = json.loads(summary.read_text())
                results[name] = entry
            finished = all(job[1].poll() is not None for job in jobs)
            state = {"phase": "COMPLETE_REVIEW_REQUIRED" if finished else "RUNNING",
                     "allow_training": False, "started_unix": started,
                     "elapsed_seconds": time.time() - started, "cells": results}
            tmp = RUN / "status.tmp"
            tmp.write_text(json.dumps(state, indent=2) + "\n")
            tmp.replace(RUN / "status.json")
            if finished:
                print(json.dumps(state, indent=2), flush=True)
                break
            time.sleep(10)
    finally:
        # Only this controller's process groups, never shared Ray or other jobs.
        import signal
        for _, process, log in jobs:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            log.close()


if __name__ == "__main__":
    main()

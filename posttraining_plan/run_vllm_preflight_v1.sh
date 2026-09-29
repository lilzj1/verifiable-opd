#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
set +e
timeout --signal=TERM --kill-after=30s 900 /home/ubuntu/anaconda3/envs/creditlab-verl-alfworld/bin/python -u \
  posttraining_plan/probe_vllm_gpu.py >> artifacts/vllm_preflight_v1_launcher.log 2>&1
result=$?
printf '%s\n' "$result" > artifacts/vllm_preflight_v1_exit_code.txt
exit "$result"

#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1
set +e
/home/ubuntu/anaconda3/envs/creditlab-verl-alfworld/bin/python -u posttraining_plan/probe_pretrained_gpu.py \
  --assets artifacts/posttraining_assets_20260926_resume_v4/status.json \
  --output artifacts/pretrained_gpu_v1 >> artifacts/pretrained_gpu_v1_launcher.log 2>&1
result=$?
printf '%s\n' "$result" > artifacts/pretrained_gpu_v1_exit_code.txt
exit "$result"

#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="$project_root/artifacts/posttraining_assets_20260926_resume_v4"
mkdir -p "$run_dir"
cd "$project_root"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=30
set +e
timeout --signal=TERM --kill-after=30s 14400 \
  /home/ubuntu/anaconda3/envs/creditlab-verl-alfworld/bin/python -u \
  posttraining_plan/stage_public_models.py --run-dir "$run_dir" \
  --weight-transport aria2 --accelerated-cache "$project_root/artifacts/accelerated_assets_v1" \
  >> "$run_dir/launcher.log" 2>&1
stage_exit=$?
printf '%s\n' "$stage_exit" > "$run_dir/exit_code.txt"
exit "$stage_exit"

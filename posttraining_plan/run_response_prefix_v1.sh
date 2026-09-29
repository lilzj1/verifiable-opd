#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export CUDA_VISIBLE_DEVICES=4 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
role="${1:-shared_student_initialization}"
out="artifacts/response_prefix_${role}_v1"
.text_runtime/bin/python -u posttraining_plan/probe_response_prefix.py --role "$role" --stage vllm --output "$out"
code=$?
if [[ "$code" == 0 ]]; then
  .text_runtime/bin/python -u posttraining_plan/probe_response_prefix.py --role "$role" --stage hf --output "$out"
  code=$?
fi
printf '%s\n' "$code" > "${out}_exit_code.txt"
exit "$code"

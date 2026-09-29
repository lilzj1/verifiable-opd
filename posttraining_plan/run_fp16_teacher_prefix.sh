#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export CUDA_VISIBLE_DEVICES=4 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
for role in math_teacher instruction_following_teacher; do
  out="artifacts/student_prefix_${role}_fp16_v1"
  .text_runtime/bin/python -u posttraining_plan/probe_response_prefix.py --dtype float16 --role "$role" --stage vllm --student-prefixes artifacts/response_prefix_student_v1/vllm.json --output "$out"
  code=$?
  if [[ "$code" == 0 ]]; then
    .text_runtime/bin/python -u posttraining_plan/probe_response_prefix.py --dtype float16 --role "$role" --stage hf --attention sdpa --output "$out"
    code=$?
  fi
  printf '%s\n' "$code" > "${out}_exit_code.txt"
done

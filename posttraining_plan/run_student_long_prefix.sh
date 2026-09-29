#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export CUDA_VISIBLE_DEVICES=4 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
run=artifacts/long_student_prefix_student_bf16_v1
[[ ! -e "$run" ]] || exit 2
for stage in vllm hf; do
  .text_runtime/bin/python -u posttraining_plan/probe_response_prefix.py \
    --role shared_student_initialization --stage "$stage" --output "$run" \
    --dtype bfloat16 --attention flash_attention_2 --max-context 8192 \
    --student-prefixes artifacts/teacher_dev_probe_v2_shared_student_initialization/predictions.jsonl
  code=$?
  if (( code != 0 )); then
    printf '%s\n' "$code" > "${run}_exit_code.txt"
    exit "$code"
  fi
done
printf '0\n' > "${run}_exit_code.txt"

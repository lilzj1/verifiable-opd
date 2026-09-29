#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
role="$1"
gpu="$2"
case "$role:$gpu" in
  shared_student_initialization:1|math_teacher:2|instruction_following_teacher:3) ;;
  *) exit 2;;
esac
run="artifacts/teacher_matrix_fp16_v1_${role}"
[[ ! -e "$run" ]] || exit 3
export CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NLTK_DATA="$PWD/data/nltk"
printf 'Started UTC: %s\n' "$(date -u --iso-8601=seconds)"
timeout --signal=TERM --kill-after=30s 2h .text_runtime/bin/python -u posttraining_plan/run_teacher_probe.py \
  --role "$role" --gpu "$gpu" --response 8192 --per-domain 64 \
  --temperature 0.6 --top-k 20 --dtype float16 --output "$run"
code=$?
printf '%s\n' "$code" > "${run}_exit_code.txt"
printf 'Finished UTC: %s; exit=%s\n' "$(date -u --iso-8601=seconds)" "$code"
exit "$code"

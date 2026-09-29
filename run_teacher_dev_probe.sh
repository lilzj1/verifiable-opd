#!/usr/bin/env bash
# Frozen-teacher capability gate before any real verl OPD profile.

set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_DIR"

TEACHER_MODEL=${TEACHER_MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
OUTPUT=${OUTPUT:-artifacts/qwen_7b_teacher_dev_probe_v1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
PYTHON=${PYTHON:-/home/ubuntu/anaconda3/envs/creditlab-verl-alfworld/bin/python}

export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" -m verifiable_opd.model_integrity "$TEACHER_MODEL"

exec "$PYTHON" -m verifiable_opd.evaluate \
  --model "$TEACHER_MODEL" \
  --data data/protocol_v2/dev.jsonl \
  --output "$OUTPUT" \
  --device cuda:0 \
  --limit 8 \
  --seed 123

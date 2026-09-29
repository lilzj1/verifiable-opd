#!/usr/bin/env bash
# Frozen full evaluation for the two 200-step models.

set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_DIR"

PYTHON=${PYTHON:-/home/ubuntu/anaconda3/envs/creditlab-verl-alfworld/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-artifacts/formal_eval_s200_v1}
mkdir -p "$OUTPUT_ROOT/logs"

evaluate_split() {
  local gpu=$1 model=$2 split=$3 output=$4 log=$5
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m verifiable_opd.evaluate \
    --model "$model" \
    --data "data/protocol_v2/${split}.jsonl" \
    --output "$output" \
    --device cuda:0 \
    --max-new-tokens 96 \
    --max-context 4096 \
    --seed 123 >>"$log" 2>&1
}

run_short_splits() {
  local gpu=$1 mode=$2 model=$3
  local log="$OUTPUT_ROOT/logs/${mode}_dev_id.log"
  evaluate_split "$gpu" "$model" dev "$OUTPUT_ROOT/$mode/dev" "$log"
  evaluate_split "$gpu" "$model" test_id "$OUTPUT_ROOT/$mode/test_id" "$log"
}

run_ood() {
  local gpu=$1 mode=$2 model=$3
  evaluate_split "$gpu" "$model" test_ood "$OUTPUT_ROOT/$mode/test_ood" \
    "$OUTPUT_ROOT/logs/${mode}_ood.log"
}

opd_model=artifacts/formal_opd_s200_seed17_v1/merged_hf
uniform_model=artifacts/formal_uniform_s200_seed17_v1/merged_hf

echo "[$(date -Is)] frozen evaluation start" | tee "$OUTPUT_ROOT/controller.log"
run_ood 1 opd "$opd_model" & p1=$!
run_ood 2 uniform "$uniform_model" & p2=$!
run_short_splits 3 opd "$opd_model" & p3=$!
run_short_splits 4 uniform "$uniform_model" & p4=$!

status=0
for pid in "$p1" "$p2" "$p3" "$p4"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "[$(date -Is)] frozen evaluation failed" | tee -a "$OUTPUT_ROOT/controller.log"
  exit "$status"
fi

for mode in opd uniform; do
  for split in dev test_id test_ood; do
    [[ -s "$OUTPUT_ROOT/$mode/$split/summary.json" ]] || {
      echo "missing summary: $mode/$split" >&2
      exit 2
    }
  done
done

echo "[$(date -Is)] FROZEN_EVALUATION_COMPLETE" | tee -a "$OUTPUT_ROOT/controller.log"

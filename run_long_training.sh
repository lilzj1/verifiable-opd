#!/usr/bin/env bash
# Continue the two retained formal arms from step 50 to step 200.
# The runs execute sequentially on physical GPUs 1,2,3,4.

set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_DIR"

PYTHON=${PYTHON:-/home/ubuntu/anaconda3/envs/creditlab-verl-alfworld/bin/python}
TOTAL_STEPS=${TOTAL_STEPS:-200}
SAVE_FREQ=${SAVE_FREQ:-25}
KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-2}
TEACHER_MODEL=${TEACHER_MODEL:-artifacts/qwen_7b_teacher_lora_s256_v1/merged_model}

run_arm() {
  local mode=$1
  local source_experiment=$2
  local target_experiment=$3
  local source_checkpoint="artifacts/$source_experiment/checkpoints/global_step_50"
  local target_root="artifacts/$target_experiment/checkpoints"
  local resume_checkpoint=$source_checkpoint

  if [[ -f "$target_root/latest_checkpointed_iteration.txt" ]]; then
    local latest
    latest=$(<"$target_root/latest_checkpointed_iteration.txt")
    if [[ "$latest" =~ ^[0-9]+$ ]] && [[ -d "$target_root/global_step_$latest/actor" ]]; then
      resume_checkpoint="$target_root/global_step_$latest"
    fi
  fi

  [[ -d "$resume_checkpoint/actor" ]] || {
    echo "Missing resume checkpoint: $resume_checkpoint" >&2
    return 2
  }

  mkdir -p "artifacts/$target_experiment"
  local log="artifacts/${target_experiment}_launcher.log"
  echo "[$(date -Is)] START mode=$mode resume=$resume_checkpoint total=$TOTAL_STEPS" | tee -a "$log"

  set +e
  CUDA_VISIBLE_DEVICES=1,2,3,4 \
  MODE="$mode" \
  TOTAL_STEPS="$TOTAL_STEPS" \
  TOTAL_EPOCHS=1000 \
  SAVE_FREQ="$SAVE_FREQ" \
  MAX_ACTOR_CKPT_TO_KEEP="$KEEP_CHECKPOINTS" \
  TEACHER_TP=1 \
  TEACHER_MODEL="$TEACHER_MODEL" \
  EXPERIMENT_NAME="$target_experiment" \
  RESUME_MODE=resume_path \
  RESUME_FROM_PATH="$resume_checkpoint" \
  bash run_verl_profile.sh 2>&1 | tee -a "$log" &
  local train_pid=$!
  "$PYTHON" checkpoint_rotator.py \
    --checkpoint-root "$target_root" \
    --keep "$KEEP_CHECKPOINTS" \
    --watch-pid "$train_pid" \
    --interval 20 >>"artifacts/${target_experiment}_rotation.log" 2>&1 &
  local rotate_pid=$!
  wait "$train_pid"
  local status=$?
  wait "$rotate_pid" || true
  set -e

  "$PYTHON" checkpoint_rotator.py --checkpoint-root "$target_root" --keep "$KEEP_CHECKPOINTS" --once
  if [[ "$status" -ne 0 ]]; then
    echo "[$(date -Is)] FAILED mode=$mode status=$status; next arm will not start" | tee -a "$log"
    return "$status"
  fi

  local final_checkpoint="$target_root/global_step_$TOTAL_STEPS"
  [[ -d "$final_checkpoint/actor" ]] || {
    echo "Final checkpoint missing: $final_checkpoint" >&2
    return 3
  }
  echo "[$(date -Is)] COMPLETE mode=$mode final=$final_checkpoint" | tee -a "$log"

  # Once a newer, recoverable run has completed, the external step-50 FSDP
  # source is redundant.  Its merged HF model and all evaluation evidence stay.
  if [[ "$source_checkpoint" != "$final_checkpoint" ]] && [[ -d "$source_checkpoint" ]]; then
    rm -rf -- "$source_checkpoint"
    echo "[$(date -Is)] REMOVED superseded source=$source_checkpoint" | tee -a "$log"
  fi
}

run_arm opd formal_opd_s50_seed17_v2 formal_opd_s200_seed17_v1
run_arm uniform formal_uniform_s50_seed17_v1 formal_uniform_s200_seed17_v1

echo "[$(date -Is)] ALL_LONG_TRAINING_COMPLETE"

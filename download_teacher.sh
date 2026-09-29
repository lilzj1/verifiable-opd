#!/usr/bin/env bash
# Resumable, integrity-checked download for the frozen Qwen2.5-7B teacher.

set -euo pipefail

REVISION=a09a35458c702b33eeacc393d103063234e8bc28
CACHE_DIR=${CACHE_DIR:-/home/ubuntu/.cache/huggingface/hub}
HF=${HF:-/home/ubuntu/anaconda3/envs/creditlab-verl-alfworld/bin/hf}
SNAPSHOT="$CACHE_DIR/models--Qwen--Qwen2.5-7B-Instruct/snapshots/$REVISION"

export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-30}

download_one() {
  local file=$1
  local expected=$2
  local attempt=0
  while true; do
    if [[ -f "$SNAPSHOT/$file" ]] && [[ $(stat -Lc %s "$SNAPSHOT/$file") -eq $expected ]]; then
      echo "TEACHER_SHARD_READY file=$file bytes=$expected attempts=$attempt"
      return 0
    fi
    attempt=$((attempt + 1))
    echo "TEACHER_SHARD_ATTEMPT file=$file attempt=$attempt"
    if "$HF" download Qwen/Qwen2.5-7B-Instruct "$file" \
      --revision "$REVISION" --cache-dir "$CACHE_DIR" --max-workers 1; then
      if [[ -f "$SNAPSHOT/$file" ]] && [[ $(stat -Lc %s "$SNAPSHOT/$file") -eq $expected ]]; then
        echo "TEACHER_SHARD_READY file=$file bytes=$expected attempts=$attempt"
        return 0
      fi
      echo "download returned success but size is not exact: $file" >&2
    else
      echo "teacher shard download retrying: $file attempt=$attempt" >&2
    fi
    sleep 3
  done
}

download_one model-00001-of-00004.safetensors 3945441440 & p1=$!
download_one model-00002-of-00004.safetensors 3864726352 & p2=$!
download_one model-00003-of-00004.safetensors 3864726424 & p3=$!
download_one model-00004-of-00004.safetensors 3556377672 & p4=$!
wait "$p1" "$p2" "$p3" "$p4"
echo QWEN_7B_DOWNLOAD_COMPLETE

#!/usr/bin/env bash
# verl v0.8.0 profile/formal launcher for the frozen OPD design.
#
# MODE=opd      : official verl k1 PG-OPD, no expert replay
# MODE=uniform  : OPD + token-budgeted uniform expert CE replay
# MODE=failure  : OPD + failure-aware expert CE replay
# CFG_ONLY=1    : compose and resolve Hydra config; allocate no GPU
# TOTAL_STEPS and SAVE_FREQ control short profiles or full runs.  The formal
# v1 comparison used TOTAL_STEPS=50 and SAVE_FREQ=50 for every arm.

set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_DIR"

MODE=${MODE:-opd}
CFG_ONLY=${CFG_ONLY:-0}
TOTAL_STEPS=${TOTAL_STEPS:-1}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-100}
SAVE_FREQ=${SAVE_FREQ:-1}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}
TEACHER_TP=${TEACHER_TP:-1}
RESUME_MODE=${RESUME_MODE:-disable}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"verl_${MODE}_profile_step${TOTAL_STEPS}_v1"}

case "$RESUME_MODE" in
  disable|auto) ;;
  resume_path)
    [[ -n "$RESUME_FROM_PATH" ]] || {
      echo "RESUME_FROM_PATH is required when RESUME_MODE=resume_path" >&2
      exit 2
    }
    ;;
  *)
    echo "RESUME_MODE must be one of: disable, auto, resume_path" >&2
    exit 2
    ;;
esac

case "$MODE" in
  opd)
    replay_strategy=none
    replay_alpha=0.0
    replay_budget=0
    replay_trajectories=0
    ;;
  uniform|failure)
    replay_strategy=$MODE
    replay_alpha=${REPLAY_ALPHA:-0.1}
    replay_budget=${REPLAY_TOKEN_BUDGET:-512}
    replay_trajectories=${REPLAY_TRAJECTORIES:-2}
    ;;
  *)
    echo "MODE must be one of: opd, uniform, failure" >&2
    exit 2
    ;;
esac

STUDENT_MODEL=${STUDENT_MODEL:-artifacts/qwen_sft_step1_probe_v1/step_000768/model}
TEACHER_MODEL=${TEACHER_MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
PYTHON=${PYTHON:-/home/ubuntu/anaconda3/envs/creditlab-verl-alfworld/bin/python}

if [[ "$CFG_ONLY" != 1 ]]; then
  [[ -f "$STUDENT_MODEL/config.json" ]] || { echo "missing student: $STUDENT_MODEL" >&2; exit 3; }
  PYTHONPATH="$PROJECT_DIR/upstream/verl-v0.8.0:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" -m verifiable_opd.model_integrity "$STUDENT_MODEL"
  PYTHONPATH="$PROJECT_DIR/upstream/verl-v0.8.0:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" -m verifiable_opd.model_integrity "$TEACHER_MODEL"
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3,4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
# This host exposes all A6000 pairs through SYS and has no direct P2P/NVLink
# path.  The same settings are required by the already validated day56
# four-GPU jobs; without them FSDP can stall in sync_module_states.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export PYTHONPATH="$PROJECT_DIR/upstream/verl-v0.8.0:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

MAX_PROMPT=512
MAX_RESPONSE=1536
MAX_MODEL_LEN=$((MAX_PROMPT + MAX_RESPONSE + 1))

ARGS=(
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  data.train_files=data/protocol_v2/train.parquet
  data.val_files=data/protocol_v2/dev.parquet
  data.train_batch_size=2
  data.val_batch_size=24
  data.max_prompt_length=$MAX_PROMPT
  data.max_response_length=$MAX_RESPONSE
  data.filter_overlong_prompts=True
  data.truncation=error
  data.return_raw_chat=True
  data.shuffle=False
  data.seed=17
  actor_rollout_ref.model.path="$STUDENT_MODEL"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.ppo_mini_batch_size=2
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.shuffle=False
  actor_rollout_ref.actor.loss_agg_mode=token-mean
  actor_rollout_ref.actor.use_dynamic_bsz=False
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.n=1
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35
  actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.enforce_eager=True
  actor_rollout_ref.rollout.enable_chunked_prefill=False
  actor_rollout_ref.rollout.free_cache_engine=False
  actor_rollout_ref.rollout.multi_turn.enable=True
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=16
  actor_rollout_ref.rollout.multi_turn.max_user_turns=16
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
  actor_rollout_ref.rollout.multi_turn.format=hermes
  actor_rollout_ref.rollout.multi_turn.tool_config_path=configs/tools.yaml
  actor_rollout_ref.rollout.agent.default_agent_loop=arithmetic_tool_agent
  actor_rollout_ref.rollout.agent.agent_loop_config_path=configs/agent_loops.yaml
  actor_rollout_ref.rollout.agent.num_workers=2
  distillation.enabled=True
  distillation.n_gpus_per_node=2
  distillation.nnodes=1
  distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
  distillation.teacher_models.teacher_model.inference.name=vllm
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=$TEACHER_TP
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.50
  distillation.teacher_models.teacher_model.inference.max_model_len=$MAX_MODEL_LEN
  distillation.teacher_models.teacher_model.inference.enforce_eager=True
  distillation.teacher_models.teacher_model.inference.enable_chunked_prefill=False
  distillation.distillation_loss.loss_mode=k1
  distillation.distillation_loss.use_task_rewards=False
  distillation.distillation_loss.use_policy_gradient=True
  distillation.distillation_loss.policy_loss_mode=vanilla
  distillation.distillation_loss.loss_max_clamp=10.0
  distillation.distillation_loss.log_prob_min_clamp=-10.0
  distillation.distillation_loss.expert_replay_strategy=$replay_strategy
  distillation.distillation_loss.expert_replay_alpha=$replay_alpha
  distillation.distillation_loss.expert_replay_token_budget=$replay_budget
  distillation.distillation_loss.expert_replay_max_trajectories=$replay_trajectories
  distillation.distillation_loss.expert_replay_seed=17
  trainer.balance_batch=True
  trainer.logger='["console"]'
  trainer.project_name=verifiable_opd
  trainer.experiment_name="$EXPERIMENT_NAME"
  trainer.n_gpus_per_node=2
  trainer.nnodes=1
  trainer.total_epochs=$TOTAL_EPOCHS
  trainer.total_training_steps=$TOTAL_STEPS
  trainer.val_before_train=False
  trainer.test_freq=-1
  trainer.save_freq=$SAVE_FREQ
  trainer.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP
  trainer.default_local_dir="artifacts/$EXPERIMENT_NAME/checkpoints"
  trainer.rollout_data_dir="artifacts/$EXPERIMENT_NAME/rollouts"
  trainer.resume_mode=$RESUME_MODE
  +ray_init.num_cpus=16
)

if [[ "$RESUME_MODE" == resume_path ]]; then
  ARGS+=(trainer.resume_from_path="$RESUME_FROM_PATH")
fi

if [[ "$CFG_ONLY" == 1 ]]; then
  exec "$PYTHON" -m verl.trainer.main_ppo "${ARGS[@]}" --cfg job --resolve
fi

mkdir -p "artifacts/$EXPERIMENT_NAME"
exec "$PYTHON" -m verl.trainer.main_ppo "${ARGS[@]}"

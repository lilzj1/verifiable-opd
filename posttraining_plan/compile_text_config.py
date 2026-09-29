"""Compose/audit new Math+IF verl config without allocating GPUs or training."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/'upstream/verl-v0.8.0'), str(ROOT)]


def build(group, run_id, updates=1, response_length=4096):
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.config import DistillationConfig
    if group not in {'S2','S3'} or not run_id.replace('_','').isalnum():
        raise ValueError('Explicit S2/S3 and safe run ID required')
    if response_length not in (4096, 8192, 16384) or updates < 1:
        raise ValueError('Use a registered length candidate and positive update count')
    context_length = 2048 + response_length + 1
    assets=json.loads((ROOT/'artifacts/posttraining_assets_20260926_resume_v4/status.json').read_text())
    with initialize_config_dir(config_dir=str(ROOT/'upstream/verl-v0.8.0/verl/trainer/config'),version_base=None):
        config=compose(config_name='ppo_trainer')
    changes={
        'algorithm.adv_estimator':'grpo','algorithm.use_kl_in_reward':False,
        'data.train_files':str(ROOT/'data/posttraining_text_v3/online_train.parquet'),
        'data.val_files':str(ROOT/'data/posttraining_text_v3/dev.parquet'),
        'data.train_batch_size':4,'data.val_batch_size':4,'data.max_prompt_length':2048,
        'data.max_response_length':response_length,'data.filter_overlong_prompts':True,
        'data.truncation':'error','data.return_raw_chat':True,'data.shuffle':False,'data.seed':17,
        'actor_rollout_ref.model.path':assets['models']['shared_student_initialization']['overlay'],
        'actor_rollout_ref.model.use_remove_padding':False,
        'actor_rollout_ref.model.enable_gradient_checkpointing':True,
        'actor_rollout_ref.actor.optim.lr':1e-6,
        'actor_rollout_ref.actor.optim.total_training_steps':updates,
        'actor_rollout_ref.actor.ppo_mini_batch_size':4,
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu':1,
        'actor_rollout_ref.actor.ppo_epochs':1,'actor_rollout_ref.actor.shuffle':False,
        'actor_rollout_ref.actor.data_loader_seed':17,
        'actor_rollout_ref.actor.loss_agg_mode':'token-mean',
        'actor_rollout_ref.actor.use_dynamic_bsz':False,'actor_rollout_ref.actor.use_kl_loss':False,
        'actor_rollout_ref.actor.fsdp_config.param_offload':False,
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload':False,
        'actor_rollout_ref.actor.fsdp_config.seed':17,
        'actor_rollout_ref.actor.fsdp_config.use_torch_compile':False,
        'actor_rollout_ref.rollout.name':'vllm','actor_rollout_ref.rollout.n':1,
        'actor_rollout_ref.rollout.seed':17,
        'actor_rollout_ref.rollout.temperature':1.0,'actor_rollout_ref.rollout.top_p':.95,
        'actor_rollout_ref.rollout.top_k':-1,'actor_rollout_ref.rollout.tensor_model_parallel_size':1,
        'actor_rollout_ref.rollout.gpu_memory_utilization':.25,
        'actor_rollout_ref.rollout.max_model_len':context_length,
        'actor_rollout_ref.rollout.max_num_batched_tokens':max(8192,context_length),
        'actor_rollout_ref.rollout.max_num_seqs':4,
        'actor_rollout_ref.rollout.log_prob_use_dynamic_bsz':False,
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu':1,
        'actor_rollout_ref.rollout.enforce_eager':True,
        'actor_rollout_ref.rollout.enable_chunked_prefill':False,
        'actor_rollout_ref.rollout.enable_prefix_caching':False,
        'actor_rollout_ref.rollout.free_cache_engine':True,
        'actor_rollout_ref.rollout.multi_turn.enable':False,
        'actor_rollout_ref.rollout.agent.default_agent_loop':'single_turn_agent',
        'actor_rollout_ref.rollout.agent.num_workers':2,
        'distillation.enabled':True,'distillation.n_gpus_per_node':2,'distillation.nnodes':1,
        'distillation.teacher_key':'data_source',
        'distillation.distillation_loss.loss_mode':'k1',
        'distillation.distillation_loss.use_task_rewards':False,
        'distillation.distillation_loss.use_policy_gradient':True,
        'distillation.distillation_loss.policy_loss_mode':'vanilla',
        'distillation.distillation_loss.loss_max_clamp':None,
        'distillation.distillation_loss.log_prob_min_clamp':None,
        'distillation.distillation_loss.single_update_per_batch':True,
        'distillation.distillation_loss.text_response_limits':{'math':16384,'if':2048} if response_length==16384 else {},
        'distillation.distillation_loss.expert_replay_adapter':'text',
        'distillation.distillation_loss.expert_replay_strategy':'uniform' if group=='S3' else 'none',
        'distillation.distillation_loss.expert_replay_alpha':.1 if group=='S3' else 0.,
        'distillation.distillation_loss.expert_replay_token_budget':8192 if group=='S3' else 0,
        'distillation.distillation_loss.expert_replay_max_trajectories':2 if group=='S3' else 0,
        'distillation.distillation_loss.expert_replay_seed':17,
        'distillation.distillation_loss.expert_replay_data_path':str(ROOT/'data/posttraining_text_v3/expert_train.jsonl'),
        'reward.custom_reward_function.path':str(ROOT/'verifiable_opd/posttraining/no_task_reward.py'),
        'reward.custom_reward_function.name':'compute_score','reward.num_workers':2,
        'trainer.balance_batch':True,'trainer.logger':['console'],
        'trainer.project_name':'opd_math_if','trainer.experiment_name':run_id,
        'trainer.n_gpus_per_node':2,'trainer.nnodes':1,'trainer.total_epochs':100,
        'trainer.total_training_steps':updates,'trainer.val_before_train':False,
        'trainer.test_freq':-1,'trainer.save_freq':1,'trainer.max_actor_ckpt_to_keep':None,
        'trainer.default_local_dir':str(ROOT/'artifacts'/run_id/'checkpoints'),
        'trainer.rollout_data_dir':str(ROOT/'artifacts'/run_id/'rollouts'),
        'trainer.resume_mode':'disable',
    }
    for key,value in changes.items():
        parent,leaf=key.rsplit('.',1)
        container=OmegaConf.select(config,parent)
        if container is None or leaf not in container.keys():
            raise KeyError(f'Unknown upstream setting: {key}')
        OmegaConf.update(config,key,value,merge=False)
    template=OmegaConf.to_container(config.distillation.teacher_models.teacher_model,resolve=True)
    for name,domain,role in [('math_teacher','math','math_teacher'),('if_teacher','if','instruction_following_teacher')]:
        teacher=copy.deepcopy(template)
        teacher.update(key=domain,model_path=assets['models'][role]['overlay'],num_replicas=1)
        teacher['inference'].update(tensor_model_parallel_size=1,gpu_memory_utilization=.50,
            dtype='float16',max_model_len=context_length,max_num_seqs=4,enforce_eager=True,enable_chunked_prefill=False,
            enable_prefix_caching=False,seed=17,max_num_batched_tokens=max(8192,context_length))
        OmegaConf.update(config,f'distillation.teacher_models.{name}',teacher,force_add=True)
    # Resolve/validate resource topology on a copy: resolver removes the template.
    distilled=omega_conf_to_dataclass(OmegaConf.create(OmegaConf.to_container(config.distillation,resolve=True)))
    assert set(distilled.teacher_models)=={'math','if'}, distilled.teacher_models.keys()
    return config


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--group',required=True,choices=['S2','S3'])
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--response',type=int,choices=[4096,8192,16384],default=4096)
    args=parser.parse_args()
    from omegaconf import OmegaConf
    config=build(args.group,args.run_id,response_length=args.response)
    args.output.mkdir(parents=True,exist_ok=False)
    yaml=OmegaConf.to_yaml(config,resolve=True)
    (args.output/'resolved.yaml').write_text(yaml)
    (args.output/'audit.json').write_text(json.dumps({'phase':'COMPOSED_NOT_AUTHORIZED_FOR_TRAINING',
        'group':args.group,'config_sha256':hashlib.sha256(yaml.encode()).hexdigest(),
        'physical_gpu_candidates':[1,2,3,4],'student_gpus':2,'teacher_gpus':2,
        'notes':['No GPU allocated','Real prefix gate/teacher matrix/checkpoint smoke still required',
                 'FP16 teacher prefix gate passed; autonomous FP16 quality remains to validate',
                 'Evaluation dependencies are isolated from Hydra training to avoid ANTLR conflict']},indent=2))
    print('Composed',args.output,flush=True)


if __name__=='__main__':
    main()

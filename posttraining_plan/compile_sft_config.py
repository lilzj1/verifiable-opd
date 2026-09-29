"""S1 smoke-only configuration compiler; token-matched formal budget NOT frozen."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'upstream/verl-v0.8.0'),str(ROOT)]


def build(run_id):
    from hydra import initialize_config_dir,compose
    from omegaconf import OmegaConf
    if not run_id.replace('_','').isalnum():raise ValueError('Safe run ID required')
    assets=json.loads((ROOT/'artifacts/posttraining_assets_20260926_resume_v4/status.json').read_text())
    with initialize_config_dir(config_dir=str(ROOT/'upstream/verl-v0.8.0/verl/trainer/config'),version_base=None):
        config=compose(config_name='sft_trainer_engine')
    changes={
        'model.path':assets['models']['shared_student_initialization']['overlay'],
        'model.use_remove_padding':False,'model.enable_gradient_checkpointing':True,
        'data.train_files':str(ROOT/'data/posttraining_text_v3/expert_train.jsonl'),
        'data.val_files':None,'data.train_batch_size':4,'data.micro_batch_size_per_gpu':1,
        'data.use_dynamic_bsz':False,'data.max_length':6144,'data.truncation':'error',
        'data.pad_mode':'no_padding','data.num_workers':0,
        'data.custom_cls.path':str(ROOT/'verifiable_opd/posttraining/sft_dataset.py'),
        'data.custom_cls.name':'TextExpertSFTDataset',
        'engine.seed':17,'engine.use_torch_compile':False,'engine.dtype':'bfloat16',
        'engine.param_offload':False,'engine.optimizer_offload':False,
        'optim.lr':1e-6,'optim.lr_scheduler_type':'constant','optim.lr_warmup_steps':0,
        'trainer.total_training_steps':1,'trainer.total_epochs':1,'trainer.seed':17,
        'trainer.logger':['console'],'trainer.n_gpus_per_node':2,
        'trainer.project_name':'opd_math_if','trainer.experiment_name':run_id,
        'trainer.save_freq':1,'trainer.test_freq':-1,'trainer.resume_mode':'disable',
        'trainer.default_local_dir':str(ROOT/'artifacts'/run_id/'checkpoints'),
    }
    for key,value in changes.items():
        parent,leaf=key.rsplit('.',1)
        if leaf not in OmegaConf.select(config,parent):raise KeyError(key)
        OmegaConf.update(config,key,value,merge=False)
    return config


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    from omegaconf import OmegaConf
    config=build(args.run_id)
    yaml=OmegaConf.to_yaml(config,resolve=True)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'resolved.yaml').write_text(yaml)
    (args.output/'audit.json').write_text(json.dumps({
        'phase':'COMPOSED_SMOKE_ONLY_NOT_AUTHORIZED_FOR_TRAINING',
        'config_sha256':hashlib.sha256(yaml.encode()).hexdigest(),
        'formal_token_budget':None,'checkpoint_rotation':'NOT_ENABLED',
        'notes':['Same complete expert pool and CE mask as S3',
                 'S1 formal target-token budget must come from S2 profile, not final results',
                 'Custom dataset CPU-tested; distributed SFT update/restore not yet validated']},indent=2)+'\n')
    print('Composed',args.output)


if __name__=='__main__':main()

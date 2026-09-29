"""Opt-in text experiment checkpoint commits; never deletes any files.

A commit checks all expected rank shards, ZIP CRC and byte hashes. Only an
actual distributed reload can establish recoverability; this module does not
claim that a valid ZIP is a successfully restored model.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path=Path(path)
    temporary=path.with_name(path.name+'.pending')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,path)


def checked_path(path):
    path=Path(path).absolute()
    if any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('Checkpoint paths must not traverse symlinks')
    if not re.fullmatch(r'global_step_\d+',path.name):
        raise ValueError('Not an exact step directory')
    return path


def safe_child(root, name):
    relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Invalid checkpoint member path')
    file=root/relative
    if file.resolve()!=file or any(p.is_symlink() for p in (file,*file.parents)):
        raise ValueError('Checkpoint member traverses symlink')
    return file


def effective_optimizer_config(optimizer, planned_updates):
    """Resolve only the unset sentinel as the trainer does, preserving strict checks."""
    result = dict(optimizer)
    if not isinstance(planned_updates, int) or planned_updates < 1:
        raise ValueError('Explicit positive planned updates required')
    steps = result.get('total_training_steps', -1)
    if steps not in (-1, planned_updates):
        raise ValueError('Explicit optimizer schedule disagrees with planned updates')
    result['total_training_steps'] = planned_updates
    return result


def replay_contract(config):
    from omegaconf import OmegaConf
    loss=config.distillation.distillation_loss
    def files(value):
        paths=[value] if isinstance(value,str) else list(value)
        return [{'path':str(Path(p).resolve(strict=True)),'sha256':file_sha(p)} for p in paths]
    return {'schema':2,'adapter':loss.expert_replay_adapter,
        'strategy':loss.expert_replay_strategy,'alpha':loss.expert_replay_alpha,
        'seed':loss.expert_replay_seed,'token_budget':loss.expert_replay_token_budget,
        'rows':loss.expert_replay_max_trajectories,
        'online':files(config.data.train_files),
        'expert':files(loss.expert_replay_data_path) if loss.expert_replay_strategy!='none' else [],
        'max_prompt':config.data.max_prompt_length,'max_response':config.data.max_response_length,
        'temperature':config.actor_rollout_ref.rollout.temperature,
        'top_p':config.actor_rollout_ref.rollout.top_p,
        'top_k':config.actor_rollout_ref.rollout.top_k,
        'rollout_n':config.actor_rollout_ref.rollout.n,
        'rollout_seed':config.actor_rollout_ref.rollout.seed,
        'planned_updates':config.trainer.total_training_steps,
        'optimizer':effective_optimizer_config(
            OmegaConf.to_container(config.actor_rollout_ref.actor.optim,resolve=True),
            config.trainer.total_training_steps),
        'initial_model_config_sha256':file_sha(Path(config.actor_rollout_ref.model.path)/'config.json'),
        'teachers':OmegaConf.to_container(config.distillation.teacher_models,resolve=True),
        'selection_state':'stateless hash(seed,global_step,example_id); pool byte hash must match'}


def commit(path, world_size, contract):
    path=checked_path(path)
    if world_size<=0:
        raise ValueError('Positive expected world size required')
    expected={'data.pt'}|{f'actor/{kind}_world_size_{world_size}_rank_{rank}.pt'
        for kind in ('model','optim','extra_state') for rank in range(world_size)}
    found={str(p.relative_to(path)) for p in (path/'actor').glob('*_world_size_*_rank_*.pt')}
    if found != expected-{'data.pt'}:
        raise ValueError('Missing rank(s) or conflicting world_size/shard names')
    entries={}
    for name in sorted(expected):
        file=safe_child(path,name)
        if file.is_symlink() or not file.is_file() or file.stat().st_size==0:
            raise ValueError(f'Missing/empty/unsafe checkpoint file: {name}')
        if not zipfile.is_zipfile(file):
            raise ValueError(f'Unsupported/corrupt torch serialization: {name}')
        with zipfile.ZipFile(file) as archive:
            if archive.testzip() is not None:
                raise ValueError(f'Checkpoint CRC mismatch: {name}')
        entries[name]={'bytes':file.stat().st_size,'sha256':file_sha(file)}
    manifest={'schema':1,'phase':'SHARDS_VERIFIED_RESTORE_NOT_YET_TESTED',
        'world_size':world_size,'step':int(path.name.removeprefix('global_step_')),
        'contract':contract,'files':entries}
    atomic_json(path/'TEXT_COMMIT.json',manifest)
    return manifest


def verify_before_restore(path, world_size, contract):
    path=checked_path(path)
    manifest=json.loads((path/'TEXT_COMMIT.json').read_text())
    if manifest['world_size']!=world_size or manifest['contract']!=contract:
        raise ValueError('Resume world size/data/replay contract changed')
    expected={'data.pt'}|{f'actor/{kind}_world_size_{world_size}_rank_{rank}.pt'
        for kind in ('model','optim','extra_state') for rank in range(world_size)}
    if set(manifest['files'])!=expected:
        raise ValueError('Commit manifest missing expected shards')
    for name,entry in manifest['files'].items():
        file=safe_child(path,name)
        if file.is_symlink() or not file.is_relative_to(path) or file_sha(file)!=entry['sha256']:
            raise ValueError(f'Checkpoint changed since commit: {name}')
    return manifest

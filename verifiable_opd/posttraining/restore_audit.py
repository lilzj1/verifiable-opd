"""Exact local state digests for opt-in real distributed restore diagnostics.

Includes all local tensor bytes, not a parameter sample. This does not establish
bitwise equivalence of a subsequent online rollout with uninterrupted training.
"""
import hashlib
import json

import numpy as np
import torch


def state_digest(value):
    digest = hashlib.sha256()

    def header(value):
        digest.update(json.dumps(value, sort_keys=True, separators=(',', ':')).encode() + b'\n')

    def visit(item):
        if hasattr(item, 'local_shards'):
            header(['sharded', list(item.size())])
            for shard in item.local_shards():
                header([shard.metadata.shard_offsets, shard.metadata.shard_sizes])
                visit(shard.tensor)
        elif hasattr(item, 'to_local'):
            header(['dtensor', list(item.shape), [str(p) for p in item.placements]])
            visit(item.to_local())
        elif isinstance(item, torch.Tensor):
            tensor = item.detach().contiguous().reshape(-1)
            header(['tensor', str(item.dtype), list(item.shape)])
            chunk = max(1, 16 * 1024**2 // tensor.element_size())
            for start in range(0, tensor.numel(), chunk):
                digest.update(tensor[start:start+chunk].cpu().view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            header(['numpy', str(item.dtype), list(item.shape)])
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            header(['dict', len(item)])
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            header([type(item).__name__, len(item)])
            for member in item:
                visit(member)
        elif item is None or isinstance(item, (str, bool, int, float)):
            header([type(item).__name__, item])
        elif isinstance(item, np.generic):
            header(['numpy_scalar', str(item.dtype), item.item()])
        elif isinstance(item, bytes):
            header(['bytes', len(item)])
            digest.update(item)
        else:
            raise TypeError(f'Unsupported checkpoint state type: {type(item).__name__}')

    visit(value)
    return digest.hexdigest()


def comparison(expected, actual):
    expected_sha = state_digest(expected)
    actual_sha = state_digest(actual)
    return {'expected_sha256': expected_sha, 'actual_sha256': actual_sha,
            'equal': expected_sha == actual_sha}


def verify_restore_reports(directory, checkpoint, contract):
    from pathlib import Path
    from .checkpoint_protocol import file_sha
    directory, checkpoint = Path(directory), Path(checkpoint).resolve()
    expected = {'rank_0.json', 'rank_1.json', 'trainer.json'}
    if {p.name for p in directory.glob('*.json')} != expected:
        raise ValueError('All rank and trainer restore reports are required')
    reports = {name: json.loads((directory / name).read_text()) for name in sorted(expected)}
    for rank in (0, 1):
        row = reports[f'rank_{rank}.json']
        if row.get('phase') != 'LOCAL_STATES_RESTORED' or row.get('rank') != rank or row.get('world_size') != 2:
            raise ValueError('Rank did not confirm distributed restore')
        if Path(row['actor_path']).resolve() != checkpoint / 'actor':
            raise ValueError('Rank loaded a different checkpoint')
        checks = row['checks']
        if set(checks) != {'model', 'optimizer', 'scheduler', 'rng'} or not all(c['equal'] for c in checks.values()):
            raise ValueError('Model/optimizer/scheduler/RNG state mismatch')
    trainer = reports['trainer.json']
    if (trainer.get('phase') != 'TRAINER_RESTORED_NO_UPDATE'
            or Path(trainer['checkpoint']).resolve() != checkpoint
            or trainer['global_step'] != int(checkpoint.name.removeprefix('global_step_'))
            or not trainer['dataloader']['equal']):
        raise ValueError('Trainer progress/data state mismatch')
    return {'phase': 'DISTRIBUTED_RESTORE_VERIFIED', 'world_size': 2, 'contract': contract,
            'checkpoint': str(checkpoint), 'reports': reports,
            'report_sha256': {name: file_sha(directory/name) for name in expected},
            'limits': ['Exact local model/optimizer/scheduler/RNG and driver data/progress states',
                       'No claim that external vLLM RNG or subsequent rollouts are bitwise equivalent',
                       'Does not authorize formal training or checkpoint deletion']}

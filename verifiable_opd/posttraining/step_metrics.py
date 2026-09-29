"""Synchronous, fail-fast training metrics; independent of Ray stdout delivery."""
import math
from pathlib import Path
from .checkpoint_protocol import atomic_json


def write_metrics(audit_dir, step, metrics):
    values={key:float(value) for key,value in metrics.items()}
    critical={'actor/loss','actor/grad_norm','actor/distillation/loss',
              'actor/expert_replay/ce','actor/expert_replay/total_loss','train/loss','train/grad_norm'}
    directory=Path(audit_dir)/'step_metrics'
    directory.mkdir(parents=True,exist_ok=True)
    atomic_json(directory/f'{step}.json',values)
    if not any(key in values for key in ('actor/grad_norm','train/grad_norm')):
        raise ValueError('Training metrics lack gradient norm')
    bad=[key for key in critical & values.keys() if not math.isfinite(values[key])]
    if bad:
        raise FloatingPointError(f'Nonfinite training metrics at step {step}: {bad}; stopped before next generation')


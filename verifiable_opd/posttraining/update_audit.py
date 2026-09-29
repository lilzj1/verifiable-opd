"""Opt-in optimizer observation; does not alter parameters, gradients or RNG."""
import json
import math
from pathlib import Path

import torch


def local(tensor):
    return tensor.to_local() if hasattr(tensor,'to_local') else tensor


def step_number(optimizer,parameter):
    value=optimizer.state.get(parameter,{}).get('step',0)
    return int(value.item()) if hasattr(value,'item') else int(value)


@torch.no_grad()
def before_step(module,optimizer):
    for name,parameter in module.named_parameters():
        if parameter.grad is None:continue
        gradient=local(parameter.grad).flatten()
        if not gradient.numel() or not torch.isfinite(gradient).all():continue
        values,indices=gradient.abs().topk(min(16,gradient.numel()))
        if values.max().item()==0:continue
        data=local(parameter).flatten()
        if data.numel()!=gradient.numel():raise ValueError('Sharded parameter/gradient shape mismatch')
        return {'name':name,'parameter':parameter,'indices':indices,
                'values':data[indices].clone(),'optimizer_step':step_number(optimizer,parameter)}
    return None


@torch.no_grad()
def after_step(snapshot,optimizer,grad_norm):
    if snapshot is None:
        return {'phase':'NO_FINITE_NONZERO_GRADIENT_OBSERVED'}
    parameter=snapshot['parameter']
    after=local(parameter).flatten()[snapshot['indices']]
    delta=(after.float()-snapshot['values'].float()).abs().max().item()
    current=step_number(optimizer,parameter)
    finite=math.isfinite(float(grad_norm)) and math.isfinite(delta)
    return {'phase':'UPDATE_OBSERVED' if finite and delta>0 and current==snapshot['optimizer_step']+1 else 'UPDATE_NOT_VERIFIED',
            'parameter':snapshot['name'],'indices':snapshot['indices'].cpu().tolist(),
            'optimizer_step_before':snapshot['optimizer_step'],'optimizer_step_after':current,
            'max_observed_parameter_delta':delta if math.isfinite(delta) else str(delta),
            'grad_norm':float(grad_norm) if math.isfinite(float(grad_norm)) else str(grad_norm),
            'limitation':'Sampled local-shard values; not all-weight equivalence or proof of training quality'}


def append_record(directory,rank,record):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/f'rank_{rank}.jsonl').open('a') as stream:
        stream.write(json.dumps(record,allow_nan=False)+'\n')


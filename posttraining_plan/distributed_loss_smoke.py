"""Real two-process CPU/Gloo/DDP check, NOT a real-model FSDP training pass."""
import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'upstream/verl-v0.8.0'),str(ROOT)]


def loss_for(model,lengths,experts,dp_size,online_total,expert_total):
    import torch
    from tensordict import TensorDict
    from omegaconf import OmegaConf
    from types import SimpleNamespace
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import left_right_2_no_padding
    from verl.workers.config.distillation import DistillationLossConfig
    from verifiable_opd.verl_mixed_loss import mixed_opd_replay_loss
    width=max(max(lengths),1)
    prompts=torch.ones(len(lengths),2,dtype=torch.long)
    responses=torch.full((len(lengths),width),2,dtype=torch.long)
    mask=torch.arange(width)[None,:]<torch.tensor(lengths)[:,None]
    attention=torch.cat([prompts,mask.long()],dim=1)
    data=left_right_2_no_padding(TensorDict({
        'prompts':prompts,'responses':responses,'input_ids':torch.cat([prompts,responses],dim=1),
        'attention_mask':attention,'position_ids':attention.cumsum(-1)-1,
        'response_mask':mask,'teacher_logprobs':torch.full((len(lengths),width+2),-1.),
        'teacher_ids':torch.cat([prompts,responses],dim=1),
        'old_log_probs':torch.full((len(lengths),width),-2.),
        'expert_replay':torch.tensor(experts)},batch_size=[len(lengths)]))
    tu.assign_non_tensor(data,dp_size=dp_size,opd_online_action_tokens=online_total,
                         opd_expert_action_tokens=expert_total)
    values=model(torch.ones(sum(lengths)+2*len(lengths),1)).flatten()
    nested=torch.nested.nested_tensor_from_jagged(values,offsets=data['input_ids'].offsets())
    settings=DistillationLossConfig(loss_mode='k1',use_task_rewards=False,use_policy_gradient=True,
        expert_replay_strategy='uniform',expert_replay_alpha=.1,
        expert_replay_token_budget=8192,expert_replay_max_trajectories=2,
        loss_max_clamp=None,log_prob_min_clamp=None)
    loss,_=mixed_opd_replay_loss(OmegaConf.create({'loss_agg_mode':'token-mean','loss_scale_factor':None}),
        SimpleNamespace(distillation_loss=settings),model_output={'log_probs':nested},data=data)
    return loss


def worker(rank,init_file,output):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+init_file,world_size=2,rank=rank)
    try:
        model=torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():model.weight.fill_(-2.)
        wrapped=DistributedDataParallel(model)
        lengths=[1,3,7] if rank==0 else [2,4,0]
        experts=[False]*3 if rank==0 else [True]*3
        counts=torch.tensor([sum(n for n,e in zip(lengths,experts) if not e),
                             sum(n for n,e in zip(lengths,experts) if e)])
        dist.all_reduce(counts)
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        total=torch.tensor(0.)
        for n,e in zip(lengths,experts):
            loss=loss_for(wrapped,[n],[e],2,*counts.tolist())
            loss.backward()
            total+=loss.detach()
        dist.all_reduce(total,op=dist.ReduceOp.AVG)
        actual_gradient=model.weight.grad.detach().clone()
        optimizer.step()
        reference=torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():reference.weight.fill_(-2.)
        expected=loss_for(reference,[1,3,7,2,4,0],[False]*3+[True]*3,1,*counts.tolist())
        expected.backward()
        torch.testing.assert_close(actual_gradient,reference.weight.grad)
        torch.testing.assert_close(total,expected.detach())
        torch.testing.assert_close(model.weight.detach(),reference.weight.detach()-.01*reference.weight.grad)
        if rank==0:
            Path(output).write_text(json.dumps({'phase':'CPU_GLOO_DDP_LOSS_PASS_NOT_FSDP',
                'global_online_tokens':counts[0].item(),'global_expert_tokens':counts[1].item(),
                'gradient':actual_gradient.tolist(),'loss':total.item(),'updated_weight':model.weight.tolist(),
                'cases':['unequal micro token counts','rank containing only CE','zero-response microbatch',
                         'all ranks execute three backward collectives'],
                'limits':['Tiny synthetic linear model only','Real transformer/FSDP engine scaling still requires GPU smoke']},indent=2)+'\n')
    finally:dist.destroy_process_group()


def main():
    import torch.multiprocessing as mp
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Refuse to overwrite prior evidence')
    with TemporaryDirectory(prefix='opd-gloo-') as temporary:
        mp.spawn(worker,args=(str(Path(temporary)/'init'),str(args.output.resolve())),nprocs=2,join=True)
    print(args.output.read_text())


if __name__=='__main__':main()

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'upstream/verl-v0.8.0'),str(ROOT)]


class MixedTextLossTests(unittest.TestCase):
    def evaluate(self, lengths, experts, *, dp_size=1, alpha=.1, poison=False, width=5, baseline=False):
        import torch
        from tensordict import TensorDict
        from omegaconf import OmegaConf
        from types import SimpleNamespace
        from verl.utils import tensordict_utils as tu
        from verl.workers.utils.padding import left_right_2_no_padding
        from verl.workers.config.distillation import DistillationLossConfig
        from verifiable_opd.verl_mixed_loss import mixed_opd_replay_loss
        prompts=torch.ones((len(lengths),2),dtype=torch.long)
        responses=torch.full((len(lengths),width),2,dtype=torch.long)
        mask=torch.arange(width)[None,:]<torch.tensor(lengths)[:,None]
        attention=torch.cat([prompts,mask.long()],dim=1)
        teacher=torch.full((len(lengths),width+2),-1.)
        old=torch.full((len(lengths),width),-2.)
        if poison:
            teacher[torch.tensor(experts)]=float('nan')
            old[torch.tensor(experts)]=float('nan')
        data=left_right_2_no_padding(TensorDict({
            'prompts':prompts,'responses':responses,'input_ids':torch.cat([prompts,responses],dim=1),
            'attention_mask':attention,'position_ids':attention.cumsum(-1)-1,
            'response_mask':mask,'teacher_logprobs':teacher,'teacher_ids':torch.cat([prompts,responses],dim=1),
            'old_log_probs':old,'expert_replay':torch.tensor(experts)},batch_size=[len(lengths)]))
        tu.assign_non_tensor(data,dp_size=dp_size,opd_online_action_tokens=11,opd_expert_action_tokens=6,
                             batch_num_tokens=11,global_batch_size=4)
        values=torch.full((sum(lengths)+2*len(lengths),),-2.,requires_grad=True)
        nested=torch.nested.nested_tensor_from_jagged(values,offsets=data['input_ids'].offsets())
        loss_cfg=DistillationLossConfig(loss_mode='k1',use_task_rewards=False,use_policy_gradient=True,
            expert_replay_strategy='uniform' if alpha else 'none',expert_replay_alpha=alpha,
            expert_replay_token_budget=20,expert_replay_max_trajectories=2,
            loss_max_clamp=None,log_prob_min_clamp=None)
        actor=OmegaConf.create({'loss_agg_mode':'token-mean','loss_scale_factor':None,'global_batch_info':{}})
        if baseline:
            from verl.trainer.distillation.losses import distillation_ppo_loss
            # This zero task-reward term is irrelevant to PG-OPD. Do NOT let its
            # implementation accidentally initialize the distillation denominator.
            with patch('verl.trainer.distillation.losses.ppo_loss', return_value=(0.,{})):
                loss,_=distillation_ppo_loss(actor,SimpleNamespace(distillation_loss=loss_cfg),
                                            model_output={'log_probs':nested},data=data)
        else:
            loss,_=mixed_opd_replay_loss(actor,SimpleNamespace(distillation_loss=loss_cfg),
                                       model_output={'log_probs':nested},data=data)
        loss.backward()
        return loss.detach(),values.grad

    def test_unequal_micro_and_rank_split(self):
        import torch
        lengths=[1,3,2,5,2,4]
        experts=[False]*4+[True]*2
        loss,grad=self.evaluate(lengths,experts)
        left,leftgrad=self.evaluate(lengths[:3],experts[:3],dp_size=2)
        right,rightgrad=self.evaluate(lengths[3:],experts[3:],dp_size=2)
        torch.testing.assert_close(loss,(left+right)/2)
        torch.testing.assert_close(grad,torch.cat([leftgrad,rightgrad])/2)
        micros=[self.evaluate([length],[expert]) for length,expert in zip(lengths,experts)]
        torch.testing.assert_close(loss,sum(x[0] for x in micros))
        torch.testing.assert_close(grad,torch.cat([x[1] for x in micros]))

    def test_irrelevant_teacher_nan_is_masked_before_arithmetic(self):
        import torch
        clean=self.evaluate([1,3,2,5,2,4],[False]*4+[True]*2)
        poison=self.evaluate([1,3,2,5,2,4],[False]*4+[True]*2,poison=True)
        for a,b in zip(clean,poison):torch.testing.assert_close(a,b)

    def test_empty_local_micro_has_finite_zero_gradient(self):
        import torch
        loss,gradient=self.evaluate([0],[False],poison=True)
        self.assertEqual(loss.item(),0)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertEqual(gradient.abs().sum().item(),0)

    def test_alpha_zero_removes_expert_gradient(self):
        loss,gradient=self.evaluate([2,4],[True,True],alpha=0)
        self.assertEqual(loss.item(),0)
        self.assertEqual(gradient.abs().sum().item(),0)

    def test_one_token_response_keeps_sequence_axis(self):
        import torch
        loss,gradient=self.evaluate([1],[False],width=1)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(gradient).all())

    def test_pure_opd_equals_zero_replay_at_first_microbatch(self):
        import torch
        for lengths, dp_size, width in (([1],2,1),([3],2,5),([1,3,2,5],1,5)):
            mixed = self.evaluate(lengths,[False]*len(lengths),alpha=0,dp_size=dp_size,width=width)
            baseline = self.evaluate(lengths,[False]*len(lengths),alpha=0,dp_size=dp_size,width=width,baseline=True)
            for actual, expected in zip(baseline, mixed):
                torch.testing.assert_close(actual, expected)


if __name__=='__main__':unittest.main()

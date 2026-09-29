import unittest
import torch
from verifiable_opd.posttraining.update_audit import before_step,after_step


class UpdateAuditTests(unittest.TestCase):
    def test_actual_adam_step_and_skipped_step(self):
        model=torch.nn.Linear(2,1,bias=False)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        model(torch.ones(1,2)).sum().backward()
        snapshot=before_step(model,optimizer)
        self.assertEqual(after_step(snapshot,optimizer,1.)['phase'],'UPDATE_NOT_VERIFIED')
        optimizer.step()
        record=after_step(snapshot,optimizer,1.)
        self.assertEqual(record['phase'],'UPDATE_OBSERVED')
        self.assertEqual(record['optimizer_step_after'],1)

    def test_zero_gradient_not_claimed_update(self):
        model=torch.nn.Linear(2,1,bias=False)
        optimizer=torch.optim.AdamW(model.parameters())
        (model(torch.ones(1,2)).sum()*0).backward()
        self.assertIsNone(before_step(model,optimizer))


if __name__=='__main__':unittest.main()

import copy
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from verifiable_opd.losses import backward_update, decision_logprobs, opd_surrogate
from verifiable_opd.tokenization import Decision


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(16, 8)
        self.head = nn.Linear(8, 16)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.head(self.emb(input_ids).cumsum(dim=1)))


class LossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.student = TinyCausalLM()
        self.teacher = TinyCausalLM().requires_grad_(False)
        self.online = [Decision([1, 2], [3, 4]), Decision([1, 5, 2], [6])]
        self.expert = [Decision([1, 2], [7])]

    def test_first_answer_uses_previous_position(self):
        d = Decision([1, 2], [3, 4])
        lp = decision_logprobs(self.student, d)
        logits = self.student(torch.tensor([[1, 2, 3, 4]])).logits
        expected = torch.log_softmax(logits[0, 1:3], -1)[torch.arange(2), torch.tensor([3, 4])]
        torch.testing.assert_close(lp, expected)

    def test_coefficient_has_no_gradient(self):
        current = torch.tensor([-2., -3.], requires_grad=True)
        old = current.detach().clone().requires_grad_(True)
        teacher = torch.tensor([-1., -4.], requires_grad=True)
        opd_surrogate(current, old, teacher).backward()
        torch.testing.assert_close(current.grad, torch.tensor([-1., 1.]))
        self.assertIsNone(old.grad)
        self.assertIsNone(teacher.grad)

    def test_update_and_frozen_teacher(self):
        before = [p.detach().clone() for p in self.student.parameters()]
        tbefore = [p.detach().clone() for p in self.teacher.parameters()]
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=.01)
        metrics = backward_update(self.student, self.teacher, optimizer, self.online, self.expert, .2)
        self.assertGreater(metrics["grad_norm"], 0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, self.student.parameters())))
        for a, b in zip(tbefore, self.teacher.parameters()):
            torch.testing.assert_close(a, b)
            self.assertIsNone(b.grad)

    def test_alpha_zero_matches_pure_opd(self):
        other = copy.deepcopy(self.student)
        for model, experts in [(self.student, []), (other, self.expert)]:
            opt = torch.optim.SGD(model.parameters(), lr=.01)
            backward_update(model, self.teacher, opt, self.online, experts, 0)
        for a, b in zip(self.student.parameters(), other.parameters()):
            torch.testing.assert_close(a, b)

    def test_token_global_mean_matches_monolithic_gradient(self):
        reference = copy.deepcopy(self.student)
        online_lp = [decision_logprobs(reference, d) for d in self.online]
        teacher_lp = [decision_logprobs(self.teacher, d) for d in self.online]
        main = sum(opd_surrogate(lp, lp.detach(), tp) for lp, tp in zip(online_lp, teacher_lp)) / 3
        ce = -decision_logprobs(reference, self.expert[0]).mean()
        (main + .2 * ce).backward()
        expected = [p.grad.clone() for p in reference.parameters()]
        # lr=0 keeps weights fixed; high clipping threshold preserves the gradient.
        opt = torch.optim.SGD(self.student.parameters(), lr=0)
        backward_update(self.student, self.teacher, opt, self.online, self.expert, .2, max_grad_norm=1e9)
        for p, grad in zip(self.student.parameters(), expected):
            torch.testing.assert_close(p.grad, grad)


if __name__ == "__main__":
    unittest.main()

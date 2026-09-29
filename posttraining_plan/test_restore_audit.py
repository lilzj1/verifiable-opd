import unittest
import numpy as np
import torch
from verifiable_opd.posttraining.restore_audit import comparison, state_digest


class RestoreDigestTests(unittest.TestCase):
    def test_full_state_not_just_parameter_shape(self):
        tensor = torch.arange(1000, dtype=torch.bfloat16)
        before = {'model': tensor, 'optimizer': {'step': torch.tensor(1.)}}
        after = {'optimizer': {'step': torch.tensor(1.)}, 'model': tensor.clone()}
        self.assertTrue(comparison(before, after)['equal'])
        after['model'][-1] = -1
        self.assertFalse(comparison(before, after)['equal'])

    def test_shapes_types_and_scalar_tensors(self):
        self.assertNotEqual(state_digest(torch.ones(2, 3)), state_digest(torch.ones(3, 2)))
        self.assertNotEqual(state_digest(torch.tensor(1)), state_digest(torch.tensor(1.)))
        self.assertNotEqual(state_digest([1, 2]), state_digest((1, 2)))
        self.assertEqual(state_digest(torch.empty(0)), state_digest(torch.empty(0)))

    def test_rng_states_are_not_consumed(self):
        import random
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        digest = state_digest(before)
        after = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        self.assertEqual(digest, state_digest(after))


if __name__ == '__main__':
    unittest.main()

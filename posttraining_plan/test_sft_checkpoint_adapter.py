import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from posttraining_plan.run_sft_smoke import CommittedSFTCheckpointHandler


class SFTCheckpointTests(unittest.TestCase):
    def test_layout_commit_and_driver_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)/'diagnostic'
            root = run/'checkpoints'
            state = {'step': torch.tensor([1])}
            loader = SimpleNamespace(state_dict=lambda: state.copy())

            def load_state(value):
                state.clear()
                state.update(value)
            loader.load_state_dict = load_state

            class FakeEngine:
                loaded = None

                def save_checkpoint(self, local_path, global_step, max_ckpt_to_keep):
                    self.assert_no_prune = max_ckpt_to_keep is None
                    path = Path(local_path)
                    path.mkdir(parents=True)
                    for rank in (0, 1):
                        for kind in ('model', 'optim', 'extra_state'):
                            torch.save({'synthetic': torch.tensor([rank])}, path/f'{kind}_world_size_2_rank_{rank}.pt')

                def load_checkpoint(self, path):
                    self.loaded = path

            engine = FakeEngine()
            trainer = SimpleNamespace(config=SimpleNamespace(trainer=SimpleNamespace(default_local_dir=str(root))),
                                      training_client=engine, train_dataloader=loader)
            handler = CommittedSFTCheckpointHandler(trainer, {'synthetic': True})
            with patch('posttraining_plan.run_sft_smoke.require_save_space'):
                handler.save_checkpoint(1)
            self.assertEqual((root/'latest_checkpointed_iteration.txt').read_text(), '1\n')
            self.assertTrue((root/'global_step_1/TEXT_COMMIT.json').exists())
            self.assertTrue(engine.assert_no_prune)
            state['step'] = torch.tensor([0])
            restore = CommittedSFTCheckpointHandler(trainer, {'synthetic': True}, root/'global_step_1', run/'restore_audit')
            self.assertEqual(restore.load_checkpoint(), 1)
            self.assertEqual(state['step'].item(), 1)
            self.assertEqual(engine.loaded, str(root/'global_step_1/actor'))
            proof = json.loads((run/'restore_audit/trainer.json').read_text())
            self.assertTrue(proof['dataloader']['equal'])


if __name__ == '__main__':
    unittest.main()

from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from verifiable_opd.posttraining.checkpoint_protocol import commit,verify_before_restore,effective_optimizer_config


class CheckpointTests(unittest.TestCase):
    def test_schedule_sentinel_normalizes_without_relaxing_real_mismatches(self):
        before={'total_training_steps':-1, 'lr':1e-6}
        after={'total_training_steps':1, 'lr':1e-6}
        self.assertEqual(effective_optimizer_config(before,1),after)
        self.assertEqual(effective_optimizer_config(after,1),after)
        self.assertEqual(before['total_training_steps'],-1)
        with self.assertRaises(ValueError): effective_optimizer_config(after,50)
        self.assertNotEqual(effective_optimizer_config({'lr':2e-6},1),after)

    def build(self,directory):
        import torch
        path=Path(directory)/'global_step_1'
        (path/'actor').mkdir(parents=True)
        torch.save({'cursor':1},path/'data.pt')
        for kind in ('model','optim','extra_state'):
            for rank in range(2):
                torch.save({'state':torch.ones(2)},path/'actor'/f'{kind}_world_size_2_rank_{rank}.pt')
        return path

    def test_complete_and_contract_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path=self.build(directory)
            commit(path,2,{'pool':'a'})
            self.assertEqual(verify_before_restore(path,2,{'pool':'a'})['step'],1)
            with self.assertRaises(ValueError):verify_before_restore(path,2,{'pool':'b'})

    def test_all_kinds_missing_same_rank_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=self.build(directory)
            for kind in ('model','optim','extra_state'):
                (path/'actor'/f'{kind}_world_size_2_rank_1.pt').unlink()
            with self.assertRaises(ValueError):commit(path,2,{})

    def test_corrupt_and_zero_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path=self.build(directory)
            file=path/'actor/model_world_size_2_rank_0.pt'
            file.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):commit(path,2,{})
            file.write_bytes(b'')
            with self.assertRaises(ValueError):commit(path,2,{})

    def test_change_after_commit_and_wrong_world(self):
        with tempfile.TemporaryDirectory() as directory:
            path=self.build(directory)
            commit(path,2,{})
            with self.assertRaises(ValueError):verify_before_restore(path,3,{})
            with (path/'data.pt').open('ab') as stream:stream.write(b'changed')
            with self.assertRaises(ValueError):verify_before_restore(path,2,{})

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=self.build(directory)
            link=Path(directory)/'global_step_2'
            link.symlink_to(path,target_is_directory=True)
            with self.assertRaises(ValueError):commit(link,2,{})


if __name__=='__main__':unittest.main()

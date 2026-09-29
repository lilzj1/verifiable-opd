import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import test_checkpoint_protocol
from verifiable_opd.posttraining.checkpoint_protocol import commit,file_sha
from verifiable_opd.posttraining.rotation import make_plan,apply_plan,rotate_after_commit,enable_automatic_rotation,require_save_space


class RotationTests(unittest.TestCase):
    def fixture(self,tmp):
        root=Path(tmp)/'synthetic_run/checkpoints'
        root.mkdir(parents=True)
        contract={'synthetic':True}
        for step in (1,2,3):
            path=test_checkpoint_protocol.CheckpointTests().build(root)
            moved=root/f'global_step_{step}'
            if moved!=path:path.rename(moved)
            commit(moved,2,contract)
            if step==1: # free the helper's fixed global_step_1 name temporarily
                moved.rename(root/'first')
        (root/'first').rename(root/'global_step_1')
        proof=root.parent/'restore.json'
        proof.write_text(json.dumps({'phase':'DISTRIBUTED_RESTORE_VERIFIED','world_size':2,'contract':contract}))
        marker={'schema':1,'project':'opd_math_if','run_id':'synthetic_run','root':str(root),
                'world_size':2,'contract':contract,'restore_report':str(proof),
                'restore_report_sha256':file_sha(proof),'protected_steps':[]}
        (root/'TEXT_RUN.json').write_text(json.dumps(marker))
        (root/'latest_checkpointed_iteration.txt').write_text('3\n')
        return root,marker

    def test_keep_two_then_apply_only_exact_targets(self):
        with TemporaryDirectory() as tmp:
            root,_=self.fixture(tmp)
            plan=make_plan(root)
            self.assertEqual([x['step'] for x in plan['delete']],[1])
            removed=apply_plan(root,plan['approval_sha256'])
            self.assertEqual(len(removed),1)
            self.assertTrue((root/'global_step_2').exists())
            self.assertTrue((root/'global_step_3').exists())

    def test_protection_change_invalidates_approval(self):
        with TemporaryDirectory() as tmp:
            root,marker=self.fixture(tmp)
            plan=make_plan(root)
            marker['protected_steps']=[1]
            (root/'TEXT_RUN.json').write_text(json.dumps(marker))
            with self.assertRaises(ValueError):apply_plan(root,plan['approval_sha256'])
            self.assertEqual(make_plan(root)['delete'],[])

    def test_stale_pointer_and_incomplete_new_checkpoint_block_cleanup(self):
        with TemporaryDirectory() as tmp:
            root,_=self.fixture(tmp)
            (root/'latest_checkpointed_iteration.txt').write_text('2')
            with self.assertRaises(ValueError):make_plan(root)
            (root/'latest_checkpointed_iteration.txt').write_text('3')
            (root/'global_step_4').mkdir()
            with self.assertRaises((ValueError,FileNotFoundError)):make_plan(root)

    def test_automatic_pruning_requires_opt_in_and_protects_latest_two(self):
        with TemporaryDirectory() as tmp:
            root, marker = self.fixture(tmp)
            self.assertEqual(rotate_after_commit(root), [])
            self.assertTrue((root/'global_step_1').exists())
            marker['automatic_keep'] = 2
            (root/'TEXT_RUN.json').write_text(json.dumps(marker))
            self.assertEqual([r['step'] for r in rotate_after_commit(root)], [1])
            self.assertTrue((root/'global_step_2').exists())
            self.assertTrue((root/'global_step_3').exists())

    def test_smoke_proof_cannot_enable_different_formal_schedule(self):
        with TemporaryDirectory() as tmp:
            root, _ = self.fixture(tmp)
            with self.assertRaises(ValueError):
                enable_automatic_rotation(root, root.parent/'restore.json', {'planned_updates': 50})

    def test_low_disk_space_refuses_save_without_deleting_old_state(self):
        with TemporaryDirectory() as tmp:
            root, _ = self.fixture(tmp)
            with patch('verifiable_opd.posttraining.rotation.shutil.disk_usage', return_value=SimpleNamespace(free=49*1024**3)):
                with self.assertRaises(RuntimeError):
                    require_save_space(root)
            self.assertTrue((root/'global_step_1').exists())


if __name__=='__main__':unittest.main()

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import test_rotation
from verifiable_opd.posttraining import rotation_v2 as v2


class RotationV2Tests(unittest.TestCase):
    def fixture(self, tmp, keep=2):
        root, marker = test_rotation.RotationTests().fixture(tmp)
        marker['automatic_keep'] = keep
        (root / 'TEXT_RUN.json').write_text(json.dumps(marker))
        return root

    def test_keep_one_retains_only_latest_and_reuses_cache(self):
        with TemporaryDirectory() as tmp:
            root = self.fixture(tmp, keep=1)
            self.assertEqual([x['step'] for x in v2.rotate_after_commit(root)], [1, 2])
            self.assertEqual([p.name for p in root.glob('global_step_*')], ['global_step_3'])
            with patch.object(v2, 'verify_before_restore', wraps=v2.verify_before_restore) as verify:
                self.assertEqual(v2.rotate_after_commit(root), [])
                self.assertEqual(verify.call_count, 0)

    def test_keep_one_failed_new_checkpoint_preserves_previous(self):
        with TemporaryDirectory() as tmp:
            root = self.fixture(tmp, keep=1)
            (root / 'global_step_3/data.pt').write_bytes(b'broken')
            with self.assertRaises(ValueError):
                v2.rotate_after_commit(root)
            self.assertTrue((root / 'global_step_2').exists())

    def test_keep_one_respects_explicit_protected_checkpoint(self):
        with TemporaryDirectory() as tmp:
            root = self.fixture(tmp, keep=1)
            path = root / 'TEXT_RUN.json'
            marker = json.loads(path.read_text()); marker['protected_steps'] = [1]
            path.write_text(json.dumps(marker))
            self.assertEqual([x['step'] for x in v2.rotate_after_commit(root)], [2])
            self.assertTrue((root / 'global_step_1').exists())
            self.assertTrue((root / 'global_step_3').exists())

    def test_invalid_retention_never_deletes(self):
        for keep in (0, -1, True, 1.0, '1'):
            with self.subTest(keep=keep), TemporaryDirectory() as tmp:
                root = self.fixture(tmp, keep=keep)
                with self.assertRaises(ValueError):
                    v2.rotate_after_commit(root)
                self.assertTrue((root / 'global_step_1').exists())

    def test_full_validation_once_then_cache(self):
        with TemporaryDirectory() as tmp:
            root = self.fixture(tmp)
            with patch.object(v2, 'verify_before_restore', wraps=v2.verify_before_restore) as verify:
                self.assertEqual([x['step'] for x in v2.rotate_after_commit(root)], [1])
                self.assertEqual(verify.call_count, 3)
                verify.reset_mock()
                self.assertEqual(v2.rotate_after_commit(root), [])
                self.assertEqual(verify.call_count, 0)

    def test_mutation_forces_full_validation_and_blocks_deletion(self):
        with TemporaryDirectory() as tmp:
            root = self.fixture(tmp)
            v2.rotate_after_commit(root)
            shard = root / 'global_step_3/data.pt'
            data = bytearray(shard.read_bytes()); data[-1] ^= 1
            shard.write_bytes(data)
            with self.assertRaises(ValueError):
                v2.rotate_after_commit(root)
            self.assertTrue((root / 'global_step_2').exists())

    def test_missing_cache_falls_back_to_full_validation(self):
        with TemporaryDirectory() as tmp:
            root = self.fixture(tmp); v2.rotate_after_commit(root)
            (root / 'ROTATION_VALIDATION_V2.json').write_text('broken')
            with patch.object(v2, 'verify_before_restore', wraps=v2.verify_before_restore) as verify:
                v2.rotate_after_commit(root)
                self.assertEqual(verify.call_count, 2)

    def test_missing_shard_or_symlink_blocks_cleanup(self):
        for mode in ('missing', 'symlink'):
            with self.subTest(mode=mode), TemporaryDirectory() as tmp:
                root = self.fixture(tmp)
                shard = root / 'global_step_3/data.pt'
                shard.unlink()
                if mode == 'symlink':
                    shard.symlink_to(root / 'global_step_2/data.pt')
                with self.assertRaises((ValueError, FileNotFoundError)):
                    v2.rotate_after_commit(root)
                self.assertTrue((root / 'global_step_1').exists())

    def test_stale_pointer_blocks_cleanup(self):
        with TemporaryDirectory() as tmp:
            root = self.fixture(tmp)
            (root / 'latest_checkpointed_iteration.txt').write_text('2')
            with self.assertRaises(ValueError):
                v2.rotate_after_commit(root)
            self.assertTrue((root / 'global_step_1').exists())


if __name__ == '__main__':
    unittest.main()

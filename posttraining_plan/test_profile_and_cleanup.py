import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from posttraining_plan.cleanup_text_smoke import cleanup
from posttraining_plan.profile_text import metrics_from_log
from posttraining_plan.run_text_smoke import profile_evidence
from verifiable_opd.posttraining.checkpoint_protocol import file_sha


class ProfileTests(unittest.TestCase):
    def test_real_log_metrics_and_no_saved_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'log'
            line = ('step:{step} - timing_s/gen:2.5 - timing_s/update_actor:1.5 - '
                    'timing_s/step:4.0 - perf/total_num_tokens:100 - '
                    'response_length/mean:20 - actor/grad_norm:np.float64(0.5)\n')
            text = ''.join(line.format(step=i) for i in range(1, 4))
            path.write_text(text)
            self.assertEqual(metrics_from_log(path)['mean_step_seconds'], 4)
            path.write_text(text.replace('timing_s/gen:2.5', 'timing_s/save_checkpoint:5 - timing_s/gen:2.5'))
            with self.assertRaises(ValueError):
                metrics_from_log(path)
            path.write_text(text + line.format(step=3))
            with self.assertRaises(ValueError):
                metrics_from_log(path)

    def test_both_restore_reports_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                profile_evidence(Path(tmp))


class CleanupTests(unittest.TestCase):
    def fixture(self, root):
        run = root / 'artifacts/text_s2_source16k_smoke_v1'
        target = run / 'checkpoints/global_step_1'
        target.mkdir(parents=True)
        (target / 'weights.pt').write_bytes(b'test only')
        (target / 'TEXT_COMMIT.json').write_text(json.dumps({'contract': {'schema': 2}}))
        (run / 'checkpoints/latest_checkpointed_iteration.txt').write_text('1')
        (run / 'smoke_result.json').write_text(json.dumps({
            'phase': 'ONE_UPDATE_AND_SAVE_VERIFIED_RESTORE_PENDING', 'group': 'S2',
            'checkpoint_commit_sha256': file_sha(target / 'TEXT_COMMIT.json')}))
        restore = run.with_name(run.name + '_restore')
        restore.mkdir()
        (restore / 'restore_result.json').write_text(json.dumps({
            'phase': 'DISTRIBUTED_RESTORE_VERIFIED', 'checkpoint': str(target),
            'contract': {'schema': 2}, 'report_sha256': {'test': 'fake'}}))
        controller = root / 'artifacts/text_source_smoke_controller_v2'
        controller.mkdir()
        (controller / 'S2_restore.exit_code.txt').write_text('0')
        return run, target, controller

    def test_dry_run_and_scope_and_retained_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, target, controller = self.fixture(root)
            with patch('posttraining_plan.cleanup_text_smoke.verify_restore_reports', return_value={'report_sha256': {'test': 'fake'}}):
                self.assertEqual(cleanup(root, 'S2')['phase'], 'PLANNED')
                self.assertTrue(target.exists())
                with self.assertRaises(ValueError):
                    cleanup(root, 'formal')
                (controller / 'S2_restore.exit_code.txt').write_text('1')
                with self.assertRaises(ValueError):
                    cleanup(root, 'S2', True)
                (controller / 'S2_restore.exit_code.txt').write_text('0')
                self.assertEqual(cleanup(root, 'S2', True)['phase'], 'DELETED')
            self.assertFalse(target.exists())
            self.assertTrue((run / 'smoke_result.json').exists())
            self.assertTrue((run / 'checkpoint_manifest_archived.json').exists())
            self.assertFalse((run / 'checkpoints/latest_checkpointed_iteration.txt').exists())


if __name__ == '__main__':
    unittest.main()

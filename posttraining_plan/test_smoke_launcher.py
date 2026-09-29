import json
from pathlib import Path
import tempfile
import unittest
from posttraining_plan.run_text_smoke import audit_updates, check_devices, require_quality


class SmokeLauncherTests(unittest.TestCase):
    def test_missing_failed_or_formal_gate_rejected(self):
        for report in ({}, {'phase': 'RUNNING_OR_INCOMPLETE', 'allow_smoke': False},
                       {'phase': 'QUALITY_SUPPORTS_BOUNDED_SMOKE', 'allow_smoke': True, 'allow_formal': True}):
            with self.assertRaises(ValueError):
                require_quality(report)
        require_quality({'phase': 'QUALITY_SUPPORTS_BOUNDED_SMOKE', 'allow_smoke': True, 'allow_formal': False})

    def test_gpu_scope_does_not_touch_other_users(self):
        gpus = '\n'.join(f'{i}, UUID-{i}, 5' for i in range(8))
        self.assertEqual(len(check_devices(gpus, 'UUID-0, 100\nUUID-6, 200')), 4)
        with self.assertRaises(ValueError):
            check_devices(gpus, 'UUID-3, 300')
        with self.assertRaises(ValueError):
            check_devices(gpus.replace('2, UUID-2, 5', '2, UUID-2, 1000'), '')
        with self.assertRaises(ValueError):
            check_devices('1, UUID-1, 5', '')

    def test_actual_updates_on_both_ranks_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {'phase': 'UPDATE_OBSERVED', 'optimizer_step_before': 0, 'optimizer_step_after': 1}
            for rank in range(2):
                (root / f'rank_{rank}.jsonl').write_text(json.dumps(row) + '\n')
            self.assertEqual(len(audit_updates(root)), 2)
            (root / 'rank_1.jsonl').write_text(json.dumps(row) + '\n' + json.dumps(row) + '\n')
            with self.assertRaises(ValueError):
                audit_updates(root)

    def test_profile_requires_three_contiguous_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [dict(phase='UPDATE_OBSERVED', optimizer_step_before=i, optimizer_step_after=i+1) for i in range(3)]
            for rank in range(2):
                (root / f'rank_{rank}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            self.assertEqual(len(audit_updates(root, 3)['rank_0.jsonl']), 3)
            rows[1]['optimizer_step_before'] = 0
            (root / 'rank_1.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaises(ValueError):
                audit_updates(root, 3)


if __name__ == '__main__':
    unittest.main()

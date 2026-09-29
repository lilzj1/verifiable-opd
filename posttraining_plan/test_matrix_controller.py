import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from posttraining_plan import watch_matrix_and_smoke as controller


class MatrixControllerTests(unittest.TestCase):
    def test_explicit_source_policy_preserves_failed_report_but_allows_engineering_smokes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'artifacts').mkdir()
            with patch.object(controller, 'ROOT', root), patch('sys.argv', ['watch', '--output', str(root/'run'),
                    '--execute-smokes', '--teacher-policy', 'published_source', '--response', '16384']), \
                    patch.object(controller, 'analyze', return_value={'phase':'QUALITY_NOT_ESTABLISHED','allow_smoke':False}), \
                    patch.object(controller, 'run_child') as child, patch.object(controller, 'wait_idle'):
                controller.main()
            self.assertEqual(child.call_count, 4)
            for call in child.call_args_list:
                self.assertIn('published_source', call.args[0])
            self.assertFalse(json.loads((root/'run/matrix_result.json').read_text())['allow_smoke'])

    def test_failed_teacher_gate_never_starts_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'artifacts').mkdir()
            output = root / 'run'
            with patch.object(controller, 'ROOT', root), patch('sys.argv', ['watch', '--output', str(output), '--execute-smokes']), \
                    patch.object(controller, 'analyze', return_value={'phase': 'QUALITY_NOT_ESTABLISHED', 'allow_smoke': False}), \
                    patch.object(controller, 'run_child') as child, patch.object(controller, 'wait_idle') as idle:
                controller.main()
            child.assert_not_called()
            idle.assert_not_called()
            self.assertEqual(json.loads((output / 'status.json').read_text())['phase'], 'STOPPED_AT_TEACHER_GATE')

    def test_passing_gate_without_execute_flag_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'artifacts').mkdir()
            with patch.object(controller, 'ROOT', root), patch('sys.argv', ['watch', '--output', str(root/'run')]), \
                    patch.object(controller, 'analyze', return_value={'phase': 'QUALITY_SUPPORTS_BOUNDED_SMOKE', 'allow_smoke': True}), \
                    patch.object(controller, 'run_child') as child:
                controller.main()
            child.assert_not_called()

    def test_child_failure_stops_before_next_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'artifacts').mkdir()
            with patch.object(controller, 'ROOT', root), patch('sys.argv', ['watch', '--output', str(root/'run'), '--execute-smokes']), \
                    patch.object(controller, 'analyze', return_value={'phase': 'QUALITY_SUPPORTS_BOUNDED_SMOKE', 'allow_smoke': True}), \
                    patch.object(controller, 'wait_idle'), \
                    patch.object(controller, 'run_child', side_effect=RuntimeError('failed update')) as child:
                with self.assertRaises(RuntimeError):
                    controller.main()
            self.assertEqual(child.call_count, 1)
            self.assertEqual(json.loads((root/'run/status.json').read_text())['phase'], 'STOPPED_NEEDS_DIAGNOSIS')


if __name__ == '__main__':
    unittest.main()

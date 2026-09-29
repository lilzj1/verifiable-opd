import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from posttraining_plan import watch_sft_smoke as controller


class SFTControllerTests(unittest.TestCase):
    def test_profile_failure_never_launches_sft(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root/'artifacts/text_profile_controller_v2'
            profile.mkdir(parents=True)
            (profile/'status.json').write_text(json.dumps({'phase': 'STOPPED_NEEDS_DIAGNOSIS'}))
            output = root/'artifacts/text_s1_smoke_controller_v1'
            with patch.object(controller, 'ROOT', root), patch('sys.argv', ['watch', '--output', str(output)]), \
                    patch.object(controller, 'run_child') as child:
                with self.assertRaises(RuntimeError):
                    controller.main()
                child.assert_not_called()

    def test_success_uses_two_gpus_then_restores_before_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root/'artifacts/text_profile_controller_v2'
            profile.mkdir(parents=True)
            (profile/'status.json').write_text(json.dumps({'phase': 'PROFILES_COMPLETE_S1_AND_FORMAL_BUDGET_PENDING'}))
            output = root/'artifacts/text_s1_smoke_controller_v1'
            with patch.object(controller, 'ROOT', root), patch('sys.argv', ['watch', '--output', str(output)]), \
                    patch.object(controller, 'run_child') as child, patch.object(controller, 'wait_idle'), \
                    patch.object(controller, 'cleanup') as cleanup:
                controller.main()
                self.assertEqual(child.call_count, 2)
                self.assertTrue(all(call.kwargs['gpu_ids'] == '1,2' for call in child.call_args_list))
                self.assertIn('--restore-from-run', child.call_args_list[1].args[0])
                cleanup.assert_called_once_with(root, 'S1', execute=True)


if __name__ == '__main__':
    unittest.main()

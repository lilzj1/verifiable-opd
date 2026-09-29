import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from posttraining_plan.finish_text_project_v2 import amended_command
from posttraining_plan.watch_rotation_handoff import process, same_process
from posttraining_plan.run_text_formal_v2 import install_runtime_hook, ROOT


class HandoffTests(unittest.TestCase):
    def test_only_s3_launch_changes(self):
        for group in ('S2', 'S3'):
            command = ['python', '-u', 'posttraining_plan/run_text_formal.py',
                       '--group', group, '--stage', 'finish', '--execute']
            actual = amended_command(command)
            expected = list(command)
            if group == 'S3':
                expected[2] = 'posttraining_plan/run_text_formal_v2.py'
            self.assertEqual(actual, expected)
        for script in ('eval_text_parallel.py', 'run_sft_formal.py', 'export_text_final.py'):
            command = ['python', 'posttraining_plan/'+script, '--group', 'S2']
            self.assertEqual(amended_command(command), command)

    def test_proc_exit_status_is_real_wait_status(self):
        child = subprocess.Popen([sys.executable, '-c', 'raise SystemExit(7)'])
        try:
            deadline = time.monotonic()+10
            while time.monotonic() < deadline:
                state = process(child.pid)
                if state['state'] == 'Z':
                    break
                time.sleep(0.02)
            self.assertEqual(state['state'], 'Z')
            self.assertEqual(os.waitstatus_to_exitcode(state['exit_status']), 7)
            with self.assertRaises(ValueError):
                same_process(child.pid, 'wrong-start')
        finally:
            child.wait(timeout=10)

    def test_stopped_parent_does_not_stop_evaluation_child(self):
        code = ('import subprocess,sys; '
                'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(0.5)"]); '
                'print(p.pid,flush=True); p.wait()')
        parent = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, text=True)
        try:
            pid = int(parent.stdout.readline())
            os.kill(parent.pid, signal.SIGSTOP)
            deadline = time.monotonic()+10
            while time.monotonic() < deadline:
                child = process(pid)
                if child and child['state'] == 'Z':
                    break
                time.sleep(0.02)
            self.assertEqual(process(parent.pid)['state'], 'T')
            self.assertEqual(child['state'], 'Z')
            self.assertEqual(child['exit_status'], 0)
        finally:
            os.kill(parent.pid, signal.SIGCONT)
            parent.wait(timeout=10)
            parent.stdout.close()

    def test_hook_changes_only_runtime_not_learning_settings(self):
        from omegaconf import OmegaConf
        from tempfile import TemporaryDirectory
        config = OmegaConf.create({'trainer': {'default_local_dir': str(ROOT/'artifacts/text_s3_formal_s100_seed17_v1/checkpoints'),
                                              'total_training_steps': 100},
                                   'actor': {'lr': 1e-6}, 'ray_kwargs': {'ray_init': {'runtime_env': {'env_vars': {'X': 'Y'}}}}})
        before = OmegaConf.to_container(config, resolve=True)
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {'OPD_TEXT_UPDATE_AUDIT_DIR': tmp+'/optimizer_audit'}), \
                patch('verifiable_opd.posttraining.checkpoint_protocol.file_sha', return_value='test-only'):
            install_runtime_hook(config)
            self.assertTrue((Path(tmp)/'io_amendment.json').exists())
        after = OmegaConf.to_container(config, resolve=True)
        hook = after['ray_kwargs']['ray_init']['runtime_env'].pop('worker_process_setup_hook')
        self.assertEqual(hook, 'posttraining_plan.rotation_handoff.worker_setup')
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()

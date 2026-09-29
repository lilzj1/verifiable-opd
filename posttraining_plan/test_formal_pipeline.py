import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from posttraining_plan.freeze_text_formal import budget_sft, engineering_evidence, checker_hashes
from posttraining_plan.finish_text_project import execute
from posttraining_plan.run_text_formal import read_protocol
from posttraining_plan import cleanup_finished_text as cleanup_module
from verifiable_opd.posttraining.checkpoint_protocol import file_sha


class FormalTests(unittest.TestCase):
    def test_metrics_fail_fast_and_keep_diagnostic(self):
        from verifiable_opd.posttraining.step_metrics import write_metrics
        with tempfile.TemporaryDirectory() as tmp:
            write_metrics(tmp,1,{'actor/loss':.5,'actor/grad_norm':2})
            self.assertEqual(json.loads((Path(tmp)/'step_metrics/1.json').read_text())['actor/loss'],.5)
            with self.assertRaises(FloatingPointError):
                write_metrics(tmp,2,{'actor/loss':float('nan'),'actor/grad_norm':2})
            self.assertTrue((Path(tmp)/'step_metrics/2.json').exists())

    def test_token_budget_exact_deterministic_complete_batches(self):
        lengths=list(range(1,17))
        result=budget_sft(lengths,1000)
        self.assertEqual(result,budget_sft(lengths,1000))
        self.assertGreaterEqual(result['scheduled_target_tokens'],1000)
        self.assertLess(result['scheduled_target_tokens']-1000,4*max(lengths))
        self.assertEqual(result['complete_expert_rows'],result['updates']*4)
        with self.assertRaises(ValueError):budget_sft([1,2,3],10)

    def test_unfinished_engineering_does_not_freeze(self):
        with tempfile.TemporaryDirectory() as tmp, patch('posttraining_plan.freeze_text_formal.ROOT',Path(tmp)):
            with self.assertRaises(FileNotFoundError):engineering_evidence()

    def test_checker_sources_exist(self):
        checks=checker_hashes()
        self.assertTrue(any('google_ifeval' in k for k in checks))
        self.assertTrue(any('ifbench-reference' in k for k in checks))
        self.assertTrue(any('verifiable_instructions' in k for k in checks))

    def test_no_duplicate_child_and_failed_child_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'ok.log'
            command=[sys.executable,'-c','print("passed")']
            execute(command,log,'',10)
            before=log.stat().st_mtime_ns
            execute(command,log,'',10)
            self.assertEqual(before,log.stat().st_mtime_ns)
            with self.assertRaises(ValueError):execute(command+['changed'],log,'',10)
            failed=Path(tmp)/'bad.log'
            with self.assertRaises(RuntimeError):execute([sys.executable,'-c','raise RuntimeError()'],failed,'',10)
            with self.assertRaises(ValueError):execute([sys.executable,'-c','raise RuntimeError()'],failed,'',10)

    def test_protocol_rejects_unfrozen_and_changed_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            path=root/'protocol.json'
            path.write_text(json.dumps({'phase':'DRAFT'}))
            with self.assertRaises(ValueError):read_protocol(path,'S2')
            source=root/'example.py'
            source.write_text('original')
            assets=root/'artifacts/posttraining_assets_20260926_resume_v4/status.json'
            assets.parent.mkdir(parents=True)
            assets.write_text('{}')
            protocol=dict(phase='FROZEN_TEXT_FORMAL',updates=100,save_freq=10,keep_full_checkpoints=2,
                          asset_manifest_sha256=file_sha(assets),
                          source_sha256={'example.py':file_sha(source)})
            path.write_text(json.dumps(protocol))
            source.write_text('changed')
            with patch('posttraining_plan.run_text_formal.ROOT',root), self.assertRaisesRegex(ValueError,'implementation changed'):
                read_protocol(path,'S2')

    def test_finished_cleanup_keeps_final_and_other_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            run=root/'artifacts/text_s2_formal_s100_seed17_v1'
            checkpoints=run/'checkpoints'
            for step in (90,100):
                path=checkpoints/f'global_step_{step}'
                path.mkdir(parents=True)
                (path/'TEXT_COMMIT.json').write_text('{}')
            (checkpoints/'latest_checkpointed_iteration.txt').write_text('100')
            sha=file_sha(checkpoints/'global_step_100/TEXT_COMMIT.json')
            stage=run/'stages/finish'
            stage.mkdir(parents=True)
            (stage/'result.json').write_text(json.dumps(dict(phase='HUNDRED_UPDATES_COMPLETE_EVALUATION_PENDING',checkpoint_commit_sha256=sha)))
            hf=run/'final_hf'
            hf.mkdir()
            (hf/'weight').write_text('synthetic')
            (run/'export_result.json').write_text(json.dumps(dict(phase='FINAL_HF_EXPORTED',target=str(hf),
                checkpoint=str(checkpoints/'global_step_100'),checkpoint_commit_sha256=sha,files={'weight':file_sha(hf/'weight')})))
            other=root/'other_project'
            other.mkdir()
            with patch.object(cleanup_module,'ROOT',root), patch.object(cleanup_module,'validate_root',return_value=(checkpoints,{'contract':{}})), \
                    patch.object(cleanup_module,'verify_before_restore',return_value={'files':{'weight':{'bytes':5}}}):
                plan=cleanup_module.cleanup('S2')
                self.assertEqual(plan['targets'],[str(checkpoints/'global_step_90')])
                self.assertTrue((checkpoints/'global_step_90').exists())
                cleanup_module.cleanup('S2',True)
                self.assertFalse((checkpoints/'global_step_90').exists())
                self.assertTrue((checkpoints/'global_step_100').exists())
                self.assertTrue(other.exists())
                self.assertTrue((run/'retired_checkpoint_manifests/global_step_90.json').exists())


if __name__=='__main__':unittest.main()

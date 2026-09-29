"""Frozen token-budget Continue-SFT control; final full save and independent restore."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/'upstream/verl-v0.8.0'), str(ROOT)]
from posttraining_plan.run_text_formal import read_protocol
from posttraining_plan.run_text_smoke import PYTHON, check_devices, audit_updates
from posttraining_plan.run_sft_smoke import CommittedSFTCheckpointHandler
from posttraining_plan.sft_resources import init_sft_ray
from verifiable_opd.posttraining.checkpoint_protocol import atomic_json, file_sha, verify_before_restore
from verifiable_opd.posttraining.restore_audit import verify_restore_reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--restore-only', action='store_true')
    args = parser.parse_args()
    protocol, config = read_protocol(args.protocol, 'S1')
    if Path(sys.executable).resolve() != Path(PYTHON).resolve() or os.environ.get('CUDA_VISIBLE_DEVICES') != '1,2' or os.environ.get('RAY_ADDRESS'):
        raise ValueError('Require training interpreter, GPU 1/2 and private Ray')
    run = ROOT/'artifacts/text_s1_formal_tokens_seed17_v1'
    if Path(config.trainer.default_local_dir) != run/'checkpoints':
        raise ValueError('Unregistered S1 run path')
    count = config.trainer.total_training_steps
    source = run/f'checkpoints/global_step_{count}' if args.restore_only else None
    contract = protocol['s1_contract']
    with (ROOT/'artifacts/.text_smoke_launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_devices(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'], text=True),
                      subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'], text=True))
        if source:
            result = json.loads((run/'train/result.json').read_text())
            if result['phase'] != 'SFT_FORMAL_SAVED_EVALUATION_PENDING':
                raise ValueError('S1 training incomplete')
            verify_before_restore(source, 2, contract)
        else:
            run.mkdir(exist_ok=False)
        stage = run/('restore' if source else 'train')
        stage.mkdir(exist_ok=False)
        from omegaconf import OmegaConf
        (stage/'resolved.yaml').write_text(OmegaConf.to_yaml(config, resolve=True))
        env = dict(OPD_TEXT_UPDATE_AUDIT_DIR=str(stage/'optimizer_audit'),
                   PYTHONPATH=f'{ROOT}/upstream/verl-v0.8.0:{ROOT}', TOKENIZERS_PARALLELISM='false',
                   OMP_NUM_THREADS='1', NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1')
        if source:
            env['OPD_TEXT_RESTORE_AUDIT_DIR'] = str(stage/'restore_audit')
        os.environ.update(env)
        import ray
        started = time.monotonic()
        try:
            from verl.trainer.sft_trainer_ray import SFTTrainer

            class TextSFTTrainer(SFTTrainer):
                def _build_dataloader(self):
                    super()._build_dataloader()
                    self.train_sampler.seed = self.config.trainer.seed

                def _build_ckpt_handler(self):
                    self.ckpt_handler = CommittedSFTCheckpointHandler(self, contract, source, stage/'restore_audit')

            init_sft_ray(config, env)
            trainer = TextSFTTrainer(config)
            if source:
                if list((stage/'optimizer_audit').glob('*.jsonl')):
                    raise ValueError('Restore unexpectedly performed updates')
                proof = verify_restore_reports(stage/'restore_audit', source, contract)
                atomic_json(stage/'restore_result.json', dict(proof, wall_seconds=time.monotonic()-started))
                return
            trainer.fit()
            records = audit_updates(stage/'optimizer_audit', count)
            checkpoint = run/f'checkpoints/global_step_{count}'
            verify_before_restore(checkpoint, 2, contract)
            tokens = sum(json.loads(p.read_text())['train/target_tokens'] for p in (stage/'optimizer_audit/step_metrics').glob('*.json'))
            if tokens != protocol['s1_budget']['scheduled_target_tokens']:
                raise ValueError('Actual S1 target-token consumption differs from frozen sampler schedule')
            atomic_json(stage/'result.json', dict(phase='SFT_FORMAL_SAVED_EVALUATION_PENDING',
                updates=count, actual_target_tokens=tokens, wall_seconds=time.monotonic()-started,
                checkpoint_commit_sha256=file_sha(checkpoint/'TEXT_COMMIT.json'), optimizer_audit=records))
        except BaseException as exc:
            atomic_json(stage/'result.json', dict(phase='SFT_FORMAL_FAILED', error=f'{type(exc).__name__}: {exc}'))
            raise
        finally:
            ray.shutdown()


if __name__ == '__main__':
    main()

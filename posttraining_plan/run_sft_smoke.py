"""One real Continue-SFT update, or independent restore-only, on GPU 1/2.

Reuses the upstream verl SFT trainer/engine/loss and the same complete expert
encoding as S3. This is NOT a token-matched formal S1 run. No teacher services.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'upstream/verl-v0.8.0'), str(ROOT)]
from posttraining_plan.compile_sft_config import build
from posttraining_plan.sft_resources import init_sft_ray
from posttraining_plan.run_text_smoke import PYTHON, audit_updates, check_devices
from verifiable_opd.posttraining.checkpoint_protocol import atomic_json, file_sha, commit, verify_before_restore
from verifiable_opd.posttraining.rotation import require_save_space, rotate_after_commit
from verifiable_opd.posttraining.restore_audit import comparison, verify_restore_reports


def sft_contract(config):
    from omegaconf import OmegaConf
    from verifiable_opd.posttraining.checkpoint_protocol import effective_optimizer_config
    return {'schema': 1, 'mode': 'continue_sft_text', 'planned_updates': config.trainer.total_training_steps,
            'expert_sha256': file_sha(config.data.train_files),
            'expert_path': str(Path(config.data.train_files).resolve()),
            'initial_model_config_sha256': file_sha(Path(config.model.path) / 'config.json'),
            'seed': config.trainer.seed, 'batch': config.data.train_batch_size,
            'max_length': config.data.max_length,
            'encoding_sha256': file_sha(ROOT / 'verifiable_opd/posttraining/text_encode.py'),
            'dataset_sha256': file_sha(ROOT / 'verifiable_opd/posttraining/sft_dataset.py'),
            'optimizer': effective_optimizer_config(OmegaConf.to_container(config.optim, resolve=True),
                                                    config.trainer.total_training_steps)}


class CommittedSFTCheckpointHandler:
    """SFT state adapter to the already audited actor/data.pt checkpoint layout."""
    def __init__(self, trainer, contract, restore=None, audit_dir=None):
        self.trainer, self.contract, self.restore, self.audit_dir = trainer, contract, restore, audit_dir
        self.root = Path(trainer.config.trainer.default_local_dir)

    def save_checkpoint(self, step):
        import torch
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / '.rotation.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            require_save_space(self.root)
            checkpoint = self.root / f'global_step_{step}'
            self.trainer.training_client.save_checkpoint(local_path=str(checkpoint/'actor'),
                                                        global_step=step, max_ckpt_to_keep=None)
            torch.save(self.trainer.train_dataloader.state_dict(), checkpoint/'data.pt')
            commit(checkpoint, 2, self.contract)
            pointer = self.root / 'latest_checkpointed_iteration.txt'
            temporary = pointer.with_suffix('.pending')
            with temporary.open('w') as stream:
                stream.write(str(step)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, pointer)
        rotate_after_commit(self.root)

    def load_checkpoint(self):
        if self.restore is None:
            return 0
        import torch
        verify_before_restore(self.restore, 2, self.contract)
        self.trainer.training_client.load_checkpoint(str(self.restore/'actor'))
        saved = torch.load(self.restore/'data.pt', weights_only=False, map_location='cpu')
        self.trainer.train_dataloader.load_state_dict(saved)
        check = comparison(saved, self.trainer.train_dataloader.state_dict())
        if not check['equal']:
            raise ValueError('SFT data state did not restore exactly')
        step = int(self.restore.name.removeprefix('global_step_'))
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(self.audit_dir/'trainer.json', {
            'phase': 'TRAINER_RESTORED_NO_UPDATE', 'checkpoint': str(self.restore),
            'global_step': step, 'dataloader': check})
        return step


def main():
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128+signum))
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--restore-from-run')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    config = build(args.run_id)
    contract = sft_contract(config)
    source = None
    if args.restore_from_run:
        if not args.restore_from_run.replace('_', '').isalnum():
            raise ValueError('Safe source ID required')
        original = ROOT/'artifacts'/args.restore_from_run
        result = json.loads((original/'smoke_result.json').read_text())
        if result.get('phase') != 'ONE_UPDATE_AND_SAVE_VERIFIED_RESTORE_PENDING' or result.get('group') != 'S1':
            raise ValueError('Successful S1 update required')
        source = original/'checkpoints/global_step_1'
        if result['checkpoint_commit_sha256'] != file_sha(source/'TEXT_COMMIT.json'):
            raise ValueError('Source checkpoint changed')
        verify_before_restore(source, 2, contract)
    if not args.execute:
        print(json.dumps({'phase': 'COMPOSED_NOT_EXECUTED', 'contract': contract}, indent=2))
        return
    if Path(sys.executable).resolve() != Path(PYTHON).resolve():
        raise ValueError('Use the Hydra-compatible training interpreter')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1,2' or os.environ.get('RAY_ADDRESS'):
        raise ValueError('Require GPU 1,2 and a private Ray cluster')
    if not source and (os.environ.get('OPD_TEXT_RESTORE_ONLY') or os.environ.get('OPD_TEXT_RESTORE_AUDIT_DIR')):
        raise ValueError('Unexpected restore flags in a fresh SFT update')
    run = ROOT/'artifacts'/args.run_id
    with (ROOT/'artifacts/.text_smoke_launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_devices(subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                      '--format=csv,noheader,nounits'], text=True),
                      subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                      '--format=csv,noheader,nounits'], text=True))
        run.mkdir(exist_ok=False)
        from omegaconf import OmegaConf
        (run/'resolved.yaml').write_text(OmegaConf.to_yaml(config, resolve=True))
        env = dict(OPD_TEXT_UPDATE_AUDIT_DIR=str(run/'optimizer_audit'),
                   PYTHONPATH=f'{ROOT}/upstream/verl-v0.8.0:{ROOT}', TOKENIZERS_PARALLELISM='false',
                   OMP_NUM_THREADS='1', NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1')
        if source:
            env.update(OPD_TEXT_RESTORE_AUDIT_DIR=str(run/'restore_audit'))
        os.environ.update(env)
        atomic_json(run/'preflight.json', {'phase': 'S1_DIAGNOSTIC_ONLY', 'contract': contract,
                    'physical_gpus': [1, 2], 'allow_formal': False})
        started = time.monotonic()
        import ray
        try:
            from verl.trainer.sft_trainer_ray import SFTTrainer

            class TextSFTTrainer(SFTTrainer):
                def _build_dataloader(self):
                    super()._build_dataloader()
                    self.train_sampler.seed = self.config.trainer.seed

                def _build_ckpt_handler(self):
                    self.ckpt_handler = CommittedSFTCheckpointHandler(self, contract, source, run/'restore_audit')

            init_sft_ray(config, env)
            trainer = TextSFTTrainer(config)
            if source:
                if list((run/'optimizer_audit').glob('*.jsonl')):
                    raise ValueError('Restore-only SFT unexpectedly updated weights')
                proof = verify_restore_reports(run/'restore_audit', source, contract)
                atomic_json(run/'restore_result.json', dict(proof, wall_seconds=time.monotonic()-started, allow_formal=False))
                return
            trainer.fit()
            records = audit_updates(run/'optimizer_audit')
            checkpoint = run/'checkpoints/global_step_1'
            manifest = verify_before_restore(checkpoint, 2, contract)
            atomic_json(run/'smoke_result.json', {
                'phase': 'ONE_UPDATE_AND_SAVE_VERIFIED_RESTORE_PENDING', 'group': 'S1',
                'optimizer_audit': records, 'checkpoint_commit_sha256': file_sha(checkpoint/'TEXT_COMMIT.json'),
                'checkpoint_bytes': sum(entry['bytes'] for entry in manifest['files'].values()),
                'wall_seconds': time.monotonic()-started, 'allow_formal': False})
        except BaseException as exc:
            atomic_json(run/'smoke_result.json', {'phase': 'SMOKE_FAILED', 'error': f'{type(exc).__name__}: {exc}',
                                                'wall_seconds': time.monotonic()-started, 'allow_formal': False})
            raise
        finally:
            ray.shutdown()


if __name__ == '__main__':
    main()

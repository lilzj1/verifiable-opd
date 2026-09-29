"""Explicit 100-update text OPD run: first10 -> restore10 -> finish, with rotation.

Requires a separately frozen protocol and completed real profiles. Never selects
hyperparameters from test results or deletes shared/other-run checkpoints.
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
sys.path[:0] = [str(ROOT/'upstream/verl-v0.8.0'), str(ROOT)]
from posttraining_plan.run_text_smoke import PYTHON, audit_updates, check_devices, profile_evidence
from verifiable_opd.posttraining.checkpoint_protocol import atomic_json, file_sha, replay_contract, verify_before_restore
from verifiable_opd.posttraining.rotation import enable_automatic_rotation, validate_root
from verifiable_opd.posttraining.restore_audit import verify_restore_reports


def read_protocol(path, group):
    protocol = json.loads(path.read_text())
    if protocol.get('phase') != 'FROZEN_TEXT_FORMAL' or protocol.get('updates') != 100:
        raise ValueError('Explicit frozen 100-update protocol required')
    if protocol.get('save_freq') != 10 or protocol.get('keep_full_checkpoints') != 2:
        raise ValueError('Unexpected save/retention policy')
    if group not in ('S1', 'S2', 'S3'):
        raise ValueError('Only frozen comparison arms')
    if file_sha(ROOT/'artifacts/posttraining_assets_20260926_resume_v4/status.json') != protocol.get('asset_manifest_sha256'):
        raise ValueError('Frozen model asset manifest changed')
    for name, digest in protocol['source_sha256'].items():
        source = (ROOT/name).resolve(strict=True)
        if not source.is_relative_to(ROOT) or file_sha(source) != digest:
            raise ValueError(f'Frozen implementation changed: {name}')
    for name, digest in protocol['evidence_sha256'].items():
        source = (ROOT/name).resolve(strict=True)
        if not source.is_relative_to(ROOT) or file_sha(source) != digest:
            raise ValueError(f'Frozen engineering evidence changed: {name}')
    from omegaconf import OmegaConf
    config_path = path.parent/f'{group}.yaml'
    if file_sha(config_path) != protocol['config_sha256'][group]:
        raise ValueError('Resolved configuration changed')
    config = OmegaConf.load(config_path)
    if group == 'S1':
        from posttraining_plan.run_sft_smoke import sft_contract
        actual, expected = sft_contract(config), protocol['s1_contract']
        if config.trainer.total_training_steps != protocol['s1_budget']['updates'] or config.trainer.save_freq != config.trainer.total_training_steps:
            raise ValueError('S1 token budget/save policy changed')
    else:
        actual, expected = replay_contract(config), protocol['contracts'][group]
    if actual != expected:
        raise ValueError('Current data/models do not match the frozen contract')
    if group != 'S1' and (config.trainer.total_training_steps != 100 or config.trainer.save_freq != 10):
        raise ValueError('Horizon/save cadence changed')
    return protocol, config


def main():
    signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(128+sig))
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--group', choices=('S2', 'S3'), required=True)
    parser.add_argument('--stage', choices=('first10', 'restore10', 'finish'), required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    protocol, config = read_protocol(args.protocol, args.group)
    profile_evidence(ROOT/'artifacts')
    if not args.execute:
        print(json.dumps({'phase': 'VALIDATED_NOT_EXECUTED', 'stage': args.stage, 'updates': 100}))
        return
    if Path(sys.executable).resolve() != Path(PYTHON).resolve():
        raise ValueError('Use the Hydra-compatible interpreter')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1,2,3,4' or os.environ.get('RAY_ADDRESS'):
        raise ValueError('Four authorized GPUs and private local Ray required')
    for name in ('OPD_TEXT_STOP_AT_STEP','OPD_TEXT_RESTORE_ONLY','OPD_TEXT_RESTORE_AUDIT_DIR'):
        if os.environ.get(name):
            raise ValueError(f'Unexpected inherited stage flag: {name}')
    root = Path(config.trainer.default_local_dir)
    run = root.parent
    expected_id = f'text_{args.group.lower()}_formal_s100_seed17_v1'
    if root != ROOT/'artifacts'/expected_id/'checkpoints':
        raise ValueError('Unregistered formal run path')
    with (ROOT/'artifacts/.text_smoke_launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_devices(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used',
            '--format=csv,noheader,nounits'],text=True), subprocess.check_output(['nvidia-smi',
            '--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True))
        if args.stage == 'first10':
            run.mkdir(exist_ok=False)
        else:
            first = json.loads((run/'stages/first10/result.json').read_text())
            if first.get('phase') != 'TEN_UPDATES_AND_SAVE_VERIFIED':
                raise ValueError('Ten-step stage has not passed')
        stage = run/'stages'/args.stage
        stage.mkdir(parents=True, exist_ok=False)
        checkpoint = root/'global_step_10'
        contract = protocol['contracts'][args.group]
        if args.stage != 'first10':
            verify_before_restore(checkpoint, 2, contract)
            config.trainer.resume_mode = 'resume_path'
            config.trainer.resume_from_path = str(checkpoint)
            config.trainer.del_local_ckpt_after_load = False
        if args.stage == 'finish':
            proof_path = run/'stages/restore10/restore_result.json'
            enable_automatic_rotation(root, proof_path, contract, protected_steps=())
            validate_root(root)
        env = dict(PYTHONPATH=f'{ROOT}/upstream/verl-v0.8.0:{ROOT}',
                   OPD_TEXT_UPDATE_AUDIT_DIR=str(stage/'optimizer_audit'),
                   TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1')
        if args.stage == 'first10':
            env['OPD_TEXT_STOP_AT_STEP'] = '10'
        elif args.stage == 'restore10':
            env.update(OPD_TEXT_RESTORE_ONLY='1', OPD_TEXT_RESTORE_AUDIT_DIR=str(stage/'restore_audit'))
        os.environ.update(env)
        from omegaconf import OmegaConf
        OmegaConf.update(config, 'ray_kwargs.ray_init', {'address': 'local', 'num_cpus': 16, 'num_gpus': 4,
            'include_dashboard': False, 'runtime_env': {'env_vars': env}}, merge=False, force_add=True)
        (stage/'resolved.yaml').write_text(OmegaConf.to_yaml(config, resolve=True))
        atomic_json(stage/'preflight.json', {'phase': 'FORMAL_STAGE_RUNNING', 'group': args.group,
                    'stage': args.stage, 'planned_updates':100, 'protocol_sha256': file_sha(args.protocol)})
        import ray
        started = time.monotonic()
        try:
            from verl.trainer.main_ppo import run_ppo
            run_ppo(config)
            if args.stage == 'restore10':
                if list((stage/'optimizer_audit').glob('*.jsonl')):
                    raise ValueError('Restore stage must not update')
                proof = verify_restore_reports(stage/'restore_audit', checkpoint, contract)
                atomic_json(stage/'restore_result.json', dict(proof, wall_seconds=time.monotonic()-started))
                return
            count, start = (10,0) if args.stage == 'first10' else (90,10)
            records = audit_updates(stage/'optimizer_audit', count, start)
            final = root/f'global_step_{start+count}'
            manifest = verify_before_restore(final, 2, contract)
            atomic_json(stage/'result.json', {
                'phase': 'TEN_UPDATES_AND_SAVE_VERIFIED' if args.stage == 'first10' else 'HUNDRED_UPDATES_COMPLETE_EVALUATION_PENDING',
                'optimizer_audit':records, 'wall_seconds':time.monotonic()-started,
                'checkpoint_commit_sha256':file_sha(final/'TEXT_COMMIT.json'),
                'checkpoint_bytes':sum(row['bytes'] for row in manifest['files'].values()),
                'limitation':'Online sampler RNG may differ after stage restart; both arms use the same restart protocol'})
        except BaseException as exc:
            atomic_json(stage/'result.json', {'phase':'FORMAL_STAGE_FAILED', 'error':f'{type(exc).__name__}: {exc}',
                                            'wall_seconds':time.monotonic()-started})
            raise
        finally:
            ray.shutdown()


if __name__ == '__main__':
    main()

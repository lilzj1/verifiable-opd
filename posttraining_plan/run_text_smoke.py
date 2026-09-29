"""Guarded Math/IF smoke or three-update disposable profile; no formal training/deletion."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'upstream/verl-v0.8.0'), str(ROOT)]
from posttraining_plan.analyze_teacher_matrix import analyze
from posttraining_plan.compile_text_config import build
from verifiable_opd.posttraining.checkpoint_protocol import atomic_json, file_sha, verify_before_restore, replay_contract

PYTHON = '/home/ubuntu/anaconda3/envs/creditlab-verl-alfworld/bin/python'
PREFIX_REPORTS = (
    'long_student_prefix_student_bf16_v1/report.json',
    'long_student_prefix_math_teacher_fp16_v1/report_sdpa.json',
    'long_student_prefix_instruction_following_teacher_fp16_v1/report_sdpa.json',
)


def require_quality(report):
    if report.get('phase') != 'QUALITY_SUPPORTS_BOUNDED_SMOKE' or report.get('allow_smoke') is not True:
        raise ValueError('Teacher matrix has not passed; no training is allowed by this launcher')
    if report.get('allow_formal') is not False:
        raise ValueError('This launcher must not be used as formal-training approval')


def check_devices(gpu_csv, processes_csv):
    devices = {}
    for line in gpu_csv.strip().splitlines():
        index, uuid, memory = [part.strip() for part in line.split(',')]
        devices[int(index)] = (uuid, int(memory))
    if not set((1, 2, 3, 4)) <= set(devices):
        raise ValueError('Four authorized GPUs are not visible')
    authorized = {devices[index][0] for index in (1, 2, 3, 4)}
    for line in processes_csv.strip().splitlines():
        uuid, pid = [part.strip() for part in line.split(',')]
        if uuid in authorized:
            raise ValueError(f'Authorized GPU remains occupied by PID {pid}; will not kill it')
    if any(devices[index][1] > 512 for index in (1, 2, 3, 4)):
        raise ValueError('Unexpected memory allocation on an authorized GPU')
    return {str(index): devices[index] for index in (1, 2, 3, 4)}


def audit_updates(directory, expected_updates=1, start_step=0):
    if not isinstance(expected_updates, int) or not 1 <= expected_updates <= 10000 or start_step < 0:
        raise ValueError('Positive bounded update count and nonnegative starting step required')
    files = sorted(directory.glob('rank_*.jsonl'))
    if {file.name for file in files} != {'rank_0.jsonl', 'rank_1.jsonl'}:
        raise ValueError('Missing or unexpected optimizer audit ranks')
    records = {}
    for file in files:
        rows = [json.loads(line) for line in file.read_text().split('\n') if line.strip()]
        if len(rows) != expected_updates or any(row.get('phase') != 'UPDATE_OBSERVED' for row in rows):
            raise ValueError('Expected verified optimizer updates on every rank')
        for index, row in enumerate(rows, start=start_step):
            if row['optimizer_step_before'] != index or row['optimizer_step_after'] != index + 1:
                raise ValueError('Updates must start at the common initialization and be contiguous')
        records[file.name] = rows[0] if expected_updates == 1 else rows
    return records


def profile_evidence(artifacts):
    """Require real restore reports for BOTH objectives, not a supplied approval flag."""
    evidence = {}
    for group in ('s2', 's3'):
        path = artifacts / f'text_{group}_source16k_smoke_v1_restore' / 'restore_result.json'
        proof = json.loads(path.read_text())
        if proof.get('phase') != 'DISTRIBUTED_RESTORE_VERIFIED' or proof.get('world_size') != 2:
            raise ValueError(f'{group} distributed restore has not passed')
        for name, digest in proof['report_sha256'].items():
            if Path(name).name != name or file_sha(path.parent / 'restore_audit' / name) != digest:
                raise ValueError('Restore evidence changed')
        if set(proof['report_sha256']) != {'rank_0.json', 'rank_1.json', 'trainer.json'}:
            raise ValueError('Incomplete restore evidence')
        evidence[group] = file_sha(path)
    return evidence


def main():
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', required=True, choices=('S2', 'S3'))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--restore-from-run', help='Reload a successful one-step smoke in a NEW process, without updates')
    parser.add_argument('--execute', action='store_true', help='Without this flag only inspect readiness')
    parser.add_argument('--teacher-policy', choices=('local_matrix', 'published_source'), default='local_matrix',
                        help='published_source records the user-approved protocol change; does not relabel old failures')
    parser.add_argument('--response', type=int, choices=(8192, 16384), default=8192)
    parser.add_argument('--profile-updates', type=int, choices=(3,),
                        help='Disposable three-update run after both restores; saves NO weights')
    args = parser.parse_args()
    if args.profile_updates and (args.restore_from_run or args.response != 16384):
        raise ValueError('Profile uses the source16k protocol and cannot be a restore job')
    config = build(args.group, args.run_id, updates=args.profile_updates or 1, response_length=args.response)
    restore_evidence = profile_evidence(ROOT / 'artifacts') if args.profile_updates else {}
    if args.profile_updates:
        config.trainer.save_freq = -1
    restore_checkpoint = None
    if args.restore_from_run:
        if not args.restore_from_run.replace('_', '').isalnum():
            raise ValueError('Safe source run ID required')
        from omegaconf import OmegaConf
        original = ROOT / 'artifacts' / args.restore_from_run
        result_path = original / 'smoke_revalidated.json'
        if not result_path.exists():
            result_path = original / 'smoke_result.json'
        result = json.loads(result_path.read_text())
        if result.get('phase') != 'ONE_UPDATE_AND_SAVE_VERIFIED_RESTORE_PENDING' or result.get('group') != args.group:
            raise ValueError('Restore source must be a successful one-update run of the same group')
        config = OmegaConf.load(original / 'resolved.yaml')
        restore_checkpoint = original / 'checkpoints/global_step_1'
        if result.get('checkpoint_commit_sha256') != file_sha(restore_checkpoint / 'TEXT_COMMIT.json'):
            raise ValueError('Restore proof no longer matches the checkpoint commit')
        verify_before_restore(restore_checkpoint, 2, replay_contract(config))
        config.trainer.resume_mode = 'resume_path'
        config.trainer.resume_from_path = str(restore_checkpoint)
        config.trainer.del_local_ckpt_after_load = False
    report = analyze()  # Recompute from all original responses; never trust a supplied pass flag.
    if not args.execute:
        print(json.dumps({'phase': 'READ_ONLY_READINESS', 'teacher_phase': report['phase'],
                          'local_matrix_allow_smoke': report['allow_smoke'],
                          'admission_policy': args.teacher_policy,
                          'engineering_checks_still_required': True, 'allow_formal': False}, indent=2))
        return
    if args.teacher_policy == 'local_matrix':
        require_quality(report)
    else:
        if report.get('phase') == 'RUNNING_OR_INCOMPLETE':
            raise ValueError('Do not overlap existing matrix jobs')
        # Explicit user-requested protocol revision, NOT a successful teacher gate.
        report = dict(report, admission_policy='published_source',
                      admission_reason='User removed small-dev teacher superiority as training prerequisite; engineering checks retained',
                      source_audit_sha256=file_sha(ROOT/'posttraining_plan/UPSTREAM_TEACHER_AUDIT.md'))
    if Path(sys.executable).resolve() != Path(PYTHON).resolve():
        raise ValueError('Use the Hydra-compatible training interpreter, not .text_runtime')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1,2,3,4':
        raise ValueError('Explicit physical GPU 1,2,3,4 visibility is required')
    if os.environ.get('RAY_ADDRESS'):
        raise ValueError('Refuse an externally configured Ray cluster')
    run = ROOT / 'artifacts' / args.run_id
    # This lock only prevents OUR launchers racing; it is not a shared-server reservation.
    with (ROOT / 'artifacts/.text_smoke_launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prefix_evidence = {}
        for name in PREFIX_REPORTS:
            path = ROOT / 'artifacts' / name
            if json.loads(path.read_text()).get('phase') != 'PASS':
                raise ValueError(f'Numerical prefix gate failed: {name}')
            prefix_evidence[name] = file_sha(path)
        gpu_csv = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
            '--format=csv,noheader,nounits'], text=True)
        processes_csv = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
            '--format=csv,noheader,nounits'], text=True)
        devices = check_devices(gpu_csv, processes_csv)
        free = shutil.disk_usage(ROOT).free
        required_gib = 10 if args.profile_updates or restore_checkpoint else 60
        if free < required_gib * 1024**3:
            raise ValueError(f'Less than {required_gib} GiB free: stop before this job')
        run.mkdir(exist_ok=False)
        from omegaconf import OmegaConf
        env = {'OPD_TEXT_UPDATE_AUDIT_DIR': str(run / 'optimizer_audit'),
               'PYTHONPATH': f'{ROOT}/upstream/verl-v0.8.0:{ROOT}',
               'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '1',
               'NCCL_P2P_DISABLE': '1', 'NCCL_IB_DISABLE': '1'}
        if restore_checkpoint:
            env.update(OPD_TEXT_RESTORE_ONLY='1', OPD_TEXT_RESTORE_AUDIT_DIR=str(run / 'restore_audit'))
        elif os.environ.get('OPD_TEXT_RESTORE_ONLY') or os.environ.get('OPD_TEXT_RESTORE_AUDIT_DIR'):
            raise ValueError('Unexpected restore environment flags in a fresh update run')
        os.environ.update(env)
        OmegaConf.update(config, 'ray_kwargs.ray_init', {
            'address': 'local', 'num_cpus': 16, 'num_gpus': 4, 'include_dashboard': False,
            'runtime_env': {'env_vars': env}}, merge=False, force_add=True)
        yaml = OmegaConf.to_yaml(config, resolve=True)
        (run / 'resolved.yaml').write_text(yaml)
        tracked_sources = [
            'posttraining_plan/run_text_smoke.py', 'posttraining_plan/compile_text_config.py',
            'verifiable_opd/verl_mixed_loss.py', 'verifiable_opd/posttraining/text_replay.py',
            'verifiable_opd/posttraining/identity.py', 'verifiable_opd/posttraining/update_audit.py',
            'verifiable_opd/posttraining/restore_audit.py',
            'verifiable_opd/posttraining/domain_sampling.py',
            'verifiable_opd/posttraining/rotation.py',
            'upstream/verl-v0.8.0/verl/trainer/distillation/losses.py',
            'upstream/verl-v0.8.0/verl/workers/engine_workers.py',
            'upstream/verl-v0.8.0/verl/trainer/ppo/ray_trainer.py',
            'upstream/verl-v0.8.0/verl/workers/engine/fsdp/transformer_impl.py',
            'upstream/verl-v0.8.0/verl/utils/checkpoint/fsdp_checkpoint_manager.py',
        ]
        atomic_json(run / 'preflight.json', {'phase': 'DISPOSABLE_THREE_UPDATE_PROFILE' if args.profile_updates else 'BOUNDED_ONE_UPDATE_ONLY',
            'group': args.group, 'config_sha256': file_sha(run / 'resolved.yaml'),
            'source_sha256': {name: file_sha(ROOT/name) for name in tracked_sources},
            'matrix': report, 'prefix_reports': prefix_evidence, 'gpus': devices,
            'free_bytes_before': free,
            'restore_evidence': restore_evidence, 'planned_updates': args.profile_updates or 1,
            'save_checkpoints': not bool(args.profile_updates),
            'limits': [f'4096-token numerical prefixes passed; {config.data.max_response_length} training remains a smoke diagnostic',
                       'Diagnostic only; no formal-training authorization or method-quality conclusion']})
        started = time.monotonic()
        import ray
        try:
            from verl.trainer.main_ppo import run_ppo
            run_ppo(config)
            if restore_checkpoint:
                from verifiable_opd.posttraining.restore_audit import verify_restore_reports
                if list((run / 'optimizer_audit').glob('*.jsonl')):
                    raise ValueError('Restore-only job unexpectedly called optimizer.step')
                proof = verify_restore_reports(run / 'restore_audit', restore_checkpoint, replay_contract(config))
                proof['wall_seconds'] = time.monotonic() - started
                proof['allow_formal'] = False
                atomic_json(run / 'restore_result.json', proof)
                return
            records = audit_updates(run / 'optimizer_audit', args.profile_updates or 1)
            if args.profile_updates:
                if list((run / 'checkpoints').glob('global_step_*')):
                    raise ValueError('Disposable profile unexpectedly saved weights')
                atomic_json(run / 'profile_result.json', {
                    'phase': 'THREE_UPDATES_VERIFIED_NO_WEIGHTS_SAVED', 'group': args.group,
                    'updates': args.profile_updates, 'wall_seconds': time.monotonic() - started,
                    'optimizer_audit': records, 'restore_evidence': restore_evidence,
                    'allow_formal': False, 'requires_cost_report_and_budget_freeze': True})
                return
            checkpoint = run / 'checkpoints/global_step_1'
            manifest = verify_before_restore(checkpoint, 2, replay_contract(config))
            atomic_json(run / 'smoke_result.json', {
                'phase': 'ONE_UPDATE_AND_SAVE_VERIFIED_RESTORE_PENDING', 'group': args.group,
                'wall_seconds': time.monotonic() - started, 'optimizer_audit': records,
                'checkpoint_commit_sha256': file_sha(checkpoint / 'TEXT_COMMIT.json'),
                'checkpoint_bytes': sum(entry['bytes'] for entry in manifest['files'].values()),
                'allow_formal': False})
        except BaseException as exc:
            atomic_json(run / ('profile_result.json' if args.profile_updates else 'smoke_result.json'), {'phase': 'PROFILE_FAILED' if args.profile_updates else 'SMOKE_FAILED',
                'error': f'{type(exc).__name__}: {exc}', 'wall_seconds': time.monotonic() - started,
                'allow_formal': False})
            raise
        finally:
            ray.shutdown()  # This process created a private local cluster, never attaches to others.


if __name__ == '__main__':
    main()

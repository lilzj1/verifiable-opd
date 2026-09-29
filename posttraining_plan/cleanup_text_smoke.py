"""Remove ONLY a completed source16k diagnostic checkpoint, preserving its evidence.

Requires the independent restore child to have exited successfully. Cannot target
formal runs, shared models, other experiments, or an un-restored checkpoint.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from verifiable_opd.posttraining.checkpoint_protocol import atomic_json, checked_path, file_sha
from verifiable_opd.posttraining.restore_audit import verify_restore_reports


def cleanup(root, group, execute=False):
    if group not in ('S1', 'S2', 'S3'):
        raise ValueError('Only the registered diagnostic checkpoints can be removed')
    run_id = 'text_s1_sft_smoke_v1' if group == 'S1' else f'text_{group.lower()}_source16k_smoke_v1'
    run = root / 'artifacts' / run_id
    target = checked_path(run / 'checkpoints/global_step_1')
    restore = run.with_name(run.name + '_restore')
    exit_file = root / 'artifacts/text_source_smoke_controller_v2' / f'{group}_restore.exit_code.txt'
    if group == 'S3' and not exit_file.exists():
        exit_file = root / 'artifacts/text_source_smoke_controller_v3/S3_restore.exit_code.txt'
    if group == 'S1':
        exit_file = root / 'artifacts/text_s1_smoke_controller_v1/S1_restore.exit_code.txt'
    if exit_file.read_text().strip() != '0':
        raise ValueError('Independent restore process has not completed successfully')
    proof = json.loads((restore / 'restore_result.json').read_text())
    if proof.get('phase') != 'DISTRIBUTED_RESTORE_VERIFIED' or Path(proof['checkpoint']) != target:
        raise ValueError('Wrong or unsuccessful restore proof')
    verified = verify_restore_reports(restore / 'restore_audit', target, proof['contract'])
    if verified['report_sha256'] != proof['report_sha256']:
        raise ValueError('Restore evidence changed')
    result_path = run / 'smoke_revalidated.json'
    if not result_path.exists():
        result_path = run / 'smoke_result.json'
    result = json.loads(result_path.read_text())
    if result.get('phase') != 'ONE_UPDATE_AND_SAVE_VERIFIED_RESTORE_PENDING' or result.get('group') != group:
        raise ValueError('Not a completed registered diagnostic update')
    if result['checkpoint_commit_sha256'] != file_sha(target / 'TEXT_COMMIT.json'):
        raise ValueError('Checkpoint manifest changed since verification')
    manifest = json.loads((target / 'TEXT_COMMIT.json').read_text())
    if manifest['contract'] != proof['contract']:
        raise ValueError('Checkpoint contract differs from restore proof')
    members = list(target.rglob('*'))
    if any(path.is_symlink() for path in members):
        raise ValueError('Refuse symlink-bearing checkpoint cleanup')
    plan = {'phase': 'PLANNED', 'target': str(target), 'group': group,
            'bytes': sum(path.stat().st_size for path in members if path.is_file()),
            'restore_report_sha256': file_sha(restore / 'restore_result.json'),
            'checkpoint_commit_sha256': file_sha(target / 'TEXT_COMMIT.json'),
            'retained': ['logs', 'resolved config', 'optimizer audit', 'restore audit', 'manifest copy', 'rollout samples'],
            'recoverable': False, 'reason': 'User-authorized deletion of completed disposable smoke weights'}
    if execute:
        with (run / '.cleanup.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            atomic_json(run / 'checkpoint_manifest_archived.json', manifest)
            atomic_json(run / 'checkpoint_cleanup.json', dict(plan, phase='DELETING'))
            shutil.rmtree(target)
            pointer = run / 'checkpoints/latest_checkpointed_iteration.txt'
            if pointer.exists() and pointer.read_text().strip() == '1':
                pointer.unlink()
            plan['phase'] = 'DELETED'
            atomic_json(run / 'checkpoint_cleanup.json', plan)
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', required=True, choices=('S1', 'S2', 'S3'))
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    print(json.dumps(cleanup(ROOT, args.group, args.execute), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

"""User-authorized retirement of superseded checkpoints in finished text arms.

Keeps the final FULL recovery checkpoint, HF export and all reports. Unlike
in-training rotation, a completed run no longer needs its penultimate state.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from verifiable_opd.posttraining.checkpoint_protocol import atomic_json, file_sha, verify_before_restore
from verifiable_opd.posttraining.rotation import validate_root


def cleanup(group, execute=False):
    if group not in ('S2','S3'):
        raise ValueError('Only the registered completed distillation runs')
    run = ROOT/f'artifacts/text_{group.lower()}_formal_s100_seed17_v1'
    root, marker = validate_root(run/'checkpoints')
    result = json.loads((run/'stages/finish/result.json').read_text())
    export = json.loads((run/'export_result.json').read_text())
    final = root/'global_step_100'
    if result['phase'] != 'HUNDRED_UPDATES_COMPLETE_EVALUATION_PENDING' or export['phase'] != 'FINAL_HF_EXPORTED':
        raise ValueError('Run has not completed and exported')
    if export['checkpoint'] != str(final) or export['target'] != str(run/'final_hf'):
        raise ValueError('Export belongs to a different checkpoint')
    if int((root/'latest_checkpointed_iteration.txt').read_text()) != 100:
        raise ValueError('Final/latest mismatch')
    with (root/'.rotation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_before_restore(final, 2, marker['contract'])
        if file_sha(final/'TEXT_COMMIT.json') != result['checkpoint_commit_sha256'] or result['checkpoint_commit_sha256'] != export['checkpoint_commit_sha256']:
            raise ValueError('Final commit changed')
        for name, sha in export['files'].items():
            if Path(name).name != name or file_sha(run/'final_hf'/name) != sha:
                raise ValueError('Final HF export changed')
        targets = []
        for path in root.glob('global_step_*'):
            if path == final:
                continue
            if path.is_symlink() or not path.is_dir() or any(p.is_symlink() for p in path.rglob('*')):
                raise ValueError('Unsafe checkpoint path')
            step = int(path.name.removeprefix('global_step_'))
            if not 0 < step < 100 or step in marker.get('protected_steps', []):
                raise ValueError('Unexpected or protected old step')
            manifest = verify_before_restore(path, 2, marker['contract'])
            targets.append((path, manifest))
        record = dict(phase='PLANNED', final_full_checkpoint=str(final),
                      targets=[str(path) for path,_ in targets],
                      bytes=sum(sum(row['bytes'] for row in m['files'].values()) for _,m in targets))
        if not execute:
            return record
        atomic_json(run/'finished_cleanup.json', dict(record, phase='DELETING'))
        archive = run/'retired_checkpoint_manifests'
        archive.mkdir(exist_ok=True)
        for path, manifest in targets:
            atomic_json(archive/f'{path.name}.json', manifest)
            shutil.rmtree(path)
        record['phase'] = 'DELETED_SUPERSEDED_FINAL_RETAINED'
        atomic_json(run/'finished_cleanup.json', record)
        return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=['S2','S3'], required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    print(json.dumps(cleanup(args.group,args.execute), indent=2))

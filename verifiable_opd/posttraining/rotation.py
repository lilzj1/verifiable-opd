"""Explicit, locked two-phase rotation for NEW text runs only.

No import-time effects. Never searches other projects. A plan is invalidated by
new checkpoints, stale latest pointers or protection changes. This checks bytes;
TEXT_RUN.json must additionally refer to a separately verified real restore.
"""
import fcntl
import json
from pathlib import Path
import shutil

from .checkpoint_protocol import file_sha, verify_before_restore, checked_path, atomic_json
from .evaluation import canonical_sha


def validate_root(root):
    root=Path(root).absolute()
    if root.name!='checkpoints' or any(p.is_symlink() for p in (root,*root.parents)):
        raise ValueError('Exact nonsymlink checkpoints run root required')
    marker=json.loads((root/'TEXT_RUN.json').read_text())
    if marker.get('schema')!=1 or marker.get('run_id')!=root.parent.name or marker.get('world_size')!=2:
        raise ValueError('Unrecognized text run marker')
    if marker.get('root')!=str(root) or marker.get('project')!='opd_math_if':
        raise ValueError('Run root/project identity mismatch')
    # A genuine reload report must be written by smoke verification, not by the
    # checkpoint byte auditor. Do not accept the checkpoint manifest as proof.
    restore=Path(marker['restore_report']).resolve(strict=True)
    if file_sha(restore)!=marker['restore_report_sha256']:
        raise ValueError('Restore evidence changed')
    proof=json.loads(restore.read_text())
    if proof.get('phase')!='DISTRIBUTED_RESTORE_VERIFIED' or proof.get('world_size')!=2:
        raise ValueError('Real distributed restore has not passed')
    if proof.get('contract')!=marker['contract']:
        raise ValueError('Restore proof belongs to a different training contract')
    return root,marker


def make_plan(root,keep=2):
    if keep<2:raise ValueError('Keep latest AND previous complete recovery points')
    root,marker=validate_root(root)
    entries=[]
    for child in root.iterdir():
        if not child.name.startswith('global_step_'):continue
        path=checked_path(child)
        if not path.is_dir():raise ValueError('Step entry is not a directory')
        manifest=verify_before_restore(path,marker['world_size'],marker['contract'])
        if manifest['step']!=int(path.name.removeprefix('global_step_')):
            raise ValueError('Checkpoint step/manifest mismatch')
        if any(p.is_symlink() for p in path.rglob('*')):
            raise ValueError('Checkpoint contains symlinks; refuse cleanup')
        entries.append({'step':manifest['step'],'path':str(path),
                        'manifest_sha256':file_sha(path/'TEXT_COMMIT.json'),
                        'bytes':sum(p.stat().st_size for p in path.rglob('*') if p.is_file())})
    entries.sort(key=lambda x:x['step'])
    if not entries:raise ValueError('No complete checkpoints; nothing to rotate')
    latest=int((root/'latest_checkpointed_iteration.txt').read_text().strip())
    if latest!=entries[-1]['step']:raise ValueError('Stale/inconsistent latest pointer')
    protected=set(marker.get('protected_steps',[]))|{x['step'] for x in entries[-keep:]}
    if not set(marker.get('protected_steps',[])).issubset({x['step'] for x in entries}):
        raise ValueError('Protected checkpoint is missing')
    plan={'root':str(root),'run_marker_sha256':file_sha(root/'TEXT_RUN.json'),
          'keep':keep,'latest':latest,'protected_steps':sorted(protected),
          'all_entries':entries,'delete':[x for x in entries if x['step'] not in protected]}
    plan['approval_sha256']=canonical_sha(plan)
    return plan


def apply_plan(root,approval_sha256,keep=2):
    root,_=validate_root(root)
    with (root/'.rotation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=make_plan(root,keep)
        if plan['approval_sha256']!=approval_sha256:
            raise ValueError('Plan changed; inspect the new dry-run before applying')
        journal=root/'ROTATION_LAST.json'
        atomic_json(journal,{'phase':'DELETING','plan':plan,'deleted':[]})
        removed=[]
        for target in plan['delete']:
            path=checked_path(target['path'])
            if path.parent!=root:raise ValueError('Deletion escaped exact run root')
            shutil.rmtree(path)
            removed.append(target)
            atomic_json(journal,{'phase':'DELETING','plan':plan,'deleted':removed})
        atomic_json(journal,{'phase':'COMPLETE','plan':plan,'deleted':removed})
        return removed


def enable_automatic_rotation(root, restore_report, contract, protected_steps=()):
    """Opt-in only after an actual restore of this EXACT formal training contract.

    A one-step smoke report cannot enable rotation for a 50-step run. The formal
    run must first save/reload its early checkpoint without changing its horizon.
    """
    root = Path(root).absolute()
    if root.name != 'checkpoints' or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('Exact nonsymlink checkpoint root required')
    proof_path = Path(restore_report).resolve(strict=True)
    proof = json.loads(proof_path.read_text())
    if proof.get('phase') != 'DISTRIBUTED_RESTORE_VERIFIED' or proof.get('contract') != contract:
        raise ValueError('Actual restore must match the formal contract, including planned updates')
    if proof.get('world_size') != 2:
        raise ValueError('Only the verified two-rank student topology is supported')
    checkpoint = checked_path(proof['checkpoint'])
    if checkpoint.parent != root:
        raise ValueError('Restore must be of a checkpoint within this exact run')
    from .restore_audit import verify_restore_reports
    actual = verify_restore_reports(proof_path.parent / 'restore_audit', checkpoint, contract)
    if actual['report_sha256'] != proof['report_sha256']:
        raise ValueError('Changed restore report files')
    verify_before_restore(checkpoint, 2, contract)
    if any(not isinstance(step, int) or step < 1 for step in protected_steps):
        raise ValueError('Protected steps must be positive integers')
    marker = {'schema': 1, 'project': 'opd_math_if', 'run_id': root.parent.name,
              'root': str(root), 'world_size': 2, 'contract': contract,
              'restore_report': str(proof_path), 'restore_report_sha256': file_sha(proof_path),
              'protected_steps': sorted(set(protected_steps)), 'automatic_keep': 2}
    with (root / '.rotation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'TEXT_RUN.json').exists():
            raise ValueError('Do not overwrite an existing rotation contract')
        atomic_json(root / 'TEXT_RUN.json', marker)
        make_plan(root)  # Audit full set and latest pointer before any deletion.
    return marker


def rotate_after_commit(root):
    """Called AFTER synchronous save and atomic latest-pointer publication.

    Native pre-save pruning stays disabled. No marker = no deletion. We do not
    hold the writer lock here; apply_plan acquires it and invalidates stale plans.
    """
    root = Path(root).absolute()
    if not (root / 'TEXT_RUN.json').exists():
        return []
    _, marker = validate_root(root)
    if marker.get('automatic_keep') is None:
        return []
    if marker['automatic_keep'] != 2:
        raise ValueError('Unexpected automatic retention policy')
    plan = make_plan(root, keep=2)
    if not plan['delete']:
        return []
    return apply_plan(root, plan['approval_sha256'], keep=2)


def require_save_space(root, margin_gib=10):
    """Allow simultaneous old checkpoints + pending new checkpoint; never pre-delete."""
    root = Path(root).absolute()
    if root.name != 'checkpoints' or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('Exact checkpoint root required for storage admission')
    estimated = 40 * 1024**3  # conservative floor for the measured 35.4 GiB 3B full state
    for path in root.glob('global_step_*/TEXT_COMMIT.json'):
        manifest = json.loads(path.read_text())
        estimated = max(estimated, sum(entry['bytes'] for entry in manifest['files'].values()))
    required = estimated + margin_gib * 1024**3
    free = shutil.disk_usage(root).free
    if free < required:
        raise RuntimeError(f'Insufficient checkpoint space: need {required} bytes, free {free}; old recovery points retained')
    return {'free_bytes': free, 'estimated_new_checkpoint_bytes': estimated, 'required_bytes': required}

"""Staged checkpoint rotation, NOT installed in the frozen training chain.

Trusted single-writer local filesystem only. Reuse previous full byte validation
while inode/size/mtime/ctime and the complete file set remain unchanged. This
detects ordinary replacement/writes, NOT silent storage corruption or a hostile
writer. Actual restore must always use verify_before_restore (full SHA256).
"""
import fcntl
import json
from pathlib import Path
import shutil
import stat

from .checkpoint_protocol import atomic_json, checked_path, file_sha, verify_before_restore
from .rotation import validate_root


def snapshot(path):
    path = checked_path(path)
    result = {}
    for item in [path, *sorted(path.rglob('*'))]:
        info = item.lstat()
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise ValueError('Unsafe checkpoint member')
        result[str(item.relative_to(path))] = [
            info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns,
        ]
    return result


def validated_entry(path, marker, cache):
    before = snapshot(path)
    manifest_hash = file_sha(path / 'TEXT_COMMIT.json')
    receipt = cache.get(path.name, {})
    if receipt.get('snapshot') == before and receipt.get('manifest_sha256') == manifest_hash:
        manifest = json.loads((path / 'TEXT_COMMIT.json').read_text())
    else:
        manifest = verify_before_restore(path, marker['world_size'], marker['contract'])
    if manifest['world_size'] != marker['world_size'] or manifest['contract'] != marker['contract']:
        raise ValueError('Rotation contract changed')
    step = int(path.name.removeprefix('global_step_'))
    if manifest['step'] != step:
        raise ValueError('Checkpoint directory/manifest mismatch')
    if snapshot(path) != before or file_sha(path / 'TEXT_COMMIT.json') != manifest_hash:
        raise ValueError('Checkpoint mutated during validation')
    receipt = {'snapshot': before, 'manifest_sha256': manifest_hash}
    return {'step': step, 'path': str(path), 'manifest_sha256': manifest_hash}, receipt


def rotate_after_commit(root):
    """Explicit opt-in; synchronous commit must already have succeeded.

    A new checkpoint gets one additional full SHA validation on first rotation.
    Old checkpoints reuse receipts. Plan/apply no longer repeats full scans.
    No background deletion. New deployments explicitly set automatic_keep=1;
    existing keep=2 markers retain their original meaning.
    """
    root = Path(root).absolute()
    if not (root / 'TEXT_RUN.json').exists():
        return []
    _, marker = validate_root(root)
    if marker.get('automatic_keep') is None:
        return []
    if type(marker['automatic_keep']) is not int or marker['automatic_keep'] not in (1, 2):
        raise ValueError('Unexpected retention policy')
    with (root / '.rotation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root, marker = validate_root(root)
        keep = marker.get('automatic_keep')
        if type(keep) is not int or keep not in (1, 2):
            raise ValueError('Unexpected retention policy inside lock')
        marker_hash = file_sha(root / 'TEXT_RUN.json')
        cache_path = root / 'ROTATION_VALIDATION_V2.json'
        cache = {}
        if cache_path.exists():
            if cache_path.is_symlink():
                raise ValueError('Unsafe validation receipt')
            try:
                saved = json.loads(cache_path.read_text())
                if saved.get('marker_sha256') == marker_hash:
                    cache = saved.get('entries', {})
                    if not isinstance(cache, dict):
                        cache = {}
            except (ValueError, AttributeError):
                pass  # Lost/broken cache costs reads, never skips validation.
        paths = sorted(root.glob('global_step_*'))
        entries, receipts = [], {}
        for path in paths:
            entry, receipt = validated_entry(checked_path(path), marker, cache)
            entries.append(entry)
            receipts[path.name] = receipt
        entries.sort(key=lambda item: item['step'])
        latest = int((root / 'latest_checkpointed_iteration.txt').read_text())
        if not entries or latest != entries[-1]['step']:
            raise ValueError('Missing checkpoints or inconsistent latest pointer')
        protected = set(marker.get('protected_steps', []))
        if not protected.issubset({item['step'] for item in entries}):
            raise ValueError('Protected checkpoint missing')
        protected.update(item['step'] for item in entries[-keep:])
        targets = [item for item in entries if item['step'] not in protected]
        # Cheap revalidation before any deletion, including all survivors.
        if (file_sha(root / 'TEXT_RUN.json') != marker_hash
                or int((root / 'latest_checkpointed_iteration.txt').read_text()) != latest
                or sorted(root.glob('global_step_*')) != paths):
            raise ValueError('Rotation plan changed')
        for path in paths:
            if snapshot(path) != receipts[path.name]['snapshot']:
                raise ValueError('Checkpoint changed before deletion')
        journal = {'phase': 'DELETING', 'implementation': 'rotation_v2',
                   'latest': latest, 'keep': keep, 'delete': targets, 'deleted': []}
        atomic_json(root / 'ROTATION_V2_LAST.json', journal)
        for item in targets:
            path = checked_path(item['path'])
            if path.parent != root:
                raise ValueError('Deletion escaped exact run root')
            shutil.rmtree(path)
            journal['deleted'].append(item)
            receipts.pop(path.name)
            atomic_json(root / 'ROTATION_V2_LAST.json', journal)
        atomic_json(cache_path, {'schema': 1, 'marker_sha256': marker_hash, 'entries': receipts})
        journal['phase'] = 'COMPLETE'
        atomic_json(root / 'ROTATION_V2_LAST.json', journal)
        return targets

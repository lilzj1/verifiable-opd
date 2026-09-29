"""Audited I/O-only amendment. Does not modify frozen training/eval files."""
import json
import os
from pathlib import Path
from verifiable_opd.posttraining.checkpoint_protocol import file_sha, atomic_json

ROOT = Path(__file__).resolve().parent.parent
AMENDMENT = ROOT / 'artifacts/text_rotation_v2_amendment.json'
SOURCES = [
    'verifiable_opd/posttraining/rotation_v2.py',
    'posttraining_plan/rotation_handoff.py',
    'posttraining_plan/run_text_formal_v2.py',
    'posttraining_plan/finish_text_project_v2.py',
    'posttraining_plan/watch_rotation_handoff.py',
]


def verify_amendment():
    record = json.loads(AMENDMENT.read_text())
    if record['scope'] != 'S3_checkpoint_IO_only' or record['keep'] != 1:
        raise ValueError('Unexpected I/O amendment')
    if record['original_protocol_sha256'] != file_sha(ROOT/'artifacts/text_formal_protocol_v1/protocol.json'):
        raise ValueError('Original protocol changed')
    if set(record['source_sha256']) != set(SOURCES):
        raise ValueError('Amendment source list changed')
    for name, sha in record['source_sha256'].items():
        if file_sha(ROOT/name) != sha:
            raise ValueError(f'Amended source changed: {name}')
    return record


def worker_setup():
    """Ray's importable setup hook: installed only by the S3 v2 launcher."""
    verify_amendment()
    from verifiable_opd.posttraining import rotation, rotation_v2
    rotation.rotate_after_commit = rotation_v2.rotate_after_commit
    audit = os.environ.get('OPD_TEXT_UPDATE_AUDIT_DIR')
    if audit:
        target = Path(audit).parent/'rotation_hook'
        target.mkdir(exist_ok=True, parents=True)
        atomic_json(target/f'{os.getpid()}.json', {
            'phase': 'ROTATION_V2_WORKER_INSTALLED', 'pid': os.getpid(),
            'keep': 1, 'amendment_sha256': file_sha(AMENDMENT),
        })


def enable_single_rotation(root, restore_report, contract, protected_steps=()):
    verify_amendment()
    expected = ROOT/'artifacts/text_s3_formal_s100_seed17_v1/checkpoints'
    if Path(root) != expected:
        raise ValueError('Amendment only authorizes S3')
    from verifiable_opd.posttraining.rotation import enable_automatic_rotation
    # Retain all original real-restore and contract validation before opt-in.
    marker = enable_automatic_rotation(root, restore_report, contract, protected_steps)
    marker.update(automatic_keep=1, rotation_implementation='rotation_v2',
                  io_amendment_sha256=file_sha(AMENDMENT))
    atomic_json(Path(root)/'TEXT_RUN.json', marker)
    return marker

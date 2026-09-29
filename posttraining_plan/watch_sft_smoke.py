"""Wait for existing S2/S3 profiles, then one S1 update/save/restore/cleanup.

This controller consumes no GPUs while waiting. Does not start formal training.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from posttraining_plan.run_text_smoke import PYTHON
from posttraining_plan.watch_matrix_and_smoke import run_child, wait_idle
from posttraining_plan.cleanup_text_smoke import cleanup
from verifiable_opd.posttraining.checkpoint_protocol import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    # Cleanup's allowlist binds this exact controller identity/exit-code path.
    if args.output.resolve() != ROOT/'artifacts/text_s1_smoke_controller_v1':
        raise ValueError('Use the registered S1 controller artifact directory')
    if not args.resume:
        args.output.mkdir(parents=True, exist_ok=False)

    def update(phase, **fields):
        record = dict(phase=phase, utc=datetime.now(timezone.utc).isoformat(), allow_formal=False, **fields)
        atomic_json(args.output/'status.json', record)
        print(json.dumps(record), flush=True)

    with (ROOT/'artifacts/.text_sft_controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.resume:
                previous = json.loads((args.output/'status.json').read_text())
                if previous['phase'] != 'STOPPED_NEEDS_DIAGNOSIS' or (ROOT/'artifacts/text_s1_sft_smoke_v1').exists():
                    raise ValueError('Resume only the stopped waiter before any S1 launch')
                atomic_json(args.output/f'status_before_resume_{time.time_ns()}.json', previous)
            update('WAITING_EXISTING_PROFILES')
            deadline = time.monotonic() + 8*3600
            while True:
                state = json.loads((ROOT/'artifacts/text_profile_controller_v2/status.json').read_text())
                if state['phase'] == 'PROFILES_COMPLETE_S1_AND_FORMAL_BUDGET_PENDING':
                    break
                if state['phase'].startswith('STOPPED'):
                    raise RuntimeError('Profile stopped; do not proceed or touch other jobs')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Wait budget exhausted; source jobs left untouched')
                time.sleep(30)
            wait_idle()
            run_id = 'text_s1_sft_smoke_v1'
            update('S1_UPDATE_RUNNING', run_id=run_id)
            run_child([PYTHON, '-u', 'posttraining_plan/run_sft_smoke.py', '--run-id', run_id, '--execute'],
                      args.output/'S1_update.log', gpu_ids='1,2')
            wait_idle()
            update('S1_RESTORE_RUNNING', run_id=run_id)
            run_child([PYTHON, '-u', 'posttraining_plan/run_sft_smoke.py', '--run-id', run_id+'_restore',
                       '--restore-from-run', run_id, '--execute'], args.output/'S1_restore.log', gpu_ids='1,2')
            update('S1_CLEANING_VERIFIED_DIAGNOSTIC')
            cleanup(ROOT, 'S1', execute=True)
            update('S1_SMOKE_AND_RESTORE_COMPLETE_FORMAL_NOT_STARTED')
        except Exception as exc:
            update('STOPPED_NEEDS_DIAGNOSIS', error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    main()

"""Finite controller: wait for existing matrix, then only gated smoke/restore.

No downloads, no process discovery/termination, no checkpoint deletion, no
formal training, no retry/tuning loop. Only its own subprocess group is stopped
on a timeout. Any failed gate is a terminal state, not a request to lower it.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from posttraining_plan.analyze_teacher_matrix import analyze, ROLES
from posttraining_plan.run_text_smoke import check_devices, PYTHON
from verifiable_opd.posttraining.checkpoint_protocol import atomic_json


def progress():
    counts = {}
    for role in ROLES:
        path = ROOT / 'artifacts' / f'teacher_matrix_fp16_v1_{role}/predictions.jsonl'
        # Count only complete JSONL records; a writer may be midway through a line.
        counts[role] = path.read_bytes().count(b'\n') if path.exists() else 0
    return counts


def run_child(command, log, timeout=7200, gpu_ids='1,2,3,4'):
    if gpu_ids not in ('1,2', '1,2,3,4'):
        raise ValueError('Only the authorized registered GPU subsets are allowed')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_ids, PYTHONUNBUFFERED='1',
               PYTHONPATH=f'{ROOT}/upstream/verl-v0.8.0:{ROOT}')
    with log.open('x') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        print('Started child', child.pid, 'log', log, flush=True)
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            code = 124
    log.with_suffix('.exit_code.txt').write_text(str(code) + '\n')
    if code:
        raise RuntimeError(f'Child stopped with exit={code}; see {log}')


def wait_idle():
    for attempt in range(5):
        try:
            gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                '--format=csv,noheader,nounits'], text=True)
            processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                '--format=csv,noheader,nounits'], text=True)
            return check_devices(gpu, processes)
        except ValueError:
            if attempt == 4:
                raise
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execute-smokes', action='store_true')
    parser.add_argument('--teacher-policy', choices=('local_matrix','published_source'), default='local_matrix')
    parser.add_argument('--response', type=int, choices=(8192,16384), default=8192)
    parser.add_argument('--resume-s2-from', help='Revalidate known post-save failure and restore it without repeating the update')
    parser.add_argument('--resume-s3-from', help='S2 already restored; revalidate and restore diagnosed S3 post-save failure')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    status = args.output / 'status.json'

    def update(phase, **details):
        record = {'phase': phase, 'utc': datetime.now(timezone.utc).isoformat(),
                  'allow_formal': False, **details}
        atomic_json(status, record)
        print(json.dumps(record), flush=True)

    with (ROOT / 'artifacts/.text_matrix_controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            deadline = time.monotonic() + 3 * 3600
            while True:
                report = analyze()
                if report['phase'] != 'RUNNING_OR_INCOMPLETE':
                    break
                update('WAITING_EXISTING_MATRIX', responses=progress())
                if time.monotonic() >= deadline:
                    raise TimeoutError('Controller wait budget exhausted; does not stop independent matrix jobs')
                time.sleep(30)
            atomic_json(args.output / 'matrix_result.json', report)
            if not report['allow_smoke'] and args.teacher_policy == 'local_matrix':
                update('STOPPED_AT_TEACHER_GATE', matrix_phase=report['phase'],
                       result=str(args.output / 'matrix_result.json'))
                return
            if not args.execute_smokes:
                update('MATRIX_COMPLETE_NO_TRAINING_REQUESTED')
                return
            for group in ('S2', 'S3'):
                if group == 'S2' and args.resume_s3_from:
                    previous = ROOT/'artifacts/text_s2_source16k_smoke_v1_restore/restore_result.json'
                    if json.loads(previous.read_text()).get('phase') != 'DISTRIBUTED_RESTORE_VERIFIED':
                        raise ValueError('Cannot skip an unverified S2 restore')
                    continue
                wait_idle()
                run_id = (f'text_{group.lower()}_source16k_smoke_v1' if args.teacher_policy == 'published_source'
                          else f'text_{group.lower()}_smoke_matrix_v1')
                recovered_run = args.resume_s2_from if group == 'S2' else args.resume_s3_from
                if recovered_run:
                    run_id = recovered_run
                    update('REVALIDATING_EXISTING_CHECKPOINT', group=group, run_id=run_id)
                    run_child([PYTHON, '-u', 'posttraining_plan/revalidate_saved_smoke.py',
                               '--run-id', run_id, '--group', group], args.output/f'{group}_revalidate.log')
                else:
                    update('ONE_UPDATE_RUNNING', group=group, run_id=run_id)
                    command = [PYTHON, '-u', 'posttraining_plan/run_text_smoke.py',
                               '--group', group, '--run-id', run_id, '--execute',
                               '--teacher-policy', args.teacher_policy, '--response', str(args.response)]
                    run_child(command, args.output / f'{group}_update.log')
                wait_idle()
                update('RESTORE_ONLY_RUNNING', group=group, run_id=run_id)
                run_child([PYTHON, '-u', 'posttraining_plan/run_text_smoke.py',
                           '--group', group, '--run-id', run_id + '_restore',
                           '--restore-from-run', run_id, '--execute', '--teacher-policy', args.teacher_policy,
                           '--response', str(args.response)], args.output / f'{group}_restore.log')
            update('SMOKES_FINISHED_PROFILE_AND_FORMAL_BUDGET_PENDING')
        except Exception as exc:
            update('STOPPED_NEEDS_DIAGNOSIS', error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    main()

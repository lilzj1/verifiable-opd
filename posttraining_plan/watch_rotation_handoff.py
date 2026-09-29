"""Wait for S2 dev; replace only the waiting controller, never its live child.

Linux /proc zombie exit status provides the real child exit code while the old
parent is stopped. S2 evaluation itself is never signalled or relaunched.
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
from posttraining_plan.rotation_handoff import verify_amendment, AMENDMENT
from verifiable_opd.posttraining.checkpoint_protocol import atomic_json, file_sha

OUT = ROOT/'artifacts/text_finish_controller_v1'
STATE = ROOT/'artifacts/rotation_handoff_status.json'


def process(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        fields = raw[raw.rfind(')')+2:].split()
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        return {'pid': pid, 'state': fields[0], 'ppid': int(fields[1]),
                'start': fields[19], 'exit_status': int(fields[49]), 'cmd': cmd}
    except FileNotFoundError:
        return None


def children(pid):
    result = []
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            try:
                item = process(int(path.name))
                if item and item['ppid'] == pid:
                    result.append(item)
            except (PermissionError, ProcessLookupError):
                continue
    return result


def same_process(pid, start):
    item = process(pid)
    if not item or item['start'] != start:
        raise ValueError('Process gone or PID reused')
    return item


def update(phase, **kwargs):
    record = dict(phase=phase, utc=datetime.now(timezone.utc).isoformat(), **kwargs)
    atomic_json(STATE, record)
    print(json.dumps(record), flush=True)


def unchanged_inputs():
    paths = [ROOT/'artifacts/text_formal_protocol_v1/protocol.json',
             ROOT/'artifacts/text_formal_protocol_v1/eval_dev.json',
             ROOT/'data/posttraining_text_v3/dev.jsonl',
             ROOT/'artifacts/text_s2_formal_s100_seed17_v1/export_result.json']
    hashes = {str(p): file_sha(p) for p in paths}
    model = ROOT/'artifacts/text_s2_formal_s100_seed17_v1/final_hf'
    stats = {str(p): (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns)
             for p in model.iterdir() if p.is_file()}
    if not stats:
        raise ValueError('S2 exported model missing')
    return hashes, stats


def main():
    signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(128+sig))
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller-pid', type=int, required=True)
    args = parser.parse_args()
    verify_amendment()
    parent = process(args.controller_pid)
    if not parent or 'posttraining_plan/finish_text_project.py ' not in parent['cmd']:
        raise ValueError('Not the original controller')
    if Path(f'/proc/{parent["pid"]}/cwd').resolve() != ROOT:
        raise ValueError('Unexpected controller working directory')
    stopped = False
    with (ROOT/'artifacts/.rotation_handoff.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            deadline = time.monotonic()+48*3600
            last_stage = None
            while True:
                same_process(parent['pid'], parent['start'])
                status = json.loads((OUT/'status.json').read_text())
                stage = status.get('stage')
                if status['phase'].startswith('STOPPED'):
                    raise RuntimeError(f'Original controller stopped: {status}')
                if list(OUT.glob('S3*.command.json')):
                    raise RuntimeError('S3 already launched; no mid-job change allowed')
                if stage != last_stage:
                    update('WAITING_FOR_S2_DEV', current_stage=stage)
                    last_stage = stage
                candidates = children(parent['pid']) if stage == 'S2_dev' else []
                candidates = [c for c in candidates if 'posttraining_plan/eval_text_parallel.py ' in c['cmd']
                              and 'text_formal_eval_v1_s2_dev' in c['cmd']]
                if len(candidates) == 1:
                    child = candidates[0]
                    os.kill(parent['pid'], signal.SIGSTOP)
                    stopped = True
                    time.sleep(0.2)
                    if same_process(parent['pid'], parent['start'])['state'] not in ('T', 't'):
                        raise RuntimeError('Controller did not stop')
                    same_process(child['pid'], child['start'])
                    if list(OUT.glob('S3*.command.json')):
                        raise RuntimeError('Late handoff: S3 already started')
                    baseline = unchanged_inputs()
                    update('S2_DEV_RUNNING_PARENT_HELD', parent_pid=parent['pid'],
                           eval_pid=child['pid'], note='Only parent stopped; evaluation child untouched')
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError('No safe S2 dev boundary within 48h')
                time.sleep(5)
            while True:
                item = same_process(child['pid'], child['start'])
                if item['state'] == 'Z':
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError('S2 dev did not finish; resume original parent')
                time.sleep(15)
            if item['exit_status'] != 0:
                raise RuntimeError(f'S2 dev actual wait status: {item["exit_status"]}')
            summary = ROOT/'artifacts/text_formal_eval_v1_s2_dev/summary.json'
            if json.loads(summary.read_text())['phase'] != 'EVALUATION_COMPLETE':
                raise ValueError('S2 evaluation summary incomplete')
            if unchanged_inputs() != baseline:
                raise ValueError('S2 model/evaluation inputs changed')
            verify_amendment()
            receipt = dict(phase='S2_DEV_COMPLETED_SAFE_HANDOFF', parent=parent, child=child,
                           actual_child_wait_status=item['exit_status'],
                           summary_sha256=file_sha(summary), amendment_sha256=file_sha(AMENDMENT),
                           input_sha256=baseline[0], model_stats=baseline[1])
            atomic_json(OUT/'rotation_handoff_receipt.json', receipt)
            # Only the old parent is terminated; its sole child has exited 0.
            os.kill(parent['pid'], signal.SIGTERM)
            os.kill(parent['pid'], signal.SIGCONT)
            stopped = False
            for _ in range(100):
                old = process(parent['pid'])
                if not old or old['start'] != parent['start'] or old['state'] == 'Z':
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError('Original controller has not exited; do not duplicate')
            # Preserve its exact command ledger; code derives from /proc, not inferred success.
            code = OUT/'S2_dev.exit_code.txt'
            if code.exists() and code.read_text().strip() != '0':
                raise ValueError('Conflicting S2 dev exit code')
            code.write_text('0\n')
            update('STARTING_V2_CONTROLLER', receipt=str(OUT/'rotation_handoff_receipt.json'))
            with (ROOT/'artifacts/text_finish_controller_v2_launcher.log').open('a') as stream:
                result = subprocess.run([sys.executable, '-u', 'posttraining_plan/finish_text_project_v2.py',
                                         '--output', str(OUT), '--resume'], cwd=ROOT,
                                        stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'V2 controller exit code {result.returncode}; inspect its log')
            update('V2_CHAIN_COMPLETE')
        except BaseException as exc:
            if stopped:
                old = process(parent['pid'])
                if old and old['start'] == parent['start']:
                    os.kill(parent['pid'], signal.SIGCONT)
            update('HANDOFF_STOPPED_NEEDS_DIAGNOSIS', error=f'{type(exc).__name__}: {exc}',
                   old_parent_resumed=stopped)
            raise


if __name__ == '__main__':
    main()

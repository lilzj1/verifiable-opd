"""Finite follow-on controller: completed real restores -> 3+3 updates, no weights.

Waits for the existing source controller without modifying/stopping it. No
formal training, teacher-quality rechecks, checkpoint deletion or tuning loops.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from posttraining_plan.run_text_smoke import PYTHON, profile_evidence, audit_updates
from posttraining_plan.watch_matrix_and_smoke import run_child, wait_idle
from verifiable_opd.posttraining.checkpoint_protocol import atomic_json, file_sha


def metrics_from_log(path, expected_updates=3):
    records = {}
    number = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'
    for line in path.read_text().splitlines():
        match = re.search(r'\bstep:(\d+) - ', line)
        if not match:
            continue
        step = int(match.group(1))
        if step in records:
            raise ValueError('Duplicate metrics step')
        row = {}
        for key, value in re.findall(r'([\w/]+):(?:np\.\w+\()?(' + number + r')', line):
            row[key] = float(value)
        required = {'timing_s/gen', 'timing_s/update_actor', 'timing_s/step',
                    'perf/total_num_tokens', 'response_length/mean', 'actor/grad_norm'}
        if not required <= row.keys():
            raise ValueError('Incomplete timing/token metrics')
        if row.get('timing_s/save_checkpoint', 0) != 0:
            raise ValueError('Disposable profile must not save checkpoints')
        records[step] = row
    if sorted(records) != list(range(1, expected_updates + 1)):
        raise ValueError('Missing completed profile steps')
    total = sum(row['timing_s/step'] for row in records.values())
    return {'steps': records, 'mean_step_seconds': total / expected_updates,
            'total_measured_step_seconds': total,
            'total_input_response_tokens': sum(row['perf/total_num_tokens'] for row in records.values()),
            'reserved_step_gpu_hours': total * 4 / 3600,
            'limits': ['Three updates are cost diagnostics, not method efficacy evidence',
                       'Step GPU hours exclude initialization/shutdown; reservation, not utilization',
                       'Input+response tokens are not target-only or teacher-only tokens',
                       'Later response lengths and costs may change during training']}


def rollout_accounting(run):
    totals = {domain: dict(samples=0, response_tokens=0, prompt_tokens=0, budget_hits=0) for domain in ('math', 'if')}
    for step in (1, 2, 3):
        rows = [json.loads(line) for line in (run / f'rollouts/{step}.jsonl').read_text().splitlines() if line.strip()]
        if len(rows) != 4 or sorted(row['domain'] for row in rows) != ['if', 'if', 'math', 'math']:
            raise ValueError('Expected each profile update to contain exactly Math2 + IF2')
        if len({row['example_id'] for row in rows}) != 4:
            raise ValueError('Duplicate task identity in one profile batch')
        for row in rows:
            bucket = totals[row['domain']]
            bucket['samples'] += 1
            bucket['response_tokens'] += row['response_tokens']
            bucket['prompt_tokens'] += row['prompt_tokens']
            bucket['budget_hits'] += int(row['generation_budget_hit'])
    return totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='Resume a stopped controller; never repeat a completed GPU run')
    parser.add_argument('--s2-worker-log', type=Path, help='Recover missing forwarded metrics from the original Ray TaskRunner log')
    parser.add_argument('--cleanup-completed-smokes', action='store_true',
                        help='User-authorized cleanup after successful restore process exits; metadata retained')
    parser.add_argument('--source-controller', choices=('text_source_smoke_controller_v2', 'text_source_smoke_controller_v3'),
                        default='text_source_smoke_controller_v2')
    args = parser.parse_args()
    if not args.resume:
        args.output.mkdir(parents=True, exist_ok=False)

    def update(phase, **fields):
        record = dict(phase=phase, utc=datetime.now(timezone.utc).isoformat(), allow_formal=False, **fields)
        atomic_json(args.output / 'status.json', record)
        print(json.dumps(record), flush=True)

    with (ROOT / 'artifacts/.text_profile_controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.resume:
                previous = json.loads((args.output/'status.json').read_text())
                if previous['phase'] != 'STOPPED_NEEDS_DIAGNOSIS':
                    raise ValueError('Only an explicitly stopped controller may resume')
                archive = args.output/f'status_before_resume_{time.time_ns()}.json'
                atomic_json(archive, previous)
            source = ROOT / 'artifacts' / args.source_controller / 'status.json'
            deadline = time.monotonic() + 4 * 3600
            last = None
            while True:
                state = json.loads(source.read_text())
                if state['phase'] == 'SMOKES_FINISHED_PROFILE_AND_FORMAL_BUDGET_PENDING':
                    break
                if state['phase'].startswith('STOPPED'):
                    raise RuntimeError('Existing smoke controller failed; profile not started')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Wait limit exceeded; existing tasks left untouched')
                key = (state['phase'], state.get('group'))
                if key != last:
                    update('WAITING_EXISTING_SMOKE', source_status=state)
                    last = key
                time.sleep(30)
            evidence = profile_evidence(ROOT / 'artifacts')
            if args.cleanup_completed_smokes:
                from posttraining_plan.cleanup_text_smoke import cleanup
                for group in ('S2', 'S3'):
                    run = ROOT / 'artifacts' / f'text_{group.lower()}_source16k_smoke_v1'
                    journal = run / 'checkpoint_cleanup.json'
                    target = run / 'checkpoints/global_step_1'
                    if journal.exists() and not target.exists() and json.loads(journal.read_text())['phase'] == 'DELETED':
                        continue
                    update('CLEANING_COMPLETED_DIAGNOSTIC_CHECKPOINT', group=group)
                    cleanup(ROOT, group, execute=True)
            summaries = {}
            for group in ('S2', 'S3'):
                wait_idle()
                run_id = f'text_{group.lower()}_source16k_profile3_v1'
                update('PROFILE_RUNNING', group=group, run_id=run_id, restore_evidence=evidence)
                log = args.output / f'{group}.log'
                run = ROOT/'artifacts'/run_id
                if args.resume and run.exists():
                    if log.with_suffix('.exit_code.txt').read_text().strip() != '0':
                        raise ValueError('Existing GPU run did not exit successfully; do not silently retry')
                    audit_updates(run/'optimizer_audit', expected_updates=3)
                else:
                    run_child([PYTHON, '-u', 'posttraining_plan/run_text_smoke.py', '--group', group,
                               '--run-id', run_id, '--execute', '--teacher-policy', 'published_source',
                               '--response', '16384', '--profile-updates', '3'], log)
                result = json.loads((run / 'profile_result.json').read_text())
                if result['phase'] != 'THREE_UPDATES_VERIFIED_NO_WEIGHTS_SAVED':
                    raise ValueError('Profile did not verify all updates')
                metrics_log = args.s2_worker_log if group == 'S2' and args.s2_worker_log else log
                if metrics_log != log:
                    raw = metrics_log.read_text()
                    if str(run/'rollouts/3.jsonl') not in raw:
                        raise ValueError('Worker log does not identify this exact run')
                    copied = args.output/'S2_original_worker.log'
                    with copied.open('x') as stream:
                        stream.write(raw)
                    metrics_log = copied
                durable = run/'optimizer_audit/step_metrics'
                if durable.is_dir():
                    # Synchronous per-step files survive Ray stdout shutdown races.
                    metrics_log = args.output/f'{group}_durable_metrics.log'
                    with metrics_log.open('x') as stream:
                        for step in (1, 2, 3):
                            record = json.loads((durable/f'{step}.json').read_text())
                            stream.write(f'step:{step} - ' + ' - '.join(f'{k}:{v}' for k,v in record.items())+'\n')
                summaries[group] = dict(metrics_from_log(metrics_log), metrics_sha256=file_sha(metrics_log),
                                        launcher_wall_seconds=result['wall_seconds'],
                                        by_domain=rollout_accounting(ROOT / 'artifacts' / run_id))
                atomic_json(args.output / 'cost_report.json', summaries)
            update('PROFILES_COMPLETE_S1_AND_FORMAL_BUDGET_PENDING', report=str(args.output / 'cost_report.json'))
        except Exception as exc:
            update('STOPPED_NEEDS_DIAGNOSIS', error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    main()

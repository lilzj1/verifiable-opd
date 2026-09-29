"""Durable, bounded project handoff: profiles -> SFT smoke -> frozen comparison -> eval.

No teacher-strength rechecks, paid APIs, downloads or other-project cleanup.
Errors stop the chain. Successful commands can be resumed without rerunning GPU
work; a failed child requires diagnosis, not an automatic tuning/retry loop.
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
sys.path.insert(0,str(ROOT))
from posttraining_plan.run_text_smoke import PYTHON
from posttraining_plan.watch_matrix_and_smoke import wait_idle
from verifiable_opd.posttraining.checkpoint_protocol import atomic_json, file_sha


def execute(command, log, gpu_ids, timeout):
    code_path = log.with_suffix('.exit_code.txt')
    command_path = log.with_suffix('.command.json')
    spec = dict(command=command, cuda_visible_devices=gpu_ids)
    if code_path.exists():
        if code_path.read_text().strip() != '0' or json.loads(command_path.read_text()) != spec:
            raise ValueError(f'Previous command failed or changed; diagnose before resuming: {log}')
        return
    if log.exists() or command_path.exists():
        raise ValueError(f'Incomplete previous child; do not launch a duplicate: {log}')
    if gpu_ids not in ('','1','1,2','1,2,3,4'):
        raise ValueError('Unauthorized GPU request')
    if gpu_ids:
        wait_idle()
    atomic_json(command_path, spec)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_ids, PYTHONUNBUFFERED='1',
               PYTHONPATH=f'{ROOT}/upstream/verl-v0.8.0:{ROOT}')
    with log.open('x') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            code = 124
    code_path.write_text(str(code)+'\n')
    if code:
        raise RuntimeError(f'Child failed ({code}), see {log}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not args.resume:
        args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    protocol_dir = ROOT/'artifacts/text_formal_protocol_v1'
    protocol = protocol_dir/'protocol.json'

    def update(phase, **extra):
        record = dict(phase=phase, utc=datetime.now(timezone.utc).isoformat(), **extra)
        atomic_json(output/'status.json',record)
        print(json.dumps(record),flush=True)

    def run(name, command, gpus='', hours=2):
        update('RUNNING', stage=name, log=str(output/f'{name}.log'))
        execute(command, output/f'{name}.log', gpus, hours*3600)

    def evaluate(group, model, split):
        manifest = ROOT/('data/posttraining_text_v3/dev.jsonl' if split=='dev' else 'data/posttraining_text_v3/test_manifest.jsonl')
        frozen = json.loads(protocol.read_text())
        if file_sha(protocol_dir/f'eval_{split}.json') != frozen['eval_protocol_sha256'][split]:
            raise ValueError('Evaluation protocol changed')
        target = ROOT/f'artifacts/text_formal_eval_v1_{group.lower()}_{split}'
        run(f'{group}_{split}', [str(ROOT/'.text_runtime/bin/python'), '-u', 'posttraining_plan/eval_text_parallel.py',
            '--model',str(model),'--manifest',str(manifest),'--protocol',str(protocol_dir/f'eval_{split}.json'),
            '--split',split,'--output',str(target)], '1,2,3,4', hours=48)
        if json.loads((target/'summary.json').read_text())['phase'] != 'EVALUATION_COMPLETE':
            raise ValueError('Evaluation incomplete')

    with (ROOT/'artifacts/.text_finish_controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            update('WAITING_ENGINEERING_CHAIN', note='S3 three-update profile, then S1 save/restore; no GPUs consumed while waiting')
            deadline = time.monotonic()+6*3600
            while True:
                states = [json.loads((ROOT/f'artifacts/{name}/status.json').read_text())
                          for name in ('text_profile_controller_v2','text_s1_smoke_controller_v1')]
                if any(state['phase'].startswith('STOPPED') for state in states):
                    raise RuntimeError(f'Engineering chain stopped: {states}')
                if states[1]['phase'] == 'S1_SMOKE_AND_RESTORE_COMPLETE_FORMAL_NOT_STARTED':
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('Engineering wait exhausted; independent job untouched')
                time.sleep(30)
            run('eval_interfaces',[str(ROOT/'.text_runtime/bin/python'),'-u','posttraining_plan/validate_text_eval.py',
                '--output',str(output/'eval_interface_audit.json')])
            run('freeze', [PYTHON,'-u','posttraining_plan/freeze_text_formal.py','--output',str(protocol_dir)])
            from posttraining_plan.run_text_formal import read_protocol
            _, cfg = read_protocol(protocol,'S2')
            models = {'S0':Path(cfg.actor_rollout_ref.model.path)}
            evaluate('S0',models['S0'],'dev')
            for group in ('S2','S3'):
                for stage in ('first10','restore10','finish'):
                    run(f'{group}_{stage}', [PYTHON,'-u','posttraining_plan/run_text_formal.py',
                        '--protocol',str(protocol),'--group',group,'--stage',stage,'--execute'], '1,2,3,4', hours=48)
                run(f'{group}_export', [PYTHON,'-u','posttraining_plan/export_text_final.py','--protocol',str(protocol),'--group',group], hours=3)
                # Completed arm keeps final full recovery state + HF; retires
                # its old penultimate state before the next arm needs disk.
                run(f'{group}_retire_old', [PYTHON,'-u','posttraining_plan/cleanup_finished_text.py','--group',group,'--execute'], hours=2)
                models[group] = ROOT/f'artifacts/text_{group.lower()}_formal_s100_seed17_v1/final_hf'
                evaluate(group,models[group],'dev')
            for restore in (False,True):
                command = [PYTHON,'-u','posttraining_plan/run_sft_formal.py','--protocol',str(protocol)]
                if restore:
                    command += ['--restore-only']
                run('S1_restore' if restore else 'S1_train',command,'1,2',hours=24)
            run('S1_export',[PYTHON,'-u','posttraining_plan/export_text_final.py','--protocol',str(protocol),'--group','S1'],hours=3)
            models['S1'] = ROOT/'artifacts/text_s1_formal_tokens_seed17_v1/final_hf'
            evaluate('S1',models['S1'],'dev')
            # Test is only opened after all checkpoint identities are fixed.
            for group in ('S0','S1','S2','S3'):
                evaluate(group,models[group],'test')
            for split in ('dev','test'):
                manifest = ROOT/('data/posttraining_text_v3/dev.jsonl' if split=='dev' else 'data/posttraining_text_v3/test_manifest.jsonl')
                for left,right in (('S0','S1'),('S0','S2'),('S0','S3'),('S2','S3')):
                    run(f'compare_{split}_{left}_{right}',[PYTHON,'posttraining_plan/summarize_text.py',
                        '--manifest',str(manifest),'--baseline',str(ROOT/f'artifacts/text_formal_eval_v1_{left.lower()}_{split}'),
                        '--candidate',str(ROOT/f'artifacts/text_formal_eval_v1_{right.lower()}_{split}'),
                        '--output',str(output/f'comparison_{split}_{left}_{right}.json')])
            run('report',[PYTHON,'posttraining_plan/report_text_formal.py','--controller',str(output)])
            update('TRAINING_AND_FROZEN_EVALUATIONS_COMPLETE', report=str(ROOT/'TEXT_FORMAL_RESULTS.md'),
                   limitation='One training seed, limited budget; preserve negative or inconclusive outcomes')
        except Exception as exc:
            update('STOPPED_NEEDS_DIAGNOSIS', error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    main()

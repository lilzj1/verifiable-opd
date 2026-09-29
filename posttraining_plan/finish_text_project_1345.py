"""Resource-only amendment for unfinished S1/S2/S3 final tests.

Preserve completed stage commands and results. Do not use GPU 2, restart
training, rerun S0, or change inference/scoring settings.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from verifiable_opd.posttraining.checkpoint_protocol import atomic_json, file_sha

GPUS = (1, 3, 4, 5)
GPU_STRING = '1,3,4,5'
AMENDMENT = ROOT/'artifacts/text_eval_gpu1345_amendment.json'


def verify_amendment():
    d = json.loads(AMENDMENT.read_text())
    if d['groups'] != ['S1', 'S2', 'S3'] or d['physical_gpus'] != list(GPUS):
        raise ValueError('Unexpected GPU amendment')
    for name, sha in d['source_sha256'].items():
        path = (ROOT/name).resolve(strict=True)
        if not path.is_relative_to(ROOT) or file_sha(path) != sha:
            raise ValueError(f'GPU amendment source changed: {name}')
    for name, sha in d['preserved_sha256'].items():
        if file_sha(ROOT/name) != sha:
            raise ValueError(f'Original evidence/protocol changed: {name}')
    return d


def check_devices(gpu_csv, process_csv):
    devices = {}
    for line in gpu_csv.strip().splitlines():
        index, uuid, memory = [x.strip() for x in line.split(',')]
        devices[int(index)] = (uuid, int(memory))
    if not set(GPUS) <= set(devices):
        raise ValueError('Required GPUs 1/3/4/5 not visible')
    selected = {devices[i][0] for i in GPUS}
    for line in process_csv.strip().splitlines():
        uuid, pid = [x.strip() for x in line.split(',')]
        if uuid in selected:
            raise ValueError(f'Selected GPU occupied by PID {pid}; do not kill it')
    if any(devices[i][1] > 512 for i in GPUS):
        raise ValueError('Selected GPU has unexpected memory usage')
    return {str(i): devices[i] for i in GPUS}


def wait_idle():
    for attempt in range(5):
        try:
            return check_devices(subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                '--format=csv,noheader,nounits'], text=True), subprocess.check_output(['nvidia-smi',
                '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True))
        except ValueError:
            if attempt == 4:
                raise
            time.sleep(30)


def remap_command(command, log):
    if log.stem not in ('S1_test', 'S2_test', 'S3_test'):
        return None
    result = list(command)
    index = result.index('posttraining_plan/eval_text_parallel.py')
    if result[result.index('--split')+1] != 'test':
        raise ValueError('Only unfinished final tests may be remapped')
    result[index] = 'posttraining_plan/eval_text_parallel_1345.py'
    return result


def execute(command, log, timeout):
    verify_amendment()
    code_path = log.with_suffix('.exit_code.txt')
    command_path = log.with_suffix('.command.json')
    spec = dict(command=command, cuda_visible_devices=GPU_STRING)
    if code_path.exists():
        if code_path.read_text().strip() != '0' or json.loads(command_path.read_text()) != spec:
            raise ValueError('Previous remapped stage failed or command changed')
        return
    if log.exists() or command_path.exists():
        raise ValueError('Incomplete remapped stage; never duplicate or selectively retry')
    devices = wait_idle()
    atomic_json(log.with_suffix('.gpu_assignment.json'),
                {'physical_gpus': list(GPUS), 'devices': devices, 'amendment_sha256': file_sha(AMENDMENT)})
    atomic_json(command_path, spec)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU_STRING, PYTHONUNBUFFERED='1',
               PYTHONPATH=f'{ROOT}/upstream/verl-v0.8.0:{ROOT}')
    with log.open('x') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
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
        raise RuntimeError(f'Remapped evaluation failed ({code}), see {log}')


def main():
    verify_amendment()
    from posttraining_plan import finish_text_project as original
    from posttraining_plan import finish_text_project_v2
    original_execute = original.execute

    def dispatch(command, log, gpu_ids, timeout):
        mapped = remap_command(command, log)
        if mapped is None:
            return original_execute(command, log, gpu_ids, timeout)
        return execute(mapped, log, timeout)

    original.execute = dispatch
    finish_text_project_v2.main()


if __name__ == '__main__':
    main()

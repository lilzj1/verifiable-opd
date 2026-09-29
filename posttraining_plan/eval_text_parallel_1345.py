"""Four independent model replicas, frozen sample partition, strict complete merge."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from verifiable_opd.posttraining.evaluation import expected_samples,summarize
from verifiable_opd.posttraining.checkpoint_protocol import atomic_json,file_sha


def merge(manifest,output,split):
    rows=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    expected=list(expected_samples(rows,split))
    common=None
    records=[]
    walls=[]
    for index in range(4):
        folder=output/f'shards/{index}'
        protocol=json.loads((folder/'protocol.json').read_text())
        if protocol.get('shard_index')!=index or protocol.get('physical_gpu')!=('1','3','4','5')[index]:
            raise ValueError('Wrong shard identity/device')
        if protocol['manifest_sha256']!=file_sha(manifest):
            raise ValueError('Evaluation manifest changed')
        normalized={k:v for k,v in protocol.items() if k not in ('shard_index','physical_gpu')}
        if common is None:common=normalized
        if common!=normalized:raise ValueError('Different model/protocol in one evaluation')
        summary=json.loads((folder/'summary.json').read_text())
        if summary['phase']!='EVALUATION_SHARD_COMPLETE':raise ValueError('Incomplete shard')
        subset=[json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines() if line.strip()]
        if [(r['example_id'],r['sample_index']) for r in subset]!=expected[index::4]:
            raise ValueError('Missing, repeated or misplaced sample in shard')
        records+=subset
        walls.append(summary['wall_seconds'])
    result=summarize(rows,records,split)
    indexed={(r['example_id'],r['sample_index']):r for r in records}
    with (output/'predictions.jsonl').open('x') as stream:
        for key in expected:stream.write(json.dumps(indexed[key],ensure_ascii=False)+'\n')
    atomic_json(output/'protocol.json',dict(common,physical_gpu='1,3,4,5'))
    result.update(phase='EVALUATION_COMPLETE',execution_shards=4,
                  replica_wall_seconds=walls,reserved_replica_gpu_hours=sum(walls)/3600)
    atomic_json(output/'summary.json',result)
    return result


def main():
    signal.signal(signal.SIGTERM,lambda sig,frame:sys.exit(128+sig))
    parser=argparse.ArgumentParser()
    for key in ('model','manifest','protocol','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--split',choices=['dev','test'],required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1,3,4,5':raise ValueError('Only assigned four GPU replicas')
    frozen=json.loads(args.protocol.read_text())
    if frozen.get('execution_shards')!=4:raise ValueError('Parallel evaluation was not frozen')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'shards').mkdir()
    children=[]
    streams=[]
    started=time.monotonic()
    try:
        for index,gpu in enumerate(('1','3','4','5')):
            log=args.output/f'shard_{index}.log'
            stream=log.open('x')
            streams.append(stream)
            command=[sys.executable,'-u',str(ROOT/'posttraining_plan/eval_text_1345.py'),'--model',str(args.model),
                     '--manifest',str(args.manifest),'--protocol',str(args.protocol),'--split',args.split,
                     '--output',str(args.output/f'shards/{index}'),'--gpu',gpu,'--shards','4','--shard-index',str(index)]
            children.append(subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),
                            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True))
        while True:
            codes=[child.poll() for child in children]
            if any(code not in (None,0) for code in codes):raise RuntimeError(f'Evaluation replica failed: {codes}')
            if all(code==0 for code in codes):break
            if time.monotonic()-started>46*3600:raise TimeoutError('Bounded evaluation timeout')
            counts=[]
            for index in range(4):
                path=args.output/f'shards/{index}/predictions.jsonl'
                counts.append(path.read_bytes().count(b'\n') if path.exists() else 0)
            print(f'completed_responses={sum(counts)} by_replica={counts}',flush=True)
            time.sleep(30)
        result=merge(args.manifest,args.output,args.split)
        result['wall_seconds']=time.monotonic()-started
        atomic_json(args.output/'summary.json',result)
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
        for child in children:
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait()
        for stream in streams:stream.close()


if __name__=='__main__':main()

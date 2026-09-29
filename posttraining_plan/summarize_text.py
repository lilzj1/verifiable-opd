"""Rebuild validated per-run metrics and paired domain deltas from raw responses."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from verifiable_opd.posttraining.evaluation import summarize,paired_report
from verifiable_opd.posttraining.checkpoint_protocol import file_sha


def read(path):return [json.loads(s) for s in path.read_text().split('\n') if s.strip()]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--baseline',required=True,type=Path)
    parser.add_argument('--candidate',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    left=json.loads((args.baseline/'protocol.json').read_text())
    right=json.loads((args.candidate/'protocol.json').read_text())
    ignored={'model_path','model_config_sha256','physical_gpu'}
    if {k:v for k,v in left.items() if k not in ignored}!={k:v for k,v in right.items() if k not in ignored}:
        raise ValueError('Protocol mismatch: do not compare different decode, tasks or checker versions')
    if file_sha(args.manifest)!=left['manifest_sha256']:
        raise ValueError('Manifest changed')
    rows=read(args.manifest)
    baseline=read(args.baseline/'predictions.jsonl')
    candidate=read(args.candidate/'predictions.jsonl')
    report={'baseline':summarize(rows,baseline,left['split']),
        'candidate':summarize(rows,candidate,left['split']),
        'paired':paired_report(rows,baseline,candidate,left['split']),
        'sources':{'baseline':str(args.baseline),'candidate':str(args.candidate)}}
    with args.output.open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

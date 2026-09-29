"""CPU-only evaluation interface audit; no generated answers, scores or GPU use."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from verifiable_opd.posttraining.checkpoint_protocol import atomic_json,file_sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from transformers import AutoTokenizer
    from verifiable_opd.posttraining.text_encode import encode_prompt
    from verifiable_opd.posttraining.checkers import score
    from verifiable_opd.posttraining.public_checkers import public_score
    from verifiable_opd.posttraining.evaluation import expected_samples
    assets=json.loads((ROOT/'artifacts/posttraining_assets_20260926_resume_v4/status.json').read_text())
    model=assets['models']['shared_student_initialization']['overlay']
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,fix_mistral_regex=False)
    result=dict(phase='CPU_EVAL_INTERFACES_VERIFIED_NO_MODEL_EVALUATION',splits={})
    for split,name in (('dev','dev.jsonl'),('test','test_manifest.jsonl')):
        path=ROOT/f'data/posttraining_text_v3/{name}'
        rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        expected=expected_samples(rows,split)
        maximum=0
        for row in rows:
            maximum=max(maximum,len(encode_prompt(tokenizer,row['messages'],2048)))
            # Exercises schema/gold parsing/constraint construction, not a real
            # candidate answer and never used for teacher/model selection.
            (score if split=='dev' else public_score)(row,'')
        result['splits'][split]=dict(tasks=len(rows),responses=len(expected),max_prompt_tokens=maximum,manifest_sha256=file_sha(path))
    if args.output.exists():raise ValueError('Do not overwrite audit evidence')
    atomic_json(args.output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

"""Generate/check a complete frozen dev or final-test evaluation, never train.

Uses the isolated .text_runtime, NOT the Hydra training interpreter. Test
requires a frozen protocol JSON; simply having test data is not authorization
to use it during development. Output directory must be new; no selective retry.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from verifiable_opd.posttraining.evaluation import canonical_sha, expected_samples, summarize, decode_settings


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--split',choices=['dev','test'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',choices=['1','2','3','4'],required=True)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--shard-index',type=int,default=0)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=args.gpu:
        raise ValueError('Evaluation only on the explicitly assigned authorized GPU')
    config=json.loads(args.protocol.read_text())
    if args.shards not in (1,4) or not 0<=args.shard_index<args.shards or config.get('execution_shards',1)!=args.shards:
        raise ValueError('Shard topology must match frozen evaluation protocol')
    raw=args.manifest.read_bytes()
    if config['manifest_sha256']!=hashlib.sha256(raw).hexdigest():
        raise ValueError('Evaluation set changed')
    if args.split=='test' and (config.get('phase')!='FROZEN_FINAL_TEST' or
                              config.get('checkpoint_selection')!='final_planned_step'):
        raise ValueError('Final test requires a frozen selection/decode/checker protocol')
    for key in ('max_prompt_tokens','max_response_tokens','temperature','top_p','seed','dtype','checker_hashes'):
        if key not in config:
            raise ValueError(f'Missing protocol setting: {key}')
    if config['dtype'] not in ('bfloat16','float16'):
        raise ValueError('Unregistered length/precision')
    decoding = {domain: decode_settings(config, domain) for domain in ('math', 'if')}
    if not config['checker_hashes']:
        raise ValueError('Freeze checker source hashes before running')
    for relative,sha in config['checker_hashes'].items():
        path=(ROOT/relative).resolve(strict=True)
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
            raise ValueError('Checker source/version changed')
    rows=[json.loads(line) for line in raw.decode().split('\n') if line.strip()]
    expected=expected_samples(rows,args.split)
    args.output.mkdir(parents=True,exist_ok=False)
    model=args.model.resolve(strict=True)
    run=dict(config,model_path=str(model),split=args.split,physical_gpu=args.gpu,
             model_config_sha256=hashlib.sha256((model/'config.json').read_bytes()).hexdigest(),
             task_ids=[row['example_id'] for row in rows],expected_responses=len(expected))
    if args.shards>1:
        run['shard_index']=args.shard_index
    (args.output/'protocol.json').write_text(json.dumps(run,indent=2)+'\n')
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from verifiable_opd.posttraining.text_encode import encode_prompt
    from verifiable_opd.posttraining.checkers import score
    from verifiable_opd.posttraining.public_checkers import public_score
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,fix_mistral_regex=False)
    start=time.monotonic()
    llm=LLM(model=str(model),dtype=config['dtype'],max_model_len=decoding['math']['max_model_len'],
            max_num_seqs=4,gpu_memory_utilization=.5,enforce_eager=True,
            enable_chunked_prefill=False,enable_prefix_caching=False,seed=config['seed'])
    init_seconds=time.monotonic()-start
    records=[]
    items=[item for index,item in enumerate(expected.items()) if index%args.shards==args.shard_index]
    try:
        with (args.output/'predictions.jsonl').open('x') as stream:
            for offset in range(0,len(items),4):
                batch=items[offset:offset+4]
                prompts=[encode_prompt(tokenizer,row['messages'],config['max_prompt_tokens']) for _,row in batch]
                params=[SamplingParams(**{k:v for k,v in decoding[row['domain']].items() if k!='max_model_len'},
                    stop_token_ids=[tokenizer.eos_token_id],
                    seed=config['seed']+sample) for (_,sample),row in batch]
                before=time.monotonic()
                outputs=llm.generate([{'prompt_token_ids':p} for p in prompts],params,use_tqdm=False)
                seconds=time.monotonic()-before
                for ((key,sample),row),prompt,out in zip(batch,prompts,outputs,strict=True):
                    answer=out.outputs[0]
                    record={'example_id':key,'sample_index':sample,'domain':row['domain'],
                        'prompt_sha256':canonical_sha(row['messages']),'response':answer.text,
                        'prompt_token_ids':prompt,'response_token_ids':list(answer.token_ids),
                        'finish_reason':answer.finish_reason,'batch_index':offset//4,'batch_wall_seconds':seconds}
                    try:
                        record['evaluation']=(score if args.split=='dev' else public_score)(row,answer.text)
                    except Exception as exc:
                        record['checker_error']=f'{type(exc).__name__}: {exc}'
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                    stream.flush()
                    records.append(record)
                print(f'{len(records)}/{len(items)} responses; batch {seconds:.2f}s',flush=True)
        summary=summarize(rows,records,args.split) if args.shards==1 else {'responses':len(records)}
        summary.update(initialization_seconds=init_seconds,wall_seconds=time.monotonic()-start,
                       phase='EVALUATION_COMPLETE' if args.shards==1 else 'EVALUATION_SHARD_COMPLETE')
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__=='__main__':
    main()

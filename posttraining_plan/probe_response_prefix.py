"""Response-token teacher-forced HF/vLLM audit on 16 fixed dev prefixes.

No tolerance relaxation: max .5 / mean .05 per response. Padding is audited
separately. All inputs are token IDs; never decode/re-encode sampled actions.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', default='shared_student_initialization')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['vllm', 'hf'], required=True)
    parser.add_argument('--fp32-head', action='store_true', help='Diagnostic variant; preserve original report')
    parser.add_argument('--attention', default='flash_attention_2', choices=['flash_attention_2','sdpa','eager'])
    parser.add_argument('--student-prefixes', type=Path, help='Score fixed student-generated token IDs, never teacher samples')
    parser.add_argument('--dtype', choices=['bfloat16','float16'], default='bfloat16')
    parser.add_argument('--max-context', type=int, default=4096)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '4':
        raise ValueError('Numerical probe restricted to GPU 4')
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from verifiable_opd.posttraining.text_encode import encode_prompt
    torch.set_num_threads(4)
    assets = json.loads((ROOT/'artifacts/posttraining_assets_20260926_resume_v4/status.json').read_text())
    path = assets['models'][args.role]['overlay']
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, fix_mistral_regex=False)
    if args.stage == 'vllm':
        args.output.mkdir(parents=True, exist_ok=False)
        from vllm import LLM, SamplingParams
        rows = [json.loads(s) for s in (ROOT/'data/posttraining_text_v1/dev.jsonl').read_text().split('\n') if s.strip()]
        rows = [r for d in ('math','if') for r in [x for x in rows if x['domain']==d][:8]]
        llm = LLM(model=path, dtype=args.dtype, max_model_len=args.max_context,
                  enforce_eager=True, max_num_seqs=4, gpu_memory_utilization=.40,
                  enable_chunked_prefill=False, enable_prefix_caching=False, seed=17)
        records = []
        if args.student_prefixes and args.student_prefixes.suffix=='.jsonl':
            generated=[json.loads(s) for s in args.student_prefixes.read_text().split('\n') if s.strip()]
            by_id={r['example_id']:r for r in generated}
            fixed=[{'example_id':r['example_id'],
                    'token_ids':by_id[r['example_id']]['prompt_token_ids']+by_id[r['example_id']]['response_token_ids']}
                   for r in rows]
        else:
            fixed = json.loads(args.student_prefixes.read_text()) if args.student_prefixes else None
        for start in range(0,16,4):
            batch = rows[start:start+4]
            prompts = [encode_prompt(tokenizer, r['messages']) for r in batch]
            if fixed is None:
                outputs = llm.generate([{'prompt_token_ids': p} for p in prompts],
                    SamplingParams(temperature=1, seed=17, max_tokens=32), use_tqdm=False)
                sequences = [p+list(o.outputs[0].token_ids) for p,o in zip(prompts,outputs,strict=True)]
            else:
                sequences = [r['token_ids'] for r in fixed[start:start+4]]
                for row,p,r,ids in zip(batch,prompts,fixed[start:start+4],sequences,strict=True):
                    assert row['example_id']==r['example_id'] and ids[:len(p)]==p
            scored = llm.generate([{'prompt_token_ids': ids} for ids in sequences],
                SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=1),use_tqdm=False)
            for row,p,ids,out in zip(batch,prompts,sequences,scored,strict=True):
                scores = [out.prompt_logprobs[j][ids[j]].logprob for j in range(len(p),len(ids))]
                records.append({'example_id':row['example_id'],'domain':row['domain'],
                    'prompt_length':len(p),'token_ids':ids,'vllm_response_scores':scores})
        (args.output/'vllm.json').write_text(json.dumps(records))
        (args.output/'protocol.json').write_text(json.dumps({'role':args.role,
            'student_prefixes':str(args.student_prefixes),'sample_count':16,
            'max_tolerance':.5,'mean_tolerance':.05,'dtype':args.dtype}))
        return
    records = json.loads((args.output/'vllm.json').read_text())
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
        torch_dtype=getattr(torch,args.dtype), attn_implementation=args.attention,device_map={'':0}).eval()
    if args.fp32_head:
        class FP32Head(torch.nn.Module):
            def __init__(self, original):
                super().__init__()
                self.register_buffer('weight', original.weight.detach().float())
            def forward(self, hidden):
                return torch.nn.functional.linear(hidden.float(), self.weight)
        model.lm_head = FP32Head(model.lm_head)
    result = {'role':args.role,'max_tolerance':.5,'mean_tolerance':.05,
              'scope':'16 real response prefixes; see protocol for source, no task-quality claim','records':[]}
    with torch.inference_mode():
        for start in range(0,16,4):
            batch = records[start:start+4]
            length = max(len(r['token_ids']) for r in batch)
            ids = torch.tensor([[tokenizer.pad_token_id]*(length-len(r['token_ids']))+r['token_ids'] for r in batch],device='cuda')
            mask = torch.tensor([[0]*(length-len(r['token_ids']))+[1]*len(r['token_ids']) for r in batch],device='cuda')
            positions = mask.cumsum(-1)-1
            positions.masked_fill_(mask==0,0)
            logits = model(ids,attention_mask=mask,position_ids=positions,use_cache=False).logits
            for i,r in enumerate(batch):
                offset=length-len(r['token_ids'])+r['prompt_length']
                scores=logits[i,offset-1:-1].float().log_softmax(-1).gather(-1,ids[i,offset:,None]).flatten().cpu()
                r['hf_batched_response_scores']=scores.tolist()
            del logits,ids,mask,positions
            for r in batch:
                ids=torch.tensor([r['token_ids']],device='cuda')
                logits=model(ids,use_cache=False).logits[0,r['prompt_length']-1:-1].float()
                scores=logits.log_softmax(-1).gather(-1,ids[0,r['prompt_length']:,None]).flatten().cpu()
                reference=torch.tensor(r['vllm_response_scores'])
                error=(scores-reference).abs()
                padding_error=(scores-torch.tensor(r['hf_batched_response_scores'])).abs()
                r.update(hf_single_response_scores=scores.tolist(),max_abs_error=error.max().item(),
                    mean_abs_error=error.mean().item(),padding_max_abs_error=padding_error.max().item(),
                    padding_mean_abs_error=padding_error.mean().item())
                result['records'].append(r)
                del ids,logits
            print(args.role,'prefixes',len(result['records']),flush=True)
    result['phase']='PASS' if all(r['max_abs_error']<=.5 and r['mean_abs_error']<=.05
        and r['padding_max_abs_error']<=.5 and r['padding_mean_abs_error']<=.05 for r in result['records']) else 'FAIL'
    result['fp32_head_diagnostic']=args.fp32_head
    report_name='report' + ('_fp32_head' if args.fp32_head else '') + ('_'+args.attention if args.attention!='flash_attention_2' else '') + '.json'
    result['attention']=args.attention
    result['dtype']=args.dtype
    (args.output/report_name).write_text(json.dumps(result,indent=2))
    print(result['phase'],[(r['max_abs_error'],r['mean_abs_error']) for r in result['records']],flush=True)
    if result['phase']!='PASS':
        raise RuntimeError('Response-prefix audit failed; do not relax thresholds automatically')


if __name__=='__main__':
    main()

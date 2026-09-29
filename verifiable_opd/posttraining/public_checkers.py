"""Frozen public benchmark adapters. No model generation or training access."""
import inspect
import random
import sys

from .checkers import ROOT, final_text


def public_score(row, response):
    checker=row['checker']
    if checker['type']=='aime_dynamic':
        from math_verify import parse,verify
        gold=parse(str(checker['answer']))
        if not gold:
            raise ValueError('Unparseable benchmark gold')
        parsed=parse(final_text(response))
        return {'score':float(bool(parsed) and verify(gold,parsed)),'parse_valid':bool(parsed)}
    import nltk
    from langdetect import DetectorFactory
    nltk.data.path.insert(0,str(ROOT/'data/nltk'))
    random.seed(17)
    DetectorFactory.seed=17
    if checker['type']=='ifeval':
        path=ROOT/'upstream/open-mopd-reference/evals/verifier/score_functions/instruction_following'
        sys.path.insert(0,str(path))
        from google_ifeval.instructions_registry import INSTRUCTION_DICT
    elif checker['type']=='ifbench':
        sys.path.insert(0,str(ROOT/'upstream/ifbench-reference'))
        from ifbench.instructions_registry import INSTRUCTION_DICT
    else:
        raise ValueError(f"Unknown benchmark checker: {checker['type']}")
    metadata=checker['metadata']
    ids=metadata['instruction_id_list']
    kwargs=metadata['kwargs']
    if not ids or len(ids)!=len(kwargs):
        raise ValueError('Invalid public instruction metadata')
    final=final_text(response)
    flags=[]
    for key,raw in zip(ids,kwargs,strict=True):
        instruction=INSTRUCTION_DICT[key](key)
        # Public aligned parquet uses a union of all keys with nulls. Retain
        # only documented arguments, as the upstream alignment wrapper does.
        allowed=set(instruction.get_instruction_args_keys() or [])
        arguments={k:int(v) if isinstance(v,float) and v.is_integer() else v
                   for k,v in (raw or {}).items() if v is not None and k in allowed}
        instruction.build_description(**arguments)
        if 'prompt' in (instruction.get_instruction_args() or {}):
            instruction.build_description(prompt=checker['original_prompt'])
        flags.append(bool(final) and bool(instruction.check_following(final)))
    return {'score':float(all(flags)),'instruction_flags':flags,
            'instruction_accuracy':sum(flags)/len(flags),'metric':'prompt_level_strict'}

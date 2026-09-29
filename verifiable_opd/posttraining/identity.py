"""Identity echoes are bound to the result, never reconstructed by row order."""
import hashlib
import json


def assign_sample_ids(batch, run_id, update):
    import numpy as np
    extras=[]
    for index,(original,domain) in enumerate(zip(batch.non_tensor_batch['extra_info'],
                                               batch.non_tensor_batch['data_source'],strict=True)):
        if domain not in {'math','if'} or original.get('domain')!=domain:
            raise ValueError('Invalid route before generation')
        extra=dict(original)
        extra['sample_id']=f'{run_id}/{update}/{index}/{domain}'
        extras.append(extra)
    batch.non_tensor_batch['extra_info']=np.array(extras,dtype=object)
    batch.non_tensor_batch['uid']=np.array([x['sample_id'] for x in extras],dtype=object)


def echo_identity(extra_info, domain, prompt_ids, response_ids):
    if not extra_info or 'example_id' not in extra_info:
        return {}
    if domain not in {'math','if'} or extra_info.get('domain') != domain:
        raise ValueError('Dataset/teacher route mismatch')
    if not extra_info.get('sample_id'):
        raise ValueError('Missing run/update/sample identity')
    return {'opd_example_id':extra_info['example_id'],'opd_domain':domain,
            'opd_sample_id':extra_info['sample_id'],
            'opd_prompt_sha256':hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest(),
            'opd_response_sha256':hashlib.sha256(json.dumps(response_ids).encode()).hexdigest()}


def validate_rollout_identity(output, input_batch):
    expected=[(r['example_id'],str(d),r['sample_id']) for r,d in zip(input_batch.non_tensor_batch['extra_info'],
        input_batch.non_tensor_batch['data_source'],strict=True)]
    if not {'opd_example_id','opd_domain','opd_sample_id'}.issubset(output.non_tensor_batch):
        raise ValueError('No result-bound identity echo; cannot assume asynchronous output order')
    actual=list(zip(output.non_tensor_batch['opd_example_id'],output.non_tensor_batch['opd_domain'],
                    output.non_tensor_batch['opd_sample_id'],strict=True))
    if len({r[2] for r in actual})!=len(actual):
        raise ValueError('Duplicate generated sample identity')
    if actual != expected:
        raise ValueError('Rollout identity/order mismatch before union')
    for index in range(len(output)):
        row=output.batch[index]
        plen=row['prompts'].shape[-1]
        prompt=row['prompts'][row['attention_mask'][:plen].bool()].tolist()
        response=row['responses'][row['response_mask'].bool()].tolist()
        for key,ids in [('opd_prompt_sha256',prompt),('opd_response_sha256',response)]:
            if output.non_tensor_batch[key][index] != hashlib.sha256(json.dumps(ids).encode()).hexdigest():
                raise ValueError(f'Corrupt {key} for {actual[index]}')

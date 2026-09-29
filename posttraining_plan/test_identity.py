import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'upstream/verl-v0.8.0'),str(ROOT)]
from verifiable_opd.posttraining.identity import echo_identity,validate_rollout_identity


class IdentityTests(unittest.TestCase):
    def fixtures(self):
        import numpy as np
        import torch
        from verl.protocol import DataProto
        extras=[{'example_id':f'id{i}','domain':d,'sample_id':f'run/1/{i}/{d}'} for i,d in enumerate(['math','if'])]
        source=DataProto.from_dict(tensors={'marker':torch.arange(2)},non_tensors={
            'extra_info':np.array(extras,dtype=object),'data_source':np.array(['math','if'],dtype=object)})
        echoes=[echo_identity(x,x['domain'],[1,2],[3,4]) for x in extras]
        result=DataProto.from_dict(tensors={'prompts':torch.tensor([[0,1,2]]*2),
            'responses':torch.tensor([[3,4,0]]*2),'response_mask':torch.tensor([[1,1,0]]*2),
            'attention_mask':torch.tensor([[0,1,1,1,1,0]]*2)},
            non_tensors={k:np.array([e[k] for e in echoes],dtype=object) for k in echoes[0]})
        return source,result

    def test_order_and_tokens_checked(self):
        source,result=self.fixtures()
        validate_rollout_identity(result,source)
        with self.assertRaises(ValueError):validate_rollout_identity(result[[1,0]],source)
        result.batch['responses'][0,0]=5
        with self.assertRaises(ValueError):validate_rollout_identity(result,source)

    def test_wrong_teacher_route(self):
        with self.assertRaises(ValueError):echo_identity({'example_id':'id','domain':'math'},'if',[1],[2])

    def test_sample_id_is_fresh_and_does_not_mutate_source_metadata(self):
        from verifiable_opd.posttraining.identity import assign_sample_ids
        source,_=self.fixtures()
        original=source.non_tensor_batch['extra_info'][0]
        assign_sample_ids(source,'new_run',2)
        self.assertEqual(original['sample_id'],'run/1/0/math')
        self.assertEqual(source.non_tensor_batch['uid'][0],'new_run/2/0/math')


if __name__=='__main__':unittest.main()

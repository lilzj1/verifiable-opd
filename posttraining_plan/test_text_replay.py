import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/'upstream/verl-v0.8.0'), str(ROOT)]
from verifiable_opd.posttraining.text_replay import TextReplayPool
from test_text_encode import TinyTokenizer


class TextReplayTests(unittest.TestCase):
    def pool(self, directory):
        path=Path(directory)/'experts.jsonl'
        rows=[{'example_id':f'{domain}{i}','domain':domain,'split':'train','kind':'expert',
               'messages':[{'role':'user','content':'question'}],'expert_response':'answer'}
              for domain in ('math','if') for i in range(4)]
        path.write_text('\n'.join(json.dumps(r) for r in rows))
        return TextReplayPool(path,TinyTokenizer())

    def test_deterministic_complete_answer_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            pool=self.pool(directory)
            first=pool.select(step=3,seed=17,token_budget=6)
            self.assertEqual(first,pool.select(step=3,seed=17,token_budget=6))
            self.assertEqual(first['tokens_by_domain'],{'math':3,'if':3})
            self.assertEqual(len(first['task_ids']),2)
            with self.assertRaises(ValueError):
                pool.select(step=3,seed=17,token_budget=4)

    def test_real_dataproto_and_shift_boundaries(self):
        import torch
        from verl.workers.utils.padding import left_right_2_no_padding
        tokenizer=TinyTokenizer()
        tokenizer.pad_token_id=99
        with tempfile.TemporaryDirectory() as directory:
            pool=self.pool(directory)
            selected=pool.select(step=3,seed=17,token_budget=6)
            batch=pool.build_batch(selected,tokenizer,max_prompt=5,max_response=4,temperature=1)
            self.assertEqual(batch.batch['prompts'][0].tolist(),[99,99,1,2,3])
            self.assertEqual(batch.batch['responses'][0].tolist(),[4,5,99,99])
            self.assertEqual(batch.batch['response_mask'][0].tolist(),[1,1,1,0])
            self.assertEqual(batch.batch['attention_mask'][0].tolist(),[0,0,1,1,1,1,1,1,0])
            self.assertEqual(int(batch.batch['response_mask'].sum()),6)
            self.assertEqual(batch.non_tensor_batch['data_source'].tolist(),['math','if'])
            nested=left_right_2_no_padding(batch.to_tensordict())
            self.assertEqual(nested['input_ids'].values().numel(),12)
            self.assertTrue(torch.isfinite(batch.batch['input_ids']).all())


if __name__=='__main__':
    unittest.main()

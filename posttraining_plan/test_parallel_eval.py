import json
from pathlib import Path
import tempfile
import unittest
from posttraining_plan.eval_text_parallel import merge
from verifiable_opd.posttraining.evaluation import canonical_sha
from verifiable_opd.posttraining.checkpoint_protocol import file_sha


class ParallelEvalTests(unittest.TestCase):
    def fixture(self,root):
        manifest=root/'dev.jsonl'
        rows=[dict(example_id=str(i),split='dev',domain='math' if i<4 else 'if',
                   checker={'type':'math' if i<4 else 'nemotron_if'},messages=[{'role':'user','content':str(i)}]) for i in range(8)]
        manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        output=root/'eval'
        for index in range(4):
            folder=output/f'shards/{index}'
            folder.mkdir(parents=True)
            (folder/'protocol.json').write_text(json.dumps(dict(shard_index=index,physical_gpu=str(index+1),
                manifest_sha256=file_sha(manifest),model_path='same_model',execution_shards=4,split='dev')))
            (folder/'summary.json').write_text(json.dumps(dict(phase='EVALUATION_SHARD_COMPLETE',wall_seconds=1)))
            records=[dict(example_id=row['example_id'],sample_index=0,domain=row['domain'],
                prompt_sha256=canonical_sha(row['messages']),evaluation={'score':1},finish_reason='stop',response_token_ids=[1])
                for row in rows[index::4]]
            (folder/'predictions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
        return manifest,output

    def test_complete_merge_is_full_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest,output=self.fixture(Path(tmp))
            result=merge(manifest,output,'dev')
            self.assertEqual(result['responses'],8)
            self.assertEqual(result['overall_domain_macro'],1)
            self.assertEqual(len((output/'predictions.jsonl').read_text().splitlines()),8)

    def test_missing_or_wrong_model_cannot_be_scored(self):
        for fault in ('missing','model'):
            with tempfile.TemporaryDirectory() as tmp:
                manifest,output=self.fixture(Path(tmp))
                if fault=='missing':
                    (output/'shards/3/predictions.jsonl').write_text('')
                else:
                    path=output/'shards/3/protocol.json'
                    config=json.loads(path.read_text())
                    config['model_path']='different_model'
                    path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):merge(manifest,output,'dev')
                self.assertFalse((output/'summary.json').exists())


if __name__=='__main__':unittest.main()

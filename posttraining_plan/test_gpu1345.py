import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import test_parallel_eval
from posttraining_plan.finish_text_project_1345 import check_devices, remap_command
from posttraining_plan.eval_text_parallel_1345 import merge

ROOT = Path(__file__).resolve().parent


class GPU1345Tests(unittest.TestCase):
    def test_busy_gpu2_is_ignored_but_selected_busy_gpu_rejected(self):
        gpu = '\n'.join(f'{i}, gpu{i}, 5' for i in range(8))
        self.assertEqual(set(check_devices(gpu, 'gpu2, 413417')), {'1','3','4','5'})
        for i in (1,3,4,5):
            with self.assertRaises(ValueError):
                check_devices(gpu, f'gpu{i}, 999')

    def test_only_pending_final_tests_remapped(self):
        cmd = ['python','posttraining_plan/eval_text_parallel.py','--split','test']
        for group in ('S1','S2','S3'):
            self.assertIn('posttraining_plan/eval_text_parallel_1345.py',remap_command(cmd,Path(group+'_test.log')))
        for name in ('S0_test','S1_train','S2_dev','S3_finish'):
            self.assertIsNone(remap_command(cmd,Path(name+'.log')))

    def test_source_diff_is_exactly_resource_mapping(self):
        original = (ROOT/'eval_text.py').read_text()
        self.assertEqual((ROOT/'eval_text_1345.py').read_text(),
                         original.replace("choices=['1','2','3','4']", "choices=['1','3','4','5']"))
        original = (ROOT/'eval_text_parallel.py').read_text()
        for a,b in [("str(index+1)","('1','3','4','5')[index]"),
                    ("physical_gpu='1,2,3,4'","physical_gpu='1,3,4,5'"),
                    ("!='1,2,3,4'","!='1,3,4,5'"),
                    ("enumerate(('1','2','3','4'))","enumerate(('1','3','4','5'))"),
                    ('posttraining_plan/eval_text.py','posttraining_plan/eval_text_1345.py')]:
            original = original.replace(a,b)
        self.assertEqual((ROOT/'eval_text_parallel_1345.py').read_text(),original)

    def fixture(self, tmp):
        manifest, output = test_parallel_eval.ParallelEvalTests().fixture(Path(tmp))
        for i,g in enumerate(('1','3','4','5')):
            p = output/f'shards/{i}/protocol.json'
            d = json.loads(p.read_text());d['physical_gpu']=g;p.write_text(json.dumps(d))
        return manifest,output

    def test_remapped_merge_preserves_coverage_and_scores(self):
        with TemporaryDirectory() as tmp:
            manifest,output = self.fixture(tmp)
            result = merge(manifest,output,'dev')
            self.assertEqual(result['responses'],8)
            self.assertEqual(result['overall_domain_macro'],1)
            self.assertEqual(json.loads((output/'protocol.json').read_text())['physical_gpu'],'1,3,4,5')

    def test_wrong_device_rejected(self):
        with TemporaryDirectory() as tmp:
            manifest,output = self.fixture(tmp)
            p=output/'shards/3/protocol.json';d=json.loads(p.read_text());d['physical_gpu']='2';p.write_text(json.dumps(d))
            with self.assertRaises(ValueError):merge(manifest,output,'dev')


if __name__=='__main__':unittest.main()

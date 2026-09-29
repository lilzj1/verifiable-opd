import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from test_text_encode import TinyTokenizer
from verifiable_opd.posttraining.sft_dataset import TextExpertSFTDataset


class SFTDatasetTests(unittest.TestCase):
    def create(self, root, **changes):
        row = dict(split='train', kind='expert', messages=[{'role':'user','content':'a\u2028b'}],
                   expert_response='answer')
        row.update(changes)
        path = Path(root)/'expert_train.jsonl'
        path.write_text(json.dumps(row, ensure_ascii=False)+'\n')
        return path

    def test_exact_mask_single_shift_and_unicode_line_separator(self):
        with TemporaryDirectory() as root:
            path = self.create(root)
            dataset = TextExpertSFTDataset(str(path), TinyTokenizer(),
                                          SimpleNamespace(pad_mode='no_padding', max_length=8))
            self.assertEqual(len(dataset), 1)
            row = dataset[0]
            self.assertEqual(row['input_ids'].tolist(), [1,2,3,4,5,99])
            self.assertEqual(row['loss_mask'].tolist(), [0,0,0,1,1,1])
            self.assertEqual(row['loss_mask'].roll(-1).tolist(), [0,0,1,1,1,0])

    def test_reject_eval_and_truncation(self):
        with TemporaryDirectory() as root:
            path = self.create(root, split='dev')
            with self.assertRaises(ValueError):
                TextExpertSFTDataset(str(path),TinyTokenizer(),SimpleNamespace(pad_mode='no_padding',max_length=8))
            path = self.create(root)
            with self.assertRaises(ValueError):
                TextExpertSFTDataset(str(path),TinyTokenizer(),SimpleNamespace(pad_mode='no_padding',max_length=5))


if __name__ == '__main__':
    unittest.main()

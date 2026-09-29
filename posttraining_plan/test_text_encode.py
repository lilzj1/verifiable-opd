import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from verifiable_opd.posttraining.text_encode import encode_expert, validate_messages


class TinyTokenizer:
    eos_token_id = 99
    eos_token = "<eos>"

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3] if messages[-1]["role"] == "user" else [1, 2, 3, 4, 5, 99, 8]


class EncodingTests(unittest.TestCase):
    def test_labels_and_eos(self):
        result = encode_expert(TinyTokenizer(), [{"role": "user", "content": "question"}], "answer")
        self.assertEqual(result.labels, [-100, -100, -100, 4, 5, 99])
        self.assertEqual(result.response_ids, [4, 5, 99])

    def test_no_truncation(self):
        with self.assertRaises(ValueError):
            encode_expert(TinyTokenizer(), [{"role": "user", "content": "q"}], "a", max_response=2)

    def test_label_leakage(self):
        with self.assertRaises(ValueError):
            validate_messages([{"role": "user", "content": "q", "ground_truth": "a"}])

    def test_injected_eos(self):
        with self.assertRaises(ValueError):
            encode_expert(TinyTokenizer(), [{"role": "user", "content": "q"}], "<eos>a")


if __name__ == "__main__":
    unittest.main()
